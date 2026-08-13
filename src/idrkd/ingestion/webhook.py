"""FastAPI webhook endpoint for commit ingestion."""

from __future__ import annotations

import hashlib
import hmac
import os
from uuid import uuid4

from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field, ValidationError, field_validator

from idrkd.ingestion.events import CommitEvent
from idrkd.ingestion.kafka import CommitEventProducer, ProducerLike


class CommitWebhookPayload(BaseModel):
    tenant_id: str = Field(default="default", min_length=1, max_length=128)
    repo_id: str = Field(min_length=1, max_length=256)
    commit_sha: str = Field(alias="after", min_length=1, max_length=128)
    changed_paths: list[str] = Field(min_length=1, max_length=10_000)

    @field_validator("changed_paths")
    @classmethod
    def validate_changed_paths(cls, paths: list[str]) -> list[str]:
        for path in paths:
            normalised = path.replace("\\", "/")
            if not path or normalised.startswith("/") or ".." in normalised.split("/"):
                raise ValueError("changed paths must be non-empty and repository-relative")
        return paths


def create_app(producer: ProducerLike, *, webhook_secret: str | None = None) -> FastAPI:
    app = FastAPI(title="IDRKD ingestion webhook")
    commit_producer = CommitEventProducer(producer)
    secret = webhook_secret or os.getenv("IDRKD_WEBHOOK_SECRET")

    @app.post("/webhooks/git/commit")
    async def receive_commit(
        request: Request,
        x_correlation_id: str | None = Header(default=None),
        x_hub_signature_256: str | None = Header(default=None),
        x_idrkd_signature_sha256: str | None = Header(default=None),
    ) -> dict[str, str]:
        body = await request.body()
        if secret:
            _verify_signature(
                body,
                secret=secret,
                signature=x_hub_signature_256 or x_idrkd_signature_sha256,
            )
        try:
            payload = CommitWebhookPayload.model_validate_json(body)
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.errors()) from exc
        correlation_id = x_correlation_id or str(uuid4())
        event = CommitEvent.create(
            tenant_id=payload.tenant_id,
            repo_id=payload.repo_id,
            commit_sha=payload.commit_sha,
            changed_paths=tuple(payload.changed_paths),
        )
        published = commit_producer.publish(event, correlation_id=correlation_id)
        wait = getattr(published, "get", None)
        if callable(wait):
            try:
                wait(timeout=10)
            except Exception as exc:
                raise HTTPException(status_code=503, detail="commit event broker unavailable") from exc
        return {"status": "accepted", "correlation_id": correlation_id}

    return app


def _verify_signature(body: bytes, *, secret: str, signature: str | None) -> None:
    if not signature:
        raise HTTPException(status_code=401, detail="missing webhook signature")
    expected = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    supplied = signature.removeprefix("sha256=").strip()
    if not hmac.compare_digest(expected, supplied):
        raise HTTPException(status_code=401, detail="invalid webhook signature")
