"""FastAPI webhook endpoint for commit ingestion."""

from __future__ import annotations

from uuid import uuid4

from fastapi import FastAPI, Header
from pydantic import BaseModel, Field

from idrkd.ingestion.events import CommitEvent
from idrkd.ingestion.kafka import CommitEventProducer, ProducerLike


class CommitWebhookPayload(BaseModel):
    tenant_id: str = "default"
    repo_id: str
    commit_sha: str = Field(alias="after")
    changed_paths: list[str]


def create_app(producer: ProducerLike) -> FastAPI:
    app = FastAPI(title="IDRKD ingestion webhook")
    commit_producer = CommitEventProducer(producer)

    @app.post("/webhooks/git/commit")
    def receive_commit(
        payload: CommitWebhookPayload,
        x_correlation_id: str | None = Header(default=None),
    ) -> dict[str, str]:
        correlation_id = x_correlation_id or str(uuid4())
        event = CommitEvent.create(
            tenant_id=payload.tenant_id,
            repo_id=payload.repo_id,
            commit_sha=payload.commit_sha,
            changed_paths=tuple(payload.changed_paths),
        )
        commit_producer.publish(event, correlation_id=correlation_id)
        return {"status": "accepted", "correlation_id": correlation_id}

    return app
