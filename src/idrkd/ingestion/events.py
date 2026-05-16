"""Kafka event contracts for the ingestion MVP."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime


@dataclass(frozen=True)
class CommitEvent:
    schema_version: int
    tenant_id: str
    repo_id: str
    commit_sha: str
    changed_paths: tuple[str, ...]
    received_at: datetime

    @classmethod
    def create(
        cls,
        *,
        tenant_id: str,
        repo_id: str,
        commit_sha: str,
        changed_paths: tuple[str, ...],
    ) -> "CommitEvent":
        return cls(
            schema_version=1,
            tenant_id=tenant_id,
            repo_id=repo_id,
            commit_sha=commit_sha,
            changed_paths=changed_paths,
            received_at=datetime.now(UTC),
        )

