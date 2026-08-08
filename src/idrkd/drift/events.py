"""Drift and re-index event contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass(frozen=True)
class EntityChangedEvent:
    tenant_id: str
    repo_id: str
    entity_id: str
    content_hash: str
    description: str
    community_id: str | None = None
    changed_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(frozen=True)
class ReindexRequest:
    tenant_id: str
    repo_id: str
    entity_id: str
    reason: str
    depth: int = 2
    requested_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def payload(self) -> dict[str, object]:
        return {
            "tenant_id": self.tenant_id,
            "repo_id": self.repo_id,
            "entity_id": self.entity_id,
            "reason": self.reason,
            "depth": self.depth,
            "requested_at": self.requested_at.isoformat(),
        }
