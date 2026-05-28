"""Week 2 ingestion SLO and Lamport clock helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime


@dataclass(frozen=True)
class IngestionSlo:
    freshness_budget_seconds: float = 5.0
    max_files: int = 20

    def check(self, *, received_at: datetime, completed_at: datetime, file_count: int) -> bool:
        elapsed = (completed_at - received_at).total_seconds()
        return file_count <= self.max_files and elapsed <= self.freshness_budget_seconds


class LamportClock:
    def __init__(self, value: int = 0) -> None:
        self.value = value

    def tick(self) -> int:
        self.value += 1
        return self.value

    def merge(self, remote_value: int) -> int:
        self.value = max(self.value, remote_value) + 1
        return self.value


def utc_now() -> datetime:
    return datetime.now(UTC)
