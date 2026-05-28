"""Kafka serialization helpers for ingestion events."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import json
from typing import Protocol

from idrkd.ingestion.events import CommitEvent


class ProducerLike(Protocol):
    def send(self, topic: str, key: bytes, value: bytes) -> object:
        ...


def commit_event_to_json(event: CommitEvent, *, correlation_id: str) -> bytes:
    payload = asdict(event)
    payload["changed_paths"] = list(event.changed_paths)
    payload["received_at"] = event.received_at.isoformat()
    payload["correlation_id"] = correlation_id
    return json.dumps(payload, sort_keys=True).encode("utf-8")


def commit_event_from_json(payload: bytes) -> tuple[CommitEvent, str]:
    data = json.loads(payload.decode("utf-8"))
    correlation_id = data.pop("correlation_id")
    data["changed_paths"] = tuple(data["changed_paths"])
    data["received_at"] = datetime.fromisoformat(data["received_at"])
    return CommitEvent(**data), correlation_id


class CommitEventProducer:
    def __init__(self, producer: ProducerLike, topic: str = "commit-events") -> None:
        self._producer = producer
        self._topic = topic

    def publish(self, event: CommitEvent, *, correlation_id: str) -> object:
        return self._producer.send(
            self._topic,
            key=event.repo_id.encode("utf-8"),
            value=commit_event_to_json(event, correlation_id=correlation_id),
        )
