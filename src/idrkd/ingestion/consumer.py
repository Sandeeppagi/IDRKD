"""Kafka consumer and outbox publisher for event-driven ingestion."""

from __future__ import annotations

import json
import hashlib
import logging
from collections.abc import Iterator
from typing import Any, Protocol

from idrkd.ingestion.kafka import ProducerLike, commit_event_from_json
from idrkd.ingestion.saga import EventIngestionSaga, SagaResult
from idrkd.ingestion.transaction_store import OutboxMessage


LOGGER = logging.getLogger("idrkd.ingestion.consumer")


class ConsumerLike(Protocol):
    def __iter__(self) -> Iterator[Any]: ...

    def commit(self) -> None: ...


class OutboxStore(Protocol):
    def pending_outbox(self, event_id: str, *, limit: int = 100) -> list[OutboxMessage]: ...

    def mark_published(self, message_id: str) -> None: ...


class OutboxPublisher:
    def __init__(self, store: OutboxStore, producer: ProducerLike) -> None:
        self._store = store
        self._producer = producer

    def publish_event(self, event_id: str) -> int:
        count = 0
        for message in self._store.pending_outbox(event_id):
            result = self._producer.send(
                message.topic,
                key=message.message_key.encode("utf-8"),
                value=json.dumps(message.payload, sort_keys=True).encode("utf-8"),
            )
            wait = getattr(result, "get", None)
            if callable(wait):
                wait(timeout=30)
            self._store.mark_published(message.id)
            count += 1
        return count

    def publish_invalid(self, value: bytes, error: Exception) -> None:
        digest = hashlib.sha256(value).hexdigest()
        payload = {
            "schema_version": 1,
            "event_id": f"invalid_{digest[:32]}",
            "error": f"{type(error).__name__}: {error}",
            "raw_sha256": digest,
        }
        result = self._producer.send(
            "ingestion-dlq",
            key=digest.encode("ascii"),
            value=json.dumps(payload, sort_keys=True).encode("utf-8"),
        )
        wait = getattr(result, "get", None)
        if callable(wait):
            wait(timeout=30)


class CommitEventConsumer:
    """Commit Kafka offsets only after the saga and its outbox are durable."""

    def __init__(
        self,
        consumer: ConsumerLike,
        saga: EventIngestionSaga,
        publisher: OutboxPublisher,
    ) -> None:
        self._consumer = consumer
        self._saga = saga
        self._publisher = publisher

    def run(self, *, max_messages: int = 0) -> None:
        processed = 0
        for message in self._consumer:
            try:
                result = self.process_value(message.value)
            except (KeyError, TypeError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
                self._publisher.publish_invalid(message.value, exc)
                self._consumer.commit()
                LOGGER.error("invalid commit event sent to DLQ error=%s", exc)
                processed += 1
                if max_messages and processed >= max_messages:
                    return
                continue
            self._consumer.commit()
            processed += 1
            LOGGER.info("ingestion event completed event_id=%s status=%s", result.event_id, result.status)
            if max_messages and processed >= max_messages:
                return

    def process_value(self, value: bytes) -> SagaResult:
        event, correlation_id = commit_event_from_json(value)
        result = self._saga.execute(event, correlation_id=correlation_id)
        self._publisher.publish_event(result.event_id)
        return result
