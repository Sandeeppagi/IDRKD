from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from idrkd.ingestion.consumer import CommitEventConsumer, OutboxPublisher
from idrkd.ingestion.events import CommitEvent
from idrkd.ingestion.kafka import commit_event_to_json
from idrkd.ingestion.object_store import ArchiveResult, MinioSourceArchive
from idrkd.ingestion.pipeline import CommitIngestionPipeline
from idrkd.ingestion.saga import EventIngestionSaga
from idrkd.ingestion.transaction_store import OutboxMessage, TransactionRecord
from idrkd.rag.embeddings import BgeM3EmbeddingAdapter
from idrkd.rag.vector_store import InMemoryVectorStore, VectorRecord


class FakeJournal:
    def __init__(self) -> None:
        self.records: dict[str, TransactionRecord] = {}
        self.outbox: list[OutboxMessage] = []
        self.published: list[str] = []
        self.stages: list[str] = []

    def begin(self, *, event_id: str, correlation_id: str, payload: dict[str, Any]) -> TransactionRecord:
        del correlation_id
        if event_id not in self.records:
            self.records[event_id] = TransactionRecord(event_id, "processing", "received", payload, {}, None)
        return self.records[event_id]

    def stage(self, event_id: str, *, stage: str, plan: dict[str, Any]) -> None:
        record = self.records[event_id]
        self.records[event_id] = TransactionRecord(event_id, "processing", stage, record.payload, plan, None)
        self.stages.append(stage)

    def mark_stage(self, event_id: str, stage: str) -> None:
        record = self.records[event_id]
        self.records[event_id] = TransactionRecord(event_id, "processing", stage, record.payload, record.plan, None)
        self.stages.append(stage)

    def finish(
        self,
        event_id: str,
        *,
        status: str,
        stage: str,
        error: str | None = None,
        outbox: list[tuple[str, str, dict[str, Any]]] | None = None,
    ) -> None:
        record = self.records[event_id]
        self.records[event_id] = TransactionRecord(event_id, status, stage, record.payload, record.plan, error)
        for index, (topic, key, payload) in enumerate(outbox or []):
            self.outbox.append(OutboxMessage(f"out-{event_id}-{index}", event_id, topic, key, payload))

    def get(self, event_id: str) -> TransactionRecord | None:
        return self.records.get(event_id)

    def repair_required(self, *, limit: int = 25) -> list[TransactionRecord]:
        return [record for record in self.records.values() if record.status == "repair_required"][:limit]

    def pending_outbox(self, event_id: str, *, limit: int = 100) -> list[OutboxMessage]:
        return [item for item in self.outbox if item.event_id == event_id and item.id not in self.published][:limit]

    def mark_published(self, message_id: str) -> None:
        self.published.append(message_id)


class FakeArchive:
    def archive(self, event: CommitEvent, *, repo_root: Path, event_id: str) -> ArchiveResult:
        assert repo_root.is_dir()
        return ArchiveResult("archive", f"commits/{event.repo_id}/{event_id}.json", ())


class FakeGraph:
    def __init__(self, *, fail_compensation: bool = False) -> None:
        self.entities: dict[str, str] = {}
        self.relations: dict[str, str] = {}
        self.schema_applied = False
        self.fail_compensation = fail_compensation
        self.compensations = 0

    def apply_schema(self) -> None:
        self.schema_applied = True

    def upsert_parsed_file(self, parsed: Any) -> dict[str, int]:
        self.entities.update({entity.id: entity.content_hash for entity in parsed.entities})
        self.relations.update({relation.id: relation.relation_type.value for relation in parsed.relations})
        return {"entities": len(parsed.entities), "relations": len(parsed.relations)}

    def snapshot_records(
        self, *, entity_ids: list[str], relation_ids: list[str]
    ) -> dict[str, list[dict[str, Any]]]:
        return {
            "entities": [
                {"labels": ["CodeEntity", "File"], "properties": {"id": key, "value": self.entities[key]}}
                for key in entity_ids
                if key in self.entities
            ],
            "relations": [
                {"properties": {"id": key, "value": self.relations[key]}}
                for key in relation_ids
                if key in self.relations
            ],
        }

    def compensate_records(
        self,
        *,
        entity_ids: list[str],
        relation_ids: list[str],
        snapshot: dict[str, list[dict[str, Any]]],
    ) -> None:
        self.compensations += 1
        if self.fail_compensation:
            raise RuntimeError("neo4j unavailable")
        for key in entity_ids:
            self.entities.pop(key, None)
        for key in relation_ids:
            self.relations.pop(key, None)
        for item in snapshot["entities"]:
            self.entities[item["properties"]["id"]] = item["properties"]["value"]
        for item in snapshot["relations"]:
            self.relations[item["properties"]["id"]] = item["properties"]["value"]


class FailAfterWriteVectorStore(InMemoryVectorStore):
    def __init__(self) -> None:
        super().__init__()
        self.fail = True
        self.compensations = 0

    def upsert_records(self, records: list[VectorRecord]) -> int:
        count = super().upsert_records(records)
        if self.fail:
            raise RuntimeError("pgvector unavailable after partial write")
        return count

    def compensate_records(self, record_ids: list[str], snapshot: list[dict[str, Any]]) -> None:
        self.compensations += 1
        self.fail = False
        super().compensate_records(record_ids, snapshot)


def _event(tmp_path: Path) -> tuple[CommitEvent, Path]:
    root = tmp_path / "repo-a"
    (root / "src").mkdir(parents=True)
    (root / "src" / "example.py").write_text("def run():\n    return 'ok'\n", encoding="utf-8")
    return (
        CommitEvent.create(
            tenant_id="tenant-a",
            repo_id="repo-a",
            commit_sha="abc123",
            changed_paths=("src/example.py",),
        ),
        root,
    )


def _saga(
    root: Path,
    journal: FakeJournal,
    graph: FakeGraph,
    vectors: InMemoryVectorStore,
) -> EventIngestionSaga:
    def pipeline(repo_root: Path) -> CommitIngestionPipeline:
        return CommitIngestionPipeline(
            repo_root=repo_root,
            writer=graph,  # type: ignore[arg-type]
            embedding_sink=vectors,
            embeddings=BgeM3EmbeddingAdapter(dimensions=8),
            embedding_model_name="test-bge",
        )

    return EventIngestionSaga(
        archive=FakeArchive(),
        journal=journal,
        graph=graph,
        vectors=vectors,
        repo_root=lambda _event: root,
        pipeline=pipeline,
    )


def test_saga_stages_and_commits_all_stores_once(tmp_path: Path) -> None:
    event, root = _event(tmp_path)
    journal = FakeJournal()
    graph = FakeGraph()
    vectors = InMemoryVectorStore()
    saga = _saga(root, journal, graph, vectors)

    result = saga.execute(event, correlation_id="corr-1")
    duplicate = saga.execute(event, correlation_id="corr-1")

    assert result.status == "committed"
    assert result.entities > 0 and result.embeddings == result.entities
    assert duplicate.status == "duplicate"
    assert journal.stages == ["archived", "staged", "graph_started", "graph_applied", "vector_started"]
    assert [message.topic for message in journal.outbox] == ["entity-changed"]
    assert graph.schema_applied


def test_partial_vector_failure_rolls_back_graph_and_vectors_and_routes_dlq(tmp_path: Path) -> None:
    event, root = _event(tmp_path)
    journal = FakeJournal()
    graph = FakeGraph()
    vectors = FailAfterWriteVectorStore()

    result = _saga(root, journal, graph, vectors).execute(event, correlation_id="corr-1")

    assert result.status == "rolled_back"
    assert graph.entities == {}
    assert vectors.snapshot_records([record.id for record in vectors._records]) == []
    assert graph.compensations == 1
    assert vectors.compensations == 1
    assert [message.topic for message in journal.outbox] == ["ingestion-dlq"]


def test_failed_compensation_creates_repair_and_dlq_messages(tmp_path: Path) -> None:
    event, root = _event(tmp_path)
    journal = FakeJournal()
    graph = FakeGraph(fail_compensation=True)
    vectors = FailAfterWriteVectorStore()

    result = _saga(root, journal, graph, vectors).execute(event, correlation_id="corr-1")

    assert result.status == "repair_required"
    assert [message.topic for message in journal.outbox] == ["ingestion-dlq", "ingestion-repair"]
    assert "graph compensation" in (journal.records[result.event_id].error or "")


class FakeObjectClient:
    def __init__(self) -> None:
        self.bucket = False
        self.objects: dict[str, bytes] = {}
        self.metadata: dict[str, dict[str, str]] = {}
        self.puts = 0

    def bucket_exists(self, bucket_name: str) -> bool:
        return self.bucket

    def make_bucket(self, bucket_name: str) -> None:
        self.bucket = True

    def stat_object(self, bucket_name: str, object_name: str) -> object:
        if object_name not in self.objects:
            raise KeyError(object_name)
        return type("Stat", (), {"metadata": self.metadata[object_name]})()

    def put_object(self, bucket_name: str, object_name: str, data: Any, length: int, content_type: str, metadata: Any = None) -> object:
        del bucket_name, content_type
        self.objects[object_name] = data.read(length)
        self.metadata[object_name] = {
            "x-amz-meta-sha256": metadata["sha256"],
        }
        self.puts += 1
        return object()


def test_minio_archive_is_content_addressed_immutable_and_tracks_deletes(tmp_path: Path) -> None:
    event, root = _event(tmp_path)
    event = CommitEvent(
        event.schema_version,
        event.tenant_id,
        event.repo_id,
        event.commit_sha,
        (*event.changed_paths, "src/deleted.py"),
        event.received_at,
    )
    client = FakeObjectClient()
    archive = MinioSourceArchive(client)

    first = archive.archive(event, repo_root=root, event_id="event-1")
    second = archive.archive(event, repo_root=root, event_id="event-1")

    assert first == second
    assert client.puts == 2  # one content blob plus one manifest
    assert first.files[1].deleted
    with pytest.raises(ValueError, match="repository-relative"):
        bad = CommitEvent(event.schema_version, event.tenant_id, event.repo_id, event.commit_sha, ("../secret",), event.received_at)
        archive.archive(bad, repo_root=root, event_id="event-2")


def test_minio_archive_rejects_oversized_and_hash_mismatched_objects(tmp_path: Path) -> None:
    event, root = _event(tmp_path)
    with pytest.raises(ValueError, match="archive limit"):
        MinioSourceArchive(FakeObjectClient(), max_file_bytes=2).archive(
            event, repo_root=root, event_id="event-1"
        )

    client = FakeObjectClient()
    archive = MinioSourceArchive(client)
    archive.archive(event, repo_root=root, event_id="event-1")
    manifest = next(key for key in client.metadata if key.startswith("commits/"))
    client.metadata[manifest]["x-amz-meta-sha256"] = "tampered"
    with pytest.raises(RuntimeError, match="hash mismatch"):
        archive.archive(event, repo_root=root, event_id="event-1")


@dataclass
class Message:
    value: bytes


class FakeConsumer:
    def __init__(self, value: bytes) -> None:
        self.messages = iter([Message(value)])
        self.commits = 0

    def __iter__(self):  # noqa: ANN204
        return self.messages

    def commit(self) -> None:
        self.commits += 1


class FakeProducer:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.messages: list[str] = []

    def send(self, topic: str, key: bytes, value: bytes) -> object:
        del key, value
        if self.fail:
            raise RuntimeError("kafka publish failed")
        self.messages.append(topic)
        return object()


def test_consumer_commits_offset_only_after_outbox_publish(tmp_path: Path) -> None:
    event, root = _event(tmp_path)
    payload = commit_event_to_json(event, correlation_id="corr-1")
    journal = FakeJournal()
    saga = _saga(root, journal, FakeGraph(), InMemoryVectorStore())
    consumer = FakeConsumer(payload)
    producer = FakeProducer()

    CommitEventConsumer(consumer, saga, OutboxPublisher(journal, producer)).run(max_messages=1)

    assert consumer.commits == 1
    assert producer.messages == ["entity-changed"]


def test_consumer_does_not_commit_when_outbox_publish_fails(tmp_path: Path) -> None:
    event, root = _event(tmp_path)
    consumer = FakeConsumer(commit_event_to_json(event, correlation_id="corr-1"))
    journal = FakeJournal()
    worker = CommitEventConsumer(
        consumer,
        _saga(root, journal, FakeGraph(), InMemoryVectorStore()),
        OutboxPublisher(journal, FakeProducer(fail=True)),
    )

    with pytest.raises(RuntimeError, match="kafka publish failed"):
        worker.run(max_messages=1)

    assert consumer.commits == 0


def test_consumer_routes_malformed_records_to_dlq_and_advances_offset(tmp_path: Path) -> None:
    _event_value, root = _event(tmp_path)
    consumer = FakeConsumer(b"not-json")
    journal = FakeJournal()
    producer = FakeProducer()
    worker = CommitEventConsumer(
        consumer,
        _saga(root, journal, FakeGraph(), InMemoryVectorStore()),
        OutboxPublisher(journal, producer),
    )

    worker.run(max_messages=1)

    assert consumer.commits == 1
    assert producer.messages == ["ingestion-dlq"]
