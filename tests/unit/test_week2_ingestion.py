from dataclasses import dataclass
import hashlib
import hmac
import json
from pathlib import Path

import httpx

from idrkd.common.models import EntityKind, RelationType
from idrkd.ingestion.document_extractor import SpanBertNerExtractor, parse_document_file
from idrkd.ingestion.events import CommitEvent
from idrkd.ingestion.javascript_extractor import parse_javascript_file
from idrkd.ingestion.kafka import commit_event_from_json, commit_event_to_json
from idrkd.ingestion.pipeline import CommitIngestionPipeline
from idrkd.ingestion.schema_extractor import parse_schema_file
from idrkd.ingestion.slo import IngestionSlo, LamportClock, utc_now
from idrkd.ingestion.webhook import create_app
from idrkd.rag.embeddings import BgeM3EmbeddingAdapter
from idrkd.rag.vector_store import VectorRecord


JAVASCRIPT_SOURCE = """
import thing from "pkg";

export function topLevel(name) {
  return name;
}

class Worker extends BaseWorker {
  run(item) {
    return item;
  }
}
"""


def test_javascript_tree_sitter_extractor_emits_typed_records() -> None:
    parsed = parse_javascript_file(
        tenant_id="tenant-a",
        repo_id="repo-a",
        path="src/example.js",
        source=JAVASCRIPT_SOURCE,
    )

    kinds = [entity.kind for entity in parsed.entities]
    relation_types = {relation.relation_type for relation in parsed.relations}

    assert parsed.language == "javascript"
    assert kinds.count(EntityKind.FILE) == 1
    assert kinds.count(EntityKind.IMPORT) == 1
    assert kinds.count(EntityKind.FUNCTION) == 2
    assert kinds.count(EntityKind.CLASS) == 1
    assert RelationType.IMPORTS in relation_types
    assert RelationType.DEFINES in relation_types
    assert RelationType.CONTAINS in relation_types


def test_schema_and_document_extractors_cover_json_csv_and_markdown() -> None:
    json_schema = parse_schema_file(
        tenant_id="tenant-a",
        repo_id="repo-a",
        path="schemas/customer.json",
        source='{"customer": {"id": 1, "name": "Ada"}}',
    )
    csv_schema = parse_schema_file(
        tenant_id="tenant-a",
        repo_id="repo-a",
        path="schemas/customer.csv",
        source="id,name\n1,Ada\n",
    )
    document = parse_document_file(
        tenant_id="tenant-a",
        repo_id="repo-a",
        path="docs/readme.md",
        source="# Telstra Messaging API\nThis service calls Neo4j and Kafka.",
        ner=SpanBertNerExtractor(),
    )

    assert json_schema.entities[0].properties["fields"] == ["customer", "customer.id", "customer.name"]
    assert csv_schema.entities[0].properties["fields"] == ["id", "name"]
    assert document.entities[0].kind is EntityKind.DOCUMENT
    assert document.entities[0].properties["named_entities"]


def test_commit_event_serialization_preserves_correlation_id() -> None:
    event = CommitEvent.create(
        tenant_id="tenant-a",
        repo_id="repo-a",
        commit_sha="abc123",
        changed_paths=("src/example.py",),
    )

    payload = commit_event_to_json(event, correlation_id="corr-1")
    restored, correlation_id = commit_event_from_json(payload)

    assert restored == event
    assert correlation_id == "corr-1"


async def test_webhook_requires_valid_hmac_signature_when_secret_is_configured() -> None:
    class Producer:
        messages: list[tuple[str, bytes, bytes]]

        def __init__(self) -> None:
            self.messages = []

        def send(self, topic: str, key: bytes, value: bytes) -> None:
            self.messages.append((topic, key, value))

    producer = Producer()
    app = create_app(producer, webhook_secret="secret")
    payload = {
        "tenant_id": "tenant-a",
        "repo_id": "repo-a",
        "after": "abc123",
        "changed_paths": ["src/example.py"],
    }
    body = json.dumps(payload).encode("utf-8")
    signature = hmac.new(b"secret", body, hashlib.sha256).hexdigest()

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        unsigned = await client.post("/webhooks/git/commit", content=body)
        signed = await client.post(
            "/webhooks/git/commit",
            content=body,
            headers={"x-hub-signature-256": f"sha256={signature}"},
        )

    assert unsigned.status_code == 401
    assert signed.status_code == 200
    assert signed.json()["status"] == "accepted"
    assert len(producer.messages) == 1


async def test_webhook_returns_503_when_kafka_does_not_acknowledge() -> None:
    class FailedFuture:
        def get(self, timeout: int) -> None:
            raise RuntimeError("broker unavailable")

    class Producer:
        def send(self, topic: str, key: bytes, value: bytes) -> FailedFuture:
            return FailedFuture()

    app = create_app(Producer(), webhook_secret="secret")
    body = json.dumps(
        {
            "tenant_id": "tenant-a",
            "repo_id": "repo-a",
            "after": "abc123",
            "changed_paths": ["src/example.py"],
        }
    ).encode("utf-8")
    signature = hmac.new(b"secret", body, hashlib.sha256).hexdigest()

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/webhooks/git/commit",
            content=body,
            headers={"x-hub-signature-256": f"sha256={signature}"},
        )

    assert response.status_code == 503


def test_slo_and_lamport_clock() -> None:
    clock = LamportClock()
    assert clock.tick() == 1
    assert clock.merge(4) == 5
    now = utc_now()
    assert IngestionSlo().check(received_at=now, completed_at=now, file_count=1)


@dataclass
class RecordingWriter:
    applied_schema: bool = False
    entities: int = 0
    relations: int = 0

    def apply_schema(self) -> None:
        self.applied_schema = True

    def upsert_parsed_file(self, parsed) -> dict[str, int]:  # noqa: ANN001
        self.entities += len(parsed.entities)
        self.relations += len(parsed.relations)
        return {"entities": len(parsed.entities), "relations": len(parsed.relations)}


@dataclass
class RecordingEmbeddingSink:
    records: list[VectorRecord]

    def upsert_records(self, records: list[VectorRecord]) -> int:
        self.records.extend(records)
        return len(records)


def test_commit_ingestion_pipeline_routes_changed_files(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "example.py").write_text("import os\n\ndef run():\n    return os.getcwd()\n", encoding="utf-8")
    (tmp_path / "src" / "example.js").write_text(JAVASCRIPT_SOURCE, encoding="utf-8")
    writer = RecordingWriter()
    event = CommitEvent.create(
        tenant_id="tenant-a",
        repo_id="repo-a",
        commit_sha="abc123",
        changed_paths=("src/example.py", "src/example.js"),
    )
    pipeline = CommitIngestionPipeline(repo_root=tmp_path, writer=writer)  # type: ignore[arg-type]

    result = pipeline.process(event, correlation_id="corr-1")

    assert writer.applied_schema
    assert result.parsed_files == 2
    assert result.entities == writer.entities
    assert result.relations == writer.relations
    assert result.embeddings == 0
    assert result.slo_passed
    assert result.lamport_clock == 2


def test_commit_ingestion_pipeline_upserts_entity_embeddings(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "example.py").write_text("def run():\n    return 'ok'\n", encoding="utf-8")
    writer = RecordingWriter()
    embedding_sink = RecordingEmbeddingSink(records=[])
    event = CommitEvent.create(
        tenant_id="tenant-a",
        repo_id="repo-a",
        commit_sha="abc123",
        changed_paths=("src/example.py",),
    )
    pipeline = CommitIngestionPipeline(
        repo_root=tmp_path,
        writer=writer,  # type: ignore[arg-type]
        embedding_sink=embedding_sink,
        embeddings=BgeM3EmbeddingAdapter(dimensions=8),
        embedding_model_name="test-bge",
    )

    result = pipeline.process(event, correlation_id="corr-1")

    assert result.embeddings == writer.entities
    assert len(embedding_sink.records) == writer.entities
    assert {record.tenant_id for record in embedding_sink.records} == {"tenant-a"}
    assert {record.repo_id for record in embedding_sink.records} == {"repo-a"}
    assert all(len(record.embedding) == 8 for record in embedding_sink.records)
    assert all(record.embedding_model == "test-bge" for record in embedding_sink.records)
