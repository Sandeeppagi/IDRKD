from dataclasses import dataclass
from pathlib import Path

from idrkd.common.models import EntityKind, RelationType
from idrkd.ingestion.document_extractor import SpanBertNerExtractor, parse_document_file
from idrkd.ingestion.events import CommitEvent
from idrkd.ingestion.javascript_extractor import parse_javascript_file
from idrkd.ingestion.kafka import commit_event_from_json, commit_event_to_json
from idrkd.ingestion.pipeline import CommitIngestionPipeline
from idrkd.ingestion.schema_extractor import parse_schema_file
from idrkd.ingestion.slo import IngestionSlo, LamportClock, utc_now


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
    assert result.slo_passed
    assert result.lamport_clock == 2
