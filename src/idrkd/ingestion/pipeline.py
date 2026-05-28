"""Commit-event to graph-write ingestion pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from idrkd.common.models import ParsedFile
from idrkd.graph.writer import Neo4jCodeGraphWriter
from idrkd.ingestion.document_extractor import parse_document_file
from idrkd.ingestion.events import CommitEvent
from idrkd.ingestion.javascript_extractor import parse_javascript_file
from idrkd.ingestion.python_extractor import parse_python_file
from idrkd.ingestion.schema_extractor import parse_schema_file
from idrkd.ingestion.slo import IngestionSlo, LamportClock, utc_now
from idrkd.observability.tracing import traced_span


@dataclass(frozen=True)
class IngestionResult:
    parsed_files: int
    entities: int
    relations: int
    slo_passed: bool
    lamport_clock: int


class CommitIngestionPipeline:
    def __init__(
        self,
        *,
        repo_root: Path,
        writer: Neo4jCodeGraphWriter,
        slo: IngestionSlo | None = None,
        clock: LamportClock | None = None,
    ) -> None:
        self._repo_root = repo_root
        self._writer = writer
        self._slo = slo or IngestionSlo()
        self._clock = clock or LamportClock()

    def process(self, event: CommitEvent, *, correlation_id: str) -> IngestionResult:
        with traced_span(
            "ingestion.commit",
            correlation_id=correlation_id,
            tenant_id=event.tenant_id,
            repo_id=event.repo_id,
            file_count=len(event.changed_paths),
        ):
            self._writer.apply_schema()
            parsed_files = [
                parsed
                for path in event.changed_paths
                if (parsed := self._parse_path(event, path)) is not None
            ]
            entities = 0
            relations = 0
            for parsed in parsed_files:
                current_clock = self._clock.tick()
                _stamp_lamport(parsed, current_clock)
                counts = self._writer.upsert_parsed_file(parsed)
                entities += counts["entities"]
                relations += counts["relations"]

            completed_at = utc_now()
            return IngestionResult(
                parsed_files=len(parsed_files),
                entities=entities,
                relations=relations,
                slo_passed=self._slo.check(
                    received_at=event.received_at,
                    completed_at=completed_at,
                    file_count=len(event.changed_paths),
                ),
                lamport_clock=self._clock.value,
            )

    def _parse_path(self, event: CommitEvent, path: str) -> ParsedFile | None:
        full_path = self._repo_root / path
        if not full_path.exists() or not full_path.is_file():
            return None
        source = full_path.read_text(encoding="utf-8")
        suffix = full_path.suffix.lower()
        if suffix == ".py":
            return parse_python_file(tenant_id=event.tenant_id, repo_id=event.repo_id, path=path, source=source)
        if suffix in {".js", ".jsx", ".mjs", ".cjs"}:
            return parse_javascript_file(tenant_id=event.tenant_id, repo_id=event.repo_id, path=path, source=source)
        if suffix in {".json", ".csv"}:
            return parse_schema_file(tenant_id=event.tenant_id, repo_id=event.repo_id, path=path, source=source)
        if suffix in {".md", ".markdown", ".txt"}:
            return parse_document_file(tenant_id=event.tenant_id, repo_id=event.repo_id, path=path, source=source)
        return None


def _stamp_lamport(parsed: ParsedFile, value: int) -> None:
    for entity in parsed.entities:
        object.__setattr__(entity, "lamport_clock", value)
    for relation in parsed.relations:
        object.__setattr__(relation, "lamport_clock", value)
