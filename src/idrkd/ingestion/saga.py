"""Compensating transaction for commit ingestion across independent stores."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Any, Callable, Protocol

from idrkd.ingestion.events import CommitEvent
from idrkd.ingestion.object_store import ArchiveResult
from idrkd.ingestion.pipeline import CommitIngestionPipeline, PreparedIngestion
from idrkd.ingestion.transaction_store import TransactionRecord
from idrkd.rag.vector_store import VectorRecord


class Archive(Protocol):
    def archive(self, event: CommitEvent, *, repo_root: Path, event_id: str) -> ArchiveResult: ...


class GraphStore(Protocol):
    def apply_schema(self) -> None: ...

    def upsert_parsed_file(self, parsed: Any) -> dict[str, int]: ...

    def snapshot_records(
        self, *, entity_ids: list[str], relation_ids: list[str]
    ) -> dict[str, list[dict[str, Any]]]: ...

    def compensate_records(
        self,
        *,
        entity_ids: list[str],
        relation_ids: list[str],
        snapshot: dict[str, list[dict[str, Any]]],
    ) -> None: ...


class VectorStore(Protocol):
    def upsert_records(self, records: list[VectorRecord]) -> int: ...

    def snapshot_records(self, record_ids: list[str]) -> list[dict[str, Any]]: ...

    def compensate_records(self, record_ids: list[str], snapshot: list[dict[str, Any]]) -> None: ...


class TransactionStore(Protocol):
    def begin(self, *, event_id: str, correlation_id: str, payload: dict[str, Any]) -> TransactionRecord: ...

    def stage(self, event_id: str, *, stage: str, plan: dict[str, Any]) -> None: ...

    def mark_stage(self, event_id: str, stage: str) -> None: ...

    def finish(
        self,
        event_id: str,
        *,
        status: str,
        stage: str,
        error: str | None = None,
        outbox: list[tuple[str, str, dict[str, Any]]] | None = None,
    ) -> None: ...

    def get(self, event_id: str) -> TransactionRecord | None: ...

    def repair_required(self, *, limit: int = 25) -> list[TransactionRecord]: ...


@dataclass(frozen=True)
class SagaResult:
    event_id: str
    status: str
    parsed_files: int = 0
    entities: int = 0
    relations: int = 0
    embeddings: int = 0
    error: str | None = None


class EventIngestionSaga:
    """Coordinate a logical transaction using durable before-images and compensation."""

    def __init__(
        self,
        *,
        archive: Archive,
        journal: TransactionStore,
        graph: GraphStore,
        vectors: VectorStore,
        repo_root: Callable[[CommitEvent], Path],
        pipeline: Callable[[Path], CommitIngestionPipeline],
    ) -> None:
        self._archive = archive
        self._journal = journal
        self._graph = graph
        self._vectors = vectors
        self._repo_root = repo_root
        self._pipeline = pipeline

    def execute(self, event: CommitEvent, *, correlation_id: str) -> SagaResult:
        event_id = ingestion_event_id(event)
        record = self._journal.begin(
            event_id=event_id,
            correlation_id=correlation_id,
            payload=_event_payload(event, correlation_id),
        )
        if record.status == "committed":
            return SagaResult(event_id, "duplicate")
        if record.plan and record.stage in {"graph_started", "graph_applied", "vector_started"}:
            try:
                self._compensate(record.plan)
            except Exception as exc:
                return self._record_repair_failure(event, event_id, exc)

        plan: dict[str, Any] = {}
        graph_started = False
        vector_started = False
        prepared: PreparedIngestion | None = None
        try:
            root = self._repo_root(event).resolve()
            archive = self._archive.archive(event, repo_root=root, event_id=event_id)
            self._journal.mark_stage(event_id, "archived")
            prepared = self._pipeline(root).prepare(event)
            entity_ids = _unique(
                entity.id for parsed in prepared.parsed_files for entity in parsed.entities
            )
            relation_ids = _unique(
                relation.id for parsed in prepared.parsed_files for relation in parsed.relations
            )
            vector_ids = _unique(record.id for record in prepared.vector_records)
            plan = {
                "archive": archive.as_dict(),
                "entity_ids": entity_ids,
                "relation_ids": relation_ids,
                "vector_ids": vector_ids,
                "graph_snapshot": self._graph.snapshot_records(
                    entity_ids=entity_ids, relation_ids=relation_ids
                ),
                "vector_snapshot": self._vectors.snapshot_records(vector_ids),
            }
            self._journal.stage(event_id, stage="staged", plan=plan)

            self._graph.apply_schema()
            self._journal.mark_stage(event_id, "graph_started")
            graph_started = True
            entities = 0
            relations = 0
            for parsed in prepared.parsed_files:
                counts = self._graph.upsert_parsed_file(parsed)
                entities += counts["entities"]
                relations += counts["relations"]
            self._journal.mark_stage(event_id, "graph_applied")

            self._journal.mark_stage(event_id, "vector_started")
            vector_started = True
            embeddings = self._vectors.upsert_records(list(prepared.vector_records))
            changed = {
                "schema_version": 1,
                "event_id": event_id,
                "correlation_id": correlation_id,
                "tenant_id": event.tenant_id,
                "repo_id": event.repo_id,
                "commit_sha": event.commit_sha,
                "entity_ids": entity_ids,
                "relation_ids": relation_ids,
                "archive_manifest": archive.manifest_key,
            }
            self._journal.finish(
                event_id,
                status="committed",
                stage="committed",
                outbox=[("entity-changed", event.repo_id, changed)],
            )
            return SagaResult(
                event_id,
                "committed",
                parsed_files=len(prepared.parsed_files),
                entities=entities,
                relations=relations,
                embeddings=embeddings,
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            compensation_errors: list[str] = []
            if plan and (graph_started or vector_started):
                compensation_errors = self._compensate(plan, collect_errors=True)
            status = "repair_required" if compensation_errors else "rolled_back"
            failure = {
                "schema_version": 1,
                "event_id": event_id,
                "correlation_id": correlation_id,
                "tenant_id": event.tenant_id,
                "repo_id": event.repo_id,
                "commit_sha": event.commit_sha,
                "error": error,
                "compensation_errors": compensation_errors,
            }
            outbox = [("ingestion-dlq", event.repo_id, failure)]
            if compensation_errors:
                outbox.append(("ingestion-repair", event.repo_id, failure))
            self._journal.finish(
                event_id,
                status=status,
                stage=status,
                error="; ".join([error, *compensation_errors]),
                outbox=outbox,
            )
            return SagaResult(event_id, status, error=error)

    def repair(self, record: TransactionRecord) -> SagaResult:
        try:
            errors = self._compensate(record.plan, collect_errors=True)
            if errors:
                raise RuntimeError("; ".join(errors))
            self._journal.finish(
                record.event_id,
                status="rolled_back",
                stage="repaired",
                error=record.error,
            )
            return SagaResult(record.event_id, "repaired")
        except Exception as exc:
            self._journal.finish(
                record.event_id,
                status="repair_required",
                stage="repair_failed",
                error=f"{type(exc).__name__}: {exc}",
            )
            return SagaResult(record.event_id, "repair_required", error=str(exc))

    def repair_pending(self, *, limit: int = 25) -> list[SagaResult]:
        return [self.repair(record) for record in self._journal.repair_required(limit=limit)]

    def _compensate(
        self, plan: dict[str, Any], *, collect_errors: bool = False
    ) -> list[str]:
        errors: list[str] = []
        try:
            self._vectors.compensate_records(plan["vector_ids"], plan["vector_snapshot"])
        except Exception as exc:
            errors.append(f"vector compensation: {type(exc).__name__}: {exc}")
        try:
            self._graph.compensate_records(
                entity_ids=plan["entity_ids"],
                relation_ids=plan["relation_ids"],
                snapshot=plan["graph_snapshot"],
            )
        except Exception as exc:
            errors.append(f"graph compensation: {type(exc).__name__}: {exc}")
        if errors and not collect_errors:
            raise RuntimeError("; ".join(errors))
        return errors

    def _record_repair_failure(
        self, event: CommitEvent, event_id: str, exc: Exception
    ) -> SagaResult:
        error = f"crash recovery compensation: {type(exc).__name__}: {exc}"
        payload = {
            "schema_version": 1,
            "event_id": event_id,
            "tenant_id": event.tenant_id,
            "repo_id": event.repo_id,
            "commit_sha": event.commit_sha,
            "error": error,
        }
        self._journal.finish(
            event_id,
            status="repair_required",
            stage="repair_required",
            error=error,
            outbox=[
                ("ingestion-dlq", event.repo_id, payload),
                ("ingestion-repair", event.repo_id, payload),
            ],
        )
        return SagaResult(event_id, "repair_required", error=error)


def ingestion_event_id(event: CommitEvent) -> str:
    value = "|".join(
        [event.tenant_id, event.repo_id, event.commit_sha, *sorted(event.changed_paths)]
    )
    return f"ing_{hashlib.sha256(value.encode()).hexdigest()[:32]}"


def _event_payload(event: CommitEvent, correlation_id: str) -> dict[str, Any]:
    payload = asdict(event)
    payload["changed_paths"] = list(event.changed_paths)
    payload["received_at"] = event.received_at.isoformat()
    payload["correlation_id"] = correlation_id
    return payload


def _unique(values: Any) -> list[str]:
    return list(dict.fromkeys(values))
