"""Synthetic schema corpus adapters for MCP evaluation.

The JSONL fixtures under ``eval/synthetic_schemas`` are intentionally simple
data files. This module turns them into executable MCP-TaskBench cases and an
in-memory backend so the fixtures participate in evaluation runs.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, cast

from idrkd.evaluation.taskbench import McpTask
from idrkd.mcp.backends import EntityRecord, InMemoryMcpStateStore, SearchResult
from idrkd.mcp.tools import McpToolRegistry


@dataclass(frozen=True)
class SyntheticSchemaCorpus:
    schemas: tuple[dict[str, Any], ...]
    conflicts: tuple[dict[str, Any], ...]

    @property
    def schema_count(self) -> int:
        return len(self.schemas)

    @property
    def conflict_count(self) -> int:
        return len(self.conflicts)


class SyntheticSchemaGraphBackend:
    """Small read backend backed by synthetic schema fixtures."""

    def __init__(self, entities: dict[str, EntityRecord]) -> None:
        self._entities = entities

    def get_entity(self, *, tenant_id: str, repo_id: str, entity_id: str) -> EntityRecord | None:
        entity = self._entities.get(entity_id)
        if entity is None or entity.properties.get("repo_id") != repo_id:
            return None
        return entity if entity.properties.get("tenant_id") == tenant_id else None

    def search_code(
        self, *, tenant_id: str, repo_id: str, query_text: str, limit: int = 10
    ) -> list[SearchResult]:
        terms = [term for term in query_text.lower().split() if term]
        hits = []
        for entity in self._entities.values():
            if entity.properties.get("tenant_id") != tenant_id or entity.properties.get("repo_id") != repo_id:
                continue
            haystack = " ".join(
                [
                    entity.name,
                    entity.qualified_name,
                    entity.path,
                    " ".join(str(field) for field in entity.properties.get("fields", [])),
                ]
            ).lower()
            score = sum(1 for term in terms if term in haystack)
            if score > 0:
                hits.append(SearchResult(entity.id, float(score), "synthetic_schema", entity.name, entity.kind))
        return sorted(hits, key=lambda hit: (-hit.score, hit.entity_id))[:limit]

    def downstream_impact(
        self, *, tenant_id: str, repo_id: str, entity_id: str, limit: int = 25
    ) -> list[SearchResult]:
        entity = self.get_entity(tenant_id=tenant_id, repo_id=repo_id, entity_id=entity_id)
        if entity is None:
            return []
        domain = str(entity.properties.get("domain", ""))
        return [
            SearchResult(other.id, 1.0, "synthetic_schema", other.name, other.kind)
            for other in self._entities.values()
            if other.id != entity.id
            and other.properties.get("tenant_id") == tenant_id
            and other.properties.get("repo_id") == repo_id
            and other.properties.get("domain") == domain
        ][:limit]

    def salience(self, *, tenant_id: str, repo_id: str, entity_id: str) -> float | None:
        entity = self.get_entity(tenant_id=tenant_id, repo_id=repo_id, entity_id=entity_id)
        if entity is None:
            return None
        return float(len(entity.properties.get("fields", [])))

    def centroid_drift(self, *, tenant_id: str, repo_id: str, community_id: str) -> float | None:
        values = [
            float(entity.properties.get("synthetic_drift", 0.0))
            for entity in self._entities.values()
            if entity.properties.get("tenant_id") == tenant_id
            and entity.properties.get("repo_id") == repo_id
            and entity.properties.get("domain") == community_id
        ]
        return max(values) if values else None

    def stale_entities(self, *, tenant_id: str, repo_id: str, limit: int = 50) -> list[EntityRecord]:
        return [
            entity
            for entity in self._entities.values()
            if entity.properties.get("tenant_id") == tenant_id
            and entity.properties.get("repo_id") == repo_id
            and entity.properties.get("stale")
        ][:limit]


def load_synthetic_schema_corpus(
    *,
    schemas_path: Path = Path("eval/synthetic_schemas/schemas.jsonl"),
    conflicts_path: Path = Path("eval/synthetic_schemas/conflicts.jsonl"),
) -> SyntheticSchemaCorpus:
    return SyntheticSchemaCorpus(
        schemas=tuple(_read_jsonl(schemas_path)),
        conflicts=tuple(_read_jsonl(conflicts_path)),
    )


def build_synthetic_schema_registry(
    corpus: SyntheticSchemaCorpus,
    *,
    tenant_id: str = "default",
    repo_id: str = "synthetic-schemas",
) -> McpToolRegistry:
    entities = _build_schema_entities(corpus, tenant_id=tenant_id, repo_id=repo_id)
    state = InMemoryMcpStateStore(conflicts=_build_conflict_state(corpus, tenant_id=tenant_id, repo_id=repo_id))
    return McpToolRegistry(
        principal_tenant_id=tenant_id,
        graph_backend=cast(Any, SyntheticSchemaGraphBackend(entities)),
        runtime_state=state,
    )


def build_synthetic_schema_tasks(
    corpus: SyntheticSchemaCorpus,
    *,
    tenant_id: str = "default",
    repo_id: str = "synthetic-schemas",
) -> list[McpTask]:
    tasks: list[McpTask] = []
    for schema in corpus.schemas:
        schema_id = str(schema["id"])
        tasks.append(
            McpTask(
                id=f"synthetic-schema-diff-{schema_id}",
                category="schema_conformance",
                prompt=(
                    f"Compare the baseline and injected variant for schema {schema['name']} "
                    f"using left_id {schema_id}:baseline, right_id {schema_id}:variant, "
                    f"tenant_id {tenant_id}, and repo_id {repo_id}."
                ),
                expected_tool="schema_diff",
                arguments={
                    "tenant_id": tenant_id,
                    "repo_id": repo_id,
                    "left_id": f"{schema_id}:baseline",
                    "right_id": f"{schema_id}:variant",
                },
                expected_result_keys=["left_found", "right_found", "added", "removed", "unchanged"],
            )
        )
    for conflict in corpus.conflicts:
        conflict_id = str(conflict["id"])
        tasks.append(
            McpTask(
                id=f"synthetic-conflict-open-{conflict_id}",
                category="conflict_resolution",
                prompt=(
                    f"Inspect the injected {conflict['conflict_type']} conflict on {conflict['field']} "
                    f"using conflict_id {conflict_id}, tenant_id {tenant_id}, and repo_id {repo_id}."
                ),
                expected_tool="reconcile",
                arguments={"tenant_id": tenant_id, "repo_id": repo_id, "conflict_id": conflict_id},
                expected_result_keys=["conflict", "recommendation"],
            )
        )
        tasks.append(
            McpTask(
                id=f"synthetic-conflict-resolve-{conflict_id}",
                category="conflict_resolution",
                prompt=(
                    f"Persist the oracle resolution {conflict['oracle_resolution']} for synthetic "
                    f"conflict_id {conflict_id}, tenant_id {tenant_id}, and repo_id {repo_id}."
                ),
                expected_tool="resolve_conflict",
                arguments={
                    "tenant_id": tenant_id,
                    "repo_id": repo_id,
                    "conflict_id": conflict_id,
                    "resolution": str(conflict["oracle_resolution"]),
                },
                expected_result_keys=["resolved", "conflict"],
            )
        )
    return tasks


def _build_schema_entities(
    corpus: SyntheticSchemaCorpus,
    *,
    tenant_id: str,
    repo_id: str,
) -> dict[str, EntityRecord]:
    entities: dict[str, EntityRecord] = {}
    for index, schema in enumerate(corpus.schemas, start=1):
        schema_id = str(schema["id"])
        fields = [str(field["name"]) for field in schema["fields"]]
        variant_fields = [*fields, f"{schema['domain']}_audit_marker"]
        for suffix, active_fields in (("baseline", fields), ("variant", variant_fields)):
            entity_id = f"{schema_id}:{suffix}"
            entities[entity_id] = EntityRecord(
                id=entity_id,
                kind="schema",
                name=str(schema["name"]),
                qualified_name=f"{schema['domain']}.{schema['name']}.{suffix}",
                path=f"eval/synthetic_schemas/{schema['name']}.{suffix}.json",
                content_hash=f"synthetic-schema-{index:02d}-{suffix}",
                properties={
                    "tenant_id": tenant_id,
                    "repo_id": repo_id,
                    "fields": active_fields,
                    "domain": schema["domain"],
                    "primary_key": schema["primary_key"],
                    "stale": suffix == "variant",
                    "synthetic_drift": 0.05 * index,
                },
            )
    return entities


def _build_conflict_state(
    corpus: SyntheticSchemaCorpus,
    *,
    tenant_id: str,
    repo_id: str,
) -> dict[str, dict[str, Any]]:
    conflicts: dict[str, dict[str, Any]] = {}
    for conflict in corpus.conflicts:
        conflict_id = str(conflict["id"])
        conflicts[conflict_id] = {
            "conflict_id": conflict_id,
            "tenant_id": tenant_id,
            "repo_id": repo_id,
            "status": "open",
            "resolution": None,
            "recommendation": conflict["oracle_resolution"],
            "schema_id": conflict["schema_id"],
            "field": conflict["field"],
            "conflict_type": conflict["conflict_type"],
            "left_source": conflict["left_source"],
            "right_source": conflict["right_source"],
        }
    return conflicts


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
