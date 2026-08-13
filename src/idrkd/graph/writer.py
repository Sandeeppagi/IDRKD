"""Neo4j writer for parsed IDRKD code entities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from neo4j import GraphDatabase

from idrkd.common.models import CodeEntity, CodeRelation, ParsedFile
from idrkd.graph.cypher import (
    ENTITY_LABELS,
    UPSERT_RELATION,
    entity_params,
    relation_params,
    upsert_entity_query,
)


SCHEMA_PATH = Path(__file__).with_name("schema.cypher")


class Neo4jCodeGraphWriter:
    """Persist typed parser records to Neo4j using idempotent MERGE writes."""

    def __init__(self, uri: str, user: str, password: str) -> None:
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        self._driver.close()

    def __enter__(self) -> Neo4jCodeGraphWriter:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def apply_schema(self, schema_path: Path = SCHEMA_PATH) -> None:
        statements = _read_cypher_statements(schema_path)
        with self._driver.session() as session:
            for statement in statements:
                session.execute_write(_run_statement, statement)

    def upsert_parsed_file(self, parsed: ParsedFile) -> dict[str, int]:
        entity_count = self.upsert_entities(parsed.entities)
        relation_count = self.upsert_relations(parsed.relations)
        return {"entities": entity_count, "relations": relation_count}

    def upsert_entities(self, entities: tuple[CodeEntity, ...]) -> int:
        with self._driver.session() as session:
            for entity in entities:
                session.execute_write(_run_statement, upsert_entity_query(entity), entity_params(entity))
        return len(entities)

    def upsert_relations(self, relations: tuple[CodeRelation, ...]) -> int:
        with self._driver.session() as session:
            for relation in relations:
                session.execute_write(_run_statement, UPSERT_RELATION, relation_params(relation))
        return len(relations)

    def snapshot_records(
        self,
        *,
        entity_ids: list[str],
        relation_ids: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        with self._driver.session() as session:
            entities = session.execute_read(_snapshot_entities, entity_ids)
            relations = session.execute_read(_snapshot_relations, relation_ids)
        return {"entities": entities, "relations": relations}

    def compensate_records(
        self,
        *,
        entity_ids: list[str],
        relation_ids: list[str],
        snapshot: dict[str, list[dict[str, Any]]],
    ) -> None:
        existing_ids = {
            str(entity["properties"]["id"])
            for entity in snapshot.get("entities", [])
        }
        new_entity_ids = [entity_id for entity_id in entity_ids if entity_id not in existing_ids]
        with self._driver.session() as session:
            session.execute_write(_delete_relations, relation_ids)
            session.execute_write(_delete_entities, new_entity_ids)
            for entity in snapshot.get("entities", []):
                session.execute_write(_restore_entity, entity)
            for relation in snapshot.get("relations", []):
                session.execute_write(_restore_relation, relation)


def _read_cypher_statements(path: Path) -> list[str]:
    return [
        statement.strip()
        for statement in path.read_text(encoding="utf-8").split(";")
        if statement.strip()
    ]


def _run_statement(tx: Any, statement: str, params: dict[str, Any] | None = None) -> None:
    tx.run(statement, params or {}).consume()


def _snapshot_entities(tx: Any, entity_ids: list[str]) -> list[dict[str, Any]]:
    rows = tx.run(
        "MATCH (n:CodeEntity) WHERE n.id IN $ids RETURN labels(n) AS labels, properties(n) AS properties",
        ids=entity_ids,
    )
    return [_json_safe_record(record) for record in rows]


def _snapshot_relations(tx: Any, relation_ids: list[str]) -> list[dict[str, Any]]:
    rows = tx.run(
        "MATCH (source:CodeEntity)-[r:RELATES_TO]->(target:CodeEntity) "
        "WHERE r.id IN $ids RETURN source.id AS source_id, target.id AS target_id, "
        "properties(r) AS properties",
        ids=relation_ids,
    )
    return [_json_safe_record(record) for record in rows]


def _delete_relations(tx: Any, relation_ids: list[str]) -> None:
    tx.run("MATCH ()-[r:RELATES_TO]->() WHERE r.id IN $ids DELETE r", ids=relation_ids).consume()


def _delete_entities(tx: Any, entity_ids: list[str]) -> None:
    tx.run("MATCH (n:CodeEntity) WHERE n.id IN $ids DETACH DELETE n", ids=entity_ids).consume()


def _restore_entity(tx: Any, entity: dict[str, Any]) -> None:
    labels = set(entity["labels"])
    typed_labels = {label for label in labels if label != "CodeEntity"}
    if len(typed_labels) != 1 or next(iter(typed_labels)) not in set(ENTITY_LABELS.values()):
        raise ValueError("Cannot restore entity with an unknown Neo4j label")
    label = next(iter(typed_labels))
    tx.run(
        f"MERGE (n:CodeEntity:{label} {{id: $id}}) SET n = $properties "  # fixed allowlist
        "SET n.created_at = datetime($properties.created_at), "
        "n.updated_at = datetime($properties.updated_at)",
        id=entity["properties"]["id"],
        properties=entity["properties"],
    ).consume()


def _restore_relation(tx: Any, relation: dict[str, Any]) -> None:
    tx.run(
        "MATCH (source:CodeEntity {id: $source_id}), (target:CodeEntity {id: $target_id}) "
        "CREATE (source)-[r:RELATES_TO]->(target) SET r = $properties "
        "SET r.created_at = datetime($properties.created_at), "
        "r.updated_at = datetime($properties.updated_at)",
        source_id=relation["source_id"],
        target_id=relation["target_id"],
        properties=relation["properties"],
    ).consume()


def _json_safe_record(record: Any) -> dict[str, Any]:
    return cast(dict[str, Any], _json_safe(dict(record)))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "iso_format"):
        return value.iso_format()
    return value
