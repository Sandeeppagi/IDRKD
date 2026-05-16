"""Neo4j writer for parsed IDRKD code entities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from neo4j import GraphDatabase

from idrkd.common.models import CodeEntity, CodeRelation, ParsedFile
from idrkd.graph.cypher import UPSERT_ENTITY, UPSERT_RELATION, entity_params, relation_params


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
                session.execute_write(_run_statement, UPSERT_ENTITY, entity_params(entity))
        return len(entities)

    def upsert_relations(self, relations: tuple[CodeRelation, ...]) -> int:
        with self._driver.session() as session:
            for relation in relations:
                session.execute_write(_run_statement, UPSERT_RELATION, relation_params(relation))
        return len(relations)


def _read_cypher_statements(path: Path) -> list[str]:
    return [
        statement.strip()
        for statement in path.read_text(encoding="utf-8").split(";")
        if statement.strip()
    ]


def _run_statement(tx: Any, statement: str, params: dict[str, Any] | None = None) -> None:
    tx.run(statement, params or {}).consume()
