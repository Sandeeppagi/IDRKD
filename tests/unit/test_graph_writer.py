from pathlib import Path

from idrkd.graph.cypher import UPSERT_RELATION, relation_params
from idrkd.graph.writer import _read_cypher_statements
from idrkd.ingestion.python_extractor import parse_python_file


def test_relation_params_include_temporal_fields_for_idempotent_merge() -> None:
    parsed = parse_python_file(
        tenant_id="tenant-a",
        repo_id="repo-a",
        path="src/example.py",
        source="import os\n",
    )

    params = relation_params(parsed.relations[0])

    assert params["created_at"].endswith("+00:00")
    assert params["updated_at"].endswith("+00:00")
    assert params["lamport_clock"] == 0
    assert params["properties_json"] == "{}"
    assert "properties" not in params
    assert "ON CREATE SET" in UPSERT_RELATION
    assert "MERGE (source)-[r:RELATES_TO {id: $id}]->(target)" in UPSERT_RELATION


def test_schema_reader_splits_cypher_statements(tmp_path: Path) -> None:
    schema_path = tmp_path / "schema.cypher"
    schema_path.write_text(
        """
        CREATE CONSTRAINT one IF NOT EXISTS FOR (n:Node) REQUIRE n.id IS UNIQUE;

        CREATE INDEX two IF NOT EXISTS FOR (n:Node) ON (n.name);
        """,
        encoding="utf-8",
    )

    statements = _read_cypher_statements(schema_path)

    assert statements == [
        "CREATE CONSTRAINT one IF NOT EXISTS FOR (n:Node) REQUIRE n.id IS UNIQUE",
        "CREATE INDEX two IF NOT EXISTS FOR (n:Node) ON (n.name)",
    ]
