"""Import Graphify graph.json output into Neo4j.

Graphify's current Docker-installable CLI can emit a local graph.json but does
not expose the HLD's planned --neo4j-push flag. This module provides the bridge
IDRKD needs for Week 1 bootstrap testing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from neo4j import GraphDatabase


@dataclass(frozen=True)
class GraphifyImportBatch:
    nodes: list[dict[str, Any]]
    relationships: list[dict[str, Any]]
    graphify_node_count: int
    unique_graphify_node_count: int
    graphify_edge_count: int
    external_stub_count: int


def build_import_batch(
    graph: dict[str, Any],
    *,
    tenant_id: str,
    repo_id: str,
    source_label: str,
    imported_at: datetime | None = None,
) -> GraphifyImportBatch:
    """Convert Graphify JSON into Neo4j-ready node and relationship rows."""
    imported_at = imported_at or datetime.now(UTC)
    imported_at_iso = imported_at.isoformat()
    raw_nodes = graph.get("nodes", [])
    raw_links = graph.get("links", [])

    if not isinstance(raw_nodes, list) or not isinstance(raw_links, list):
        raise ValueError("Graphify graph must contain list-valued 'nodes' and 'links'")

    nodes_by_id: dict[str, dict[str, Any]] = {}
    for raw_node in raw_nodes:
        if not isinstance(raw_node, dict) or not raw_node.get("id"):
            continue
        graphify_id = str(raw_node["id"])
        nodes_by_id[graphify_id] = {
            "id": _node_id(tenant_id, repo_id, graphify_id),
            "graphify_id": graphify_id,
            "tenant_id": tenant_id,
            "repo_id": repo_id,
            "source": "graphify",
            "source_label": source_label,
            "label": _optional_str(raw_node.get("label")) or graphify_id,
            "file_type": _optional_str(raw_node.get("file_type")) or "unknown",
            "source_file": _optional_str(raw_node.get("source_file")),
            "source_location": _optional_str(raw_node.get("source_location")),
            "is_external": False,
            "created_at": imported_at_iso,
            "updated_at": imported_at_iso,
            "lamport_clock": 0,
        }

    external_ids = sorted(
        {
            endpoint
            for raw_link in raw_links
            if isinstance(raw_link, dict)
            for endpoint in (_optional_str(raw_link.get("source")), _optional_str(raw_link.get("target")))
            if endpoint and endpoint not in nodes_by_id
        }
    )
    for graphify_id in external_ids:
        nodes_by_id[graphify_id] = {
            "id": _node_id(tenant_id, repo_id, graphify_id),
            "graphify_id": graphify_id,
            "tenant_id": tenant_id,
            "repo_id": repo_id,
            "source": "graphify",
            "source_label": source_label,
            "label": graphify_id,
            "file_type": "external",
            "source_file": None,
            "source_location": None,
            "is_external": True,
            "created_at": imported_at_iso,
            "updated_at": imported_at_iso,
            "lamport_clock": 0,
        }

    relationships: list[dict[str, Any]] = []
    for raw_link in raw_links:
        if not isinstance(raw_link, dict):
            continue
        source = _optional_str(raw_link.get("source"))
        target = _optional_str(raw_link.get("target"))
        if not source or not target:
            continue
        relation = _optional_str(raw_link.get("relation")) or "related_to"
        row = {
            "id": _relationship_id(tenant_id, repo_id, raw_link),
            "tenant_id": tenant_id,
            "repo_id": repo_id,
            "source_id": _node_id(tenant_id, repo_id, source),
            "target_id": _node_id(tenant_id, repo_id, target),
            "relation": relation,
            "context": _optional_str(raw_link.get("context")),
            "confidence": _optional_str(raw_link.get("confidence")),
            "source_file": _optional_str(raw_link.get("source_file")),
            "source_location": _optional_str(raw_link.get("source_location")),
            "weight": _optional_float(raw_link.get("weight")),
            "created_at": imported_at_iso,
            "updated_at": imported_at_iso,
            "lamport_clock": 0,
        }
        relationships.append(row)

    return GraphifyImportBatch(
        nodes=list(nodes_by_id.values()),
        relationships=relationships,
        graphify_node_count=len(raw_nodes),
        unique_graphify_node_count=len(nodes_by_id) - len(external_ids),
        graphify_edge_count=len(raw_links),
        external_stub_count=len(external_ids),
    )


def import_graphify_json(
    graph_path: Path,
    *,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    tenant_id: str,
    repo_id: str,
    source_label: str,
    batch_size: int = 500,
) -> GraphifyImportBatch:
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    batch = build_import_batch(
        graph,
        tenant_id=tenant_id,
        repo_id=repo_id,
        source_label=source_label,
    )

    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    try:
        _wait_for_neo4j(driver)
        with driver.session() as session:
            session.execute_write(_create_schema)
            for chunk in _chunks(batch.nodes, batch_size):
                session.execute_write(_upsert_nodes, chunk)
            for chunk in _chunks(batch.relationships, batch_size):
                session.execute_write(_upsert_relationships, chunk)
    finally:
        driver.close()

    return batch


def _create_schema(tx: Any) -> None:
    tx.run(
        """
        CREATE CONSTRAINT graphify_node_id IF NOT EXISTS
        FOR (n:GraphifyNode)
        REQUIRE n.id IS UNIQUE
        """
    )
    tx.run(
        """
        CREATE CONSTRAINT graphify_relationship_id IF NOT EXISTS
        FOR ()-[r:GRAPHIFY_RELATION]-()
        REQUIRE r.id IS UNIQUE
        """
    )
    tx.run(
        """
        CREATE INDEX graphify_node_tenant_repo IF NOT EXISTS
        FOR (n:GraphifyNode)
        ON (n.tenant_id, n.repo_id)
        """
    )


def _upsert_nodes(tx: Any, rows: list[dict[str, Any]]) -> None:
    tx.run(
        """
        UNWIND $rows AS row
        MERGE (n:GraphifyNode {id: row.id})
        ON CREATE SET
          n.created_at = datetime(row.created_at),
          n.lamport_clock = row.lamport_clock
        SET
          n.graphify_id = row.graphify_id,
          n.tenant_id = row.tenant_id,
          n.repo_id = row.repo_id,
          n.source = row.source,
          n.source_label = row.source_label,
          n.label = row.label,
          n.file_type = row.file_type,
          n.source_file = row.source_file,
          n.source_location = row.source_location,
          n.is_external = row.is_external,
          n.updated_at = datetime(row.updated_at)
        """,
        rows=rows,
    ).consume()


def _upsert_relationships(tx: Any, rows: list[dict[str, Any]]) -> None:
    tx.run(
        """
        UNWIND $rows AS row
        MATCH (source:GraphifyNode {id: row.source_id})
        MATCH (target:GraphifyNode {id: row.target_id})
        MERGE (source)-[r:GRAPHIFY_RELATION {id: row.id}]->(target)
        ON CREATE SET
          r.created_at = datetime(row.created_at),
          r.lamport_clock = row.lamport_clock
        SET
          r.tenant_id = row.tenant_id,
          r.repo_id = row.repo_id,
          r.relation = row.relation,
          r.context = row.context,
          r.confidence = row.confidence,
          r.source_file = row.source_file,
          r.source_location = row.source_location,
          r.weight = row.weight,
          r.updated_at = datetime(row.updated_at)
        """,
        rows=rows,
    ).consume()


def _wait_for_neo4j(driver: Any, *, attempts: int = 30, sleep_seconds: float = 1.0) -> None:
    last_error: Exception | None = None
    for _ in range(attempts):
        try:
            driver.verify_connectivity()
            return
        except Exception as exc:  # pragma: no cover - exercised by Docker startup timing
            last_error = exc
            time.sleep(sleep_seconds)
    raise RuntimeError("Neo4j did not become available for Graphify import") from last_error


def _chunks(rows: Sequence[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [list(rows[index : index + size]) for index in range(0, len(rows), size)]


def _node_id(tenant_id: str, repo_id: str, graphify_id: str) -> str:
    return f"graphify:{tenant_id}:{repo_id}:{graphify_id}"


def _relationship_id(tenant_id: str, repo_id: str, raw_link: dict[str, Any]) -> str:
    payload = {
        "tenant_id": tenant_id,
        "repo_id": repo_id,
        "source": raw_link.get("source"),
        "target": raw_link.get("target"),
        "relation": raw_link.get("relation"),
        "context": raw_link.get("context"),
        "confidence": raw_link.get("confidence"),
        "source_file": raw_link.get("source_file"),
        "source_location": raw_link.get("source_location"),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return f"graphify-rel:{digest}"


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Import Graphify graph.json into Neo4j")
    parser.add_argument("--graph", required=True, type=Path)
    parser.add_argument("--neo4j-uri", default=os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    parser.add_argument("--neo4j-user", default=os.getenv("NEO4J_USER", "neo4j"))
    parser.add_argument("--neo4j-password", default=os.getenv("NEO4J_PASSWORD", "change-me"))
    parser.add_argument("--tenant-id", default=os.getenv("TENANT_ID", "default"))
    parser.add_argument("--repo-id", default=os.getenv("REPO_ID", "graphify-bootstrap"))
    parser.add_argument("--source-label", default=os.getenv("SOURCE_LABEL", "graphify-bootstrap"))
    args = parser.parse_args()

    batch = import_graphify_json(
        args.graph,
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
        tenant_id=args.tenant_id,
        repo_id=args.repo_id,
        source_label=args.source_label,
    )
    print(
        "Imported Graphify graph into Neo4j: "
        f"{len(batch.nodes)} Neo4j nodes "
        f"({batch.unique_graphify_node_count} unique Graphify nodes "
        f"from {batch.graphify_node_count} entries + {batch.external_stub_count} stubs), "
        f"{len(batch.relationships)} relationships "
        f"({batch.graphify_edge_count} Graphify edges)."
    )


if __name__ == "__main__":
    main()
