"""Cypher builders for idempotent Week 1 graph writes."""

from __future__ import annotations

from dataclasses import asdict
import json
from typing import Any

from idrkd.common.models import CodeEntity, CodeRelation, EntityKind


ENTITY_LABELS: dict[EntityKind, str] = {
    EntityKind.FILE: "File",
    EntityKind.MODULE: "Module",
    EntityKind.CLASS: "Class",
    EntityKind.FUNCTION: "Function",
    EntityKind.IMPORT: "Import",
    EntityKind.SCHEMA: "Schema",
    EntityKind.DOCUMENT: "Document",
}


UPSERT_ENTITY_TEMPLATE = """
MERGE (n:CodeEntity:{typed_label} {{id: $id}})
ON CREATE SET
  n.created_at = datetime($created_at),
  n.lamport_clock = $lamport_clock
SET
  n.tenant_id = $tenant_id,
  n.repo_id = $repo_id,
  n.kind = $kind,
  n.name = $name,
  n.qualified_name = $qualified_name,
  n.path = $path,
  n.start_line = $start_line,
  n.end_line = $end_line,
  n.content_hash = $content_hash,
  n.language = $language,
  n.properties_json = $properties_json,
  n.updated_at = datetime($updated_at)
RETURN n.id AS id
"""


UPSERT_RELATION = """
MATCH (source:CodeEntity {id: $source_id})
MATCH (target:CodeEntity {id: $target_id})
MERGE (source)-[r:RELATES_TO {id: $id}]->(target)
ON CREATE SET
  r.created_at = datetime($created_at),
  r.lamport_clock = $lamport_clock
SET
  r.tenant_id = $tenant_id,
  r.repo_id = $repo_id,
  r.relation_type = $relation_type,
  r.properties_json = $properties_json,
  r.updated_at = datetime($updated_at)
RETURN r.id AS id
"""


def entity_params(entity: CodeEntity) -> dict[str, Any]:
    data = asdict(entity)
    location = data.pop("location")
    data["kind"] = entity.kind.value
    data["path"] = location["path"]
    data["start_line"] = location["start_line"]
    data["end_line"] = location["end_line"]
    data["created_at"] = entity.created_at.isoformat()
    data["updated_at"] = entity.updated_at.isoformat()
    data["properties_json"] = json.dumps(entity.properties, sort_keys=True)
    data.pop("properties")
    return data


def typed_entity_label(kind: EntityKind) -> str:
    return ENTITY_LABELS[kind]


def upsert_entity_query(entity: CodeEntity) -> str:
    return UPSERT_ENTITY_TEMPLATE.format(typed_label=typed_entity_label(entity.kind))


def relation_params(relation: CodeRelation) -> dict[str, Any]:
    data = asdict(relation)
    data["relation_type"] = relation.relation_type.value
    data["created_at"] = relation.created_at.isoformat()
    data["updated_at"] = relation.updated_at.isoformat()
    data["properties_json"] = json.dumps(relation.properties, sort_keys=True)
    data.pop("properties")
    return data


# Week 5 read-only Graph Traversal Agent queries. Every value that varies per
# call is bound through a `$param`; the only text ever `.format()`-spliced
# into a template is a range bound that has been validated as a positive
# int, never a caller-supplied string.

BFS_NEIGHBORS_TEMPLATE = """
MATCH (start:CodeEntity {{id: $entity_id, tenant_id: $tenant_id, repo_id: $repo_id}})
MATCH path = (start)-[:RELATES_TO*1..{depth}]-(neighbor:CodeEntity)
WHERE neighbor.tenant_id = $tenant_id AND neighbor.repo_id = $repo_id
RETURN DISTINCT neighbor.id AS id, neighbor.kind AS kind, min(length(path)) AS distance
ORDER BY distance ASC, id ASC
LIMIT $limit
"""


SHORTEST_PATH_TEMPLATE = """
MATCH (source:CodeEntity {{id: $source_id, tenant_id: $tenant_id, repo_id: $repo_id}})
MATCH (target:CodeEntity {{id: $target_id, tenant_id: $tenant_id, repo_id: $repo_id}})
MATCH path = shortestPath((source)-[:RELATES_TO*..{max_hops}]-(target))
RETURN [node IN nodes(path) | node.id] AS node_ids, length(path) AS hop_count
"""


COMMUNITY_SUBGRAPH_QUERY = """
MATCH (n:CodeEntity {tenant_id: $tenant_id, repo_id: $repo_id, community_id: $community_id})
RETURN n.id AS id, n.kind AS kind, n.name AS name
ORDER BY id ASC
LIMIT $limit
"""


COMMUNITY_FOR_ENTITY_QUERY = """
MATCH (start:CodeEntity {id: $entity_id, tenant_id: $tenant_id, repo_id: $repo_id})
WITH start.community_id AS community_id
MATCH (n:CodeEntity {tenant_id: $tenant_id, repo_id: $repo_id, community_id: community_id})
RETURN n.id AS id, n.kind AS kind, n.name AS name
ORDER BY id ASC
LIMIT $limit
"""


def bfs_neighbors_query(*, depth: int) -> str:
    if depth < 1:
        raise ValueError("depth must be >= 1")
    return BFS_NEIGHBORS_TEMPLATE.format(depth=depth)


def bfs_neighbors_params(*, tenant_id: str, repo_id: str, entity_id: str, limit: int = 10) -> dict[str, Any]:
    return {"tenant_id": tenant_id, "repo_id": repo_id, "entity_id": entity_id, "limit": limit}


def shortest_path_query(*, max_hops: int) -> str:
    if max_hops < 1:
        raise ValueError("max_hops must be >= 1")
    return SHORTEST_PATH_TEMPLATE.format(max_hops=max_hops)


def shortest_path_params(*, tenant_id: str, repo_id: str, source_id: str, target_id: str) -> dict[str, Any]:
    return {"tenant_id": tenant_id, "repo_id": repo_id, "source_id": source_id, "target_id": target_id}


def community_subgraph_params(*, tenant_id: str, repo_id: str, community_id: int, limit: int = 25) -> dict[str, Any]:
    return {"tenant_id": tenant_id, "repo_id": repo_id, "community_id": community_id, "limit": limit}


def community_for_entity_params(*, tenant_id: str, repo_id: str, entity_id: str, limit: int = 25) -> dict[str, Any]:
    return {"tenant_id": tenant_id, "repo_id": repo_id, "entity_id": entity_id, "limit": limit}
