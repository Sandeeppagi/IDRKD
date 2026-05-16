"""Cypher builders for idempotent Week 1 graph writes."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from idrkd.common.models import CodeEntity, CodeRelation


UPSERT_ENTITY = """
MERGE (n:CodeEntity {id: $id})
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
  n.properties = $properties,
  n.updated_at = datetime($updated_at)
RETURN n.id AS id
"""


UPSERT_RELATION = """
MATCH (source:CodeEntity {id: $source_id})
MATCH (target:CodeEntity {id: $target_id})
MERGE (source)-[r:RELATES_TO {id: $id}]->(target)
SET
  r.tenant_id = $tenant_id,
  r.repo_id = $repo_id,
  r.relation_type = $relation_type,
  r.properties = $properties
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
    return data


def relation_params(relation: CodeRelation) -> dict[str, Any]:
    data = asdict(relation)
    data["relation_type"] = relation.relation_type.value
    return data

