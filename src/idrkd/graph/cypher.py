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
