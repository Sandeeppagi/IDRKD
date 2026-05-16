# Spec: Neo4j Schema and Writer

## Goal

Persist typed Python ingestion records into Neo4j using idempotent `MERGE`
writes, tenant-scoped metadata, temporal fields, and typed labels.

## Graph Contract

Every code entity node must have:

- Base label `:CodeEntity`
- One typed label:
  - `:File`
  - `:Module`
  - `:Class`
  - `:Function`
  - `:Import`
  - `:Schema`
  - `:Document`
- `id`
- `tenant_id`
- `repo_id`
- `kind`
- `name`
- `qualified_name`
- `path`
- `start_line`
- `end_line`
- `content_hash`
- `language`
- `properties_json`
- `created_at`
- `updated_at`
- `lamport_clock`

Every relationship must:

- Use type `:RELATES_TO`
- Have stable `id`
- Have `tenant_id`, `repo_id`, `relation_type`
- Have `properties_json`
- Have `created_at`, `updated_at`, `lamport_clock`

## Schema Contract

Required uniqueness constraints:

- `CodeEntity.id`
- typed label `id` constraints for File, Module, Class, Function, Import,
  Schema, and Document
- `RELATES_TO.id`

Required indexes:

- `CodeEntity(tenant_id, repo_id)`
- typed label `(tenant_id, repo_id)` indexes
- `CodeEntity(kind)`
- `CodeEntity(qualified_name)`
- `CodeEntity(content_hash)`
- `CodeEntity(created_at)`
- `CodeEntity(updated_at)`
- `CodeEntity(lamport_clock)`

## Implementation

- `src/idrkd/graph/schema.cypher`
- `src/idrkd/graph/cypher.py`
- `src/idrkd/graph/writer.py`
- `tests/unit/test_graph_writer.py`

## Acceptance Criteria

- `Neo4jCodeGraphWriter.apply_schema()` applies schema DDL.
- `Neo4jCodeGraphWriter.upsert_parsed_file()` writes entities before
  relationships.
- Entity writes use `MERGE`.
- Relationship writes use `MERGE`.
- Repeating the same write does not duplicate nodes or relationships.
- New code entity nodes have additive typed labels such as
  `:CodeEntity:Function`.

## Verification

Automated:

```bash
uv run pytest tests/unit/test_graph_writer.py
```

Live smoke write:

```bash
uv run python - <<'PY'
from idrkd.graph.writer import Neo4jCodeGraphWriter
from idrkd.ingestion.python_extractor import parse_python_file

source = """
import os

def top_level(name: str) -> str:
    return name.upper()

class Worker:
    def run(self, item):
        return item
"""

parsed = parse_python_file(
    tenant_id="tenant-w1",
    repo_id="typed-label-smoke",
    path="src/example.py",
    source=source,
)

with Neo4jCodeGraphWriter("bolt://localhost:7687", "neo4j", "change-me") as writer:
    writer.apply_schema()
    writer.upsert_parsed_file(parsed)
    writer.upsert_parsed_file(parsed)
PY
```

Verify labels and relation count:

```cypher
MATCH (n:CodeEntity {tenant_id: 'tenant-w1', repo_id: 'typed-label-smoke'})
RETURN labels(n) AS labels, n.kind AS kind, count(*) AS count
ORDER BY kind;

MATCH (:CodeEntity {tenant_id: 'tenant-w1', repo_id: 'typed-label-smoke'})
      -[r:RELATES_TO]->
      (:CodeEntity {tenant_id: 'tenant-w1', repo_id: 'typed-label-smoke'})
RETURN count(r) AS code_relations;
```

Expected labels include:

- `["CodeEntity", "File"]`
- `["CodeEntity", "Import"]`
- `["CodeEntity", "Function"]`
- `["CodeEntity", "Class"]`
