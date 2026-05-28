# Spec: Commit Ingestion Pipeline

## Goal

Process Git commit events into typed parsed files and Neo4j writes.

## Contract

Input:

- `CommitEvent`
- repo root
- changed paths
- correlation id

Output:

- `IngestionResult`
- schema applied before writes
- parsed entities and relations written through `Neo4jCodeGraphWriter`

Supported file types:

- `.py`
- `.js`, `.jsx`, `.mjs`, `.cjs`
- `.json`
- `.csv`
- `.md`, `.markdown`, `.txt`

## Implementation

- `src/idrkd/ingestion/events.py`
- `src/idrkd/ingestion/kafka.py`
- `src/idrkd/ingestion/webhook.py`
- `src/idrkd/ingestion/pipeline.py`
- `tests/unit/test_week2_ingestion.py`

## Acceptance Criteria

- Commit event serialization preserves `correlation_id`.
- Webhook payload can publish a commit event to Kafka producer interface.
- Parser pipeline routes changed paths by file extension.
- Writer schema is applied before graph writes.
- Result reports parsed file, entity, relation, SLO, and Lamport counts.

## Verification

```bash
uv run pytest tests/unit/test_week2_ingestion.py
```
