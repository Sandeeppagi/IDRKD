# Spec: Tracing and SLO

## Goal

Propagate correlation metadata through ingestion and enforce the Week 2
freshness SLO contract.

## Contract

Tracing:

- all commit pipeline work runs inside an OpenTelemetry span
- span attributes include `correlation_id`, `tenant_id`, `repo_id`, and
  changed file count

SLO:

- freshness p95 target is represented as `IngestionSlo`
- default budget is 5 seconds for at most 20 files

Conflict ordering:

- `LamportClock` supports local `tick` and remote `merge`
- pipeline stamps parsed entities and relations with the current Lamport value

## Implementation

- `src/idrkd/observability/tracing.py`
- `src/idrkd/ingestion/slo.py`
- `src/idrkd/ingestion/pipeline.py`
- `tests/unit/test_week2_ingestion.py`

## Acceptance Criteria

- Correlation id is accepted from webhook or generated.
- Commit serialization includes correlation id.
- Pipeline returns `slo_passed`.
- Lamport clock increments once per parsed file.

## Verification

```bash
uv run pytest tests/unit/test_week2_ingestion.py
```
