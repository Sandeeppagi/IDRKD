# Week 2 Specs

Week 2 completes the ingestion MVP beyond Python-only parsing:

- Git webhook to commit-event serialization.
- Parser worker pipeline from changed paths to graph writes.
- JavaScript Tree-sitter extraction.
- JSON/CSV schema extraction.
- Markdown/document NER facade.
- Correlation-aware tracing and ingestion SLO checks.
- Lamport clock stamping for entity and relation writes.

## Spec Index

| Spec | Status | Primary Verification |
|---|---:|---|
| [Commit Ingestion Pipeline](commit-ingestion-pipeline.spec.md) | Implemented | `tests/unit/test_week2_ingestion.py` |
| [Schema and Document Extraction](schema-document-extraction.spec.md) | Implemented | `tests/unit/test_week2_ingestion.py` |
| [Tracing and SLO](tracing-slo.spec.md) | Implemented | `tests/unit/test_week2_ingestion.py` |
