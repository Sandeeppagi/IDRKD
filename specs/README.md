# IDRKD Specs

This directory holds implementation specs that convert the HLD/LLD and project
plan into verifiable engineering contracts.

Each spec should answer:

- What capability is being built?
- What data/API/graph contract must be preserved?
- What acceptance criteria define done?
- Which tests and live commands verify the result?
- Which code paths implement the contract?

## Done Rule

A spec is complete only when it has:

- A written contract.
- Automated tests where practical.
- A live verification command for infrastructure-backed behavior.
- Evidence recorded in the spec or a linked reference note.

## Layout

```text
specs/
  week-1/
    README.md
    graphify-bootstrap.spec.md
    docker-stack.spec.md
    pgvector-hnsw.spec.md
    python-ingestion.spec.md
    neo4j-schema-writer.spec.md
  week-2/
    README.md
    commit-ingestion-pipeline.spec.md
    schema-document-extraction.spec.md
    tracing-slo.spec.md
  week-3/
    README.md
    embedding-vector-retrieval.spec.md
    hybrid-rrf-retrieval.spec.md
    graph-analytics.spec.md
```
