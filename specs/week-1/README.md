# Week 1 Specs

Week 1 establishes the local foundation for IDRKD:

- A full Docker Compose developer stack.
- Graphify bootstrap data loaded into Neo4j for early integration testing.
- PostgreSQL with pgvector and an HNSW index.
- Tree-sitter Python ingestion into typed records.
- Idempotent Neo4j writes with typed labels, temporal properties, and tenant scoping.

## Spec Index

| Spec | Status | Primary Verification |
|---|---:|---|
| [Graphify Bootstrap](graphify-bootstrap.spec.md) | Implemented | Neo4j Telstra counts |
| [Docker Stack](docker-stack.spec.md) | Implemented | Compose status and service health checks |
| [pgvector HNSW](pgvector-hnsw.spec.md) | Implemented | Postgres index inspection |
| [Python Ingestion](python-ingestion.spec.md) | Implemented | Unit tests and Tree-sitter dependency check |
| [Neo4j Schema and Writer](neo4j-schema-writer.spec.md) | Implemented | Live typed-label smoke write |

## Common Verification Commands

```bash
uv run pytest tests/unit
docker compose -f docker/docker-compose.yml ps --all
```
