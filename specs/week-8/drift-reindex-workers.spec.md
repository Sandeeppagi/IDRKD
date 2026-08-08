# Spec: Drift Reindex Workers

## Goal

Implement real drift and re-index worker contracts for the W7-W8 drift phase,
using Neo4j for drift/stale writeback and pgvector for embedding comparisons.

## Contract

- `EntityChangedEvent` represents a tenant/repo scoped changed entity with a
  new textual description and content hash.
- `EntityDriftWorker.process(...)` fetches the previous pgvector embedding,
  embeds the new description, computes cosine drift, writes the updated
  embedding, updates Neo4j `drift_score`/`stale`, and enqueues a
  `ReindexRequest` when drift exceeds `ENTITY_DRIFT_THRESHOLD = 0.15`.
- `CommunityCentroidDriftWorker.process(...)` loads Neo4j community members,
  reads their pgvector embeddings, computes centroid shift, writes
  `centroid_drift`/`community_centroid`/`stale` to Neo4j, and queues all
  community members when drift exceeds `CENTROID_DRIFT_THRESHOLD = 0.10`.
- `ReindexWorker.process(...)` expands the changed entity through real Neo4j
  BFS, reads each entity, re-embeds it, upserts pgvector rows, and clears
  `stale` flags in Neo4j.
- `RedisReindexQueue` provides a Redis list-backed queue; `InMemoryReindexQueue`
  is the deterministic test adapter with the same interface.
- `RedisMcpReindexQueue` consumes MCP `enqueue_reindex` queue items from
  `idrkd:mcp:reindex`, preserving the MCP-facing queue while converting payloads
  to `ReindexRequest` objects for the worker.
- `idrkd.drift.cli` exposes `idrkd-drift-worker` commands:
  `reindex-loop`, `reindex-once`, `healthcheck`, and `live-smoke`.
- Docker Compose runs `reindex-worker` by default against Redis/Neo4j/Postgres,
  and exposes an optional `drift-worker` profile for drift-generated
  `idrkd:reindex` requests.

## Implementation

- `src/idrkd/drift/events.py`
- `src/idrkd/drift/scoring.py`
- `src/idrkd/drift/queue.py`
- `src/idrkd/drift/store.py`
- `src/idrkd/drift/workers.py`
- `src/idrkd/drift/cli.py`
- `src/idrkd/rag/vector_store.py`
- `docker/docker-compose.yml`
- `tests/unit/test_drift_reindex_workers.py`

## Acceptance Criteria

- Entity drift triggers when cosine distance exceeds `0.15`.
- Entity drift writes the replacement embedding and Neo4j drift properties.
- Community centroid drift queues every member when the centroid shift exceeds
  `0.10`.
- Re-index expands a bounded graph neighbourhood, writes pgvector rows, and
  clears stale flags.
- Redis and in-memory queues expose the same enqueue/dequeue contract.
- `reindex-worker` healthcheck verifies Redis, PostgreSQL, and Neo4j
  connectivity.
- A live smoke can enqueue through MCP, wait for the Dockerized worker to
  consume the queue item, verify queue depth returns to zero, confirm pgvector
  has a `source='reindex'` row, and confirm Neo4j `stale=false`.

## Verification

```bash
uv run pytest tests/unit/test_drift_reindex_workers.py
uv run mypy src/idrkd/drift
docker compose -f docker/docker-compose.yml up -d --build reindex-worker
docker exec docker-reindex-worker-1 idrkd-drift-worker healthcheck
uv run idrkd-drift-worker live-smoke --external-worker --mcp-base-url http://localhost:8080 --tenant-id tenant-live --repo-id week5-e2e
```

Live Docker evidence from 2026-08-05:

- `reindex-worker` started `idrkd-drift-worker reindex-loop` and polled
  `idrkd:mcp:reindex`.
- `live-smoke --external-worker` enqueued through MCP and returned
  `status=ok`, `queue_depth=0`, `external_worker=true`.
- Worker logs recorded processing
  `tenant=tenant-live repo=week5-e2e entity=ent_830163b93b3f05426b98a589`
  and wrote `embeddings=5`, `cleared_stale=5`.
- Neo4j returned `stale=false`, `reindexed=true` for the smoke entity.
- PostgreSQL returned a `knowledge_embeddings` row for the smoke entity with
  `source='reindex'` and `embedding_model='bge-m3'`.
