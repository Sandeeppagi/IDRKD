# Week 8 Specs

Week 8 starts the drift detection and re-indexing phase:

- Entity-level cosine drift detection over pgvector embeddings.
- Community centroid drift over Neo4j community membership plus pgvector
  vectors.
- Redis-compatible re-index queue.
- Re-index worker that re-embeds the changed entity and bounded graph
  neighbourhood, writes pgvector rows, and clears Neo4j stale flags.

## Spec Index

| Spec | Status | Primary Verification |
|---|---:|---|
| [Drift Reindex Workers](drift-reindex-workers.spec.md) | Implemented | `tests/unit/test_drift_reindex_workers.py`; `idrkd-drift-worker live-smoke --external-worker` |
