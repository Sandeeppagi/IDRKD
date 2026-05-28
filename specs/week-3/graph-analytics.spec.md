# Spec: Graph Analytics

## Goal

Provide the Week 3 graph analytics skeleton for salience and community
properties.

## Contract

Algorithms:

- BFS neighborhood expansion.
- PageRank salience.
- Louvain-compatible community detection fallback.

Output:

- `pagerank: dict[str, float]`
- `communities: dict[str, int]`

Scheduling:

- `NightlyGraphAnalyticsJob.run(edges)` provides the callable unit that can be
  scheduled by APScheduler or Celery Beat.

## Implementation

- `src/idrkd/graph/analytics.py`
- `tests/unit/test_week3_rag.py`

## Acceptance Criteria

- BFS respects depth.
- PageRank returns all graph nodes.
- Community fallback groups connected components.
- Result can be written back to Neo4j by a later persistence adapter.

## Verification

```bash
uv run pytest tests/unit/test_week3_rag.py
```
