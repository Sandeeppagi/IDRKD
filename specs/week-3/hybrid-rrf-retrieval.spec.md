# Spec: Hybrid RRF Retrieval

## Goal

Fuse vector retrieval and graph traversal results using Reciprocal Rank Fusion.

## Contract

Inputs:

- natural-language query
- vector search results
- graph BFS/search results

Output:

- ordered `HybridHit` records
- accumulated RRF score
- source provenance from vector and graph channels

## Implementation

- `src/idrkd/rag/retrieval.py`
- `src/idrkd/rag/reranker.py`
- `tests/unit/test_week3_rag.py`

## Acceptance Criteria

- Shared entities across vector and graph channels rank higher.
- Source provenance is preserved.
- MiniLM reranker facade can reorder fused hits.
- Retrieval remains deterministic under test.

## Verification

```bash
uv run pytest tests/unit/test_week3_rag.py
```
