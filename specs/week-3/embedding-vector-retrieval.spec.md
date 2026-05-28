# Spec: Embedding and Vector Retrieval

## Goal

Provide the Week 3 embedding and vector-search contract for RAG integration.

## Contract

Embedding adapter:

- class: `BgeM3EmbeddingAdapter`
- method: `embed(text: str) -> list[float]`
- default dimension: `1536`
- deterministic local fallback for tests

Vector storage:

- table: `knowledge_embeddings`
- HNSW index exists from Week 1/2 foundation
- SQL contract uses pgvector cosine distance

## Implementation

- `src/idrkd/rag/embeddings.py`
- `src/idrkd/rag/vector_store.py`
- `configs/postgres-init/001_pgvector_hnsw.sql`
- `tests/unit/test_week3_rag.py`

## Acceptance Criteria

- Embeddings are deterministic.
- Cosine similarity works.
- Vector search returns ranked hits.
- pgvector search SQL references `knowledge_embeddings` and cosine ordering.

## Verification

```bash
uv run pytest tests/unit/test_week3_rag.py
```
