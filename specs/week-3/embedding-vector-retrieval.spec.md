# Spec: Embedding and Vector Retrieval

## Goal

Provide the Week 3 embedding and vector-search contract for RAG integration.

## Contract

Embedding adapter:

- class: `BgeM3EmbeddingAdapter`
- method: `embed(text: str) -> list[float]`
- default dimension: `1536`
- optional real-model path: `BgeM3EmbeddingAdapter(model=...)` or
  `from_sentence_transformers("BAAI/bge-m3")`
- deterministic local fallback for tests

Vector storage:

- table: `knowledge_embeddings`
- HNSW index exists from Week 1/2 foundation
- SQL contract uses pgvector cosine distance
- `PostgresVectorStore.upsert_records(...)` writes entity embeddings into
  `knowledge_embeddings` using `ON CONFLICT` idempotency

## Implementation

- `src/idrkd/rag/embeddings.py`
- `src/idrkd/rag/vector_store.py`
- `src/idrkd/ingestion/pipeline.py`
- `configs/postgres-init/001_pgvector_hnsw.sql`
- `tests/unit/test_week3_rag.py`

## Acceptance Criteria

- Embeddings are deterministic.
- A supplied model object is used for real embedding inference while preserving
  the fixed-dimension output contract.
- Cosine similarity works.
- Vector search returns ranked hits.
- Commit ingestion can automatically upsert one pgvector row per parsed entity.
- pgvector search SQL references `knowledge_embeddings` and cosine ordering.

## Verification

```bash
uv run pytest tests/unit/test_week3_rag.py tests/unit/test_mcp_server_and_real_adapters.py
```
