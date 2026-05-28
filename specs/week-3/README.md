# Week 3 Specs

Week 3 establishes the agentic RAG skeleton:

- BGE-M3 embedding adapter contract with a deterministic local fallback.
- pgvector search SQL contract and in-memory test store.
- Hybrid vector + graph retrieval with Reciprocal Rank Fusion.
- MiniLM reranker facade with deterministic local fallback.
- BFS, PageRank, and Louvain/community fallback analytics.

## Spec Index

| Spec | Status | Primary Verification |
|---|---:|---|
| [Embedding and Vector Retrieval](embedding-vector-retrieval.spec.md) | Implemented | `tests/unit/test_week3_rag.py` |
| [Hybrid RRF Retrieval](hybrid-rrf-retrieval.spec.md) | Implemented | `tests/unit/test_week3_rag.py` |
| [Graph Analytics](graph-analytics.spec.md) | Implemented | `tests/unit/test_week3_rag.py` |
