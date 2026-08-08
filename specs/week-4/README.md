# Week 4 Specs

Week 4 closes out the agentic RAG skeleton (HLD phase 2, W3-W4):

- 5-agent LangGraph-style state machine with a bounded re-retrieve loop.
- DeBERTa-v3-large NLI faithfulness critic gating synthesis output.
- Query-path SLO gate and HotpotQA-lite recall evaluation helpers.

The FastAPI MCP server skeleton and its 6 initial tools (`search_code`,
`get_entity`, `graph_bfs`, `graph_path`, `get_community`, `enqueue_reindex`)
called for in the Week 4 plan were already delivered ahead of schedule as
part of the June 26 W5-W6 MCP/A2A milestone; see `src/idrkd/mcp/tools.py`
and `specs/README.md`.

## Spec Index

| Spec | Status | Primary Verification |
|---|---:|---|
| [Agentic RAG Orchestrator](agentic-rag-orchestrator.spec.md) | Implemented | `tests/unit/test_week4_rag_orchestration.py` |
| [Faithfulness Critic](faithfulness-critic.spec.md) | Implemented | `tests/unit/test_week4_rag_orchestration.py` |
| [Query SLO and Recall Gate](query-slo.spec.md) | Implemented | `tests/unit/test_week4_rag_orchestration.py` |
