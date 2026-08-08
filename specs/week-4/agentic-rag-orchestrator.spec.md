# Spec: Agentic RAG Orchestrator

## Goal

Provide the Week 4 5-agent LangGraph-style state machine that turns a query
into a critic-gated, evidence-grounded answer with a bounded re-retrieve
loop.

## Contract

Agents, run in sequence per round:

- `RouterAgent` - marks the entry hop of the state machine.
- `VectorRetrieverAgent` - BGE-M3 embedding search over the vector store.
- `GraphTraversalAgent` - graph BFS/keyword traversal.
- `SynthesisAgent` - Reciprocal Rank Fusion + MiniLM rerank + answer draft.
- `CriticAgent` - faithfulness gate (see `faithfulness-critic.spec.md`).

State:

- `QueryState` is a `TypedDict` carrying `query`, `rounds`, per-channel hits,
  fused/reranked hits, `answer`, `faithfulness`, `accepted`, and an ordered
  `trace` of agent hops.
- `rounds` starts at 0 and increments once per loop iteration; the loop is
  capped at `MAX_ROUNDS = 2` (one initial pass plus one bounded re-retrieve).
- The loop exits early when the critic accepts the answer
  (`state["accepted"] is True`).

## Implementation

- `src/idrkd/rag/orchestrator.py`
- `tests/unit/test_week4_rag_orchestration.py`

## Acceptance Criteria

- `AgenticRagOrchestrator.run` always executes agents in router -> vector ->
  graph -> synthesis -> critic order for every round.
- A grounded query converges with `accepted is True` in a single round.
- An unmatched/unsupported query never exceeds `MAX_ROUNDS` rounds and ends
  with `accepted is False` rather than looping indefinitely.
- `trace` reflects every agent hop for the final round, in order.

## Verification

```bash
uv run pytest tests/unit/test_week4_rag_orchestration.py
```
