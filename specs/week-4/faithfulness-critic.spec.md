# Spec: Faithfulness Critic

## Goal

Gate synthesized answers on evidence support before they leave the Week 4
orchestrator, using an optional DeBERTa-v3-large NLI entailment model with a
deterministic lexical fallback and an AlignScore-style target of >= 0.78.

## Contract

Inputs:

- `answer: str`
- `evidence_texts: list[str]` - the top reranked hit labels for the round

Output: `FaithfulnessResult(score: float, entailed: bool)`

- `score` in `[0, 1]`; with an NLI pipeline it is the model's entailment
  score, otherwise it is the fraction of answer terms supported by evidence
  terms.
- `entailed` is `True` iff `score >= threshold` (default `0.78`, matching
  the AlignScore target in the project plan).

## Implementation

- `src/idrkd/rag/critic.py`
- `tests/unit/test_week4_rag_orchestration.py`

## Acceptance Criteria

- An answer whose terms appear in the evidence text scores above the
  default threshold and is marked `entailed`.
- An answer with no term overlap in the evidence scores below the
  threshold and is marked not `entailed`.
- An empty answer never spuriously entails (`score == 0.0`).
- A supplied NLI pipeline is used for model-backed entailment while preserving
  the `FaithfulnessResult` contract.

## Verification

```bash
uv run pytest tests/unit/test_week4_rag_orchestration.py tests/unit/test_mcp_server_and_real_adapters.py
```
