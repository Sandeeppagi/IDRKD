# Spec: Query SLO and Recall Gate

## Goal

Encode the Week 4 query-path SLO (`p50 <= 3.0s`, `p95 <= 8.0s` with at most
one re-retrieve round) and the HotpotQA-lite top-10 recall target
(`>= 0.70` on a 100-query subset) as testable helpers.

## Contract

- `QuerySlo(p50_budget_seconds=3.0, p95_budget_seconds=8.0, max_reretrieve_rounds=1)`
  - `check(p50_seconds, p95_seconds, rounds) -> bool`: `True` only if both
    latency budgets hold and `rounds` does not exceed
    `max_reretrieve_rounds + 1` (the initial pass plus the allowed
    re-retrieve).
- `percentile(latencies_seconds, percentile_rank) -> float`: nearest-rank
  percentile over a latency sample; `0.0` for an empty sample.
- `recall_at_k(predicted_ids, gold_ids, k=10) -> float`: fraction of gold
  IDs present in the top-`k` predicted IDs; `0.0` when `gold_ids` is empty.

## Implementation

- `src/idrkd/rag/query_slo.py`
- `tests/unit/test_week4_rag_orchestration.py`

## Acceptance Criteria

- `QuerySlo.check` rejects a run that breaches either latency budget.
- `QuerySlo.check` rejects a run that uses more than one re-retrieve round.
- `percentile` matches a hand-computed nearest-rank value on a small sample.
- `recall_at_k` matches a hand-computed recall value against a HotpotQA-lite
  style gold/predicted ID set, to be run over the 100-query subset once the
  live evaluation harness lands.

## Verification

```bash
uv run pytest tests/unit/test_week4_rag_orchestration.py
```
