"""Week 4 query-path SLO gate and retrieval-quality evaluation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QuerySlo:
    p50_budget_seconds: float = 3.0
    p95_budget_seconds: float = 8.0
    max_reretrieve_rounds: int = 1

    def check(self, *, p50_seconds: float, p95_seconds: float, rounds: int) -> bool:
        return (
            p50_seconds <= self.p50_budget_seconds
            and p95_seconds <= self.p95_budget_seconds
            and rounds <= self.max_reretrieve_rounds + 1
        )


def percentile(latencies_seconds: list[float], percentile_rank: float) -> float:
    if not latencies_seconds:
        return 0.0
    ordered = sorted(latencies_seconds)
    index = min(len(ordered) - 1, max(0, round(percentile_rank / 100 * (len(ordered) - 1))))
    return ordered[index]


def recall_at_k(predicted_ids: list[str], gold_ids: set[str], *, k: int = 10) -> float:
    if not gold_ids:
        return 0.0
    top_k = set(predicted_ids[:k])
    return len(top_k & gold_ids) / len(gold_ids)
