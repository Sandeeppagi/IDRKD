"""Distillation quality gates for BFCL and serving SLOs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BfclMetrics:
    true_positives: int
    false_positives: int
    false_negatives: int

    @property
    def precision(self) -> float:
        denominator = self.true_positives + self.false_positives
        return self.true_positives / denominator if denominator else 0.0

    @property
    def recall(self) -> float:
        denominator = self.true_positives + self.false_negatives
        return self.true_positives / denominator if denominator else 0.0

    @property
    def f1(self) -> float:
        denominator = self.precision + self.recall
        return 2 * self.precision * self.recall / denominator if denominator else 0.0


@dataclass(frozen=True)
class DistillationGate:
    first_pass_bfcl_f1: float = 0.75
    post_dpo_bfcl_f1: float = 0.82
    align_score: float = 0.78
    ttft_seconds: float = 1.2

    def check_first_pass(self, metrics: BfclMetrics) -> bool:
        return metrics.f1 >= self.first_pass_bfcl_f1

    def check_release(
        self,
        *,
        bfcl_metrics: BfclMetrics,
        align_score: float,
        ttft_seconds: float,
    ) -> bool:
        return (
            bfcl_metrics.f1 >= self.post_dpo_bfcl_f1
            and align_score >= self.align_score
            and ttft_seconds <= self.ttft_seconds
        )
