"""Promotion gates for student model artifacts."""

from __future__ import annotations

from dataclasses import dataclass

from idrkd.evaluation.taskbench import EvalSummary


@dataclass(frozen=True)
class PromotionCriteria:
    min_tool_f1: float = 0.82
    min_faithfulness: float = 0.78
    max_ttft_seconds: float = 1.2
    max_latency_p95_seconds: float = 8.0
    max_tool_f1_regression: float = 0.02


@dataclass(frozen=True)
class PromotionInputs:
    summary: EvalSummary
    faithfulness_score: float
    tenant_security_passed: bool
    ttft_seconds: float
    latency_p95_seconds: float
    previous_tool_f1: float | None = None


@dataclass(frozen=True)
class PromotionDecision:
    promoted: bool
    reasons: tuple[str, ...]


def evaluate_promotion(
    inputs: PromotionInputs,
    *,
    criteria: PromotionCriteria = PromotionCriteria(),
) -> PromotionDecision:
    reasons: list[str] = []
    if inputs.summary.tool_f1 < criteria.min_tool_f1:
        reasons.append(
            f"tool_f1 {inputs.summary.tool_f1:.3f} < {criteria.min_tool_f1:.3f}"
        )
    if inputs.faithfulness_score < criteria.min_faithfulness:
        reasons.append(
            f"faithfulness {inputs.faithfulness_score:.3f} < {criteria.min_faithfulness:.3f}"
        )
    if not inputs.tenant_security_passed:
        reasons.append("tenant/security tests failed")
    if inputs.ttft_seconds > criteria.max_ttft_seconds:
        reasons.append(f"ttft {inputs.ttft_seconds:.3f}s > {criteria.max_ttft_seconds:.3f}s")
    if inputs.latency_p95_seconds > criteria.max_latency_p95_seconds:
        reasons.append(
            f"latency_p95 {inputs.latency_p95_seconds:.3f}s > {criteria.max_latency_p95_seconds:.3f}s"
        )
    if inputs.previous_tool_f1 is not None:
        regression = inputs.previous_tool_f1 - inputs.summary.tool_f1
        if regression > criteria.max_tool_f1_regression:
            reasons.append(
                f"tool_f1 regression {regression:.3f} > {criteria.max_tool_f1_regression:.3f}"
            )
    return PromotionDecision(promoted=not reasons, reasons=tuple(reasons))
