"""BFCL-style function-call scoring primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FunctionCallPrediction:
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolCallMetrics:
    true_positives: int
    false_positives: int
    false_negatives: int
    argument_matches: int

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

    @property
    def argument_accuracy(self) -> float:
        return self.argument_matches / self.true_positives if self.true_positives else 0.0


def score_function_calls(
    expected: list[FunctionCallPrediction],
    predicted: list[FunctionCallPrediction | None],
) -> ToolCallMetrics:
    """Score one prediction against the expected call from the same case.

    Tool calls are deliberately not matched as an unordered bag. TaskBench
    contains many repeated tool names, so bag matching can incorrectly credit a
    prediction from one case against another case's oracle.
    """

    if len(expected) != len(predicted):
        raise ValueError("expected and predicted must contain one entry per evaluation case")

    true_positives = 0
    false_positives = 0
    false_negatives = 0
    argument_matches = 0
    for expected_call, predicted_call in zip(expected, predicted, strict=True):
        if predicted_call is None:
            false_negatives += 1
            continue
        if predicted_call.name != expected_call.name:
            false_positives += 1
            false_negatives += 1
            continue
        true_positives += 1
        if predicted_call.arguments == expected_call.arguments:
            argument_matches += 1
    return ToolCallMetrics(
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        argument_matches=argument_matches,
    )
