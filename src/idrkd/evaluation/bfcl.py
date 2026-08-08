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
    predicted: list[FunctionCallPrediction],
) -> ToolCallMetrics:
    remaining = predicted.copy()
    true_positives = 0
    argument_matches = 0
    for expected_call in expected:
        match_index = next((index for index, call in enumerate(remaining) if call.name == expected_call.name), None)
        if match_index is None:
            continue
        matched = remaining.pop(match_index)
        true_positives += 1
        if matched.arguments == expected_call.arguments:
            argument_matches += 1
    return ToolCallMetrics(
        true_positives=true_positives,
        false_positives=len(remaining),
        false_negatives=len(expected) - true_positives,
        argument_matches=argument_matches,
    )
