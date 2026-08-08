"""Entity and centroid drift scoring."""

from __future__ import annotations

from dataclasses import dataclass
import math


ENTITY_DRIFT_THRESHOLD = 0.15
CENTROID_DRIFT_THRESHOLD = 0.10


@dataclass(frozen=True)
class DriftDecision:
    score: float
    threshold: float
    triggered: bool


def cosine_distance(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 1.0
    dot = sum(a * b for a, b in zip(left, right, strict=False))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 1.0
    return 1.0 - dot / (left_norm * right_norm)


def entity_drift_decision(
    *,
    previous_embedding: list[float] | None,
    current_embedding: list[float],
    threshold: float = ENTITY_DRIFT_THRESHOLD,
) -> DriftDecision:
    score = 1.0 if previous_embedding is None else cosine_distance(previous_embedding, current_embedding)
    return DriftDecision(score=score, threshold=threshold, triggered=score >= threshold)


def centroid(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    width = min(len(vector) for vector in vectors)
    return [sum(vector[index] for vector in vectors) / len(vectors) for index in range(width)]
