"""Embedding adapters for Week 3 retrieval."""

from __future__ import annotations

from collections.abc import Iterable
import hashlib
import math

import numpy as np


class BgeM3EmbeddingAdapter:
    """BGE-M3 facade with a deterministic local fallback.

    The test path uses hashing so the project remains runnable without model
    downloads. A production adapter can replace `embed` with a real BGE-M3
    encoder while preserving dimensions and call shape.
    """

    def __init__(self, dimensions: int = 1536) -> None:
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        vector = np.zeros(self.dimensions, dtype=np.float32)
        tokens = [token.lower() for token in text.split() if token.strip()]
        for token in tokens or [text]:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector /= norm
        return vector.tolist()


def cosine_similarity(left: Iterable[float], right: Iterable[float]) -> float:
    left_values = list(left)
    right_values = list(right)
    dot = sum(a * b for a, b in zip(left_values, right_values, strict=False))
    left_norm = math.sqrt(sum(a * a for a in left_values))
    right_norm = math.sqrt(sum(b * b for b in right_values))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)
