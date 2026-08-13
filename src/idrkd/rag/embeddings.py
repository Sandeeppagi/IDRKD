"""Embedding adapters for Week 3 retrieval."""

from __future__ import annotations

from collections.abc import Iterable
import hashlib
import math

import numpy as np


class BgeM3EmbeddingAdapter:
    """BGE-M3 adapter with optional real-model inference and deterministic fallback.

    `model` may be any object exposing `encode(text)` (for example a
    SentenceTransformer BGE-M3 instance). When no model is supplied the adapter
    uses hashing so local tests remain runnable without model downloads.
    """

    def __init__(
        self,
        dimensions: int = 1536,
        model: object | None = None,
        batch_size: int = 32,
    ) -> None:
        self.dimensions = dimensions
        self._model = model
        self._batch_size = batch_size

    def embed(self, text: str) -> list[float]:
        if self._model is not None:
            return _normalise_dimensions(_encode_model(self._model, text), self.dimensions)
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

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self._model is None:
            return [self.embed(text) for text in texts]
        encode = getattr(self._model, "encode")
        raw = encode(texts, batch_size=self._batch_size, show_progress_bar=False)
        values = raw.tolist() if hasattr(raw, "tolist") else raw
        return [
            _normalise_dimensions([float(value) for value in row], self.dimensions)
            for row in values
        ]

    @classmethod
    def from_sentence_transformers(
        cls,
        model_name: str = "BAAI/bge-m3",
        *,
        dimensions: int = 1536,
        batch_size: int = 32,
        device: str | None = None,
        local_files_only: bool = False,
    ) -> BgeM3EmbeddingAdapter:
        from sentence_transformers import SentenceTransformer

        return cls(
            dimensions=dimensions,
            model=SentenceTransformer(
                model_name,
                device=device,
                local_files_only=local_files_only,
            ),
            batch_size=batch_size,
        )


def cosine_similarity(left: Iterable[float], right: Iterable[float]) -> float:
    left_values = list(left)
    right_values = list(right)
    dot = sum(a * b for a, b in zip(left_values, right_values, strict=False))
    left_norm = math.sqrt(sum(a * a for a in left_values))
    right_norm = math.sqrt(sum(b * b for b in right_values))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _encode_model(model: object, text: str) -> list[float]:
    encode = getattr(model, "encode")
    raw = encode(text)
    if hasattr(raw, "tolist"):
        values = raw.tolist()
    else:
        values = raw
    if values and isinstance(values[0], list):
        values = values[0]
    return [float(value) for value in values]


def _normalise_dimensions(values: list[float], dimensions: int) -> list[float]:
    if len(values) == dimensions:
        return values
    if len(values) > dimensions:
        return values[:dimensions]
    return values + [0.0] * (dimensions - len(values))
