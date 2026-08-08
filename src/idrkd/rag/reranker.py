"""MiniLM-style cross-encoder reranker."""

from __future__ import annotations

from idrkd.rag.retrieval import HybridHit


class MiniLmReranker:
    """Reranker with optional real cross-encoder inference and deterministic fallback."""

    def __init__(self, model: object | None = None) -> None:
        self._model = model

    @classmethod
    def from_sentence_transformers(
        cls,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        *,
        local_files_only: bool = False,
    ) -> MiniLmReranker:
        from sentence_transformers import CrossEncoder

        return cls(CrossEncoder(model_name, local_files_only=local_files_only))

    def rerank(self, query: str, hits: list[HybridHit], labels: dict[str, str]) -> list[HybridHit]:
        if self._model is not None:
            pairs = [(query, labels.get(hit.entity_id, hit.entity_id)) for hit in hits]
            predict = getattr(self._model, "predict")
            raw_scores = predict(pairs)
            if hasattr(raw_scores, "tolist"):
                raw_scores = raw_scores.tolist()
            scored = [
                (hit.score + float(score), hit)
                for hit, score in zip(hits, raw_scores, strict=True)
            ]
            return [hit for _score, hit in sorted(scored, key=lambda item: item[0], reverse=True)]

        query_terms = {term.lower() for term in query.split()}

        def score(hit: HybridHit) -> float:
            label_terms = set(labels.get(hit.entity_id, "").lower().replace(".", " ").split())
            return hit.score + 0.01 * len(query_terms & label_terms)

        return sorted(hits, key=score, reverse=True)
