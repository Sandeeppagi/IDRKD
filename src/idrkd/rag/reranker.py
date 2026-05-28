"""MiniLM-style cross-encoder reranker facade."""

from __future__ import annotations

from idrkd.rag.retrieval import HybridHit


class MiniLmReranker:
    """Deterministic lexical fallback behind the MiniLM reranker contract."""

    def rerank(self, query: str, hits: list[HybridHit], labels: dict[str, str]) -> list[HybridHit]:
        query_terms = {term.lower() for term in query.split()}

        def score(hit: HybridHit) -> float:
            label_terms = set(labels.get(hit.entity_id, "").lower().replace(".", " ").split())
            return hit.score + 0.01 * len(query_terms & label_terms)

        return sorted(hits, key=score, reverse=True)
