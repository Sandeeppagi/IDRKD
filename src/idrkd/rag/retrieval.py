"""Hybrid vector + graph retrieval with Reciprocal Rank Fusion."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from idrkd.rag.embeddings import BgeM3EmbeddingAdapter
from idrkd.rag.vector_store import InMemoryVectorStore, SearchHit


class GraphSearch(Protocol):
    def bfs(self, query: str, *, limit: int = 10) -> list[SearchHit]:
        ...


@dataclass(frozen=True)
class HybridHit:
    entity_id: str
    score: float
    sources: tuple[str, ...] = field(default_factory=tuple)


def reciprocal_rank_fusion(rankings: list[list[SearchHit]], *, k: int = 60, limit: int = 10) -> list[HybridHit]:
    scores: dict[str, float] = {}
    sources: dict[str, set[str]] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            scores[hit.entity_id] = scores.get(hit.entity_id, 0.0) + 1.0 / (k + rank)
            sources.setdefault(hit.entity_id, set()).add(hit.source)
    fused = [
        HybridHit(entity_id=entity_id, score=score, sources=tuple(sorted(sources[entity_id])))
        for entity_id, score in scores.items()
    ]
    return sorted(fused, key=lambda hit: hit.score, reverse=True)[:limit]


class KeywordGraphSearch:
    def __init__(self, labels_by_entity: dict[str, str]) -> None:
        self._labels_by_entity = labels_by_entity

    def bfs(self, query: str, *, limit: int = 10) -> list[SearchHit]:
        terms = {term.lower() for term in query.split()}
        hits: list[SearchHit] = []
        for entity_id, label in self._labels_by_entity.items():
            label_terms = set(label.lower().replace(".", " ").split())
            score = len(terms & label_terms)
            if score:
                hits.append(SearchHit(entity_id=entity_id, score=float(score), source="graph"))
        return sorted(hits, key=lambda hit: hit.score, reverse=True)[:limit]


class HybridRetriever:
    def __init__(
        self,
        *,
        embeddings: BgeM3EmbeddingAdapter,
        vector_store: InMemoryVectorStore,
        graph_search: GraphSearch,
    ) -> None:
        self._embeddings = embeddings
        self._vector_store = vector_store
        self._graph_search = graph_search

    def search(self, query: str, *, limit: int = 10) -> list[HybridHit]:
        vector_hits = self._vector_store.search(self._embeddings.embed(query), limit=limit)
        graph_hits = self._graph_search.bfs(query, limit=limit)
        return reciprocal_rank_fusion([vector_hits, graph_hits], limit=limit)
