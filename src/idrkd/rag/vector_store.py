"""In-memory vector store and pgvector SQL contracts for Week 3."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from idrkd.rag.embeddings import cosine_similarity


PGVECTOR_SEARCH_SQL = """
SELECT id, entity_id, source, metadata, 1 - (embedding <=> %(embedding)s::vector) AS score
FROM knowledge_embeddings
WHERE tenant_id = %(tenant_id)s
  AND repo_id = %(repo_id)s
ORDER BY embedding <=> %(embedding)s::vector
LIMIT %(limit)s
"""


@dataclass(frozen=True)
class VectorRecord:
    id: str
    entity_id: str
    text: str
    embedding: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchHit:
    entity_id: str
    score: float
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._records: list[VectorRecord] = []

    def upsert(self, record: VectorRecord) -> None:
        self._records = [existing for existing in self._records if existing.id != record.id]
        self._records.append(record)

    def search(self, embedding: list[float], *, limit: int = 10) -> list[SearchHit]:
        hits = [
            SearchHit(
                entity_id=record.entity_id,
                score=cosine_similarity(embedding, record.embedding),
                source="vector",
                metadata=record.metadata,
            )
            for record in self._records
        ]
        return sorted(hits, key=lambda hit: hit.score, reverse=True)[:limit]
