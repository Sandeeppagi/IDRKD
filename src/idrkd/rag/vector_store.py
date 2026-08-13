"""Vector stores and pgvector SQL contracts for Week 3 retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
import json
from typing import Any

import psycopg

from idrkd.common.models import CodeEntity
from idrkd.observability.metrics import EMBEDDING_UPSERTS
from idrkd.observability.tracing import traced_span
from idrkd.rag.embeddings import cosine_similarity


PGVECTOR_SEARCH_SQL = """
SELECT id, entity_id, source, metadata, 1 - (embedding <=> %(embedding)s::vector) AS score
FROM knowledge_embeddings
WHERE tenant_id = %(tenant_id)s
  AND repo_id = %(repo_id)s
ORDER BY embedding <=> %(embedding)s::vector
LIMIT %(limit)s
"""

PGVECTOR_UPSERT_SQL = """
INSERT INTO knowledge_embeddings (
  id, tenant_id, repo_id, entity_id, source, content_hash, embedding_model,
  embedding, metadata, created_at, updated_at
)
VALUES (
  %(id)s, %(tenant_id)s, %(repo_id)s, %(entity_id)s, %(source)s,
  %(content_hash)s, %(embedding_model)s, %(embedding)s::vector,
  %(metadata)s::jsonb, %(created_at)s, %(updated_at)s
)
ON CONFLICT (id) DO UPDATE SET
  tenant_id = EXCLUDED.tenant_id,
  repo_id = EXCLUDED.repo_id,
  entity_id = EXCLUDED.entity_id,
  source = EXCLUDED.source,
  content_hash = EXCLUDED.content_hash,
  embedding_model = EXCLUDED.embedding_model,
  embedding = EXCLUDED.embedding,
  metadata = EXCLUDED.metadata,
  updated_at = EXCLUDED.updated_at
"""


@dataclass(frozen=True)
class VectorRecord:
    id: str
    entity_id: str
    text: str
    embedding: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)
    tenant_id: str = ""
    repo_id: str = ""
    source: str = "code_entity"
    content_hash: str = ""
    embedding_model: str = "bge-m3"


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

    def upsert_records(self, records: list[VectorRecord]) -> int:
        for record in records:
            self.upsert(record)
        if records:
            EMBEDDING_UPSERTS.labels("in_memory").inc(len(records))
        return len(records)

    def snapshot_records(self, record_ids: list[str]) -> list[dict[str, Any]]:
        selected = set(record_ids)
        return [_vector_record_to_dict(record) for record in self._records if record.id in selected]

    def compensate_records(self, record_ids: list[str], snapshot: list[dict[str, Any]]) -> None:
        selected = set(record_ids)
        self._records = [record for record in self._records if record.id not in selected]
        self.upsert_records([VectorRecord(**record) for record in snapshot])

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


class PostgresVectorStore:
    """pgvector-backed embedding sink/search store."""

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn

    def upsert_records(self, records: list[VectorRecord]) -> int:
        if not records:
            return 0
        rows = _vector_rows(records)
        with traced_span(
            "pgvector.embedding_upsert",
            correlation_id="",
            record_count=len(records),
            source="postgres",
        ):
            with psycopg.connect(self._dsn) as conn:
                with conn.cursor() as cursor:
                    cursor.executemany(PGVECTOR_UPSERT_SQL, rows)
        EMBEDDING_UPSERTS.labels("postgres").inc(len(records))
        return len(records)

    def snapshot_records(self, record_ids: list[str]) -> list[dict[str, Any]]:
        if not record_ids:
            return []
        sql = """
        SELECT id, entity_id, metadata, tenant_id, repo_id, source, content_hash,
               embedding_model, embedding::text
        FROM knowledge_embeddings
        WHERE id = ANY(%(record_ids)s)
        """
        with psycopg.connect(self._dsn) as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql, {"record_ids": record_ids})
                rows = cursor.fetchall()
        return [
            _vector_record_to_dict(
                VectorRecord(
                    id=str(row[0]),
                    entity_id=str(row[1]),
                    text="",
                    metadata=dict(row[2]),
                    tenant_id=str(row[3]),
                    repo_id=str(row[4]),
                    source=str(row[5]),
                    content_hash=str(row[6]),
                    embedding_model=str(row[7]),
                    embedding=parse_vector_literal(str(row[8])),
                )
            )
            for row in rows
        ]

    def compensate_records(self, record_ids: list[str], snapshot: list[dict[str, Any]]) -> None:
        restored = [VectorRecord(**record) for record in snapshot]
        with psycopg.connect(self._dsn) as conn:
            with conn.cursor() as cursor:
                cursor.execute("DELETE FROM knowledge_embeddings WHERE id = ANY(%s)", (record_ids,))
                if restored:
                    cursor.executemany(PGVECTOR_UPSERT_SQL, _vector_rows(restored))

    def search(
        self,
        *,
        tenant_id: str,
        repo_id: str,
        embedding: list[float],
        limit: int = 10,
    ) -> list[SearchHit]:
        with traced_span(
            "pgvector.search",
            correlation_id="",
            tenant_id=tenant_id,
            repo_id=repo_id,
            limit=limit,
        ):
            with psycopg.connect(self._dsn) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        PGVECTOR_SEARCH_SQL,
                        {
                            "tenant_id": tenant_id,
                            "repo_id": repo_id,
                            "embedding": vector_literal(embedding),
                            "limit": limit,
                        },
                    )
                    rows = cursor.fetchall()
        return [
            SearchHit(entity_id=str(row[1]), score=float(row[4]), source=str(row[2]), metadata=row[3])
            for row in rows
        ]

    def get_entity_embedding(
        self,
        *,
        tenant_id: str,
        repo_id: str,
        entity_id: str,
        embedding_model: str = "bge-m3",
    ) -> list[float] | None:
        sql = """
        SELECT embedding::text
        FROM knowledge_embeddings
        WHERE tenant_id = %(tenant_id)s
          AND repo_id = %(repo_id)s
          AND entity_id = %(entity_id)s
          AND embedding_model = %(embedding_model)s
        ORDER BY updated_at DESC
        LIMIT 1
        """
        with psycopg.connect(self._dsn) as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    sql,
                    {
                        "tenant_id": tenant_id,
                        "repo_id": repo_id,
                        "entity_id": entity_id,
                        "embedding_model": embedding_model,
                    },
                )
                row = cursor.fetchone()
        return parse_vector_literal(str(row[0])) if row else None

    def get_embeddings_for_entities(
        self,
        *,
        tenant_id: str,
        repo_id: str,
        entity_ids: list[str],
        embedding_model: str = "bge-m3",
    ) -> dict[str, list[float]]:
        if not entity_ids:
            return {}
        sql = """
        SELECT entity_id, embedding::text
        FROM knowledge_embeddings
        WHERE tenant_id = %(tenant_id)s
          AND repo_id = %(repo_id)s
          AND entity_id = ANY(%(entity_ids)s)
          AND embedding_model = %(embedding_model)s
        """
        with psycopg.connect(self._dsn) as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    sql,
                    {
                        "tenant_id": tenant_id,
                        "repo_id": repo_id,
                        "entity_ids": entity_ids,
                        "embedding_model": embedding_model,
                    },
                )
                rows = cursor.fetchall()
        return {str(entity_id): parse_vector_literal(str(vector)) for entity_id, vector in rows}


def vector_record_from_entity(
    entity: CodeEntity,
    *,
    embedding: list[float],
    text: str,
    embedding_model: str = "bge-m3",
) -> VectorRecord:
    return VectorRecord(
        id=f"emb_{entity.id}_{embedding_model.replace('/', '_')}",
        tenant_id=entity.tenant_id,
        repo_id=entity.repo_id,
        entity_id=entity.id,
        text=text,
        embedding=embedding,
        source="code_entity",
        content_hash=entity.content_hash,
        embedding_model=embedding_model,
        metadata={
            "kind": entity.kind.value,
            "name": entity.name,
            "qualified_name": entity.qualified_name,
            "path": entity.location.path,
            "lamport_clock": entity.lamport_clock,
        },
    )


def _vector_record_to_dict(record: VectorRecord) -> dict[str, Any]:
    return {
        "id": record.id,
        "entity_id": record.entity_id,
        "text": record.text,
        "embedding": record.embedding,
        "metadata": record.metadata,
        "tenant_id": record.tenant_id,
        "repo_id": record.repo_id,
        "source": record.source,
        "content_hash": record.content_hash,
        "embedding_model": record.embedding_model,
    }


def _vector_rows(records: list[VectorRecord]) -> list[dict[str, Any]]:
    now = datetime.now(UTC)
    return [
        {
            "id": record.id,
            "tenant_id": record.tenant_id,
            "repo_id": record.repo_id,
            "entity_id": record.entity_id,
            "source": record.source,
            "content_hash": record.content_hash,
            "embedding_model": record.embedding_model,
            "embedding": vector_literal(record.embedding),
            "metadata": json.dumps(record.metadata, sort_keys=True),
            "created_at": now,
            "updated_at": now,
        }
        for record in records
    ]


def embedding_text_for_entity(entity: CodeEntity) -> str:
    property_text = json.dumps(entity.properties, sort_keys=True)
    return " ".join(
        part
        for part in (
            entity.kind.value,
            entity.name,
            entity.qualified_name,
            entity.location.path,
            entity.language,
            property_text,
        )
        if part
    )


def vector_literal(values: list[float]) -> str:
    return "[" + ",".join(str(float(value)) for value in values) + "]"


def parse_vector_literal(value: str) -> list[float]:
    stripped = value.strip().removeprefix("[").removesuffix("]")
    if not stripped:
        return []
    return [float(part) for part in stripped.split(",")]
