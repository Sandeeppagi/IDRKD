"""Backing services for concrete MCP tool handlers."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, cast

from neo4j import GraphDatabase
import psycopg
import redis

from idrkd.rag.embeddings import BgeM3EmbeddingAdapter


@dataclass(frozen=True)
class EntityRecord:
    id: str
    kind: str
    name: str
    qualified_name: str
    path: str
    content_hash: str
    properties: dict[str, Any]


@dataclass(frozen=True)
class SearchResult:
    entity_id: str
    score: float
    source: str
    name: str
    kind: str


class Neo4jMcpBackend:
    """Read-only Neo4j adapter used by default MCP business handlers."""

    def __init__(self, uri: str, user: str, password: str) -> None:
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        self._driver.close()

    def __enter__(self) -> Neo4jMcpBackend:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def get_entity(self, *, tenant_id: str, repo_id: str, entity_id: str) -> EntityRecord | None:
        query = """
        MATCH (n:CodeEntity {tenant_id: $tenant_id, repo_id: $repo_id, id: $entity_id})
        RETURN n.id AS id, n.kind AS kind, n.name AS name, n.qualified_name AS qualified_name,
               n.path AS path, n.content_hash AS content_hash, n.properties_json AS properties_json
        LIMIT 1
        """
        with self._driver.session() as session:
            record = session.run(
                query,
                {"tenant_id": tenant_id, "repo_id": repo_id, "entity_id": entity_id},
            ).single()
        if record is None:
            return None
        return EntityRecord(
            id=record["id"],
            kind=record["kind"],
            name=record["name"],
            qualified_name=record["qualified_name"],
            path=record["path"],
            content_hash=record["content_hash"],
            properties=_safe_json(record["properties_json"]),
        )

    def search_code(
        self, *, tenant_id: str, repo_id: str, query_text: str, limit: int = 10
    ) -> list[SearchResult]:
        query = """
        MATCH (n:CodeEntity {tenant_id: $tenant_id, repo_id: $repo_id})
        WITH n,
             toLower(coalesce(n.name, '') + ' ' + coalesce(n.qualified_name, '') + ' ' +
                     coalesce(n.path, '') + ' ' + coalesce(n.properties_json, '')) AS haystack,
             [term IN split(toLower($query_text), ' ') WHERE term <> ''] AS terms
        WITH n, reduce(score = 0, term IN terms |
             score + CASE WHEN haystack CONTAINS term THEN 1 ELSE 0 END) AS score
        WHERE score > 0
        RETURN n.id AS id, n.kind AS kind, n.name AS name, score
        ORDER BY score DESC, n.id ASC
        LIMIT $limit
        """
        with self._driver.session() as session:
            records = list(
                session.run(
                    query,
                    {
                        "tenant_id": tenant_id,
                        "repo_id": repo_id,
                        "query_text": query_text,
                        "limit": limit,
                    },
                )
            )
        return [
            SearchResult(
                entity_id=record["id"],
                score=float(record["score"]),
                source="neo4j_keyword",
                name=record["name"],
                kind=record["kind"],
            )
            for record in records
        ]

    def downstream_impact(
        self, *, tenant_id: str, repo_id: str, entity_id: str, limit: int = 25
    ) -> list[SearchResult]:
        query = """
        MATCH (start:CodeEntity {tenant_id: $tenant_id, repo_id: $repo_id, id: $entity_id})
        MATCH path = (start)-[:RELATES_TO*1..3]->(n:CodeEntity)
        WHERE n.tenant_id = $tenant_id AND n.repo_id = $repo_id
        RETURN DISTINCT n.id AS id, n.kind AS kind, n.name AS name, min(length(path)) AS distance
        ORDER BY distance ASC, id ASC
        LIMIT $limit
        """
        with self._driver.session() as session:
            records = list(
                session.run(
                    query,
                    {"tenant_id": tenant_id, "repo_id": repo_id, "entity_id": entity_id, "limit": limit},
                )
            )
        return [
            SearchResult(
                entity_id=record["id"],
                score=1.0 / float(record["distance"]),
                source="neo4j_downstream",
                name=record["name"],
                kind=record["kind"],
            )
            for record in records
        ]

    def salience(self, *, tenant_id: str, repo_id: str, entity_id: str) -> float | None:
        query = """
        MATCH (n:CodeEntity {tenant_id: $tenant_id, repo_id: $repo_id, id: $entity_id})
        RETURN coalesce(n.salience, n.pagerank, n.page_rank) AS salience
        LIMIT 1
        """
        with self._driver.session() as session:
            record = session.run(
                query,
                {"tenant_id": tenant_id, "repo_id": repo_id, "entity_id": entity_id},
            ).single()
        if record is None or record["salience"] is None:
            return None
        return float(record["salience"])

    def centroid_drift(self, *, tenant_id: str, repo_id: str, community_id: str) -> float | None:
        query = """
        MATCH (n:CodeEntity {tenant_id: $tenant_id, repo_id: $repo_id})
        WHERE toString(n.community_id) = $community_id
        RETURN max(coalesce(n.centroid_drift, n.drift_score, 0.0)) AS drift
        """
        with self._driver.session() as session:
            record = session.run(
                query,
                {"tenant_id": tenant_id, "repo_id": repo_id, "community_id": community_id},
            ).single()
        if record is None or record["drift"] is None:
            return None
        return float(record["drift"])

    def stale_entities(self, *, tenant_id: str, repo_id: str, limit: int = 50) -> list[EntityRecord]:
        query = """
        MATCH (n:CodeEntity {tenant_id: $tenant_id, repo_id: $repo_id})
        WHERE coalesce(n.stale, false) = true OR coalesce(n.drift_score, 0.0) > 0.2
        RETURN n.id AS id, n.kind AS kind, n.name AS name, n.qualified_name AS qualified_name,
               n.path AS path, n.content_hash AS content_hash, n.properties_json AS properties_json
        ORDER BY coalesce(n.drift_score, 0.0) DESC, n.id ASC
        LIMIT $limit
        """
        with self._driver.session() as session:
            records = list(session.run(query, {"tenant_id": tenant_id, "repo_id": repo_id, "limit": limit}))
        return [
            EntityRecord(
                id=record["id"],
                kind=record["kind"],
                name=record["name"],
                qualified_name=record["qualified_name"],
                path=record["path"],
                content_hash=record["content_hash"],
                properties=_safe_json(record["properties_json"]),
            )
            for record in records
        ]


class PgvectorSearchBackend:
    """pgvector-backed semantic search for `search_code`."""

    def __init__(self, dsn: str, *, embeddings: BgeM3EmbeddingAdapter | None = None) -> None:
        self._dsn = dsn
        self._embeddings = embeddings or BgeM3EmbeddingAdapter()

    def search_code(
        self, *, tenant_id: str, repo_id: str, query_text: str, limit: int = 10
    ) -> list[SearchResult]:
        embedding = "[" + ",".join(str(value) for value in self._embeddings.embed(query_text)) + "]"
        sql = """
        SELECT entity_id, source, 1 - (embedding <=> %(embedding)s::vector) AS score
        FROM knowledge_embeddings
        WHERE tenant_id = %(tenant_id)s AND repo_id = %(repo_id)s
        ORDER BY embedding <=> %(embedding)s::vector
        LIMIT %(limit)s
        """
        with psycopg.connect(self._dsn) as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    sql,
                    {
                        "tenant_id": tenant_id,
                        "repo_id": repo_id,
                        "embedding": embedding,
                        "limit": limit,
                    },
                )
                rows = cursor.fetchall()
        return [
            SearchResult(
                entity_id=str(row[0]),
                score=float(row[2]),
                source=f"pgvector:{row[1]}",
                name=str(row[0]),
                kind="embedding",
            )
            for row in rows
        ]


class McpStateStore:
    def enqueue_reindex(self, item: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def get_conflict(self, conflict_id: str) -> dict[str, Any] | None:
        raise NotImplementedError

    def ensure_conflict(self, conflict: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def resolve_conflict(self, conflict: dict[str, Any], resolution: str) -> dict[str, Any]:
        raise NotImplementedError


@dataclass
class InMemoryMcpStateStore(McpStateStore):
    reindex_queue: list[dict[str, Any]] = field(default_factory=list)
    conflicts: dict[str, dict[str, Any]] = field(default_factory=dict)

    def enqueue_reindex(self, item: dict[str, Any]) -> dict[str, Any]:
        queued = {**item, "status": "queued", "queue_position": len(self.reindex_queue) + 1}
        self.reindex_queue.append(queued)
        return queued

    def get_conflict(self, conflict_id: str) -> dict[str, Any] | None:
        return self.conflicts.get(conflict_id)

    def ensure_conflict(self, conflict: dict[str, Any]) -> dict[str, Any]:
        return self.conflicts.setdefault(str(conflict["conflict_id"]), conflict)

    def resolve_conflict(self, conflict: dict[str, Any], resolution: str) -> dict[str, Any]:
        stored = self.ensure_conflict(conflict)
        stored["status"] = "resolved"
        stored["resolution"] = resolution
        return stored


class RedisMcpStateStore(McpStateStore):
    """Redis-backed MCP queue/conflict state for the Dockerized server."""

    def __init__(
        self,
        redis_url: str,
        *,
        reindex_key: str = "idrkd:mcp:reindex",
        conflict_prefix: str = "idrkd:mcp:conflict:",
    ) -> None:
        self._client = redis.Redis.from_url(redis_url)
        self._reindex_key = reindex_key
        self._conflict_prefix = conflict_prefix

    def enqueue_reindex(self, item: dict[str, Any]) -> dict[str, Any]:
        queue_position = cast(int, self._client.llen(self._reindex_key)) + 1
        queued = {**item, "status": "queued", "queue_position": queue_position}
        self._client.rpush(self._reindex_key, json.dumps(queued, sort_keys=True))
        return queued

    def get_conflict(self, conflict_id: str) -> dict[str, Any] | None:
        raw = self._client.get(self._conflict_key(conflict_id))
        if raw is None:
            return None
        text = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
        loaded = json.loads(text)
        return loaded if isinstance(loaded, dict) else None

    def ensure_conflict(self, conflict: dict[str, Any]) -> dict[str, Any]:
        key = self._conflict_key(str(conflict["conflict_id"]))
        existing = self.get_conflict(str(conflict["conflict_id"]))
        if existing is not None:
            return existing
        self._client.set(key, json.dumps(conflict, sort_keys=True))
        return conflict

    def resolve_conflict(self, conflict: dict[str, Any], resolution: str) -> dict[str, Any]:
        stored = self.ensure_conflict(conflict)
        stored["status"] = "resolved"
        stored["resolution"] = resolution
        self._client.set(self._conflict_key(str(stored["conflict_id"])), json.dumps(stored, sort_keys=True))
        return stored

    def _conflict_key(self, conflict_id: str) -> str:
        return f"{self._conflict_prefix}{conflict_id}"


def _safe_json(value: str | None) -> dict[str, Any]:
    if not value:
        return {}

    loaded = json.loads(value)
    return loaded if isinstance(loaded, dict) else {"value": loaded}
