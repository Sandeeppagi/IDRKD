"""Neo4j drift state writeback."""

from __future__ import annotations

from typing import Any, cast

from neo4j import GraphDatabase


class Neo4jDriftStore:
    def __init__(self, uri: str, user: str, password: str) -> None:
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        self._driver.close()

    def __enter__(self) -> Neo4jDriftStore:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def update_entity_drift(
        self,
        *,
        tenant_id: str,
        repo_id: str,
        entity_id: str,
        drift_score: float,
        stale: bool,
        community_id: str | None = None,
    ) -> None:
        query = """
        MATCH (n:CodeEntity {tenant_id: $tenant_id, repo_id: $repo_id, id: $entity_id})
        SET n.drift_score = $drift_score,
            n.stale = $stale,
            n.drift_checked_at = datetime()
        FOREACH (_ IN CASE WHEN $community_id IS NULL THEN [] ELSE [1] END |
          SET n.community_id = $community_id
        )
        """
        self._write(
            query,
            {
                "tenant_id": tenant_id,
                "repo_id": repo_id,
                "entity_id": entity_id,
                "drift_score": drift_score,
                "stale": stale,
                "community_id": community_id,
            },
        )

    def update_centroid_drift(
        self,
        *,
        tenant_id: str,
        repo_id: str,
        community_id: str,
        centroid_drift: float,
        stale: bool,
        centroid_vector: list[float],
    ) -> int:
        query = """
        MATCH (n:CodeEntity {tenant_id: $tenant_id, repo_id: $repo_id, community_id: $community_id})
        SET n.centroid_drift = $centroid_drift,
            n.community_centroid = $centroid_vector,
            n.centroid_checked_at = datetime(),
            n.stale = coalesce(n.stale, false) OR $stale
        RETURN count(n) AS updated
        """
        value = self._write_value(
            query,
            {
                "tenant_id": tenant_id,
                "repo_id": repo_id,
                "community_id": community_id,
                "centroid_drift": centroid_drift,
                "centroid_vector": centroid_vector,
                "stale": stale,
            },
            "updated",
        )
        return int(cast(int | str, value or 0))

    def clear_stale(self, *, tenant_id: str, repo_id: str, entity_ids: list[str]) -> int:
        if not entity_ids:
            return 0
        query = """
        MATCH (n:CodeEntity {tenant_id: $tenant_id, repo_id: $repo_id})
        WHERE n.id IN $entity_ids
        SET n.stale = false,
            n.reindexed_at = datetime()
        RETURN count(n) AS updated
        """
        value = self._write_value(
            query,
            {"tenant_id": tenant_id, "repo_id": repo_id, "entity_ids": entity_ids},
            "updated",
        )
        return int(cast(int | str, value or 0))

    def community_member_ids(self, *, tenant_id: str, repo_id: str, community_id: str) -> list[str]:
        query = """
        MATCH (n:CodeEntity {tenant_id: $tenant_id, repo_id: $repo_id, community_id: $community_id})
        RETURN n.id AS id
        ORDER BY id ASC
        """
        with self._driver.session() as session:
            records = list(
                session.run(
                    query,
                    {"tenant_id": tenant_id, "repo_id": repo_id, "community_id": community_id},
                )
            )
        return [str(record["id"]) for record in records]

    def _write(self, query: str, params: dict[str, Any]) -> None:
        with self._driver.session() as session:
            session.run(query, params).consume()

    def _write_value(self, query: str, params: dict[str, Any], key: str) -> object:
        with self._driver.session() as session:
            record = session.run(query, params).single()
        return None if record is None else record[key]
