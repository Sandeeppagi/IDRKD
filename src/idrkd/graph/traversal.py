"""Week 5 Graph Traversal Agent: real Cypher BFS, shortestPath, and community subgraph."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from neo4j import GraphDatabase

from idrkd.graph.cypher import (
    COMMUNITY_FOR_ENTITY_QUERY,
    COMMUNITY_SUBGRAPH_QUERY,
    bfs_neighbors_params,
    bfs_neighbors_query,
    community_for_entity_params,
    community_subgraph_params,
    shortest_path_params,
    shortest_path_query,
)
from idrkd.observability.metrics import GRAPH_TRAVERSAL_LATENCY, observe_histogram
from idrkd.observability.tracing import traced_span
from idrkd.rag.vector_store import SearchHit


@dataclass(frozen=True)
class BfsNeighbor:
    entity_id: str
    kind: str
    distance: int


@dataclass(frozen=True)
class ShortestPath:
    node_ids: tuple[str, ...]
    hop_count: int


@dataclass(frozen=True)
class CommunityMember:
    entity_id: str
    kind: str
    name: str


def _run_and_collect(tx: Any, statement: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    return [record.data() for record in tx.run(statement, params)]


class Neo4jGraphTraversal:
    """Runs the Week 5 read-only Cypher traversal queries against Neo4j."""

    def __init__(self, uri: str, user: str, password: str) -> None:
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        self._driver.close()

    def __enter__(self) -> Neo4jGraphTraversal:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def bfs_neighbors(
        self, *, tenant_id: str, repo_id: str, entity_id: str, depth: int = 2, limit: int = 10
    ) -> list[BfsNeighbor]:
        with observe_histogram(GRAPH_TRAVERSAL_LATENCY, "bfs_neighbors"), traced_span(
            "graph.traversal.bfs_neighbors",
            correlation_id="",
            tenant_id=tenant_id,
            repo_id=repo_id,
            entity_id=entity_id,
            depth=depth,
            limit=limit,
        ):
            query = bfs_neighbors_query(depth=depth)
            params = bfs_neighbors_params(tenant_id=tenant_id, repo_id=repo_id, entity_id=entity_id, limit=limit)
            with self._driver.session() as session:
                records = session.execute_read(_run_and_collect, query, params)
        return [BfsNeighbor(entity_id=r["id"], kind=r["kind"], distance=r["distance"]) for r in records]

    def shortest_path(
        self, *, tenant_id: str, repo_id: str, source_id: str, target_id: str, max_hops: int = 6
    ) -> ShortestPath | None:
        with observe_histogram(GRAPH_TRAVERSAL_LATENCY, "shortest_path"), traced_span(
            "graph.traversal.shortest_path",
            correlation_id="",
            tenant_id=tenant_id,
            repo_id=repo_id,
            source_id=source_id,
            target_id=target_id,
            max_hops=max_hops,
        ):
            query = shortest_path_query(max_hops=max_hops)
            params = shortest_path_params(
                tenant_id=tenant_id, repo_id=repo_id, source_id=source_id, target_id=target_id
            )
            with self._driver.session() as session:
                records = session.execute_read(_run_and_collect, query, params)
        if not records:
            return None
        record = records[0]
        return ShortestPath(node_ids=tuple(record["node_ids"]), hop_count=record["hop_count"])

    def community_subgraph(
        self, *, tenant_id: str, repo_id: str, community_id: int, limit: int = 25
    ) -> list[CommunityMember]:
        with observe_histogram(GRAPH_TRAVERSAL_LATENCY, "community_subgraph"), traced_span(
            "graph.traversal.community_subgraph",
            correlation_id="",
            tenant_id=tenant_id,
            repo_id=repo_id,
            community_id=community_id,
            limit=limit,
        ):
            params = community_subgraph_params(
                tenant_id=tenant_id, repo_id=repo_id, community_id=community_id, limit=limit
            )
            with self._driver.session() as session:
                records = session.execute_read(_run_and_collect, COMMUNITY_SUBGRAPH_QUERY, params)
        return [CommunityMember(entity_id=r["id"], kind=r["kind"], name=r["name"]) for r in records]

    def community_for_entity(
        self, *, tenant_id: str, repo_id: str, entity_id: str, limit: int = 25
    ) -> list[CommunityMember]:
        with observe_histogram(GRAPH_TRAVERSAL_LATENCY, "community_for_entity"), traced_span(
            "graph.traversal.community_for_entity",
            correlation_id="",
            tenant_id=tenant_id,
            repo_id=repo_id,
            entity_id=entity_id,
            limit=limit,
        ):
            params = community_for_entity_params(tenant_id=tenant_id, repo_id=repo_id, entity_id=entity_id, limit=limit)
            with self._driver.session() as session:
                records = session.execute_read(_run_and_collect, COMMUNITY_FOR_ENTITY_QUERY, params)
        return [CommunityMember(entity_id=r["id"], kind=r["kind"], name=r["name"]) for r in records]


class CypherGraphSearch:
    """Adapts real Cypher BFS to the `GraphSearch` protocol used by hybrid retrieval.

    Free-text queries are resolved to seed entities by term overlap against
    `labels_by_entity` (the same lexical match `KeywordGraphSearch` uses),
    then each seed is BFS-expanded through real Cypher so results reflect
    the live graph rather than a pure lexical match.
    """

    def __init__(
        self,
        traversal: Neo4jGraphTraversal,
        *,
        tenant_id: str,
        repo_id: str,
        labels_by_entity: dict[str, str],
    ) -> None:
        self._traversal = traversal
        self._tenant_id = tenant_id
        self._repo_id = repo_id
        self._labels_by_entity = labels_by_entity

    def _resolve_seeds(self, query: str) -> list[str]:
        terms = {term.lower() for term in query.split()}
        scored = []
        for entity_id, label in self._labels_by_entity.items():
            label_terms = set(label.lower().replace(".", " ").split())
            overlap = len(terms & label_terms)
            if overlap:
                scored.append((overlap, entity_id))
        return [entity_id for _, entity_id in sorted(scored, reverse=True)]

    def bfs(self, query: str, *, limit: int = 10) -> list[SearchHit]:
        best_distance: dict[str, int] = {}
        for seed in self._resolve_seeds(query):
            for neighbor in self._traversal.bfs_neighbors(
                tenant_id=self._tenant_id, repo_id=self._repo_id, entity_id=seed, depth=1, limit=limit
            ):
                if neighbor.entity_id not in best_distance or neighbor.distance < best_distance[neighbor.entity_id]:
                    best_distance[neighbor.entity_id] = neighbor.distance
        hits = [
            SearchHit(entity_id=entity_id, score=1.0 / (1 + distance), source="graph")
            for entity_id, distance in best_distance.items()
        ]
        return sorted(hits, key=lambda hit: hit.score, reverse=True)[:limit]
