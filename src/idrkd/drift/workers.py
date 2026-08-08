"""Drift detection and re-index workers."""

from __future__ import annotations

from dataclasses import dataclass

from idrkd.drift.events import EntityChangedEvent, ReindexRequest
from idrkd.drift.queue import ReindexQueue
from idrkd.drift.scoring import (
    CENTROID_DRIFT_THRESHOLD,
    ENTITY_DRIFT_THRESHOLD,
    DriftDecision,
    centroid,
    cosine_distance,
    entity_drift_decision,
)
from idrkd.drift.store import Neo4jDriftStore
from idrkd.graph.traversal import Neo4jGraphTraversal
from idrkd.mcp.backends import Neo4jMcpBackend
from idrkd.observability.metrics import REINDEX_JOBS
from idrkd.observability.tracing import traced_span
from idrkd.rag.embeddings import BgeM3EmbeddingAdapter
from idrkd.rag.vector_store import PostgresVectorStore, VectorRecord


@dataclass(frozen=True)
class EntityDriftResult:
    decision: DriftDecision
    queued: bool
    queue_depth: int


@dataclass(frozen=True)
class CommunityDriftResult:
    community_id: str
    score: float
    triggered: bool
    member_count: int
    queued: int


@dataclass(frozen=True)
class ReindexResult:
    request: ReindexRequest
    entity_ids: tuple[str, ...]
    embeddings: int
    cleared_stale: int


class EntityDriftWorker:
    def __init__(
        self,
        *,
        vector_store: PostgresVectorStore,
        drift_store: Neo4jDriftStore,
        queue: ReindexQueue,
        embeddings: BgeM3EmbeddingAdapter | None = None,
        embedding_model: str = "bge-m3",
        threshold: float = ENTITY_DRIFT_THRESHOLD,
    ) -> None:
        self._vector_store = vector_store
        self._drift_store = drift_store
        self._queue = queue
        self._embeddings = embeddings or BgeM3EmbeddingAdapter()
        self._embedding_model = embedding_model
        self._threshold = threshold

    def process(self, event: EntityChangedEvent) -> EntityDriftResult:
        with traced_span(
            "drift.entity.process",
            correlation_id="",
            tenant_id=event.tenant_id,
            repo_id=event.repo_id,
            entity_id=event.entity_id,
        ):
            previous = self._vector_store.get_entity_embedding(
                tenant_id=event.tenant_id,
                repo_id=event.repo_id,
                entity_id=event.entity_id,
                embedding_model=self._embedding_model,
            )
            current = self._embeddings.embed(event.description)
            decision = entity_drift_decision(
                previous_embedding=previous,
                current_embedding=current,
                threshold=self._threshold,
            )
            self._vector_store.upsert_records(
                [
                    VectorRecord(
                        id=f"emb_{event.entity_id}_{self._embedding_model.replace('/', '_')}",
                        tenant_id=event.tenant_id,
                        repo_id=event.repo_id,
                        entity_id=event.entity_id,
                        text=event.description,
                        embedding=current,
                        source="drift_entity",
                        content_hash=event.content_hash,
                        embedding_model=self._embedding_model,
                        metadata={"community_id": event.community_id, "changed_at": event.changed_at.isoformat()},
                    )
                ]
            )
            self._drift_store.update_entity_drift(
                tenant_id=event.tenant_id,
                repo_id=event.repo_id,
                entity_id=event.entity_id,
                drift_score=decision.score,
                stale=decision.triggered,
                community_id=event.community_id,
            )
            queue_depth = 0
            if decision.triggered:
                queue_depth = self._queue.enqueue(
                    ReindexRequest(
                        tenant_id=event.tenant_id,
                        repo_id=event.repo_id,
                        entity_id=event.entity_id,
                        reason=f"entity_drift:{decision.score:.4f}",
                    )
                )
                REINDEX_JOBS.labels("entity_drift", "queued").inc()
            return EntityDriftResult(decision=decision, queued=decision.triggered, queue_depth=queue_depth)


class CommunityCentroidDriftWorker:
    def __init__(
        self,
        *,
        vector_store: PostgresVectorStore,
        drift_store: Neo4jDriftStore,
        queue: ReindexQueue,
        threshold: float = CENTROID_DRIFT_THRESHOLD,
        embedding_model: str = "bge-m3",
    ) -> None:
        self._vector_store = vector_store
        self._drift_store = drift_store
        self._queue = queue
        self._threshold = threshold
        self._embedding_model = embedding_model
        self._previous: dict[tuple[str, str, str], list[float]] = {}

    def process(self, *, tenant_id: str, repo_id: str, community_id: str) -> CommunityDriftResult:
        with traced_span(
            "drift.community.process",
            correlation_id="",
            tenant_id=tenant_id,
            repo_id=repo_id,
            community_id=community_id,
        ):
            member_ids = self._drift_store.community_member_ids(
                tenant_id=tenant_id,
                repo_id=repo_id,
                community_id=community_id,
            )
            embeddings = self._vector_store.get_embeddings_for_entities(
                tenant_id=tenant_id,
                repo_id=repo_id,
                entity_ids=member_ids,
                embedding_model=self._embedding_model,
            )
            current = centroid(list(embeddings.values()))
            key = (tenant_id, repo_id, community_id)
            previous = self._previous.get(key)
            score = 1.0 if previous is None and current else (cosine_distance(previous or [], current) if current else 0.0)
            triggered = score >= self._threshold
            self._previous[key] = current
            self._drift_store.update_centroid_drift(
                tenant_id=tenant_id,
                repo_id=repo_id,
                community_id=community_id,
                centroid_drift=score,
                stale=triggered,
                centroid_vector=current,
            )
            queued = 0
            if triggered:
                for entity_id in member_ids:
                    self._queue.enqueue(
                        ReindexRequest(
                            tenant_id=tenant_id,
                            repo_id=repo_id,
                            entity_id=entity_id,
                            reason=f"centroid_drift:{community_id}:{score:.4f}",
                        )
                    )
                    queued += 1
                REINDEX_JOBS.labels("centroid_drift", "queued").inc(queued)
            return CommunityDriftResult(
                community_id=community_id,
                score=score,
                triggered=triggered,
                member_count=len(member_ids),
                queued=queued,
            )


class ReindexWorker:
    def __init__(
        self,
        *,
        graph_backend: Neo4jMcpBackend,
        traversal: Neo4jGraphTraversal,
        vector_store: PostgresVectorStore,
        drift_store: Neo4jDriftStore,
        embeddings: BgeM3EmbeddingAdapter | None = None,
        embedding_model: str = "bge-m3",
    ) -> None:
        self._graph_backend = graph_backend
        self._traversal = traversal
        self._vector_store = vector_store
        self._drift_store = drift_store
        self._embeddings = embeddings or BgeM3EmbeddingAdapter()
        self._embedding_model = embedding_model

    def process(self, request: ReindexRequest) -> ReindexResult:
        with traced_span(
            "drift.reindex.process",
            correlation_id="",
            tenant_id=request.tenant_id,
            repo_id=request.repo_id,
            entity_id=request.entity_id,
            reason=request.reason,
        ):
            neighbors = self._traversal.bfs_neighbors(
                tenant_id=request.tenant_id,
                repo_id=request.repo_id,
                entity_id=request.entity_id,
                depth=request.depth,
                limit=50,
            )
            entity_ids = [request.entity_id, *(neighbor.entity_id for neighbor in neighbors)]
            records = []
            for entity_id in dict.fromkeys(entity_ids):
                entity = self._graph_backend.get_entity(
                    tenant_id=request.tenant_id,
                    repo_id=request.repo_id,
                    entity_id=entity_id,
                )
                if entity is None:
                    continue
                text = " ".join(
                    part
                    for part in (
                        entity.kind,
                        entity.name,
                        entity.qualified_name,
                        entity.path,
                        str(entity.properties),
                    )
                    if part
                )
                records.append(
                    VectorRecord(
                        id=f"emb_{entity.id}_{self._embedding_model.replace('/', '_')}",
                        tenant_id=request.tenant_id,
                        repo_id=request.repo_id,
                        entity_id=entity.id,
                        text=text,
                        embedding=self._embeddings.embed(text),
                        source="reindex",
                        content_hash=entity.content_hash,
                        embedding_model=self._embedding_model,
                        metadata={
                            "kind": entity.kind,
                            "name": entity.name,
                            "qualified_name": entity.qualified_name,
                            "path": entity.path,
                            "reason": request.reason,
                        },
                    )
                )
            written = self._vector_store.upsert_records(records)
            cleared = self._drift_store.clear_stale(
                tenant_id=request.tenant_id,
                repo_id=request.repo_id,
                entity_ids=[record.entity_id for record in records],
            )
            return ReindexResult(
                request=request,
                entity_ids=tuple(record.entity_id for record in records),
                embeddings=written,
                cleared_stale=cleared,
            )
