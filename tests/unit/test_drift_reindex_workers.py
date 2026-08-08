from dataclasses import dataclass, field

from idrkd.drift import (
    CommunityCentroidDriftWorker,
    EntityChangedEvent,
    EntityDriftWorker,
    InMemoryReindexQueue,
    ReindexRequest,
    ReindexWorker,
    centroid,
    cosine_distance,
    entity_drift_decision,
)
from idrkd.drift.queue import reindex_request_from_payload
from idrkd.mcp.backends import EntityRecord
from idrkd.rag.embeddings import BgeM3EmbeddingAdapter
from idrkd.rag.vector_store import VectorRecord


class _TinyEmbedding(BgeM3EmbeddingAdapter):
    def __init__(self, values_by_text: dict[str, list[float]]) -> None:
        super().__init__(dimensions=2)
        self._values_by_text = values_by_text

    def embed(self, text: str) -> list[float]:
        return self._values_by_text.get(text, [1.0, 0.0])


@dataclass
class _VectorStore:
    embeddings: dict[str, list[float]] = field(default_factory=dict)
    records: list[VectorRecord] = field(default_factory=list)

    def get_entity_embedding(self, *, entity_id: str, **_kwargs) -> list[float] | None:
        return self.embeddings.get(entity_id)

    def get_embeddings_for_entities(self, *, entity_ids: list[str], **_kwargs) -> dict[str, list[float]]:
        return {entity_id: self.embeddings[entity_id] for entity_id in entity_ids if entity_id in self.embeddings}

    def upsert_records(self, records: list[VectorRecord]) -> int:
        self.records.extend(records)
        for record in records:
            self.embeddings[record.entity_id] = record.embedding
        return len(records)


@dataclass
class _DriftStore:
    entity_updates: list[dict[str, object]] = field(default_factory=list)
    centroid_updates: list[dict[str, object]] = field(default_factory=list)
    cleared: list[str] = field(default_factory=list)
    members: list[str] = field(default_factory=lambda: ["a", "b"])

    def update_entity_drift(self, **kwargs) -> None:
        self.entity_updates.append(kwargs)

    def update_centroid_drift(self, **kwargs) -> int:
        self.centroid_updates.append(kwargs)
        return len(self.members)

    def clear_stale(self, *, entity_ids: list[str], **_kwargs) -> int:
        self.cleared.extend(entity_ids)
        return len(entity_ids)

    def community_member_ids(self, **_kwargs) -> list[str]:
        return self.members


class _GraphBackend:
    def get_entity(self, *, entity_id: str, **_kwargs) -> EntityRecord | None:
        return EntityRecord(
            id=entity_id,
            kind="function",
            name=f"name-{entity_id}",
            qualified_name=f"pkg.{entity_id}",
            path="src/example.py",
            content_hash=f"hash-{entity_id}",
            properties={"args": []},
        )


class _Traversal:
    def bfs_neighbors(self, **_kwargs):
        class Neighbor:
            entity_id = "b"

        return [Neighbor()]


def test_cosine_and_entity_drift_decision() -> None:
    assert cosine_distance([1.0, 0.0], [1.0, 0.0]) == 0.0

    decision = entity_drift_decision(previous_embedding=[1.0, 0.0], current_embedding=[0.0, 1.0])

    assert decision.triggered is True
    assert round(decision.score, 2) == 1.0


def test_entity_drift_worker_updates_pgvector_state_and_enqueues_reindex() -> None:
    vector_store = _VectorStore(embeddings={"entity-a": [1.0, 0.0]})
    drift_store = _DriftStore()
    queue = InMemoryReindexQueue()
    worker = EntityDriftWorker(
        vector_store=vector_store,  # type: ignore[arg-type]
        drift_store=drift_store,  # type: ignore[arg-type]
        queue=queue,
        embeddings=_TinyEmbedding({"changed": [0.0, 1.0]}),
        threshold=0.15,
    )

    result = worker.process(
        EntityChangedEvent(
            tenant_id="tenant-a",
            repo_id="repo-a",
            entity_id="entity-a",
            content_hash="hash-new",
            description="changed",
            community_id="c1",
        )
    )

    assert result.queued is True
    assert queue.dequeue() is not None
    assert drift_store.entity_updates[0]["stale"] is True
    assert vector_store.records[0].source == "drift_entity"


def test_community_centroid_worker_marks_members_and_queues_them() -> None:
    vector_store = _VectorStore(embeddings={"a": [1.0, 0.0], "b": [1.0, 0.0]})
    drift_store = _DriftStore(members=["a", "b"])
    queue = InMemoryReindexQueue()
    worker = CommunityCentroidDriftWorker(
        vector_store=vector_store,  # type: ignore[arg-type]
        drift_store=drift_store,  # type: ignore[arg-type]
        queue=queue,
        threshold=0.10,
    )

    result = worker.process(tenant_id="tenant-a", repo_id="repo-a", community_id="c1")

    assert centroid([[1.0, 0.0], [0.0, 1.0]]) == [0.5, 0.5]
    assert result.triggered is True
    assert result.queued == 2
    assert drift_store.centroid_updates[0]["stale"] is True


def test_reindex_worker_reembeds_requested_entity_and_neighbors() -> None:
    vector_store = _VectorStore()
    drift_store = _DriftStore()
    worker = ReindexWorker(
        graph_backend=_GraphBackend(),  # type: ignore[arg-type]
        traversal=_Traversal(),  # type: ignore[arg-type]
        vector_store=vector_store,  # type: ignore[arg-type]
        drift_store=drift_store,  # type: ignore[arg-type]
        embeddings=BgeM3EmbeddingAdapter(dimensions=4),
    )

    result = worker.process(
        ReindexRequest(
            tenant_id="tenant-a",
            repo_id="repo-a",
            entity_id="a",
            reason="test",
        )
    )

    assert result.embeddings == 2
    assert result.cleared_stale == 2
    assert result.entity_ids == ("a", "b")


def test_mcp_reindex_payload_becomes_worker_request() -> None:
    request = reindex_request_from_payload(
        {
            "tenant_id": "tenant-live",
            "repo_id": "week5-e2e",
            "entity_id": "entity-a",
            "status": "queued",
        }
    )

    assert request.tenant_id == "tenant-live"
    assert request.repo_id == "week5-e2e"
    assert request.entity_id == "entity-a"
    assert request.reason == "mcp_enqueue_reindex"
    assert request.depth == 2
