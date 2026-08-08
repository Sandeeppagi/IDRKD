import httpx

from idrkd.drift import EntityChangedEvent, EntityDriftWorker, InMemoryReindexQueue
from idrkd.mcp.server import create_mcp_app
from idrkd.mcp.tools import McpToolRegistry
from idrkd.observability.metrics import metrics_response
from idrkd.rag.embeddings import BgeM3EmbeddingAdapter
from idrkd.rag.vector_store import InMemoryVectorStore, VectorRecord


class _TinyEmbedding(BgeM3EmbeddingAdapter):
    def __init__(self) -> None:
        super().__init__(dimensions=2)

    def embed(self, text: str) -> list[float]:
        return [0.0, 1.0] if text == "changed" else [1.0, 0.0]


class _DriftStore:
    def update_entity_drift(self, **_kwargs) -> None:  # noqa: ANN003
        return None


class _VectorStore:
    def __init__(self) -> None:
        self.embeddings: dict[str, list[float]] = {}

    def get_entity_embedding(self, *, entity_id: str, **_kwargs) -> list[float] | None:  # noqa: ANN003
        return self.embeddings.get(entity_id)

    def upsert_records(self, records: list[VectorRecord]) -> int:
        for record in records:
            self.embeddings[record.entity_id] = record.embedding
        return len(records)


async def test_metrics_endpoint_is_non_empty_after_mcp_workload() -> None:
    registry = McpToolRegistry(
        principal_tenant_id="tenant-a",
        handlers={"search_code": lambda params: {"hits": [params["query"]]}},
    )
    app = create_mcp_app(registry)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": "metrics-smoke",
                "method": "tools/call",
                "params": {
                    "name": "search_code",
                    "arguments": {"tenant_id": "tenant-a", "repo_id": "repo", "query": "customer api"},
                },
            },
        )
        metrics = await client.get("/metrics")

    assert response.status_code == 200
    assert response.json()["result"]["hits"] == ["customer api"]
    body = metrics.text
    assert "idrkd_mcp_tool_calls_total" in body
    assert "idrkd_mcp_tool_latency_seconds" in body
    assert 'tool="search_code"' in body


def test_component_workload_emits_embedding_and_reindex_metrics() -> None:
    instrumented_store = InMemoryVectorStore()
    instrumented_store.upsert_records(
        [
            VectorRecord(
                id="emb-a",
                tenant_id="tenant-a",
                repo_id="repo",
                entity_id="entity-a",
                text="entity a",
                embedding=[1.0, 0.0],
                source="unit",
            )
        ]
    )
    vector_store = _VectorStore()
    vector_store.upsert_records(
        [
            VectorRecord(
                id="emb-a",
                tenant_id="tenant-a",
                repo_id="repo",
                entity_id="entity-a",
                text="entity a",
                embedding=[1.0, 0.0],
                source="unit",
            )
        ]
    )
    worker = EntityDriftWorker(
        vector_store=vector_store,  # type: ignore[arg-type]
        drift_store=_DriftStore(),  # type: ignore[arg-type]
        queue=InMemoryReindexQueue(),
        embeddings=_TinyEmbedding(),
        threshold=0.1,
    )

    result = worker.process(
        EntityChangedEvent(
            tenant_id="tenant-a",
            repo_id="repo",
            entity_id="entity-a",
            content_hash="hash-new",
            description="changed",
        )
    )

    metrics = metrics_response()[0].decode()
    assert result.queued is True
    assert "idrkd_embedding_upserts_total" in metrics
    assert "idrkd_reindex_jobs_total" in metrics
