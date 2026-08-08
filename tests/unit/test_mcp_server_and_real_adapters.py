import httpx

from idrkd.ingestion.document_extractor import SpanBertNerExtractor
from idrkd.mcp.server import create_mcp_app
from idrkd.mcp.tools import McpToolRegistry
from idrkd.rag.critic import FaithfulnessCritic
from idrkd.rag.embeddings import BgeM3EmbeddingAdapter
from idrkd.rag.reranker import MiniLmReranker
from idrkd.rag.retrieval import HybridHit


async def test_standalone_mcp_fastapi_server_handles_json_rpc() -> None:
    registry = McpToolRegistry(principal_tenant_id="tenant-a")
    app = create_mcp_app(registry)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post("/mcp", json={"jsonrpc": "2.0", "method": "tools/list", "id": "1"})
        health = await client.get("/healthz")

    assert response.status_code == 200
    assert len(response.json()["result"]["tools"]) == 14
    assert health.json() == {"status": "ok"}


class _EmbeddingModel:
    def encode(self, _text: str) -> list[float]:
        return [0.5, 0.25]


def test_bge_adapter_can_use_real_model_shape() -> None:
    adapter = BgeM3EmbeddingAdapter(dimensions=4, model=_EmbeddingModel())

    assert adapter.embed("customer api") == [0.5, 0.25, 0.0, 0.0]


class _CrossEncoder:
    def predict(self, _pairs):
        return [0.1, 0.9]


def test_minilm_reranker_can_use_cross_encoder_scores() -> None:
    reranker = MiniLmReranker(model=_CrossEncoder())
    hits = [HybridHit("a", 0.1), HybridHit("b", 0.1)]

    reranked = reranker.rerank("query", hits, {"a": "weak", "b": "strong"})

    assert [hit.entity_id for hit in reranked] == ["b", "a"]


class _NerPipeline:
    def __call__(self, _text: str):
        return [{"word": "Customer API", "entity_group": "ORG", "start": 0, "end": 12, "score": 0.98}]


def test_spanbert_extractor_can_use_transformer_pipeline() -> None:
    extractor = SpanBertNerExtractor(pipeline=_NerPipeline())

    entities = extractor.extract("Customer API")

    assert entities[0].text == "Customer API"
    assert entities[0].confidence == 0.98


class _NliPipeline:
    def __call__(self, *_args, **_kwargs):
        return {"labels": ["entailed", "unsupported"], "scores": [0.91, 0.09]}


def test_faithfulness_critic_can_use_nli_pipeline() -> None:
    critic = FaithfulnessCritic(nli_pipeline=_NliPipeline())

    result = critic.evaluate("Customer API handles billing", ["Customer API handles billing"])

    assert result.entailed is True
    assert result.score == 0.91
