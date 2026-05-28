from idrkd.graph.analytics import NightlyGraphAnalyticsJob, bfs_neighbors
from idrkd.rag.embeddings import BgeM3EmbeddingAdapter, cosine_similarity
from idrkd.rag.reranker import MiniLmReranker
from idrkd.rag.retrieval import HybridRetriever, KeywordGraphSearch, reciprocal_rank_fusion
from idrkd.rag.vector_store import InMemoryVectorStore, SearchHit, VectorRecord


def test_bge_m3_embedding_adapter_is_deterministic_and_normalized() -> None:
    adapter = BgeM3EmbeddingAdapter(dimensions=32)

    first = adapter.embed("customer API")
    second = adapter.embed("customer API")

    assert first == second
    assert round(cosine_similarity(first, first), 6) == 1.0


def test_reciprocal_rank_fusion_combines_vector_and_graph_hits() -> None:
    fused = reciprocal_rank_fusion(
        [
            [SearchHit("a", 0.9, "vector"), SearchHit("b", 0.8, "vector")],
            [SearchHit("b", 1.0, "graph"), SearchHit("c", 0.7, "graph")],
        ],
        limit=3,
    )

    assert fused[0].entity_id == "b"
    assert fused[0].sources == ("graph", "vector")


def test_hybrid_retriever_and_reranker() -> None:
    embeddings = BgeM3EmbeddingAdapter(dimensions=32)
    store = InMemoryVectorStore()
    store.upsert(
        VectorRecord(
            id="vec-a",
            entity_id="entity-a",
            text="customer API",
            embedding=embeddings.embed("customer API"),
            metadata={},
        )
    )
    graph = KeywordGraphSearch({"entity-a": "Customer API", "entity-b": "Billing Worker"})
    retriever = HybridRetriever(embeddings=embeddings, vector_store=store, graph_search=graph)

    hits = retriever.search("customer API")
    reranked = MiniLmReranker().rerank("customer API", hits, {"entity-a": "Customer API"})

    assert reranked[0].entity_id == "entity-a"


def test_graph_analytics_bfs_pagerank_and_communities() -> None:
    edges = [("a", "b"), ("b", "c"), ("x", "y")]
    result = NightlyGraphAnalyticsJob().run(edges)

    assert bfs_neighbors(edges, "a", depth=2) == ["b", "c"]
    assert set(result.pagerank) == {"a", "b", "c", "x", "y"}
    assert result.communities["a"] == result.communities["c"]
    assert result.communities["x"] == result.communities["y"]
    assert result.communities["a"] != result.communities["x"]
