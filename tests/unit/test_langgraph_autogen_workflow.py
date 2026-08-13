import httpx

from idrkd.a2a import A2ABridge, IdrkdA2AClient, build_a2a_app, build_idrkd_agent_card
from idrkd.a2a.autogen_reconciler import (
    A2AReconciliationClient,
    AutoGenReconciliationAgent,
    AutoGenReconciliationExecutor,
    ReconciliationRequest,
    ReconciliationResult,
)
from idrkd.mcp.tools import McpToolRegistry
from idrkd.rag.embeddings import BgeM3EmbeddingAdapter
from idrkd.rag.langgraph_orchestrator import LangGraphAutoGenOrchestrator, decompose_query
from idrkd.rag.retrieval import KeywordGraphSearch
from idrkd.rag.vector_store import InMemoryVectorStore, VectorRecord


class _StudentModel:
    def generate(self, *, query: str, evidence: list[str]) -> str:
        return evidence[0]


class _RecordingDelegate:
    def __init__(self) -> None:
        self.requests: list[ReconciliationRequest] = []

    async def reconcile(self, request: ReconciliationRequest) -> ReconciliationResult:
        self.requests.append(request)
        return ReconciliationResult(
            conflict_id=request.conflict_id,
            recommendation="review_latest_entity_version",
            details={"source": "test"},
        )


def _retrieval_components():  # noqa: ANN202
    embeddings = BgeM3EmbeddingAdapter(dimensions=32)
    store = InMemoryVectorStore()
    store.upsert(
        VectorRecord(
            id="vec-a",
            entity_id="entity-a",
            text="Customer API handles billing",
            embedding=embeddings.embed("Customer API handles billing"),
            metadata={},
        )
    )
    labels = {"entity-a": "Customer API handles billing"}
    return embeddings, store, KeywordGraphSearch(labels), labels


def test_query_decomposition_produces_bounded_unique_subqueries() -> None:
    assert decompose_query("Find Customer API and then inspect billing; reconcile conflict") == (
        "Find Customer API",
        "inspect billing",
        "reconcile conflict",
    )


async def test_langgraph_runs_parallel_retrieval_without_unneeded_delegation() -> None:
    embeddings, store, graph, labels = _retrieval_components()
    delegate = _RecordingDelegate()
    orchestrator = LangGraphAutoGenOrchestrator(
        embeddings=embeddings,
        vector_store=store,
        graph_search=graph,
        reconciliation_delegate=delegate,
        student_model=_StudentModel(),
    )

    state = await orchestrator.run(
        "Customer API billing",
        tenant_id="tenant-a",
        repo_id="repo-a",
        labels=labels,
    )

    assert state["accepted"] is True
    assert state["route"] == "answer"
    assert delegate.requests == []
    assert "langgraph:vector_retrieval" in state["trace"]
    assert "langgraph:graph_retrieval" in state["trace"]
    assert state["trace"].index("langgraph:synthesize") > state["trace"].index(
        "langgraph:vector_retrieval"
    )
    assert state["trace"].index("langgraph:synthesize") > state["trace"].index(
        "langgraph:graph_retrieval"
    )


async def test_langgraph_delegates_reconciliation_through_a2a_to_autogen() -> None:
    registry = McpToolRegistry(principal_tenant_id="tenant-a")
    reconciler_card = build_idrkd_agent_card(
        name="IDRKD AutoGen Reconciler",
        description="Runs governed reconciliation.",
        version="0.1.0",
        endpoint="http://testserver/",
        capabilities=("reconcile",),
    )
    app = build_a2a_app(
        reconciler_card,
        agent_executor=AutoGenReconciliationExecutor(AutoGenReconciliationAgent(registry)),
    )
    planner_card = build_idrkd_agent_card(
        name="IDRKD LangGraph Planner",
        description="Routes retrieval and reconciliation.",
        version="0.1.0",
        endpoint="http://planner.local/a2a",
        capabilities=("query.plan",),
    )
    transport = httpx.ASGITransport(app=app)
    http_client = httpx.AsyncClient(transport=transport, base_url="http://testserver")
    client = IdrkdA2AClient(reconciler_card, httpx_client=http_client)
    delegate = A2AReconciliationClient(
        client=client,
        bridge=A2ABridge(local_card=planner_card, shared_secret="test-secret"),
    )
    embeddings, store, graph, labels = _retrieval_components()
    orchestrator = LangGraphAutoGenOrchestrator(
        embeddings=embeddings,
        vector_store=store,
        graph_search=graph,
        reconciliation_delegate=delegate,
        student_model=_StudentModel(),
    )

    try:
        state = await orchestrator.run(
            "Find Customer API and reconcile conflict conf-1",
            tenant_id="tenant-a",
            repo_id="repo-a",
            labels=labels,
            conflict_id="conf-1",
        )
    finally:
        await delegate.close()

    assert state["route"] == "reconcile"
    assert state["reconciliation"] is not None
    assert state["reconciliation"].framework == "autogen"
    assert state["reconciliation"].recommendation == "review_latest_entity_version"
    assert state["accepted"] is True
    assert "a2a:autogen_delegate" in state["trace"]
    assert "langgraph:reconcile" in state["trace"]
    assert "Reconciliation recommendation" in state["answer"]
