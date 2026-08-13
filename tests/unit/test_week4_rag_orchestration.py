from idrkd.rag.critic import FaithfulnessCritic
from idrkd.rag.embeddings import BgeM3EmbeddingAdapter
from idrkd.rag.orchestrator import MAX_ROUNDS, AgenticRagOrchestrator, new_query_state
from idrkd.rag.query_slo import QuerySlo, percentile, recall_at_k
from idrkd.rag.retrieval import KeywordGraphSearch
from idrkd.rag.vector_store import InMemoryVectorStore, VectorRecord


class _StudentModel:
    def generate(self, *, query: str, evidence: list[str]) -> str:
        return evidence[0]


class _UnsupportedStudentModel:
    def generate(self, *, query: str, evidence: list[str]) -> str:
        return "Order API deletes invoices."


def _build_orchestrator() -> tuple[AgenticRagOrchestrator, dict[str, str]]:
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
    labels = {"entity-a": "Customer API", "entity-b": "Billing Worker"}
    graph = KeywordGraphSearch(labels)
    orchestrator = AgenticRagOrchestrator(embeddings=embeddings, vector_store=store, graph_search=graph)
    return orchestrator, labels


def test_new_query_state_starts_at_zero_rounds() -> None:
    state = new_query_state("customer API")

    assert state["rounds"] == 0
    assert state["accepted"] is False
    assert state["trace"] == []


def test_orchestrator_runs_five_agent_trace_and_accepts_grounded_answer() -> None:
    orchestrator, labels = _build_orchestrator()

    state = orchestrator.run("customer API", labels=labels)

    assert state["trace"] == ["router", "vector_retriever", "graph_traversal", "synthesis", "critic"]
    assert state["accepted"] is True
    assert state["faithfulness"] is not None
    assert state["faithfulness"].score >= 0.78
    assert "Customer API" in state["answer"]
    assert state["rounds"] == 1


def test_orchestrator_can_synthesize_with_served_student_model() -> None:
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
    labels = {"entity-a": "Customer API"}
    orchestrator = AgenticRagOrchestrator(
        embeddings=embeddings,
        vector_store=store,
        graph_search=KeywordGraphSearch(labels),
        student_model=_StudentModel(),
    )

    state = orchestrator.run("customer API", labels=labels)

    assert state["answer"] == "Customer API"
    assert state["accepted"] is True


def test_orchestrator_bounds_re_retrieve_loop_at_max_rounds() -> None:
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
    graph = KeywordGraphSearch({"entity-a": "Customer API handles billing."})
    orchestrator = AgenticRagOrchestrator(
        embeddings=embeddings,
        vector_store=store,
        graph_search=graph,
        critic=FaithfulnessCritic(threshold=0.78),
        student_model=_UnsupportedStudentModel(),
    )

    state = orchestrator.run("customer API", labels={"entity-a": "Customer API handles billing."})

    assert state["rounds"] == MAX_ROUNDS
    assert state["accepted"] is False
    assert state["retrieval_query"] != state["query"]
    assert "evidence for" in state["retrieval_query"]


def test_faithfulness_critic_scores_supported_answer_higher() -> None:
    critic = FaithfulnessCritic(threshold=0.78)

    grounded = critic.evaluate("Customer API", ["Customer API handles billing lookups"])
    ungrounded = critic.evaluate("Customer API", ["Totally unrelated evidence text"])

    assert grounded.entailed is True
    assert ungrounded.entailed is False
    assert grounded.score > ungrounded.score


def test_faithfulness_critic_scores_each_claim_independently() -> None:
    critic = FaithfulnessCritic(threshold=0.78)

    result = critic.evaluate(
        "Customer API handles billing. Order API deletes invoices.",
        ["Customer API handles billing."],
    )

    assert result.entailed is False
    assert len(result.claim_scores) == 2
    assert result.claim_scores[0] > result.claim_scores[1]


def test_query_slo_gate_checks_latency_and_bounded_rounds() -> None:
    slo = QuerySlo()

    assert slo.check(p50_seconds=2.5, p95_seconds=7.5, rounds=2) is True
    assert slo.check(p50_seconds=3.5, p95_seconds=7.5, rounds=2) is False
    assert slo.check(p50_seconds=2.5, p95_seconds=7.5, rounds=3) is False


def test_percentile_and_recall_at_k() -> None:
    latencies = [1.0, 2.0, 3.0, 4.0, 5.0]

    assert percentile(latencies, 50) == 3.0
    assert percentile([], 50) == 0.0
    assert recall_at_k(["a", "b", "c"], {"a", "c", "z"}, k=10) == 2 / 3
