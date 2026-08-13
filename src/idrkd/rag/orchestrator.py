"""Week 4 agentic RAG orchestrator: 5-agent LangGraph-style state machine.

RouterAgent -> VectorRetrieverAgent -> GraphTraversalAgent -> SynthesisAgent
-> CriticAgent, with a bounded re-retrieve loop gated by the faithfulness
critic and capped by `QueryState["rounds"]`.
"""

from __future__ import annotations

from typing import Protocol, TypedDict

from idrkd.distillation.serving import StudentModelClient, student_model_client_from_env
from idrkd.rag.critic import FaithfulnessCritic, FaithfulnessResult
from idrkd.rag.embeddings import BgeM3EmbeddingAdapter
from idrkd.rag.reranker import MiniLmReranker
from idrkd.rag.retrieval import GraphSearch, HybridHit, reciprocal_rank_fusion
from idrkd.rag.vector_store import SearchHit


MAX_ROUNDS = 2


class VectorSearch(Protocol):
    def search(self, embedding: list[float], *, limit: int = 10) -> list[SearchHit]:
        ...


class QueryState(TypedDict):
    query: str
    retrieval_query: str
    rounds: int
    vector_hits: list[SearchHit]
    graph_hits: list[SearchHit]
    fused_hits: list[HybridHit]
    reranked_hits: list[HybridHit]
    answer: str
    faithfulness: FaithfulnessResult | None
    accepted: bool
    trace: list[str]


def new_query_state(query: str) -> QueryState:
    return QueryState(
        query=query,
        retrieval_query=query,
        rounds=0,
        vector_hits=[],
        graph_hits=[],
        fused_hits=[],
        reranked_hits=[],
        answer="",
        faithfulness=None,
        accepted=False,
        trace=[],
    )


class RouterAgent:
    """Marks the entry hop of the state machine; routing is single-strategy today."""

    def route(self, state: QueryState) -> QueryState:
        state["trace"].append("router")
        return state


class VectorRetrieverAgent:
    def __init__(self, *, embeddings: BgeM3EmbeddingAdapter, vector_store: VectorSearch) -> None:
        self._embeddings = embeddings
        self._vector_store = vector_store

    def retrieve(self, state: QueryState, *, limit: int = 10) -> QueryState:
        state["vector_hits"] = self._vector_store.search(
            self._embeddings.embed(state["retrieval_query"]),
            limit=limit,
        )
        state["trace"].append("vector_retriever")
        return state


class GraphTraversalAgent:
    def __init__(self, graph_search: GraphSearch) -> None:
        self._graph_search = graph_search

    def traverse(self, state: QueryState, *, limit: int = 10) -> QueryState:
        state["graph_hits"] = self._graph_search.bfs(state["retrieval_query"], limit=limit)
        state["trace"].append("graph_traversal")
        return state


class SynthesisAgent:
    def __init__(
        self,
        reranker: MiniLmReranker | None = None,
        student_model: StudentModelClient | None = None,
    ) -> None:
        self._reranker = reranker or MiniLmReranker()
        self._student_model = student_model

    def synthesize(self, state: QueryState, *, labels: dict[str, str], top_n: int = 3) -> QueryState:
        fused = reciprocal_rank_fusion([state["vector_hits"], state["graph_hits"]])
        reranked = self._reranker.rerank(state["query"], fused, labels)
        evidence = [labels.get(hit.entity_id, hit.entity_id) for hit in reranked[:top_n]]
        state["fused_hits"] = fused
        state["reranked_hits"] = reranked
        if self._student_model is not None and evidence:
            state["answer"] = self._student_model.generate(query=state["query"], evidence=evidence)
        else:
            state["answer"] = "; ".join(evidence)
        state["trace"].append("synthesis")
        return state


class CriticAgent:
    def __init__(self, critic: FaithfulnessCritic | None = None) -> None:
        self._critic = critic or FaithfulnessCritic()

    def review(self, state: QueryState, *, labels: dict[str, str], evidence_top_n: int = 5) -> QueryState:
        evidence = [labels.get(hit.entity_id, hit.entity_id) for hit in state["reranked_hits"][:evidence_top_n]]
        result = self._critic.evaluate(state["answer"], evidence)
        state["faithfulness"] = result
        state["accepted"] = result.entailed
        if not result.entailed:
            state["retrieval_query"] = _targeted_follow_up_query(state)
        state["trace"].append("critic")
        return state


def _targeted_follow_up_query(state: QueryState) -> str:
    faithfulness = state["faithfulness"]
    if faithfulness is None or not faithfulness.claim_scores:
        return state["query"]
    claims = tuple(claim.strip() for claim in state["answer"].replace(";", ".").split(".") if claim.strip())
    if len(claims) != len(faithfulness.claim_scores):
        return state["query"]
    weakest_index = min(
        range(len(faithfulness.claim_scores)),
        key=lambda index: faithfulness.claim_scores[index],
    )
    return f"{state['query']} evidence for {claims[weakest_index]}"


class AgenticRagOrchestrator:
    """Bounded 5-agent state machine over hybrid vector/graph retrieval."""

    def __init__(
        self,
        *,
        embeddings: BgeM3EmbeddingAdapter,
        vector_store: VectorSearch,
        graph_search: GraphSearch,
        reranker: MiniLmReranker | None = None,
        critic: FaithfulnessCritic | None = None,
        student_model: StudentModelClient | None = None,
        max_rounds: int = MAX_ROUNDS,
    ) -> None:
        self._router = RouterAgent()
        self._vector_agent = VectorRetrieverAgent(embeddings=embeddings, vector_store=vector_store)
        self._graph_agent = GraphTraversalAgent(graph_search)
        self._synthesis_agent = SynthesisAgent(reranker, student_model or student_model_client_from_env())
        self._critic_agent = CriticAgent(critic)
        self._max_rounds = max_rounds

    def run(self, query: str, *, labels: dict[str, str], limit: int = 10) -> QueryState:
        state = new_query_state(query)
        while True:
            state["rounds"] += 1
            state = self._router.route(state)
            state = self._vector_agent.retrieve(state, limit=limit)
            state = self._graph_agent.traverse(state, limit=limit)
            state = self._synthesis_agent.synthesize(state, labels=labels)
            state = self._critic_agent.review(state, labels=labels)
            if state["accepted"] or state["rounds"] >= self._max_rounds:
                return state
