"""Connected LangGraph RAG workflow with conditional AutoGen delegation."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
import hashlib
import operator
import re
from typing import Annotated, Any, Literal, Protocol, TypedDict, cast

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from idrkd.a2a.autogen_reconciler import (
    ReconciliationDelegate,
    ReconciliationRequest,
    ReconciliationResult,
)
from idrkd.distillation.serving import StudentModelClient, student_model_client_from_env
from idrkd.rag.critic import FaithfulnessCritic, FaithfulnessResult
from idrkd.rag.embeddings import BgeM3EmbeddingAdapter
from idrkd.rag.orchestrator import VectorSearch
from idrkd.rag.reranker import MiniLmReranker
from idrkd.rag.retrieval import GraphSearch, HybridHit, reciprocal_rank_fusion
from idrkd.rag.vector_store import SearchHit


QueryRoute = Literal["answer", "reconcile"]
NextStep = Literal["delegate", "critic", "retry", "end"]


class EvidenceSource(Protocol):
    def labels(self, entity_ids: list[str]) -> dict[str, str]:
        ...


class LangGraphQueryState(TypedDict):
    query: str
    tenant_id: str
    repo_id: str
    conflict_id: str
    traceparent: str
    route: QueryRoute
    subqueries: tuple[str, ...]
    retrieval_query: str
    rounds: int
    limit: int
    vector_hits: list[SearchHit]
    graph_hits: list[SearchHit]
    fused_hits: list[HybridHit]
    reranked_hits: list[HybridHit]
    labels: dict[str, str]
    answer: str
    reconciliation: ReconciliationResult | None
    faithfulness: FaithfulnessResult | None
    accepted: bool
    trace: Annotated[list[str], operator.add]


def new_langgraph_state(
    query: str,
    *,
    tenant_id: str,
    repo_id: str,
    labels: dict[str, str] | None = None,
    conflict_id: str | None = None,
    traceparent: str = "",
    limit: int = 10,
) -> LangGraphQueryState:
    return LangGraphQueryState(
        query=query,
        tenant_id=tenant_id,
        repo_id=repo_id,
        conflict_id=conflict_id or "",
        traceparent=traceparent,
        route="answer",
        subqueries=(),
        retrieval_query=query,
        rounds=0,
        limit=limit,
        vector_hits=[],
        graph_hits=[],
        fused_hits=[],
        reranked_hits=[],
        labels=dict(labels or {}),
        answer="",
        reconciliation=None,
        faithfulness=None,
        accepted=False,
        trace=[],
    )


class LangGraphAutoGenOrchestrator:
    """StateGraph with decomposition, parallel retrieval, A2A, and critique."""

    def __init__(
        self,
        *,
        embeddings: BgeM3EmbeddingAdapter,
        vector_store: VectorSearch,
        graph_search: GraphSearch,
        reconciliation_delegate: ReconciliationDelegate,
        evidence_source: EvidenceSource | None = None,
        reranker: MiniLmReranker | None = None,
        critic: FaithfulnessCritic | None = None,
        student_model: StudentModelClient | None = None,
        max_rounds: int = 2,
    ) -> None:
        if max_rounds < 1:
            raise ValueError("max_rounds must be at least one")
        self._embeddings = embeddings
        self._vector_store = vector_store
        self._graph_search = graph_search
        self._delegate = reconciliation_delegate
        self._evidence_source = evidence_source
        self._reranker = reranker or MiniLmReranker()
        self._critic = critic or FaithfulnessCritic()
        self._student_model = student_model or student_model_client_from_env()
        self._max_rounds = max_rounds
        self._graph = self._build_graph()

    async def run(
        self,
        query: str,
        *,
        tenant_id: str,
        repo_id: str,
        labels: dict[str, str] | None = None,
        conflict_id: str | None = None,
        traceparent: str = "",
        limit: int = 10,
    ) -> LangGraphQueryState:
        state = new_langgraph_state(
            query,
            tenant_id=tenant_id,
            repo_id=repo_id,
            labels=labels,
            conflict_id=conflict_id,
            traceparent=traceparent,
            limit=limit,
        )
        result = await self._graph.ainvoke(state)
        return cast(LangGraphQueryState, result)

    def _build_graph(self) -> CompiledStateGraph[Any, Any, Any, Any]:
        graph = StateGraph(LangGraphQueryState)
        graph.add_node("classify", self._classify)
        graph.add_node("prepare_retrieval", self._prepare_retrieval)
        graph.add_node("vector_retrieval", self._vector_retrieval)
        graph.add_node("graph_retrieval", self._graph_retrieval)
        graph.add_node("synthesize", self._synthesize)
        graph.add_node("delegate_reconciliation", self._delegate_reconciliation)
        graph.add_node("reconcile", self._reconcile)
        graph.add_node("critic", self._review)
        graph.add_edge(START, "classify")
        graph.add_edge("classify", "prepare_retrieval")
        graph.add_edge("prepare_retrieval", "vector_retrieval")
        graph.add_edge("prepare_retrieval", "graph_retrieval")
        graph.add_edge(["vector_retrieval", "graph_retrieval"], "synthesize")
        graph.add_conditional_edges(
            "synthesize",
            self._after_synthesis,
            {"delegate": "delegate_reconciliation", "critic": "critic"},
        )
        graph.add_edge("delegate_reconciliation", "reconcile")
        graph.add_edge("reconcile", "critic")
        graph.add_conditional_edges(
            "critic",
            self._after_critic,
            {"retry": "prepare_retrieval", "end": END},
        )
        return graph.compile()

    def _classify(self, state: LangGraphQueryState) -> dict[str, Any]:
        route: QueryRoute = "reconcile" if _requires_reconciliation(state["query"]) else "answer"
        conflict_id = state["conflict_id"]
        if route == "reconcile" and not conflict_id:
            conflict_id = _stable_conflict_id(state)
        return {"route": route, "conflict_id": conflict_id, "trace": ["langgraph:classify"]}

    def _prepare_retrieval(self, state: LangGraphQueryState) -> dict[str, Any]:
        subqueries = decompose_query(state["retrieval_query"])
        return {
            "rounds": state["rounds"] + 1,
            "subqueries": subqueries,
            "vector_hits": [],
            "graph_hits": [],
            "trace": ["langgraph:decompose"],
        }

    async def _vector_retrieval(self, state: LangGraphQueryState) -> dict[str, Any]:
        def search() -> list[SearchHit]:
            rankings = [
                self._vector_store.search(
                    self._embeddings.embed(subquery),
                    limit=state["limit"],
                )
                for subquery in state["subqueries"]
            ]
            return _merge_search_hits(rankings, limit=state["limit"])

        hits = await asyncio.to_thread(search)
        return {"vector_hits": hits, "trace": ["langgraph:vector_retrieval"]}

    async def _graph_retrieval(self, state: LangGraphQueryState) -> dict[str, Any]:
        def search() -> list[SearchHit]:
            rankings = [
                self._graph_search.bfs(subquery, limit=state["limit"])
                for subquery in state["subqueries"]
            ]
            return _merge_search_hits(rankings, limit=state["limit"])

        hits = await asyncio.to_thread(search)
        return {"graph_hits": hits, "trace": ["langgraph:graph_retrieval"]}

    def _synthesize(self, state: LangGraphQueryState) -> dict[str, Any]:
        fused = reciprocal_rank_fusion(
            [state["vector_hits"], state["graph_hits"]],
            limit=state["limit"],
        )
        entity_ids = [hit.entity_id for hit in fused]
        labels = dict(state["labels"])
        if self._evidence_source is not None:
            labels.update(self._evidence_source.labels(entity_ids))
        reranked = self._reranker.rerank(state["query"], fused, labels)
        evidence = [labels.get(hit.entity_id, hit.entity_id) for hit in reranked[:3]]
        if self._student_model is not None and evidence:
            answer = self._student_model.generate(query=state["query"], evidence=evidence)
        else:
            answer = "; ".join(evidence)
        return {
            "labels": labels,
            "fused_hits": fused,
            "reranked_hits": reranked,
            "answer": answer,
            "trace": ["langgraph:synthesize"],
        }

    def _after_synthesis(self, state: LangGraphQueryState) -> NextStep:
        if state["route"] == "reconcile" and state["reconciliation"] is None:
            return "delegate"
        return "critic"

    async def _delegate_reconciliation(self, state: LangGraphQueryState) -> dict[str, Any]:
        evidence = tuple(
            state["labels"].get(hit.entity_id, hit.entity_id)
            for hit in state["reranked_hits"][:5]
        )
        result = await self._delegate.reconcile(
            ReconciliationRequest(
                tenant_id=state["tenant_id"],
                repo_id=state["repo_id"],
                conflict_id=state["conflict_id"],
                query=state["query"],
                evidence=evidence,
                traceparent=state["traceparent"],
            )
        )
        return {"reconciliation": result, "trace": ["a2a:autogen_delegate"]}

    def _reconcile(self, state: LangGraphQueryState) -> dict[str, Any]:
        result = state["reconciliation"]
        if result is None:
            raise RuntimeError("reconciliation node requires an AutoGen result")
        answer = state["answer"].strip()
        recommendation = f"Reconciliation recommendation {result.recommendation}"
        return {
            "answer": f"{answer}. {recommendation}".strip(" ."),
            "trace": ["langgraph:reconcile"],
        }

    def _review(self, state: LangGraphQueryState) -> dict[str, Any]:
        evidence = [
            state["labels"].get(hit.entity_id, hit.entity_id)
            for hit in state["reranked_hits"][:5]
        ]
        if state["reconciliation"] is not None:
            evidence.append(
                f"Reconciliation recommendation {state['reconciliation'].recommendation}"
            )
        result = self._critic.evaluate(state["answer"], evidence)
        update: dict[str, Any] = {
            "faithfulness": result,
            "accepted": result.entailed,
            "trace": ["langgraph:critic"],
        }
        if not result.entailed:
            update["retrieval_query"] = _targeted_follow_up_query(state, result)
        return update

    def _after_critic(self, state: LangGraphQueryState) -> NextStep:
        if state["accepted"] or state["rounds"] >= self._max_rounds:
            return "end"
        return "retry"


def decompose_query(query: str) -> tuple[str, ...]:
    parts = [
        part.strip(" ,.")
        for part in re.split(r"\s+(?:and then|then|and)\s+|[;?]+", query, flags=re.IGNORECASE)
        if part.strip(" ,.")
    ]
    return tuple(dict.fromkeys(parts[:4])) or (query,)


def _requires_reconciliation(query: str) -> bool:
    terms = query.casefold()
    return any(
        marker in terms
        for marker in ("conflict", "reconcile", "disagree", "inconsistent", "which version")
    )


def _stable_conflict_id(state: LangGraphQueryState) -> str:
    value = f"{state['tenant_id']}\0{state['repo_id']}\0{state['query']}".encode()
    return f"query-{hashlib.sha256(value).hexdigest()[:16]}"


def _merge_search_hits(
    rankings: Sequence[Sequence[SearchHit]],
    *,
    limit: int,
) -> list[SearchHit]:
    by_entity: dict[str, SearchHit] = {}
    for ranking in rankings:
        for hit in ranking:
            current = by_entity.get(hit.entity_id)
            if current is None or hit.score > current.score:
                by_entity[hit.entity_id] = hit
    return sorted(by_entity.values(), key=lambda hit: hit.score, reverse=True)[:limit]


def _targeted_follow_up_query(
    state: LangGraphQueryState,
    faithfulness: FaithfulnessResult,
) -> str:
    claims = tuple(
        claim.strip()
        for claim in state["answer"].replace(";", ".").split(".")
        if claim.strip()
    )
    if not claims or len(claims) != len(faithfulness.claim_scores):
        return state["query"]
    weakest = min(
        range(len(faithfulness.claim_scores)),
        key=lambda index: faithfulness.claim_scores[index],
    )
    return f"{state['query']} evidence for {claims[weakest]}"
