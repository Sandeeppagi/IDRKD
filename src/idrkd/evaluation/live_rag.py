"""Live repository RAG faithfulness benchmark.

The release path deliberately uses the production pgvector and Neo4j adapters.
It does not substitute the deterministic in-memory stores used by unit tests.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import statistics
import time
from typing import Any, Protocol

from idrkd.graph.traversal import Neo4jGraphTraversal
from idrkd.mcp.backends import EntityRecord, Neo4jMcpBackend
from idrkd.rag.critic import FaithfulnessCritic
from idrkd.rag.embeddings import BgeM3EmbeddingAdapter
from idrkd.rag.orchestrator import (
    CriticAgent,
    GraphTraversalAgent,
    RouterAgent,
    SynthesisAgent,
    VectorRetrieverAgent,
    new_query_state,
)
from idrkd.rag.reranker import MiniLmReranker
from idrkd.rag.vector_store import PostgresVectorStore, SearchHit
from idrkd.distillation.serving import OpenAICompatibleStudentClient


@dataclass(frozen=True)
class LiveRagCase:
    case_id: str
    query: str
    tenant_id: str
    repo_id: str
    expected_entity_ids: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, value: dict[str, Any], *, line_number: int) -> LiveRagCase:
        required = ("id", "query", "tenant_id", "repo_id")
        missing = [key for key in required if not str(value.get(key, "")).strip()]
        if missing:
            raise ValueError(f"Live RAG case line {line_number} is missing: {', '.join(missing)}")
        expected = value.get("expected_entity_ids", [])
        if not isinstance(expected, list) or not all(
            isinstance(item, str) and item.strip() for item in expected
        ):
            raise ValueError(
                f"Live RAG case line {line_number} expected_entity_ids must be a string list"
            )
        return cls(
            case_id=str(value["id"]),
            query=str(value["query"]),
            tenant_id=str(value["tenant_id"]),
            repo_id=str(value["repo_id"]),
            expected_entity_ids=tuple(expected),
        )


def load_live_rag_cases(
    path: Path,
    *,
    require_expected_entities: bool = False,
) -> list[LiveRagCase]:
    cases: list[LiveRagCase] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Live RAG case line {line_number} must be a JSON object")
        cases.append(LiveRagCase.from_dict(value, line_number=line_number))
    if not cases:
        raise ValueError(f"Live RAG case file is empty: {path}")
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("Live RAG case IDs must be unique")
    if require_expected_entities:
        missing = [case.case_id for case in cases if not case.expected_entity_ids]
        if missing:
            raise ValueError(
                "Live RAG release cases require non-empty expected_entity_ids: "
                + ", ".join(missing)
            )
    return cases


def annotate_live_rag_retrieval(
    artifact: dict[str, Any],
    cases: list[LiveRagCase],
    *,
    limit: int = 10,
) -> dict[str, Any]:
    """Attach curated retrieval oracles without rewriting historical model scores."""

    if limit <= 0:
        raise ValueError("Retrieval limit must be positive")
    result = copy.deepcopy(artifact)
    raw_results = result.get("cases")
    if not isinstance(raw_results, list):
        raise ValueError("Live RAG artifact cases must be a list")
    oracles = {case.case_id: case for case in cases}
    result_ids = {str(item.get("case_id", "")) for item in raw_results if isinstance(item, dict)}
    if len(raw_results) != len(oracles) or result_ids != set(oracles):
        raise ValueError("Live RAG artifact and oracle case IDs must match exactly")

    recalls: list[float] = []
    query_mismatch_count = 0
    for item in raw_results:
        if not isinstance(item, dict):
            raise ValueError("Live RAG artifact case must be a JSON object")
        oracle = oracles[str(item["case_id"])]
        if item.get("tenant_id") != oracle.tenant_id or item.get("repo_id") != oracle.repo_id:
            raise ValueError(f"Live RAG scope does not match oracle case: {oracle.case_id}")
        if str(item.get("query", "")) != oracle.query:
            item["oracle_query"] = oracle.query
            query_mismatch_count += 1
        expected = set(oracle.expected_entity_ids)
        if not expected:
            raise ValueError(f"Live RAG oracle case has no expected entities: {oracle.case_id}")
        hits = item.get("reranked_hits", [])
        retrieved = {
            str(hit["entity_id"])
            for hit in hits[:limit]
            if isinstance(hit, dict) and hit.get("entity_id")
        }
        recall = len(expected & retrieved) / len(expected)
        item["expected_entity_ids"] = list(oracle.expected_entity_ids)
        item["retrieval_recall"] = recall
        item["retrieval_recall_at_k"] = recall
        recalls.append(recall)

    atomic_claim_case_count = sum(
        bool(item.get("faithfulness_claim_scores"))
        for item in raw_results
        if isinstance(item, dict)
    )
    result["retrieval_k"] = limit
    result["retrieval_recall_at_k_mean"] = statistics.fmean(recalls) if recalls else None
    result["retrieval_recall_mean"] = result["retrieval_recall_at_k_mean"]
    result["retrieval_recall_case_count"] = len(recalls)
    result["expected_entity_case_count"] = len(recalls)
    result["atomic_claim_case_count"] = atomic_claim_case_count
    result["atomic_claim_scoring"] = atomic_claim_case_count == len(raw_results)
    result["faithfulness_aggregation"] = (
        "minimum-atomic-claim-score"
        if result["atomic_claim_scoring"]
        else "legacy-whole-answer-score"
    )
    result["retrieval_oracle_annotation"] = {
        "method": "post-hoc-against-recorded-reranked-hits",
        "faithfulness_recomputed": False,
        "query_mismatch_count": query_mismatch_count,
    }
    return result


class EvidenceSource(Protocol):
    def labels(self, entity_ids: list[str]) -> dict[str, str]:
        ...


class ScopedPostgresVectorSearch:
    def __init__(self, store: PostgresVectorStore, *, tenant_id: str, repo_id: str) -> None:
        self._store = store
        self._tenant_id = tenant_id
        self._repo_id = repo_id

    def search(self, embedding: list[float], *, limit: int = 10) -> list[SearchHit]:
        return self._store.search(
            tenant_id=self._tenant_id,
            repo_id=self._repo_id,
            embedding=embedding,
            limit=limit,
        )


class ScopedNeo4jGraphSearch:
    """Resolve lexical seeds in Neo4j, then perform real bounded BFS."""

    def __init__(
        self,
        backend: Neo4jMcpBackend,
        traversal: Neo4jGraphTraversal,
        *,
        tenant_id: str,
        repo_id: str,
    ) -> None:
        self._backend = backend
        self._traversal = traversal
        self._tenant_id = tenant_id
        self._repo_id = repo_id

    def bfs(self, query: str, *, limit: int = 10) -> list[SearchHit]:
        seeds = self._backend.search_code(
            tenant_id=self._tenant_id,
            repo_id=self._repo_id,
            query_text=query,
            limit=max(1, min(limit, 5)),
        )
        scores: dict[str, float] = {seed.entity_id: float(seed.score) for seed in seeds}
        for seed in seeds:
            neighbors = self._traversal.bfs_neighbors(
                tenant_id=self._tenant_id,
                repo_id=self._repo_id,
                entity_id=seed.entity_id,
                depth=1,
                limit=limit,
            )
            for neighbor in neighbors:
                score = 1.0 / (1 + neighbor.distance)
                scores[neighbor.entity_id] = max(scores.get(neighbor.entity_id, 0.0), score)
        hits = [SearchHit(entity_id=entity_id, score=score, source="neo4j_bfs") for entity_id, score in scores.items()]
        return sorted(hits, key=lambda hit: hit.score, reverse=True)[:limit]


class Neo4jEvidenceSource:
    def __init__(self, backend: Neo4jMcpBackend, *, tenant_id: str, repo_id: str) -> None:
        self._backend = backend
        self._tenant_id = tenant_id
        self._repo_id = repo_id
        self._cache: dict[str, str] = {}

    def labels(self, entity_ids: list[str]) -> dict[str, str]:
        for entity_id in entity_ids:
            if entity_id in self._cache:
                continue
            entity = self._backend.get_entity(
                tenant_id=self._tenant_id,
                repo_id=self._repo_id,
                entity_id=entity_id,
            )
            self._cache[entity_id] = _entity_evidence(entity) if entity else entity_id
        return {entity_id: self._cache[entity_id] for entity_id in entity_ids}


def _entity_evidence(entity: EntityRecord) -> str:
    properties = json.dumps(entity.properties, sort_keys=True)
    return (
        f"Entity {entity.id}: {entity.kind} {entity.qualified_name or entity.name}; "
        f"path={entity.path}; properties={properties}"
    )


class LiveRagPipeline:
    """One scoped production RAG pipeline used by the benchmark runner."""

    def __init__(
        self,
        *,
        embeddings: BgeM3EmbeddingAdapter,
        vector_search: ScopedPostgresVectorSearch,
        graph_search: ScopedNeo4jGraphSearch,
        evidence_source: EvidenceSource,
        student_model: OpenAICompatibleStudentClient,
        reranker: MiniLmReranker,
        critic: FaithfulnessCritic,
        max_rounds: int = 2,
    ) -> None:
        self._router = RouterAgent()
        self._vector = VectorRetrieverAgent(embeddings=embeddings, vector_store=vector_search)
        self._graph = GraphTraversalAgent(graph_search)
        self._synthesis = SynthesisAgent(reranker, student_model)
        self._critic = CriticAgent(critic)
        self._evidence_source = evidence_source
        self._max_rounds = max_rounds

    def run(self, query: str, *, limit: int = 10) -> dict[str, Any]:
        state = new_query_state(query)
        labels: dict[str, str] = {}
        while True:
            state["rounds"] += 1
            self._router.route(state)
            self._vector.retrieve(state, limit=limit)
            self._graph.traverse(state, limit=limit)
            entity_ids = list(
                dict.fromkeys(
                    hit.entity_id for hit in [*state["vector_hits"], *state["graph_hits"]]
                )
            )
            labels.update(self._evidence_source.labels(entity_ids))
            self._synthesis.synthesize(state, labels=labels)
            self._critic.review(state, labels=labels)
            if state["accepted"] or state["rounds"] >= self._max_rounds:
                faithfulness = state["faithfulness"]
                return {
                    "answer": state["answer"],
                    "accepted": state["accepted"],
                    "rounds": state["rounds"],
                    "trace": list(state["trace"]),
                    "retrieval_query": state["retrieval_query"],
                    "faithfulness_score": faithfulness.score if faithfulness else 0.0,
                    "faithfulness_claim_scores": list(faithfulness.claim_scores)
                    if faithfulness
                    else [],
                    "vector_hits": [_search_hit_dict(hit) for hit in state["vector_hits"]],
                    "graph_hits": [_search_hit_dict(hit) for hit in state["graph_hits"]],
                    "reranked_hits": [
                        {
                            "entity_id": hit.entity_id,
                            "score": hit.score,
                            "sources": list(hit.sources),
                        }
                        for hit in state["reranked_hits"]
                    ],
                    "synthesis_evidence": [
                        labels.get(hit.entity_id, hit.entity_id)
                        for hit in state["reranked_hits"][:3]
                    ],
                    "critic_evidence": [
                        labels.get(hit.entity_id, hit.entity_id)
                        for hit in state["reranked_hits"][:5]
                    ],
                    "evidence": [
                        labels.get(hit.entity_id, hit.entity_id)
                        for hit in state["reranked_hits"][:3]
                    ],
                }


def _search_hit_dict(hit: SearchHit) -> dict[str, Any]:
    return {
        "entity_id": hit.entity_id,
        "score": hit.score,
        "source": hit.source,
        "metadata": hit.metadata,
    }


class RagPipelineFactory(Protocol):
    def __call__(self, case: LiveRagCase) -> LiveRagPipeline:
        ...


def run_live_rag_benchmark(
    cases: list[LiveRagCase],
    pipeline_factory: RagPipelineFactory,
    *,
    critic_model: str,
    embedding_model: str = "BAAI/bge-m3",
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    reranker_device: str = "cpu",
    threshold: float = 0.78,
    limit: int = 10,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for case in cases:
        started = time.perf_counter()
        try:
            result = pipeline_factory(case).run(case.query, limit=limit)
            retrieved = [str(hit["entity_id"]) for hit in result["reranked_hits"]]
            expected = set(case.expected_entity_ids)
            recall = len(expected & set(retrieved)) / len(expected) if expected else None
            results.append(
                {
                    **asdict(case),
                    **result,
                    "retrieval_recall": recall,
                    "retrieval_recall_at_k": recall,
                    "latency_seconds": time.perf_counter() - started,
                    "error": None,
                }
            )
        except Exception as exc:
            results.append(
                {
                    **asdict(case),
                    "answer": "",
                    "accepted": False,
                    "faithfulness_score": 0.0,
                    "latency_seconds": time.perf_counter() - started,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    scores = [float(result["faithfulness_score"]) for result in results]
    recalls = [
        float(result["retrieval_recall_at_k"])
        for result in results
        if result.get("retrieval_recall_at_k") is not None
    ]
    recall_case_count = len(recalls)
    error_count = sum(result["error"] is not None for result in results)
    accepted_count = sum(bool(result["accepted"]) for result in results)
    atomic_claim_case_count = sum(
        bool(result.get("faithfulness_claim_scores")) for result in results
    )
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "benchmark": "live-rag-faithfulness",
        "backend": {
            "vector": "postgres-pgvector",
            "graph": "neo4j-bfs",
            "embedding_model": embedding_model,
            "reranker_model": reranker_model,
            "reranker_device": reranker_device,
        },
        "critic": {"backend": "transformers-nli", "model": critic_model, "threshold": threshold},
        "case_count": len(results),
        "accepted_count": accepted_count,
        "error_count": error_count,
        "faithfulness_mean": statistics.fmean(scores) if scores else 0.0,
        "faithfulness_min": min(scores, default=0.0),
        "faithfulness_pass_rate": accepted_count / len(results) if results else 0.0,
        "faithfulness_aggregation": "minimum-atomic-claim-score",
        "atomic_claim_scoring": atomic_claim_case_count == len(results),
        "atomic_claim_case_count": atomic_claim_case_count,
        "retrieval_k": limit,
        "retrieval_recall_at_k_mean": statistics.fmean(recalls) if recalls else None,
        "retrieval_recall_mean": statistics.fmean(recalls) if recalls else None,
        "retrieval_recall_case_count": recall_case_count,
        "expected_entity_case_count": sum(
            bool(result.get("expected_entity_ids")) for result in results
        ),
        "cases": results,
    }
