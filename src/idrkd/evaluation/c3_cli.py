"""Live CLI for the paired C3 monolithic versus LangGraph-AutoGen experiment."""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
from typing import Any

from idrkd.a2a import A2ABridge, IdrkdA2AClient, build_idrkd_agent_card
from idrkd.a2a.autogen_reconciler import A2AReconciliationClient
from idrkd.distillation.serving import OpenAICompatibleStudentClient
from idrkd.evaluation.c3 import C3Case, load_c3_cases, run_c3_benchmark
from idrkd.evaluation.live_rag import (
    LiveRagPipeline,
    Neo4jEvidenceSource,
    ScopedNeo4jGraphSearch,
    ScopedPostgresVectorSearch,
)
from idrkd.evaluation.release_record import write_json
from idrkd.graph.traversal import Neo4jGraphTraversal
from idrkd.mcp.backends import Neo4jMcpBackend
from idrkd.rag.critic import FaithfulnessCritic
from idrkd.rag.embeddings import BgeM3EmbeddingAdapter
from idrkd.rag.langgraph_orchestrator import LangGraphAutoGenOrchestrator
from idrkd.rag.reranker import MiniLmReranker
from idrkd.rag.vector_store import PostgresVectorStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run paired monolithic and LangGraph-AutoGen C3 evaluation cases."
    )
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model-api-key", default=os.getenv("IDRKD_STUDENT_MODEL_API_KEY", "idrkd-local"))
    parser.add_argument("--model-timeout", type=float, default=60.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--autogen-a2a-url", default="http://127.0.0.1:8090/")
    parser.add_argument("--a2a-shared-secret", default=os.getenv("IDRKD_A2A_SHARED_SECRET", "local-development"))
    parser.add_argument(
        "--postgres-dsn",
        default=os.getenv("POSTGRES_DSN", "postgresql://idrkd:idrkd@localhost:5432/idrkd"),
    )
    parser.add_argument("--neo4j-uri", default=os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    parser.add_argument("--neo4j-user", default=os.getenv("NEO4J_USER", "neo4j"))
    parser.add_argument("--neo4j-password", default=os.getenv("NEO4J_PASSWORD", "change-me"))
    parser.add_argument("--embedding-model", default="BAAI/bge-m3")
    parser.add_argument("--embedding-device", default="cpu")
    parser.add_argument("--reranker-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--reranker-device", default="cpu")
    parser.add_argument("--critic-model", default="cross-encoder/nli-deberta-v3-large")
    parser.add_argument("--faithfulness-threshold", type=float, default=0.78)
    parser.add_argument("--retrieval-limit", type=int, default=10)
    parser.add_argument("--max-rounds", type=int, default=2)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=17)
    parser.add_argument("--local-files-only", action="store_true")
    return parser


class LiveC3Experiment:
    def __init__(self, args: argparse.Namespace) -> None:
        self._args = args
        self._vector_store = PostgresVectorStore(args.postgres_dsn)
        self._neo4j = Neo4jMcpBackend(args.neo4j_uri, args.neo4j_user, args.neo4j_password)
        self._traversal = Neo4jGraphTraversal(
            args.neo4j_uri,
            args.neo4j_user,
            args.neo4j_password,
        )
        self._embeddings = BgeM3EmbeddingAdapter.from_sentence_transformers(
            args.embedding_model,
            device=args.embedding_device,
            local_files_only=args.local_files_only,
        )
        self._reranker = MiniLmReranker.from_sentence_transformers(
            args.reranker_model,
            device=args.reranker_device,
            local_files_only=args.local_files_only,
        )
        self._critic = FaithfulnessCritic.from_transformers(
            args.critic_model,
            threshold=args.faithfulness_threshold,
            local_files_only=args.local_files_only,
        )
        self._student = OpenAICompatibleStudentClient(
            base_url=args.model_base_url,
            model=args.model_id,
            api_key=args.model_api_key,
            timeout_seconds=args.model_timeout,
            max_tokens=args.max_tokens,
        )
        reconciler_card = build_idrkd_agent_card(
            name="IDRKD AutoGen Reconciler",
            description="Runs tenant-scoped conflict reconciliation through MCP.",
            version="0.1.0",
            endpoint=args.autogen_a2a_url,
            capabilities=("reconcile",),
        )
        planner_card = build_idrkd_agent_card(
            name="IDRKD LangGraph Planner",
            description="Runs decomposed retrieval and A2A delegation.",
            version="0.1.0",
            endpoint="http://127.0.0.1/unused-planner-card",
            capabilities=("query.plan",),
        )
        self._delegate = A2AReconciliationClient(
            client=IdrkdA2AClient(reconciler_card),
            bridge=A2ABridge(local_card=planner_card, shared_secret=args.a2a_shared_secret),
        )

    async def monolithic(self, case: C3Case) -> dict[str, Any]:
        pipeline = LiveRagPipeline(
            embeddings=self._embeddings,
            vector_search=self._vector_search(case),
            graph_search=self._graph_search(case),
            evidence_source=self._evidence_source(case),
            student_model=self._student,
            reranker=self._reranker,
            critic=self._critic,
            max_rounds=self._args.max_rounds,
        )
        result = await asyncio.to_thread(
            pipeline.run,
            case.query,
            limit=self._args.retrieval_limit,
        )
        result["retrieved_entity_ids"] = [
            str(hit["entity_id"]) for hit in result["reranked_hits"]
        ]
        return result

    async def decomposed(self, case: C3Case) -> dict[str, Any]:
        orchestrator = LangGraphAutoGenOrchestrator(
            embeddings=self._embeddings,
            vector_store=self._vector_search(case),
            graph_search=self._graph_search(case),
            reconciliation_delegate=self._delegate,
            evidence_source=self._evidence_source(case),
            reranker=self._reranker,
            critic=self._critic,
            student_model=self._student,
            max_rounds=self._args.max_rounds,
        )
        state = await orchestrator.run(
            case.query,
            tenant_id=case.tenant_id,
            repo_id=case.repo_id,
            conflict_id=case.conflict_id,
            limit=self._args.retrieval_limit,
        )
        return {
            "answer": state["answer"],
            "accepted": state["accepted"],
            "faithfulness_score": state["faithfulness"].score
            if state["faithfulness"]
            else 0.0,
            "retrieved_entity_ids": [hit.entity_id for hit in state["reranked_hits"]],
            "trace": state["trace"],
            "rounds": state["rounds"],
            "route": state["route"],
            "reconciliation": (
                {
                    "conflict_id": state["reconciliation"].conflict_id,
                    "recommendation": state["reconciliation"].recommendation,
                    "framework": state["reconciliation"].framework,
                }
                if state["reconciliation"]
                else None
            ),
        }

    def _vector_search(self, case: C3Case) -> ScopedPostgresVectorSearch:
        return ScopedPostgresVectorSearch(
            self._vector_store,
            tenant_id=case.tenant_id,
            repo_id=case.repo_id,
        )

    def _graph_search(self, case: C3Case) -> ScopedNeo4jGraphSearch:
        return ScopedNeo4jGraphSearch(
            self._neo4j,
            self._traversal,
            tenant_id=case.tenant_id,
            repo_id=case.repo_id,
        )

    def _evidence_source(self, case: C3Case) -> Neo4jEvidenceSource:
        return Neo4jEvidenceSource(
            self._neo4j,
            tenant_id=case.tenant_id,
            repo_id=case.repo_id,
        )

    async def close(self) -> None:
        await self._delegate.close()
        self._traversal.close()
        self._neo4j.close()


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    experiment = LiveC3Experiment(args)
    try:
        artifact = await run_c3_benchmark(
            load_c3_cases(args.cases),
            monolithic_runner=experiment.monolithic,
            decomposed_runner=experiment.decomposed,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
        )
    finally:
        await experiment.close()
    artifact["configuration"] = {
        "model_id": args.model_id,
        "model_base_url": args.model_base_url,
        "autogen_a2a_url": args.autogen_a2a_url,
        "embedding_model": args.embedding_model,
        "reranker_model": args.reranker_model,
        "critic_model": args.critic_model,
        "cases_path": str(args.cases),
    }
    return artifact


def main() -> None:
    args = build_parser().parse_args()
    write_json(args.out, asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
