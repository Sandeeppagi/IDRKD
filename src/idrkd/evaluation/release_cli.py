"""CLI for live release evidence and model promotion."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from idrkd.distillation.serving import OpenAICompatibleStudentClient
from idrkd.evaluation.live_rag import (
    LiveRagCase,
    LiveRagPipeline,
    Neo4jEvidenceSource,
    ScopedNeo4jGraphSearch,
    ScopedPostgresVectorSearch,
    annotate_live_rag_retrieval,
    load_live_rag_cases,
    run_live_rag_benchmark,
)
from idrkd.evaluation.live_taskbench import run_live_taskbench_benchmark
from idrkd.evaluation.artifact_security import sign_release, verify_release
from idrkd.evaluation.release_record import (
    build_promotion_record,
    collect_model_provenance,
    collect_runtime_metadata,
    evidence_file,
    read_json_object,
    read_json_object_or_error,
    write_json,
)
from idrkd.evaluation.security_runner import run_security_suite
from idrkd.evaluation.streaming import run_streaming_benchmark
from idrkd.graph.traversal import Neo4jGraphTraversal
from idrkd.mcp.backends import Neo4jMcpBackend
from idrkd.rag.critic import FaithfulnessCritic
from idrkd.rag.embeddings import BgeM3EmbeddingAdapter
from idrkd.rag.reranker import MiniLmReranker
from idrkd.rag.vector_store import PostgresVectorStore


DEFAULT_CHECKPOINT = Path(
    "artifacts/models/checkpoints/phi4-mini-dpo-tooljson-split-v2-llmc-awq"
)
DEFAULT_POSTGRES_DSN = "postgresql://idrkd:idrkd@localhost:5432/idrkd"
DEFAULT_NEO4J_URI = "bolt://localhost:7687"
DEFAULT_NEO4J_USER = "neo4j"
DEFAULT_NEO4J_PASSWORD = "change-me"


def _environment_value(name: str, default: str) -> str:
    value = os.getenv(name, "").strip()
    return value or default


class _LivePipelineFactory:
    def __init__(self, args: argparse.Namespace) -> None:
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

    def __call__(self, case: LiveRagCase) -> LiveRagPipeline:
        return LiveRagPipeline(
            embeddings=self._embeddings,
            vector_search=ScopedPostgresVectorSearch(
                self._vector_store,
                tenant_id=case.tenant_id,
                repo_id=case.repo_id,
            ),
            graph_search=ScopedNeo4jGraphSearch(
                self._neo4j,
                self._traversal,
                tenant_id=case.tenant_id,
                repo_id=case.repo_id,
            ),
            evidence_source=Neo4jEvidenceSource(
                self._neo4j,
                tenant_id=case.tenant_id,
                repo_id=case.repo_id,
            ),
            student_model=self._student,
            reranker=self._reranker,
            critic=self._critic,
        )

    def close(self) -> None:
        self._traversal.close()
        self._neo4j.close()


def _run_rag(args: argparse.Namespace) -> dict[str, Any]:
    factory = _LivePipelineFactory(args)
    try:
        result = run_live_rag_benchmark(
            load_live_rag_cases(args.rag_cases, require_expected_entities=True),
            factory,
            critic_model=args.critic_model,
            embedding_model=args.embedding_model,
            reranker_model=args.reranker_model,
            reranker_device=args.reranker_device,
            threshold=args.faithfulness_threshold,
            limit=args.retrieval_limit,
        )
    finally:
        factory.close()
    return result


def _run_taskbench(args: argparse.Namespace) -> dict[str, Any]:
    return run_live_taskbench_benchmark(
        base_url=args.model_base_url,
        model=args.model_id,
        tasks_path=args.taskbench_tasks,
        schemas_path=args.synthetic_schemas,
        conflicts_path=args.synthetic_conflicts,
        api_key=args.model_api_key,
        timeout_seconds=args.model_timeout,
        max_tokens=args.max_tokens,
    )


def _run_record(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = args.checkpoint / "idrkd-model-manifest.json"
    manifest = read_json_object_or_error(manifest_path)
    try:
        provenance = collect_model_provenance(
            repo_root=args.repo_root,
            checkpoint_dir=args.checkpoint,
            manifest=manifest,
        )
    except Exception as exc:
        provenance = {"artifact_error": f"{type(exc).__name__}: {exc}"}
    taskbench_path = getattr(args, "taskbench", None)
    taskbench = read_json_object_or_error(taskbench_path) if taskbench_path else None
    artifacts = {
        "runtime": evidence_file(args.runtime),
        "holdout": evidence_file(args.holdout),
        "live_rag": evidence_file(args.rag),
        "performance": evidence_file(args.performance),
        "security": evidence_file(args.security),
    }
    if taskbench_path:
        artifacts["taskbench_all"] = evidence_file(taskbench_path)
    return build_promotion_record(
        provenance=provenance,
        runtime=read_json_object_or_error(args.runtime),
        holdout=read_json_object_or_error(args.holdout),
        taskbench=taskbench,
        rag=read_json_object_or_error(args.rag),
        performance=read_json_object_or_error(args.performance),
        security=read_json_object_or_error(args.security),
        evidence_artifacts=artifacts,
        previous_tool_f1=args.previous_tool_f1,
        expected_holdout_cases=args.expected_holdout_cases,
        expected_taskbench_cases=args.expected_taskbench_cases,
    )


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-api-key", default=os.getenv("IDRKD_STUDENT_MODEL_API_KEY", "idrkd-local"))
    parser.add_argument("--model-timeout", type=float, default=60.0)
    parser.add_argument("--max-tokens", type=int, default=256)


def _add_rag_args(parser: argparse.ArgumentParser) -> None:
    _add_model_args(parser)
    parser.add_argument("--rag-cases", type=Path, required=True)
    parser.add_argument(
        "--postgres-dsn",
        default=_environment_value("POSTGRES_DSN", DEFAULT_POSTGRES_DSN),
    )
    parser.add_argument(
        "--neo4j-uri",
        default=_environment_value("NEO4J_URI", DEFAULT_NEO4J_URI),
    )
    parser.add_argument(
        "--neo4j-user",
        default=_environment_value("NEO4J_USER", DEFAULT_NEO4J_USER),
    )
    parser.add_argument(
        "--neo4j-password",
        default=_environment_value("NEO4J_PASSWORD", DEFAULT_NEO4J_PASSWORD),
    )
    parser.add_argument("--embedding-model", default="BAAI/bge-m3")
    parser.add_argument("--embedding-device", default="cpu")
    parser.add_argument("--reranker-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--reranker-device", default="cpu")
    parser.add_argument("--critic-model", default="cross-encoder/nli-deberta-v3-large")
    parser.add_argument("--faithfulness-threshold", type=float, default=0.78)
    parser.add_argument("--retrieval-limit", type=int, default=10)
    parser.add_argument("--local-files-only", action="store_true")


def _add_taskbench_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--taskbench-tasks",
        type=Path,
        default=Path("eval/taskbench/seed_tasks.jsonl"),
    )
    parser.add_argument(
        "--synthetic-schemas",
        type=Path,
        default=Path("eval/synthetic_schemas/schemas.jsonl"),
    )
    parser.add_argument(
        "--synthetic-conflicts",
        type=Path,
        default=Path("eval/synthetic_schemas/conflicts.jsonl"),
    )


def _add_record_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, default=Path("eval/distillation/llmc-awq-holdout.json"))
    parser.add_argument("--taskbench", type=Path)
    parser.add_argument("--rag", type=Path, required=True)
    parser.add_argument("--performance", type=Path, required=True)
    parser.add_argument("--security", type=Path, required=True)
    parser.add_argument("--previous-tool-f1", type=float)
    parser.add_argument("--expected-holdout-cases", type=int, default=89)
    parser.add_argument("--expected-taskbench-cases", type=int, default=440)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run release benchmarks and generate a promotion record.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    runtime = subparsers.add_parser("runtime", help="Capture Torch, CUDA, GPU, and vLLM versions.")
    runtime.add_argument("--out", type=Path, required=True)

    sign = subparsers.add_parser(
        "sign",
        help="Create a canonical release descriptor and sign it with Cosign.",
    )
    sign.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    sign.add_argument("--promotion-record", type=Path, required=True)
    sign.add_argument("--key", required=True, help="Cosign private key path or KMS URI.")
    sign.add_argument("--descriptor", type=Path, required=True)
    sign.add_argument("--bundle", type=Path, required=True)
    sign.add_argument("--cosign", default="cosign")

    verify = subparsers.add_parser(
        "verify",
        help="Verify a Cosign bundle and every signed release artifact.",
    )
    verify.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    verify.add_argument("--promotion-record", type=Path, required=True)
    verify.add_argument("--public-key", required=True, help="Cosign public key path or KMS URI.")
    verify.add_argument("--descriptor", type=Path, required=True)
    verify.add_argument("--bundle", type=Path, required=True)
    verify.add_argument("--cosign", default="cosign")

    rag = subparsers.add_parser("rag", help="Run live pgvector/Neo4j/synthesis/NLI faithfulness.")
    _add_rag_args(rag)
    rag.add_argument("--out", type=Path, required=True)

    rag_oracles = subparsers.add_parser(
        "rag-oracles",
        help="Audit recorded RAG rankings against curated retrieval oracles.",
    )
    rag_oracles.add_argument("--rag", type=Path, required=True)
    rag_oracles.add_argument("--cases", type=Path, required=True)
    rag_oracles.add_argument("--retrieval-limit", type=int, default=10)
    rag_oracles.add_argument("--out", type=Path, required=True)

    taskbench = subparsers.add_parser(
        "taskbench",
        help="Run the full no-split MCP-TaskBench suite against a live model.",
    )
    _add_model_args(taskbench)
    _add_taskbench_data_args(taskbench)
    taskbench.add_argument("--out", type=Path, required=True)

    performance = subparsers.add_parser("performance", help="Measure true streaming TTFT and latency.")
    _add_model_args(performance)
    performance.add_argument("--rag", type=Path, required=True)
    performance.add_argument("--samples", type=int, default=20)
    performance.add_argument("--warmups", type=int, default=2)
    performance.add_argument("--evidence-limit", type=int, default=3)
    performance.add_argument("--out", type=Path, required=True)

    security = subparsers.add_parser("security", help="Execute release security suites.")
    security.add_argument("--repo-root", type=Path, default=Path.cwd())
    security.add_argument("--out", type=Path, required=True)

    record = subparsers.add_parser("record", help="Bind evidence and provenance into a decision.")
    _add_record_args(record)
    record.add_argument("--out", type=Path, required=True)

    run = subparsers.add_parser("run", help="Run RAG, performance, security, and promotion end to end.")
    _add_rag_args(run)
    run.add_argument("--repo-root", type=Path, default=Path.cwd())
    run.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    run.add_argument("--runtime", type=Path, required=True)
    run.add_argument("--holdout", type=Path, default=Path("eval/distillation/llmc-awq-holdout.json"))
    run.add_argument("--previous-tool-f1", type=float)
    run.add_argument("--expected-holdout-cases", type=int, default=89)
    run.add_argument("--expected-taskbench-cases", type=int, default=440)
    _add_taskbench_data_args(run)
    run.add_argument("--performance-samples", type=int, default=20)
    run.add_argument("--performance-warmups", type=int, default=2)
    run.add_argument("--performance-evidence-limit", type=int, default=3)
    run.add_argument("--out-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "runtime":
        write_json(args.out, collect_runtime_metadata())
        return
    if args.command == "sign":
        result = sign_release(
            checkpoint_dir=args.checkpoint,
            promotion_record_path=args.promotion_record,
            descriptor_path=args.descriptor,
            bundle_path=args.bundle,
            key=args.key,
            cosign=args.cosign,
        )
        print(json.dumps(result, sort_keys=True))
        return
    if args.command == "verify":
        result = verify_release(
            checkpoint_dir=args.checkpoint,
            promotion_record_path=args.promotion_record,
            descriptor_path=args.descriptor,
            bundle_path=args.bundle,
            public_key=args.public_key,
            cosign=args.cosign,
        )
        print(json.dumps(result, sort_keys=True))
        return
    if args.command == "rag":
        write_json(args.out, _run_rag(args))
        return
    if args.command == "rag-oracles":
        artifact = annotate_live_rag_retrieval(
            read_json_object(args.rag),
            load_live_rag_cases(args.cases, require_expected_entities=True),
            limit=args.retrieval_limit,
        )
        write_json(args.out, artifact)
        return
    if args.command == "taskbench":
        artifact = _run_taskbench(args)
        write_json(args.out, artifact)
        print(
            f"mode={artifact['mode']} split={artifact['split']} "
            f"pass_rate={artifact['pass_rate']:.3f} "
            f"tool_f1={artifact['tool_f1']:.3f} cases={artifact['case_count']}"
        )
        return
    if args.command == "performance":
        artifact = run_streaming_benchmark(
            read_json_object(args.rag),
            base_url=args.model_base_url,
            model=args.model_id,
            samples=args.samples,
            warmups=args.warmups,
            api_key=args.model_api_key,
            max_tokens=args.max_tokens,
            timeout_seconds=args.model_timeout,
            evidence_limit=args.evidence_limit,
        )
        write_json(args.out, artifact)
        return
    if args.command == "security":
        write_json(args.out, run_security_suite(cwd=str(args.repo_root)))
        return
    if args.command == "record":
        write_json(args.out, _run_record(args))
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    taskbench_path = args.out_dir / "taskbench-all.json"
    rag_path = args.out_dir / "live-rag.json"
    performance_path = args.out_dir / "streaming-performance.json"
    security_path = args.out_dir / "security.json"
    record_path = args.out_dir / "promotion-record.json"
    try:
        taskbench_artifact = _run_taskbench(args)
    except Exception as exc:
        taskbench_artifact = {
            "benchmark": "mcp-taskbench-live",
            "split": "all",
            "artifact_error": f"{type(exc).__name__}: {exc}",
            "case_count": 0,
            "error_count": 1,
            "pass_rate": 0.0,
            "tool_call_pass_rate": 0.0,
            "semantic_outcome_rate": 0.0,
            "evaluation_scope": "live-model-tool-call-conformance-with-fixture-execution",
            "generalization_claim": False,
            "schema_valid_rate": 0.0,
            "tool_precision": 0.0,
            "tool_recall": 0.0,
            "tool_f1": 0.0,
            "argument_accuracy": 0.0,
            "by_category": {},
            "cases": [],
        }
    write_json(taskbench_path, taskbench_artifact)
    try:
        rag_artifact = _run_rag(args)
    except Exception as exc:
        rag_artifact = {
            "benchmark": "live-rag-faithfulness",
            "artifact_error": f"{type(exc).__name__}: {exc}",
            "case_count": 0,
            "error_count": 1,
            "faithfulness_min": 0.0,
            "faithfulness_mean": 0.0,
            "faithfulness_pass_rate": 0.0,
            "critic": {"backend": "transformers-nli", "model": args.critic_model},
            "cases": [],
        }
    write_json(rag_path, rag_artifact)
    try:
        performance_artifact = run_streaming_benchmark(
            rag_artifact,
            base_url=args.model_base_url,
            model=args.model_id,
            samples=args.performance_samples,
            warmups=args.performance_warmups,
            api_key=args.model_api_key,
            max_tokens=args.max_tokens,
            timeout_seconds=args.model_timeout,
            evidence_limit=args.performance_evidence_limit,
        )
    except Exception as exc:
        performance_artifact = {
            "benchmark": "openai-sse-streaming",
            "artifact_error": f"{type(exc).__name__}: {exc}",
            "sample_count": 0,
            "error_count": 1,
            "ttft": {"p95_seconds": 0.0},
            "latency": {"p95_seconds": 0.0},
            "measurements": [],
        }
    write_json(performance_path, performance_artifact)
    try:
        security_artifact = run_security_suite(cwd=str(args.repo_root))
    except Exception as exc:
        security_artifact = {
            "suite": "tenant-and-agent-security",
            "artifact_error": f"{type(exc).__name__}: {exc}",
            "passed": False,
            "tests": [],
        }
    write_json(security_path, security_artifact)
    args.rag = rag_path
    args.taskbench = taskbench_path
    args.performance = performance_path
    args.security = security_path
    record_artifact = _run_record(args)
    write_json(record_path, record_artifact)
    print(
        f"decision={record_artifact['decision']['status']} "
        f"record={record_path} digest={record_artifact['record_digest']}"
    )


if __name__ == "__main__":
    main()
