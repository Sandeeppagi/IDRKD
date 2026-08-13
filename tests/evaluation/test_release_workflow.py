from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from idrkd.evaluation.bfcl import FunctionCallPrediction
from idrkd.evaluation.live_rag import (
    LiveRagCase,
    annotate_live_rag_retrieval,
    load_live_rag_cases,
    run_live_rag_benchmark,
)
from idrkd.evaluation.live_taskbench import run_live_taskbench_benchmark
from idrkd.evaluation.release_cli import DEFAULT_NEO4J_URI, _environment_value
from idrkd.evaluation.release_record import build_promotion_record, evidence_file
from idrkd.evaluation.security_runner import run_security_suite
from idrkd.evaluation.streaming import run_streaming_benchmark, stream_chat_completion


class _FakePipeline:
    def run(self, query: str, *, limit: int = 10):
        del query, limit
        return {
            "answer": "CustomerService calls BillingClient.",
            "accepted": True,
            "rounds": 1,
            "trace": ["router", "vector_retriever", "graph_traversal", "synthesis", "critic"],
            "faithfulness_score": 0.91,
            "faithfulness_claim_scores": [0.91],
            "vector_hits": [],
            "graph_hits": [],
            "reranked_hits": [
                {"entity_id": "entity-a", "score": 1.0, "sources": ["vector", "neo4j_bfs"]}
            ],
            "evidence": ["CustomerService calls BillingClient."],
        }


class _SseResponse:
    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return None

    def __iter__(self):
        return iter(
            [
                b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n',
                b'data: {"choices":[{"delta":{"content":"Customer"}}]}\n',
                b'data: {"choices":[{"delta":{"content":" API"}}]}\n',
                b"data: [DONE]\n",
            ]
        )


class _SearchPredictor:
    def predict_tool_call(self, *, prompt: str, tools: list[dict]):
        del prompt, tools
        prediction = FunctionCallPrediction(
            "search_code",
            {
                "tenant_id": "default",
                "repo_id": "repo-a",
                "query": "customer lookup",
                "limit": 3,
            },
        )
        return '{"name":"search_code","arguments":{}}', prediction


def test_live_rag_cases_validate_and_benchmark_retrieval_recall(tmp_path: Path) -> None:
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(
        json.dumps(
            {
                "id": "case-a",
                "query": "How does billing work?",
                "tenant_id": "tenant-a",
                "repo_id": "repo-a",
                "expected_entity_ids": ["entity-a"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    cases = load_live_rag_cases(cases_path)
    artifact = run_live_rag_benchmark(
        cases,
        lambda _case: _FakePipeline(),  # type: ignore[arg-type,return-value]
        critic_model="test-nli",
    )

    assert artifact["error_count"] == 0
    assert artifact["faithfulness_min"] == pytest.approx(0.91)
    assert artifact["retrieval_recall_mean"] == pytest.approx(1.0)
    assert artifact["retrieval_recall_at_k_mean"] == pytest.approx(1.0)
    assert artifact["retrieval_recall_case_count"] == 1
    assert artifact["expected_entity_case_count"] == 1
    assert artifact["retrieval_k"] == 10
    assert artifact["atomic_claim_scoring"] is True
    assert artifact["atomic_claim_case_count"] == 1
    assert artifact["cases"][0]["trace"][-1] == "critic"


def test_streaming_measurement_uses_first_content_delta() -> None:
    requests = []

    def opener(http_request, *, timeout):
        requests.append((http_request, timeout))
        return _SseResponse()

    result = stream_chat_completion(
        base_url="http://model.test/v1",
        model="student",
        query="query",
        evidence=["evidence"],
        opener=opener,
    )

    payload = json.loads(requests[0][0].data)
    assert payload["stream"] is True
    assert result["output"] == "Customer API"
    assert 0 <= result["ttft_seconds"] <= result["latency_seconds"]


def test_live_rag_oracle_audit_preserves_historical_faithfulness() -> None:
    artifact = {
        "faithfulness_min": 0.91,
        "cases": [
            {
                "case_id": "case-a",
                "query": "old wording",
                "tenant_id": "tenant-a",
                "repo_id": "repo-a",
                "faithfulness_score": 0.91,
                "reranked_hits": [{"entity_id": "entity-a"}, {"entity_id": "other"}],
            }
        ],
    }
    cases = [
        LiveRagCase(
            case_id="case-a",
            query="new wording",
            tenant_id="tenant-a",
            repo_id="repo-a",
            expected_entity_ids=("entity-a", "entity-b"),
        )
    ]

    audited = annotate_live_rag_retrieval(artifact, cases, limit=10)

    assert audited["faithfulness_min"] == 0.91
    assert audited["retrieval_recall_at_k_mean"] == 0.5
    assert audited["retrieval_recall_case_count"] == 1
    assert audited["expected_entity_case_count"] == 1
    assert audited["atomic_claim_scoring"] is False
    assert audited["faithfulness_aggregation"] == "legacy-whole-answer-score"
    assert audited["retrieval_oracle_annotation"]["faithfulness_recomputed"] is False
    assert audited["cases"][0]["oracle_query"] == "new wording"


def test_live_taskbench_runs_explicit_all_split(tmp_path: Path) -> None:
    tasks = tmp_path / "tasks.jsonl"
    schemas = tmp_path / "schemas.jsonl"
    conflicts = tmp_path / "conflicts.jsonl"
    tasks.write_text(
        json.dumps(
            {
                "id": "search-1",
                "category": "tool_selection",
                "prompt": "Search for customer lookup.",
                "expected_tool": "search_code",
                "arguments": {
                    "tenant_id": "default",
                    "repo_id": "repo-a",
                    "query": "customer lookup",
                    "limit": 3,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    schemas.write_text("", encoding="utf-8")
    conflicts.write_text("", encoding="utf-8")

    artifact = run_live_taskbench_benchmark(
        base_url="http://model.test/v1",
        model="student",
        tasks_path=tasks,
        schemas_path=schemas,
        conflicts_path=conflicts,
        predictor=_SearchPredictor(),
    )

    assert artifact["benchmark"] == "mcp-taskbench-live"
    assert artifact["split"] == "all"
    assert artifact["model_id"] == "student"
    assert artifact["seed_case_count"] == 1
    assert artifact["synthetic_case_count"] == 0
    assert artifact["case_count"] == 1
    assert artifact["error_count"] == 0
    assert artifact["pass_rate"] == 1.0
    assert artifact["tool_call_pass_rate"] == 1.0
    assert artifact["semantic_outcome_rate"] == 1.0
    assert artifact["generalization_claim"] is False
    assert artifact["tool_f1"] == 1.0


def test_streaming_benchmark_reports_p95_distributions() -> None:
    requests = []

    def opener(http_request, *, timeout):
        requests.append((http_request, timeout))
        return _SseResponse()

    rag = {
        "cases": [
            {
                "query": "query",
                "evidence": ["one", "two", "three", "four", "five"],
                "error": None,
            }
        ]
    }

    artifact = run_streaming_benchmark(
        rag,
        base_url="http://model.test/v1",
        model="student",
        samples=3,
        warmups=1,
        opener=opener,
    )

    assert artifact["success_count"] == 3
    assert artifact["error_count"] == 0
    assert artifact["evidence_limit"] == 3
    assert artifact["ttft"]["p95_seconds"] >= 0
    assert artifact["latency"]["p95_seconds"] >= artifact["ttft"]["p95_seconds"]
    payload = json.loads(requests[-1][0].data)
    assert "- three" in payload["messages"][1]["content"]
    assert "- four" not in payload["messages"][1]["content"]


def test_security_runner_records_executed_command() -> None:
    def runner(command, **_kwargs):
        return subprocess.CompletedProcess(command, 0, stdout="12 passed\n", stderr="")

    artifact = run_security_suite(tests=("tests/security.py",), runner=runner)

    assert artifact["passed"] is True
    assert "pytest -q tests/security.py" in artifact["command"]
    assert artifact["stdout"] == "12 passed\n"


def test_promotion_record_binds_evidence_and_promotes() -> None:
    record = build_promotion_record(
        provenance={
            "artifact_git_commit": "8aeb3c3",
            "manifest_digest": "7c784750",
            "lfs_objects": [{"oid": "sha256:abc", "path": "model.safetensors"}],
        },
        runtime={"vllm": "0.27.1", "torch": "2.13.0+cu129"},
        holdout={"cases": [{}] * 89, "pass_rate": 1.0, "tool_f1": 1.0, "argument_accuracy": 1.0},
        taskbench={
            "split": "all",
            "case_count": 440,
            "error_count": 0,
            "pass_rate": 1.0,
            "schema_valid_rate": 1.0,
            "tool_precision": 1.0,
            "tool_recall": 1.0,
            "tool_f1": 1.0,
            "argument_accuracy": 1.0,
            "by_category": {"tool_selection": 1.0},
        },
        rag={
            "case_count": 5,
            "error_count": 0,
            "faithfulness_min": 0.81,
            "faithfulness_mean": 0.9,
            "faithfulness_pass_rate": 1.0,
            "faithfulness_aggregation": "minimum-atomic-claim-score",
            "atomic_claim_scoring": True,
            "atomic_claim_case_count": 5,
            "retrieval_k": 10,
            "retrieval_recall_at_k_mean": 0.84,
            "retrieval_recall_mean": 0.84,
            "retrieval_recall_case_count": 5,
            "expected_entity_case_count": 5,
            "critic": {"backend": "transformers-nli", "model": "deberta"},
        },
        performance={
            "sample_count": 20,
            "error_count": 0,
            "ttft": {"p95_seconds": 0.8},
            "latency": {"p95_seconds": 4.2},
        },
        security={"passed": True, "tests": ["security.py"]},
    )

    assert record["decision"] == {"status": "promoted", "reasons": []}
    assert record["record_digest"].startswith("sha256:")
    assert record["schema_version"] == 2
    assert record["evaluation"]["holdout"]["cases"] == 89
    assert record["evaluation"]["holdout"]["claim_scope"] == "held-out tool-call conformance"
    assert record["evaluation"]["taskbench_all"]["cases"] == 440
    assert record["evaluation"]["taskbench_all"]["split"] == "all"
    assert record["evaluation"]["taskbench_all"]["generalization_claim"] is False
    assert record["evaluation"]["faithfulness"]["retrieval_recall_at_k_mean"] == 0.84
    assert record["criteria"]["min_retrieval_recall"] == 0.8


def test_promotion_record_rejects_incomplete_full_taskbench() -> None:
    record = build_promotion_record(
        provenance={"manifest_digest": "digest", "lfs_objects": [{}]},
        runtime={"vllm": "0.27.1", "torch": "2.13.0"},
        holdout={"cases": [{}] * 89, "pass_rate": 1.0, "tool_f1": 1.0, "argument_accuracy": 1.0},
        taskbench={
            "split": "holdout",
            "case_count": 89,
            "error_count": 2,
            "tool_f1": 0.5,
        },
        rag={
            "error_count": 0,
            "faithfulness_min": 0.9,
            "retrieval_recall_case_count": 1,
            "critic": {"backend": "transformers-nli"},
        },
        performance={
            "error_count": 0,
            "ttft": {"p95_seconds": 0.1},
            "latency": {"p95_seconds": 0.2},
        },
        security={"passed": True},
    )

    reasons = record["decision"]["reasons"]
    assert record["decision"]["status"] == "rejected"
    assert "full TaskBench split is not all" in reasons
    assert "full TaskBench cases 89 != 440" in reasons
    assert "full TaskBench errors: 2" in reasons
    assert "full TaskBench tool_f1 0.500 < 0.820" in reasons


def test_promotion_record_rejects_non_nli_and_streaming_errors() -> None:
    record = build_promotion_record(
        provenance={"manifest_digest": "digest", "lfs_objects": [{}]},
        runtime={"vllm": "0.27.1", "torch": "2.13.0"},
        holdout={"cases": [{}] * 89, "tool_f1": 1.0},
        rag={
            "error_count": 0,
            "faithfulness_min": 0.9,
            "retrieval_recall_case_count": 1,
            "critic": {"backend": "lexical"},
        },
        performance={
            "error_count": 1,
            "ttft": {"p95_seconds": 0.1},
            "latency": {"p95_seconds": 0.2},
        },
        security={"passed": True},
    )

    assert record["decision"]["status"] == "rejected"
    assert "faithfulness critic was not transformers NLI" in record["decision"]["reasons"]
    assert "streaming measurement errors: 1" in record["decision"]["reasons"]


def test_promotion_record_rejects_live_rag_without_complete_recall_coverage() -> None:
    record = build_promotion_record(
        provenance={"manifest_digest": "digest", "lfs_objects": [{}]},
        runtime={"vllm": "0.27.1", "torch": "2.13.0"},
        holdout={"cases": [{}] * 89, "tool_f1": 1.0, "argument_accuracy": 1.0},
        rag={
            "case_count": 5,
            "error_count": 0,
            "faithfulness_min": 0.9,
            "critic": {"backend": "transformers-nli"},
            "atomic_claim_scoring": True,
            "atomic_claim_case_count": 5,
            "retrieval_k": 10,
            "retrieval_recall_at_k_mean": 1.0,
            "expected_entity_case_count": 1,
            "retrieval_recall_case_count": 1,
        },
        performance={
            "error_count": 0,
            "ttft": {"p95_seconds": 0.1},
            "latency": {"p95_seconds": 0.2},
        },
        security={"passed": True},
    )

    assert record["decision"]["status"] == "rejected"
    assert "live RAG expected-entity coverage 1/5 is incomplete" in record["decision"]["reasons"]
    assert "live RAG recall coverage 1/5 is incomplete" in record["decision"]["reasons"]


def test_promotion_record_rejects_low_retrieval_recall() -> None:
    record = build_promotion_record(
        provenance={"manifest_digest": "digest", "lfs_objects": [{}]},
        runtime={"vllm": "0.27.1", "torch": "2.13.0"},
        holdout={
            "cases": [{}] * 89,
            "pass_rate": 1.0,
            "tool_f1": 1.0,
            "argument_accuracy": 1.0,
        },
        rag={
            "case_count": 5,
            "error_count": 0,
            "faithfulness_min": 0.9,
            "critic": {"backend": "transformers-nli"},
            "atomic_claim_scoring": True,
            "atomic_claim_case_count": 5,
            "retrieval_k": 10,
            "retrieval_recall_at_k_mean": 0.79,
            "expected_entity_case_count": 5,
            "retrieval_recall_case_count": 5,
        },
        performance={
            "error_count": 0,
            "ttft": {"p95_seconds": 0.1},
            "latency": {"p95_seconds": 0.2},
        },
        security={"passed": True},
    )

    assert record["decision"]["status"] == "rejected"
    assert "live RAG recall@10 0.790 < 0.800" in record["decision"]["reasons"]


def test_live_rag_case_rejects_missing_scope() -> None:
    with pytest.raises(ValueError, match="tenant_id"):
        LiveRagCase.from_dict({"id": "x", "query": "q", "repo_id": "r"}, line_number=1)


def test_release_live_rag_cases_require_retrieval_oracles(tmp_path: Path) -> None:
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(
        json.dumps(
            {
                "id": "case-a",
                "query": "How does billing work?",
                "tenant_id": "tenant-a",
                "repo_id": "repo-a",
                "expected_entity_ids": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="case-a"):
        load_live_rag_cases(cases_path, require_expected_entities=True)


def test_committed_live_rag_suite_is_fully_oracled() -> None:
    repo_root = Path(__file__).parents[2]

    cases = load_live_rag_cases(
        repo_root / "eval/rag/live_repository_queries.jsonl",
        require_expected_entities=True,
    )
    provenance = json.loads(
        (repo_root / "eval/rag/live_repository_oracles.json").read_text(encoding="utf-8")
    )

    assert len(cases) == 5
    assert all(case.expected_entity_ids for case in cases)
    assert {case.case_id: set(case.expected_entity_ids) for case in cases} == {
        case_id: {entity["entity_id"] for entity in entities}
        for case_id, entities in provenance["cases"].items()
    }


def test_evidence_file_hashes_complete_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "rag.json"
    artifact.write_text('{"score": 0.9}\n', encoding="utf-8")

    evidence = evidence_file(artifact)

    assert evidence["size_bytes"] == artifact.stat().st_size
    assert len(evidence["sha256"]) == 64


def test_release_cli_uses_default_for_blank_environment_value(monkeypatch) -> None:
    monkeypatch.setenv("NEO4J_URI", "")

    assert _environment_value("NEO4J_URI", DEFAULT_NEO4J_URI) == "bolt://localhost:7687"


def test_release_rag_defaults_reranker_to_cpu() -> None:
    from idrkd.evaluation.release_cli import build_parser

    args = build_parser().parse_args(
        [
            "rag",
            "--model-id",
            "student",
            "--rag-cases",
            "cases.jsonl",
            "--out",
            "rag.json",
        ]
    )

    assert args.embedding_device == "cpu"
    assert args.reranker_device == "cpu"


def test_release_taskbench_defaults_to_full_corpus() -> None:
    from idrkd.evaluation.release_cli import build_parser

    args = build_parser().parse_args(
        [
            "taskbench",
            "--model-id",
            "student",
            "--out",
            "taskbench.json",
        ]
    )

    assert args.taskbench_tasks == Path("eval/taskbench/seed_tasks.jsonl")
    assert args.synthetic_schemas == Path("eval/synthetic_schemas/schemas.jsonl")
    assert args.synthetic_conflicts == Path("eval/synthetic_schemas/conflicts.jsonl")
