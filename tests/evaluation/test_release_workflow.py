from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from idrkd.evaluation.live_rag import LiveRagCase, load_live_rag_cases, run_live_rag_benchmark
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


def test_streaming_benchmark_reports_p95_distributions() -> None:
    rag = {"cases": [{"query": "query", "evidence": ["evidence"], "error": None}]}

    artifact = run_streaming_benchmark(
        rag,
        base_url="http://model.test/v1",
        model="student",
        samples=3,
        warmups=1,
        opener=lambda _request, *, timeout: _SseResponse(),
    )

    assert artifact["success_count"] == 3
    assert artifact["error_count"] == 0
    assert artifact["ttft"]["p95_seconds"] >= 0
    assert artifact["latency"]["p95_seconds"] >= artifact["ttft"]["p95_seconds"]


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
        rag={
            "case_count": 5,
            "error_count": 0,
            "faithfulness_min": 0.81,
            "faithfulness_mean": 0.9,
            "faithfulness_pass_rate": 1.0,
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
    assert record["evaluation"]["holdout"]["cases"] == 89


def test_promotion_record_rejects_non_nli_and_streaming_errors() -> None:
    record = build_promotion_record(
        provenance={"manifest_digest": "digest", "lfs_objects": [{}]},
        runtime={"vllm": "0.27.1", "torch": "2.13.0"},
        holdout={"cases": [{}] * 89, "tool_f1": 1.0},
        rag={
            "error_count": 0,
            "faithfulness_min": 0.9,
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


def test_live_rag_case_rejects_missing_scope() -> None:
    with pytest.raises(ValueError, match="tenant_id"):
        LiveRagCase.from_dict({"id": "x", "query": "q", "repo_id": "r"}, line_number=1)


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
