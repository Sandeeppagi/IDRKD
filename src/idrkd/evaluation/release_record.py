"""Provenance collection and automatic release promotion records."""

from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import subprocess
from typing import Any

from idrkd.evaluation.promotion import PromotionCriteria


LFS_LINE = re.compile(r"^(?P<oid>[0-9a-f]{64})\s+[-*]\s+(?P<path>.+)$")
LFS_SIZE = re.compile(r"^size (?P<size>\d+)$", re.MULTILINE)


def collect_runtime_metadata() -> dict[str, Any]:
    import torch

    try:
        vllm_version = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError:
        vllm_version = None
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "python": sys_version(),
        "vllm": vllm_version,
        "torch": str(torch.__version__),
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": gpu,
    }


def sys_version() -> str:
    import platform

    return platform.python_version()


def collect_model_provenance(
    *,
    repo_root: Path,
    checkpoint_dir: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    relative_checkpoint = checkpoint_dir.resolve().relative_to(repo_root.resolve()).as_posix()
    artifact_commit = _git(
        repo_root,
        "log",
        "-1",
        "--format=%H",
        "--",
        relative_checkpoint,
    ).strip()
    if not artifact_commit:
        raise ValueError(f"Checkpoint is not committed: {relative_checkpoint}")
    lfs_entries = _lfs_entries(repo_root, relative_checkpoint, artifact_commit)
    if not lfs_entries:
        raise ValueError(f"Checkpoint has no Git LFS objects: {relative_checkpoint}")
    return {
        "artifact_git_commit": artifact_commit,
        "artifact_git_commit_short": artifact_commit[:7],
        "checkpoint_path": relative_checkpoint,
        "model_id": manifest.get("model_id"),
        "manifest_digest": manifest.get("digest"),
        "base_model_id": manifest.get("base_model_id"),
        "adapter_path": manifest.get("adapter_path"),
        "quantization": manifest.get("quantization"),
        "lfs_objects": lfs_entries,
    }


def _lfs_entries(repo_root: Path, prefix: str, commit: str) -> list[dict[str, Any]]:
    output = _git(repo_root, "lfs", "ls-files", "-l")
    entries: list[dict[str, Any]] = []
    for line in output.splitlines():
        match = LFS_LINE.match(line)
        if match is None or not match.group("path").startswith(prefix + "/"):
            continue
        path = match.group("path")
        pointer = _git(repo_root, "show", f"{commit}:{path}")
        size_match = LFS_SIZE.search(pointer)
        entries.append(
            {
                "path": path,
                "oid": f"sha256:{match.group('oid')}",
                "size_bytes": int(size_match.group("size")) if size_match else None,
            }
        )
    return sorted(entries, key=lambda item: str(item["path"]))


def _git(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {completed.stderr.strip()}")
    return completed.stdout


def build_promotion_record(
    *,
    provenance: dict[str, Any],
    runtime: dict[str, Any],
    holdout: dict[str, Any],
    taskbench: dict[str, Any] | None = None,
    rag: dict[str, Any],
    performance: dict[str, Any],
    security: dict[str, Any],
    evidence_artifacts: dict[str, Any] | None = None,
    previous_tool_f1: float | None = None,
    expected_holdout_cases: int = 89,
    expected_taskbench_cases: int = 440,
    criteria: PromotionCriteria = PromotionCriteria(),
) -> dict[str, Any]:
    reasons: list[str] = []
    artifacts = [
        ("provenance", provenance),
        ("runtime", runtime),
        ("holdout", holdout),
        ("live RAG", rag),
        ("performance", performance),
        ("security", security),
    ]
    if taskbench is not None:
        artifacts.append(("full TaskBench", taskbench))
    for name, artifact in artifacts:
        if artifact.get("artifact_error"):
            reasons.append(f"{name} artifact error: {artifact['artifact_error']}")
    holdout_cases = len(holdout.get("cases", []))
    holdout_tool_call_pass_rate = float(
        holdout.get("tool_call_pass_rate", holdout.get("pass_rate", 0.0))
    )
    holdout_semantic_outcome_rate = holdout.get("semantic_outcome_rate")
    tool_f1 = float(holdout.get("tool_f1", 0.0))
    argument_accuracy = float(holdout.get("argument_accuracy", 0.0))
    faithfulness = float(rag.get("faithfulness_min", 0.0))
    rag_case_count = int(rag.get("case_count", 0))
    expected_entity_case_count = int(rag.get("expected_entity_case_count", 0))
    retrieval_recall_case_count = int(rag.get("retrieval_recall_case_count", 0))
    retrieval_recall = rag.get("retrieval_recall_at_k_mean", rag.get("retrieval_recall_mean"))
    retrieval_k = int(rag.get("retrieval_k", 0))
    atomic_claim_case_count = int(rag.get("atomic_claim_case_count", 0))
    ttft_p95 = float(performance.get("ttft", {}).get("p95_seconds", 0.0))
    latency_p95 = float(performance.get("latency", {}).get("p95_seconds", 0.0))

    if taskbench is not None:
        taskbench_cases = int(taskbench.get("case_count", len(taskbench.get("cases", []))))
        taskbench_tool_f1 = float(taskbench.get("tool_f1", 0.0))
        taskbench_errors = int(taskbench.get("error_count", 0))
        if taskbench.get("split") != "all":
            reasons.append("full TaskBench split is not all")
        if taskbench_cases != expected_taskbench_cases:
            reasons.append(
                f"full TaskBench cases {taskbench_cases} != {expected_taskbench_cases}"
            )
        if taskbench_errors != 0:
            reasons.append(f"full TaskBench errors: {taskbench_errors}")
        if taskbench_tool_f1 < criteria.min_tool_f1:
            reasons.append(
                f"full TaskBench tool_f1 {taskbench_tool_f1:.3f} "
                f"< {criteria.min_tool_f1:.3f}"
            )

    if holdout_cases != expected_holdout_cases:
        reasons.append(f"holdout cases {holdout_cases} != {expected_holdout_cases}")
    if holdout_tool_call_pass_rate < 1.0:
        reasons.append(
            f"holdout tool_call_pass_rate {holdout_tool_call_pass_rate:.3f} < 1.000"
        )
    if tool_f1 < criteria.min_tool_f1:
        reasons.append(f"tool_f1 {tool_f1:.3f} < {criteria.min_tool_f1:.3f}")
    if argument_accuracy < 1.0:
        reasons.append(f"argument_accuracy {argument_accuracy:.3f} < 1.000")
    if faithfulness < criteria.min_faithfulness:
        reasons.append(f"faithfulness_min {faithfulness:.3f} < {criteria.min_faithfulness:.3f}")
    if rag.get("critic", {}).get("backend") != "transformers-nli":
        reasons.append("faithfulness critic was not transformers NLI")
    if int(rag.get("error_count", 0)) != 0:
        reasons.append(f"live RAG errors: {rag.get('error_count')}")
    if rag_case_count == 0:
        reasons.append("live RAG has no cases")
    if expected_entity_case_count != rag_case_count:
        reasons.append(
            "live RAG expected-entity coverage "
            f"{expected_entity_case_count}/{rag_case_count} is incomplete"
        )
    if retrieval_recall_case_count != rag_case_count:
        reasons.append(
            f"live RAG recall coverage {retrieval_recall_case_count}/{rag_case_count} is incomplete"
        )
    if retrieval_k != criteria.retrieval_k:
        reasons.append(f"live RAG retrieval_k {retrieval_k} != {criteria.retrieval_k}")
    if retrieval_recall is None:
        reasons.append("live RAG retrieval recall is missing")
    elif float(retrieval_recall) < criteria.min_retrieval_recall:
        reasons.append(
            f"live RAG recall@{criteria.retrieval_k} {float(retrieval_recall):.3f} "
            f"< {criteria.min_retrieval_recall:.3f}"
        )
    if not bool(rag.get("atomic_claim_scoring")):
        reasons.append("live RAG did not use atomic claim scoring")
    if atomic_claim_case_count != rag_case_count:
        reasons.append(
            "live RAG atomic-claim coverage "
            f"{atomic_claim_case_count}/{rag_case_count} is incomplete"
        )
    if not bool(security.get("passed")):
        reasons.append("tenant/security tests failed")
    if int(performance.get("error_count", 0)) != 0:
        reasons.append(f"streaming measurement errors: {performance.get('error_count')}")
    if ttft_p95 > criteria.max_ttft_seconds:
        reasons.append(f"ttft_p95 {ttft_p95:.3f}s > {criteria.max_ttft_seconds:.3f}s")
    if latency_p95 > criteria.max_latency_p95_seconds:
        reasons.append(f"latency_p95 {latency_p95:.3f}s > {criteria.max_latency_p95_seconds:.3f}s")
    if previous_tool_f1 is not None:
        regression = previous_tool_f1 - tool_f1
        if regression > criteria.max_tool_f1_regression:
            reasons.append(
                f"tool_f1 regression {regression:.3f} > {criteria.max_tool_f1_regression:.3f}"
            )
    if not runtime.get("vllm") or not runtime.get("torch"):
        reasons.append("runtime metadata is incomplete")
    if not provenance.get("manifest_digest") or not provenance.get("lfs_objects"):
        reasons.append("model provenance is incomplete")

    record: dict[str, Any] = {
        "schema_version": 2 if taskbench is not None else 1,
        "created_at": datetime.now(UTC).isoformat(),
        "model": provenance,
        "runtime": runtime,
        "evidence_artifacts": evidence_artifacts or {},
        "evaluation": {
            "holdout": {
                "cases": holdout_cases,
                "tool_call_pass_rate": holdout_tool_call_pass_rate,
                "semantic_outcome_rate": holdout_semantic_outcome_rate,
                "tool_f1": tool_f1,
                "argument_accuracy": argument_accuracy,
                "claim_scope": "held-out tool-call conformance",
            },
            "faithfulness": {
                "cases": rag_case_count,
                "minimum": faithfulness,
                "mean": float(rag.get("faithfulness_mean", 0.0)),
                "pass_rate": float(rag.get("faithfulness_pass_rate", 0.0)),
                "critic": rag.get("critic"),
                "faithfulness_aggregation": rag.get("faithfulness_aggregation"),
                "atomic_claim_scoring": bool(rag.get("atomic_claim_scoring")),
                "atomic_claim_case_count": atomic_claim_case_count,
                "retrieval_k": retrieval_k,
                "retrieval_recall_at_k_mean": retrieval_recall,
                "retrieval_recall_mean": rag.get("retrieval_recall_mean"),
                "retrieval_recall_case_count": retrieval_recall_case_count,
                "expected_entity_case_count": expected_entity_case_count,
            },
            "performance": {
                "samples": int(performance.get("sample_count", 0)),
                "ttft_p95_seconds": ttft_p95,
                "latency_p95_seconds": latency_p95,
            },
            "security": {
                "passed": bool(security.get("passed")),
                "tests": security.get("tests", []),
                "duration_seconds": security.get("duration_seconds"),
            },
        },
        "criteria": {
            "expected_holdout_cases": expected_holdout_cases,
            "min_tool_f1": criteria.min_tool_f1,
            "min_faithfulness": criteria.min_faithfulness,
            "min_retrieval_recall": criteria.min_retrieval_recall,
            "retrieval_k": criteria.retrieval_k,
            "max_ttft_p95_seconds": criteria.max_ttft_seconds,
            "max_latency_p95_seconds": criteria.max_latency_p95_seconds,
            "max_tool_f1_regression": criteria.max_tool_f1_regression,
        },
        "decision": {"status": "promoted" if not reasons else "rejected", "reasons": reasons},
    }
    if taskbench is not None:
        record["evaluation"]["taskbench_all"] = {
            "split": taskbench.get("split"),
            "cases": int(taskbench.get("case_count", len(taskbench.get("cases", [])))),
            "pass_rate": float(taskbench.get("pass_rate", 0.0)),
            "tool_call_pass_rate": float(
                taskbench.get("tool_call_pass_rate", taskbench.get("pass_rate", 0.0))
            ),
            "semantic_outcome_rate": taskbench.get("semantic_outcome_rate"),
            "evaluation_scope": taskbench.get(
                "evaluation_scope",
                "legacy artifact without explicit scope",
            ),
            "generalization_claim": bool(taskbench.get("generalization_claim", False)),
            "schema_valid_rate": float(taskbench.get("schema_valid_rate", 0.0)),
            "tool_precision": float(taskbench.get("tool_precision", 0.0)),
            "tool_recall": float(taskbench.get("tool_recall", 0.0)),
            "tool_f1": float(taskbench.get("tool_f1", 0.0)),
            "argument_accuracy": float(taskbench.get("argument_accuracy", 0.0)),
            "error_count": int(taskbench.get("error_count", 0)),
            "by_category": taskbench.get("by_category", {}),
        }
        record["criteria"]["expected_taskbench_cases"] = expected_taskbench_cases
    canonical = json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
    record["record_digest"] = f"sha256:{hashlib.sha256(canonical).hexdigest()}"
    return record


def read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def read_json_object_or_error(path: Path) -> dict[str, Any]:
    try:
        return read_json_object(path)
    except Exception as exc:
        return {"artifact_error": f"{type(exc).__name__}: {exc}", "path": str(path)}


def evidence_file(path: Path) -> dict[str, Any]:
    try:
        content = path.read_bytes()
    except Exception as exc:
        return {"path": str(path), "artifact_error": f"{type(exc).__name__}: {exc}"}
    return {
        "path": str(path),
        "sha256": hashlib.sha256(content).hexdigest(),
        "size_bytes": len(content),
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
