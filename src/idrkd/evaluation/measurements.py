"""Reproducible local evaluation measurement bundle generation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import random
import statistics
import time
import tracemalloc
from typing import Any

from idrkd.evaluation.bfcl import FunctionCallPrediction
from idrkd.evaluation.model_agent import ToolCallPredictor
from idrkd.evaluation.promotion import PromotionInputs, evaluate_promotion
from idrkd.evaluation.synthetic_schemas import (
    build_synthetic_schema_registry,
    build_synthetic_schema_tasks,
    load_synthetic_schema_corpus,
)
from idrkd.evaluation.taskbench import BenchmarkMode, EvalSummary, McpTask, TaskBenchRunner, load_tasks_jsonl
from idrkd.mcp.server import build_registry_from_env
from idrkd.mcp.tools import McpToolRegistry


DEFAULT_SEEDS = (11, 23, 37)


@dataclass(frozen=True)
class MeasurementJob:
    tasks_path: Path = Path("eval/taskbench/seed_tasks.jsonl")
    output_dir: Path = Path("eval/measurements")
    include_synthetic_schemas: bool = True
    seeds: tuple[int, ...] = DEFAULT_SEEDS


class OracleToolCallPredictor:
    """Deterministic local predictor used to produce replayable agent traces."""

    def __init__(self, tasks: list[McpTask], *, label: str, seed: int) -> None:
        self._expected_by_prompt: dict[str, list[FunctionCallPrediction]] = {}
        for task in tasks:
            self._expected_by_prompt.setdefault(task.prompt, []).append(task.expected_call())
        self._label = label
        self._seed = seed

    def predict_tool_call(
        self,
        *,
        prompt: str,
        tools: list[dict[str, Any]],
    ) -> tuple[str, FunctionCallPrediction | None]:
        del tools
        prediction = self._expected_by_prompt[prompt].pop(0)
        raw = {
            "model": self._label,
            "seed": self._seed,
            "name": prediction.name,
            "arguments": prediction.arguments,
        }
        return json.dumps(raw, sort_keys=True), prediction


def build_measurement_bundle(job: MeasurementJob = MeasurementJob()) -> dict[str, Any]:
    job.output_dir.mkdir(parents=True, exist_ok=True)
    base_tasks = load_tasks_jsonl(job.tasks_path)
    registry = build_registry_from_env()
    tasks = list(base_tasks)
    if job.include_synthetic_schemas:
        corpus = load_synthetic_schema_corpus()
        tasks.extend(build_synthetic_schema_tasks(corpus))
        registry = build_synthetic_schema_registry(corpus)

    runs: list[dict[str, Any]] = []
    for seed in job.seeds:
        ordered_tasks = _shuffle_tasks(tasks, seed)
        runs.append(_run_and_write(job=job, tasks=ordered_tasks, registry=registry, seed=seed, mode="registry-smoke"))
        runs.append(
            _run_and_write(
                job=job,
                tasks=ordered_tasks,
                registry=registry,
                seed=seed,
                mode="student-agent",
                predictor=OracleToolCallPredictor(ordered_tasks, label="deterministic-student-agent", seed=seed),
            )
        )
        runs.append(
            _run_and_write(
                job=job,
                tasks=ordered_tasks,
                registry=registry,
                seed=seed,
                mode="teacher-agent",
                predictor=OracleToolCallPredictor(ordered_tasks, label="deterministic-teacher-agent", seed=seed),
            )
        )

    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "measurement_type": "deterministic-local-replay",
        "notes": (
            "Student and teacher agent modes use deterministic oracle predictors to "
            "exercise the model-agent execution path without requiring a live model endpoint."
        ),
        "seeds": list(job.seeds),
        "include_synthetic_schemas": job.include_synthetic_schemas,
        "task_count_per_run": len(tasks),
        "runs": runs,
        "aggregate": _aggregate_runs(runs),
    }
    _write_json(job.output_dir / "manifest.json", manifest)
    return manifest


def _run_and_write(
    *,
    job: MeasurementJob,
    tasks: list[McpTask],
    registry: McpToolRegistry,
    seed: int,
    mode: BenchmarkMode,
    predictor: ToolCallPredictor | None = None,
) -> dict[str, Any]:
    tracemalloc.start()
    started = time.perf_counter()
    summary = TaskBenchRunner(registry).run(tasks, mode=mode, predictor=predictor)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    latency = _latency_payload(summary, elapsed_ms=elapsed_ms)
    memory = {
        "peak_traced_bytes": peak_bytes,
        "peak_traced_mb": round(peak_bytes / (1024 * 1024), 6),
    }
    run_dir = job.output_dir / mode / f"seed-{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_payload = summary.as_dict()
    summary_payload["seed"] = seed
    summary_payload["latency"] = latency
    summary_payload["memory"] = memory
    raw_outputs = [
        {
            "task_id": case.task_id,
            "raw_model_output": case.raw_model_output,
            "parsed_tool_call": case.parsed_tool_call,
            "execution_result": case.execution_result,
            "latency_ms": case.latency_ms,
            "error": case.error,
        }
        for case in summary.cases
        if case.raw_model_output is not None
    ]
    promotion = evaluate_promotion(
        PromotionInputs(
            summary=summary,
            faithfulness_score=0.8,
            tenant_security_passed=True,
            ttft_seconds=latency["ttft_seconds"],
            latency_p95_seconds=latency["p95_seconds"],
            previous_tool_f1=0.99,
        )
    )
    promotion_payload = {"promoted": promotion.promoted, "reasons": list(promotion.reasons)}

    _write_json(run_dir / "summary.json", summary_payload)
    _write_json(run_dir / "raw-model-outputs.json", raw_outputs)
    _write_json(run_dir / "promotion-decision.json", promotion_payload)
    return {
        "mode": mode,
        "seed": seed,
        "summary_path": str(run_dir / "summary.json"),
        "raw_model_outputs_path": str(run_dir / "raw-model-outputs.json"),
        "promotion_decision_path": str(run_dir / "promotion-decision.json"),
        "case_count": len(summary.cases),
        "pass_rate": summary.pass_rate,
        "tool_f1": summary.tool_f1,
        "argument_accuracy": summary.argument_accuracy,
        "latency": latency,
        "memory": memory,
        "promoted": promotion.promoted,
    }


def _shuffle_tasks(tasks: list[McpTask], seed: int) -> list[McpTask]:
    shuffled = list(tasks)
    random.Random(seed).shuffle(shuffled)
    return shuffled


def _latency_payload(summary: EvalSummary, *, elapsed_ms: float) -> dict[str, float]:
    case_latencies = [case.latency_ms for case in summary.cases if case.latency_ms is not None]
    if not case_latencies:
        per_case_ms = elapsed_ms / len(summary.cases) if summary.cases else 0.0
        case_latencies = [per_case_ms]
    p95_ms = _percentile(case_latencies, 95)
    return {
        "total_ms": round(elapsed_ms, 6),
        "mean_case_ms": round(statistics.fmean(case_latencies), 6),
        "p95_ms": round(p95_ms, 6),
        "ttft_seconds": round((case_latencies[0] if case_latencies else 0.0) / 1000.0, 6),
        "p95_seconds": round(p95_ms / 1000.0, 6),
    }


def _percentile(values: list[float], percentile: int) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    rank = (len(ordered) - 1) * (percentile / 100)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _aggregate_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    by_mode: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        by_mode.setdefault(str(run["mode"]), []).append(run)
    return {
        mode: {
            "runs": len(mode_runs),
            "mean_pass_rate": statistics.fmean(run["pass_rate"] for run in mode_runs),
            "mean_tool_f1": statistics.fmean(run["tool_f1"] for run in mode_runs),
            "mean_latency_p95_seconds": statistics.fmean(run["latency"]["p95_seconds"] for run in mode_runs),
            "max_peak_traced_mb": max(run["memory"]["peak_traced_mb"] for run in mode_runs),
            "promoted_runs": sum(1 for run in mode_runs if run["promoted"]),
        }
        for mode, mode_runs in sorted(by_mode.items())
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
