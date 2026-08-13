"""Paired C3 evaluation for monolithic and LangGraph-AutoGen RAG paths."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import random
import re
import statistics
import time
from typing import Any


C3Runner = Callable[["C3Case"], Awaitable[Mapping[str, Any]]]


@dataclass(frozen=True)
class C3Case:
    case_id: str
    query: str
    expected_answer: str
    tenant_id: str
    repo_id: str
    conflict_id: str | None = None
    expected_entity_ids: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, value: dict[str, Any], *, line_number: int) -> C3Case:
        required = ("id", "query", "expected_answer", "tenant_id", "repo_id")
        missing = [key for key in required if not str(value.get(key, "")).strip()]
        if missing:
            raise ValueError(f"C3 case line {line_number} is missing: {', '.join(missing)}")
        expected_entities = value.get("expected_entity_ids", [])
        if not isinstance(expected_entities, list) or not all(
            isinstance(item, str) for item in expected_entities
        ):
            raise ValueError(
                f"C3 case line {line_number} expected_entity_ids must be a string list"
            )
        conflict_id = value.get("conflict_id")
        return cls(
            case_id=str(value["id"]),
            query=str(value["query"]),
            expected_answer=str(value["expected_answer"]),
            tenant_id=str(value["tenant_id"]),
            repo_id=str(value["repo_id"]),
            conflict_id=str(conflict_id) if conflict_id else None,
            expected_entity_ids=tuple(expected_entities),
        )


def load_c3_cases(path: Path) -> list[C3Case]:
    cases: list[C3Case] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"C3 case line {line_number} must be a JSON object")
        cases.append(C3Case.from_dict(value, line_number=line_number))
    if not cases:
        raise ValueError(f"C3 case file is empty: {path}")
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("C3 case IDs must be unique")
    return cases


async def run_c3_benchmark(
    cases: list[C3Case],
    *,
    monolithic_runner: C3Runner,
    decomposed_runner: C3Runner,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 17,
) -> dict[str, Any]:
    if not cases:
        raise ValueError("C3 benchmark requires at least one case")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be at least one")

    results: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        if index % 2 == 0:
            monolithic = await _run_one(monolithic_runner, case)
            decomposed = await _run_one(decomposed_runner, case)
            execution_order = ["monolithic", "langgraph-autogen"]
        else:
            decomposed = await _run_one(decomposed_runner, case)
            monolithic = await _run_one(monolithic_runner, case)
            execution_order = ["langgraph-autogen", "monolithic"]
        results.append(
            {
                "case": asdict(case),
                "execution_order": execution_order,
                "monolithic": _score_result(case, monolithic),
                "langgraph_autogen": _score_result(case, decomposed),
            }
        )

    metric_names = ("exact_match", "task_completion", "faithfulness")
    metrics: dict[str, Any] = {}
    for metric in metric_names:
        baseline_values = [float(result["monolithic"][metric]) for result in results]
        decomposed_values = [float(result["langgraph_autogen"][metric]) for result in results]
        metrics[metric] = _paired_metric(
            baseline_values,
            decomposed_values,
            samples=bootstrap_samples,
            seed=bootstrap_seed,
        )

    baseline_latency = [float(result["monolithic"]["latency_seconds"]) for result in results]
    decomposed_latency = [
        float(result["langgraph_autogen"]["latency_seconds"]) for result in results
    ]
    metrics["latency_seconds"] = _paired_metric(
        baseline_latency,
        decomposed_latency,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    error_count = sum(
        result[system]["error"] is not None
        for result in results
        for system in ("monolithic", "langgraph_autogen")
    )
    criterion_metrics = [metrics["exact_match"], metrics["task_completion"]]
    criterion_met = any(
        metric["delta"] >= 0.05 and metric["confidence_interval_95"][0] > 0.0
        for metric in criterion_metrics
    )
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "benchmark": "c3-monolithic-vs-langgraph-autogen",
        "case_count": len(cases),
        "error_count": error_count,
        "execution": {
            "paired": True,
            "order": "alternating",
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": bootstrap_seed,
            "monolithic_framework": "python-fixed-loop",
            "decomposed_frameworks": ["langgraph", "a2a-sdk", "autogen-agentchat"],
        },
        "metrics": metrics,
        "c3_criterion": {
            "required_delta": 0.05,
            "requires_confidence_interval_above_zero": True,
            "met": criterion_met,
        },
        "claim_scope": (
            "paired C3 outputs; model, dataset, service, and commit provenance must be bound "
            "with the artifact before reporting an empirical result"
        ),
        "cases": results,
    }


async def _run_one(runner: C3Runner, case: C3Case) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        output = dict(await runner(case))
        return {
            **output,
            "latency_seconds": time.perf_counter() - started,
            "error": None,
        }
    except Exception as exc:
        return {
            "answer": "",
            "accepted": False,
            "faithfulness_score": 0.0,
            "retrieved_entity_ids": [],
            "trace": [],
            "latency_seconds": time.perf_counter() - started,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _score_result(case: C3Case, result: dict[str, Any]) -> dict[str, Any]:
    answer = str(result.get("answer", ""))
    retrieved = [str(item) for item in result.get("retrieved_entity_ids", [])]
    expected_entities = set(case.expected_entity_ids)
    recall = (
        len(expected_entities & set(retrieved)) / len(expected_entities)
        if expected_entities
        else None
    )
    return {
        **result,
        "answer": answer,
        "exact_match": float(_normalize_answer(answer) == _normalize_answer(case.expected_answer)),
        "task_completion": float(bool(answer.strip()) and bool(result.get("accepted"))),
        "faithfulness": float(result.get("faithfulness_score", 0.0)),
        "retrieval_recall": recall,
    }


def _paired_metric(
    baseline: list[float],
    decomposed: list[float],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    if len(baseline) != len(decomposed) or not baseline:
        raise ValueError("paired metrics require non-empty, equal-length samples")
    deltas = [candidate - control for control, candidate in zip(baseline, decomposed, strict=True)]
    bootstrap = _paired_bootstrap(deltas, samples=samples, seed=seed)
    return {
        "monolithic_mean": statistics.fmean(baseline),
        "langgraph_autogen_mean": statistics.fmean(decomposed),
        "delta": statistics.fmean(deltas),
        "confidence_interval_95": [_percentile(bootstrap, 2.5), _percentile(bootstrap, 97.5)],
    }


def _paired_bootstrap(deltas: list[float], *, samples: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    count = len(deltas)
    return [
        statistics.fmean(deltas[rng.randrange(count)] for _ in range(count))
        for _ in range(samples)
    ]


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _normalize_answer(answer: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", answer.casefold()))
