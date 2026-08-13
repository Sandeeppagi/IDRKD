"""Live, no-split MCP-TaskBench release evaluation."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from idrkd.evaluation.model_agent import (
    OpenAICompatibleToolCallPredictor,
    ToolCallPredictor,
)
from idrkd.evaluation.synthetic_schemas import (
    build_synthetic_schema_registry,
    build_synthetic_schema_tasks,
    load_synthetic_schema_corpus,
)
from idrkd.evaluation.taskbench import (
    TaskBenchRunner,
    load_tasks_jsonl,
    split_taskbench_tasks,
)


def run_live_taskbench_benchmark(
    *,
    base_url: str,
    model: str,
    tasks_path: Path = Path("eval/taskbench/seed_tasks.jsonl"),
    schemas_path: Path = Path("eval/synthetic_schemas/schemas.jsonl"),
    conflicts_path: Path = Path("eval/synthetic_schemas/conflicts.jsonl"),
    api_key: str = "idrkd-local",
    timeout_seconds: float = 60.0,
    max_tokens: int = 256,
    predictor: ToolCallPredictor | None = None,
) -> dict[str, Any]:
    """Run every seed and synthetic TaskBench case against a live model."""

    seed_tasks = load_tasks_jsonl(tasks_path)
    corpus = load_synthetic_schema_corpus(
        schemas_path=schemas_path,
        conflicts_path=conflicts_path,
    )
    synthetic_tasks = build_synthetic_schema_tasks(corpus)
    tasks = split_taskbench_tasks([*seed_tasks, *synthetic_tasks], split="all")

    active_predictor: ToolCallPredictor
    if predictor is None:
        model_predictor = OpenAICompatibleToolCallPredictor(
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            max_tokens=max_tokens,
        )
        model_predictor.verify_model_available()
        active_predictor = model_predictor
    else:
        active_predictor = predictor

    summary = TaskBenchRunner(build_synthetic_schema_registry(corpus)).run(
        tasks,
        mode="student-agent",
        predictor=active_predictor,
    )
    summary_data = summary.as_dict()
    return {
        "benchmark": "mcp-taskbench-live",
        "evaluation_scope": "live-model-tool-call-conformance-with-fixture-execution",
        "generalization_claim": False,
        "execution_backend": "synthetic-fixture",
        "created_at": datetime.now(UTC).isoformat(),
        "split": "all",
        "model_id": model,
        "model_base_url": base_url,
        "tasks_path": str(tasks_path),
        "synthetic_schemas_path": str(schemas_path),
        "synthetic_conflicts_path": str(conflicts_path),
        "seed_case_count": len(seed_tasks),
        "synthetic_case_count": len(synthetic_tasks),
        "case_count": len(summary.cases),
        "error_count": sum(1 for case in summary.cases if case.error is not None),
        **summary_data,
    }
