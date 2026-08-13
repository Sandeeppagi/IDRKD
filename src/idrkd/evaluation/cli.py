"""Command line entry point for local MCP-TaskBench runs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from idrkd.evaluation.measurements import MeasurementJob, build_measurement_bundle
from idrkd.evaluation.model_agent import OpenAICompatibleToolCallPredictor
from idrkd.evaluation.synthetic_schemas import (
    build_synthetic_schema_registry,
    build_synthetic_schema_tasks,
    load_synthetic_schema_corpus,
)
from idrkd.evaluation.taskbench import (
    TaskBenchRunner,
    load_tasks_jsonl,
    split_taskbench_tasks,
    write_summary,
)
from idrkd.mcp.server import build_registry_from_env


def main() -> None:
    parser = argparse.ArgumentParser(description="Run IDRKD MCP-TaskBench tasks.")
    subparsers = parser.add_subparsers(dest="command")
    measurements_parser = subparsers.add_parser(
        "build-measurements",
        help="Write reproducible local evaluation measurement bundles.",
    )
    measurements_parser.add_argument("--tasks", type=Path, default=Path("eval/taskbench/seed_tasks.jsonl"))
    measurements_parser.add_argument("--out-dir", type=Path, default=Path("eval/measurements"))
    measurements_parser.add_argument("--seed", action="append", type=int, default=[])
    measurements_parser.add_argument("--without-synthetic-schemas", action="store_true")
    _add_split_args(measurements_parser, default="all")
    parser.add_argument("--tasks", type=Path, default=Path("eval/taskbench/seed_tasks.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("eval/taskbench/latest-summary.json"))
    parser.add_argument(
        "--mode",
        choices=("registry-smoke", "student-agent", "teacher-agent", "ablation"),
        default="registry-smoke",
    )
    parser.add_argument("--model-base-url", help="OpenAI-compatible /v1 base URL for agent modes.")
    parser.add_argument("--model-id", help="Model id for student-agent or ablation modes.")
    parser.add_argument("--teacher-model-id", help="Model id for teacher-agent mode.")
    parser.add_argument("--api-key", default=os.getenv("IDRKD_EVAL_MODEL_API_KEY", "idrkd-local"))
    parser.add_argument("--ablation", action="append", default=[])
    parser.add_argument(
        "--include-synthetic-schemas",
        action="store_true",
        help="Append synthetic schema/conflict fixture tasks and use the fixture-backed registry.",
    )
    parser.add_argument("--synthetic-schemas", type=Path, default=Path("eval/synthetic_schemas/schemas.jsonl"))
    parser.add_argument("--synthetic-conflicts", type=Path, default=Path("eval/synthetic_schemas/conflicts.jsonl"))
    _add_split_args(parser, default="all")
    args = parser.parse_args()

    if args.command == "build-measurements":
        manifest = build_measurement_bundle(
            MeasurementJob(
                tasks_path=args.tasks,
                output_dir=args.out_dir,
                include_synthetic_schemas=not args.without_synthetic_schemas,
                seeds=tuple(args.seed) if args.seed else (11, 23, 37),
                split=args.split,
                holdout_fraction=args.holdout_fraction,
                split_seed=args.split_seed,
            )
        )
        print(json.dumps({"manifest": str(args.out_dir / "manifest.json"), "runs": len(manifest["runs"])}, sort_keys=True))
        return

    tasks = load_tasks_jsonl(args.tasks)
    registry = build_registry_from_env()
    if args.include_synthetic_schemas:
        corpus = load_synthetic_schema_corpus(
            schemas_path=args.synthetic_schemas,
            conflicts_path=args.synthetic_conflicts,
        )
        tasks.extend(build_synthetic_schema_tasks(corpus))
        registry = build_synthetic_schema_registry(corpus)
    tasks = split_taskbench_tasks(
        tasks,
        split=args.split,
        holdout_fraction=args.holdout_fraction,
        seed=args.split_seed,
    )
    predictor = None
    if args.mode != "registry-smoke":
        base_url = args.model_base_url or os.getenv("IDRKD_EVAL_MODEL_BASE_URL")
        model_id = (
            args.teacher_model_id
            if args.mode == "teacher-agent"
            else args.model_id
        ) or os.getenv(
            "IDRKD_EVAL_TEACHER_MODEL_ID" if args.mode == "teacher-agent" else "IDRKD_EVAL_STUDENT_MODEL_ID"
        )
        if not base_url or not model_id:
            raise SystemExit(
                f"{args.mode} requires --model-base-url and a model id "
                "(--model-id or --teacher-model-id)."
            )
        predictor = OpenAICompatibleToolCallPredictor(
            base_url=base_url,
            model=model_id,
            api_key=args.api_key,
        )
        predictor.verify_model_available()
    summary = TaskBenchRunner(registry).run(
        tasks,
        mode=args.mode,
        predictor=predictor,
        ablations=tuple(args.ablation),
    )
    write_summary(args.out, summary)
    print(
        f"mode={summary.mode} pass_rate={summary.pass_rate:.3f} "
        f"tool_f1={summary.tool_f1:.3f} cases={len(summary.cases)}"
    )


def _add_split_args(parser: argparse.ArgumentParser, *, default: str) -> None:
    parser.add_argument("--split", choices=("train", "holdout", "all"), default=default)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=17)


if __name__ == "__main__":
    main()
