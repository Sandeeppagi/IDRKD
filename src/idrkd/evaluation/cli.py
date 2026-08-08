"""Command line entry point for local MCP-TaskBench runs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from idrkd.evaluation.model_agent import OpenAICompatibleToolCallPredictor
from idrkd.evaluation.taskbench import TaskBenchRunner, load_tasks_jsonl, write_summary
from idrkd.mcp.server import build_registry_from_env


def main() -> None:
    parser = argparse.ArgumentParser(description="Run IDRKD MCP-TaskBench tasks.")
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
    args = parser.parse_args()

    tasks = load_tasks_jsonl(args.tasks)
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
    summary = TaskBenchRunner(build_registry_from_env()).run(
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


if __name__ == "__main__":
    main()
