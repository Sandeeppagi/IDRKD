"""Command line entry point for local MCP-TaskBench runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from idrkd.evaluation.taskbench import TaskBenchRunner, load_tasks_jsonl, write_summary
from idrkd.mcp.server import build_registry_from_env


def main() -> None:
    parser = argparse.ArgumentParser(description="Run IDRKD MCP-TaskBench tasks.")
    parser.add_argument("--tasks", type=Path, default=Path("eval/taskbench/seed_tasks.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("eval/taskbench/latest-summary.json"))
    args = parser.parse_args()

    tasks = load_tasks_jsonl(args.tasks)
    summary = TaskBenchRunner(build_registry_from_env()).run(tasks)
    write_summary(args.out, summary)
    print(f"pass_rate={summary.pass_rate:.3f} tool_f1={summary.tool_f1:.3f} cases={len(summary.cases)}")


if __name__ == "__main__":
    main()
