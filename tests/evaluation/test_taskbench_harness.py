from pathlib import Path

from idrkd.evaluation import FunctionCallPrediction, TaskBenchRunner, load_tasks_jsonl, score_function_calls
from idrkd.mcp.tools import McpToolRegistry


def test_bfcl_style_function_call_scoring() -> None:
    metrics = score_function_calls(
        expected=[
            FunctionCallPrediction("search_code", {"query": "customer"}),
            FunctionCallPrediction("get_entity", {"entity_id": "a"}),
        ],
        predicted=[
            FunctionCallPrediction("search_code", {"query": "customer"}),
            FunctionCallPrediction("graph_bfs", {"entity_id": "a"}),
        ],
    )

    assert metrics.true_positives == 1
    assert metrics.false_positives == 1
    assert metrics.false_negatives == 1
    assert metrics.argument_accuracy == 1.0


def test_taskbench_runner_executes_seed_tasks_against_registry() -> None:
    registry = McpToolRegistry(principal_tenant_id="default")
    tasks = load_tasks_jsonl(Path("eval/taskbench/seed_tasks.jsonl"))

    summary = TaskBenchRunner(registry).run(tasks)

    assert len(summary.cases) == 6
    assert summary.tool_f1 == 1.0
    assert summary.argument_accuracy == 1.0
    assert summary.schema_valid_rate == 1.0
    assert summary.pass_rate == 1.0
    assert set(summary.by_category()) >= {"tool_selection", "drift_trigger", "conflict_resolution"}
