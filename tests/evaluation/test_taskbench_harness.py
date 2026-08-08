from pathlib import Path

from idrkd.evaluation import (
    FunctionCallPrediction,
    PromotionInputs,
    TaskBenchRunner,
    evaluate_promotion,
    load_tasks_jsonl,
    parse_tool_call,
    score_function_calls,
)
from idrkd.mcp.tools import McpToolRegistry


class _StaticPredictor:
    def __init__(self, predictions: dict[str, FunctionCallPrediction]) -> None:
        self.predictions = predictions

    def predict_tool_call(self, *, prompt: str, tools: list[dict]):
        prediction = self.predictions[prompt]
        return (
            f'{{"name": "{prediction.name}", "arguments": {prediction.arguments!r}}}'.replace("'", '"'),
            prediction,
        )


class _FirstToolPredictor:
    def predict_tool_call(self, *, prompt: str, tools: list[dict]):
        tool = tools[0]
        return f'{{"name":"{tool["name"]}","arguments":{{}}}}', FunctionCallPrediction(tool["name"], {})


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


def test_parse_tool_call_from_raw_model_output() -> None:
    parsed = parse_tool_call(
        'The call is {"name":"search_code","arguments":{"tenant_id":"default","repo_id":"repo-a","query":"x"}}'
    )

    assert parsed == FunctionCallPrediction(
        "search_code",
        {"tenant_id": "default", "repo_id": "repo-a", "query": "x"},
    )


def test_student_agent_mode_uses_model_prediction_and_executes_tool() -> None:
    registry = McpToolRegistry(principal_tenant_id="default")
    tasks = load_tasks_jsonl(Path("eval/taskbench/seed_tasks.jsonl"))[:1]
    predictor = _StaticPredictor({tasks[0].prompt: tasks[0].expected_call()})

    summary = TaskBenchRunner(registry).run(tasks, mode="student-agent", predictor=predictor)
    case = summary.cases[0]

    assert summary.mode == "student-agent"
    assert summary.tool_f1 == 1.0
    assert case.raw_model_output is not None
    assert case.parsed_tool_call == {"name": "search_code", "arguments": tasks[0].arguments}
    assert case.execution_result is not None
    assert case.latency_ms is not None


def test_ablation_mode_can_remove_graph_tools_from_prompt_schema() -> None:
    registry = McpToolRegistry(principal_tenant_id="default")
    tasks = [task for task in load_tasks_jsonl(Path("eval/taskbench/seed_tasks.jsonl")) if task.expected_tool == "graph_bfs"]

    summary = TaskBenchRunner(registry).run(
        tasks,
        mode="ablation",
        predictor=_FirstToolPredictor(),
        ablations=("no_graph",),
    )

    assert summary.mode == "ablation"
    assert summary.ablations == ("no_graph",)
    assert summary.cases[0].schema_valid is False
    assert summary.tool_f1 == 0.0


def test_promotion_gate_checks_metrics_security_latency_and_regression() -> None:
    registry = McpToolRegistry(principal_tenant_id="default")
    summary = TaskBenchRunner(registry).run(load_tasks_jsonl(Path("eval/taskbench/seed_tasks.jsonl")))

    passed = evaluate_promotion(
        PromotionInputs(
            summary=summary,
            faithfulness_score=0.8,
            tenant_security_passed=True,
            ttft_seconds=1.0,
            latency_p95_seconds=7.0,
            previous_tool_f1=0.99,
        )
    )
    failed = evaluate_promotion(
        PromotionInputs(
            summary=summary,
            faithfulness_score=0.7,
            tenant_security_passed=False,
            ttft_seconds=1.4,
            latency_p95_seconds=9.0,
            previous_tool_f1=1.0,
        )
    )

    assert passed.promoted is True
    assert failed.promoted is False
    assert len(failed.reasons) >= 3
