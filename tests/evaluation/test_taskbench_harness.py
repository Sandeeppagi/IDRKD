from pathlib import Path

import pytest

from idrkd.evaluation import (
    FunctionCallPrediction,
    PromotionInputs,
    TaskBenchRunner,
    build_synthetic_schema_tasks,
    evaluate_promotion,
    load_synthetic_schema_corpus,
    load_tasks_jsonl,
    parse_tool_call,
    score_function_calls,
    split_taskbench_tasks,
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


def test_function_call_scoring_is_case_aligned_for_repeated_tools() -> None:
    metrics = score_function_calls(
        expected=[
            FunctionCallPrediction("search_code", {"query": "customer"}),
            FunctionCallPrediction("search_code", {"query": "billing"}),
        ],
        predicted=[
            FunctionCallPrediction("search_code", {"query": "billing"}),
            FunctionCallPrediction("search_code", {"query": "customer"}),
        ],
    )

    assert metrics.true_positives == 2
    assert metrics.argument_accuracy == 0.0


def test_function_call_scoring_preserves_missing_case_predictions() -> None:
    metrics = score_function_calls(
        expected=[FunctionCallPrediction("search_code", {"query": "customer"})],
        predicted=[None],
    )

    assert metrics.false_negatives == 1
    assert metrics.f1 == 0.0


def test_taskbench_runner_executes_seed_tasks_against_registry() -> None:
    registry = McpToolRegistry(principal_tenant_id="default")
    tasks = load_tasks_jsonl(Path("eval/taskbench/seed_tasks.jsonl"))

    summary = TaskBenchRunner(registry).run(tasks)

    assert len(summary.cases) == 360
    assert summary.tool_f1 == 1.0
    assert summary.argument_accuracy == 1.0
    assert summary.schema_valid_rate == 1.0
    assert summary.tool_call_pass_rate == 1.0
    assert summary.semantic_outcome_rate == 0.5
    assert summary.pass_rate == 0.5
    assert summary.cases[0].benchmark_schema_version == 2
    assert summary.cases[0].protocol_revision == "2025-03-26"
    assert summary.cases[0].tool_catalog_revision == "idrkd-mcp-tools-v1"
    unavailable = next(case for case in summary.cases if case.task_id == "tb-graph-bfs-003")
    assert unavailable.execution_success is True
    assert unavailable.execution_outcome_success is False
    assert unavailable.outcome_valid is False
    assert set(summary.by_category()) >= {
        "tool_selection",
        "schema_conformance",
        "multi_hop_planning",
        "conflict_resolution",
        "drift_trigger",
        "a2a_delegation",
    }


def test_taskbench_prompts_do_not_expose_expected_arguments_by_default() -> None:
    task = load_tasks_jsonl(Path("eval/taskbench/seed_tasks.jsonl"))[0]
    leaked_task = load_tasks_jsonl(
        Path("eval/taskbench/seed_tasks.jsonl"),
        expose_expected_arguments=True,
    )[0]

    assert '"tenant_id": "default"' in task.prompt
    assert '"repo_id": "repo-a"' in task.prompt
    assert '"query": "customer lookup"' not in task.prompt
    assert '"query": "customer lookup"' in leaked_task.prompt


def test_taskbench_split_is_disjoint_stratified_and_balances_conflict_tools() -> None:
    tasks = load_tasks_jsonl(Path("eval/taskbench/seed_tasks.jsonl"))
    tasks.extend(build_synthetic_schema_tasks(load_synthetic_schema_corpus()))

    train = split_taskbench_tasks(tasks, split="train", holdout_fraction=0.2, seed=17)
    holdout = split_taskbench_tasks(tasks, split="holdout", holdout_fraction=0.2, seed=17)

    assert len(tasks) == 440
    assert {task.id for task in train}.isdisjoint(task.id for task in holdout)
    assert {task.id for task in train} | {task.id for task in holdout} == {
        task.id for task in tasks
    }

    scope_marker = "\n\nTask scope and identifiers as JSON:"
    train_prompts = {task.prompt.partition(scope_marker)[0] for task in train}
    holdout_prompts = {task.prompt.partition(scope_marker)[0] for task in holdout}
    assert train_prompts.isdisjoint(holdout_prompts)

    expected_tools = {task.expected_tool for task in tasks}
    assert {task.expected_tool for task in train} == expected_tools
    assert {task.expected_tool for task in holdout} == expected_tools
    assert sum(task.expected_tool == "reconcile" for task in tasks) == 54
    assert sum(task.expected_tool == "get_conflict" for task in tasks) == 54
    assert sum(task.expected_tool == "reconcile" for task in train) == 43
    assert sum(task.expected_tool == "get_conflict" for task in train) == 43
    assert sum(task.expected_tool == "reconcile" for task in holdout) == 11
    assert sum(task.expected_tool == "get_conflict" for task in holdout) == 11


def test_taskbench_split_validates_configuration() -> None:
    tasks = load_tasks_jsonl(Path("eval/taskbench/seed_tasks.jsonl"))

    with pytest.raises(ValueError, match="holdout_fraction"):
        split_taskbench_tasks(tasks, split="train", holdout_fraction=1.0)


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
