"""MCP-TaskBench runner over the real IDRKD MCP registry."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from idrkd.evaluation.bfcl import FunctionCallPrediction, score_function_calls
from idrkd.evaluation.model_agent import (
    ToolCallPredictor,
    filter_tools_for_ablations,
    run_model_agent_case,
)
from idrkd.mcp.tools import JsonRpcRequest, McpToolRegistry


BenchmarkMode = Literal["registry-smoke", "student-agent", "teacher-agent", "ablation"]
TaskBenchSplit = Literal["all", "train", "holdout"]
TASKBENCH_SCHEMA_VERSION = 2
MCP_PROTOCOL_REVISION = "2025-03-26"
TOOL_CATALOG_REVISION = "idrkd-mcp-tools-v1"


class McpTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    category: Literal[
        "tool_selection",
        "schema_conformance",
        "multi_hop_planning",
        "conflict_resolution",
        "drift_trigger",
        "a2a_delegation",
    ]
    prompt: str
    expected_tool: str
    arguments: dict[str, Any]
    expected_result_keys: list[str] = Field(default_factory=list)
    expected_result_values: dict[str, Any] = Field(default_factory=dict)
    should_succeed: bool = True
    benchmark_schema_version: int = TASKBENCH_SCHEMA_VERSION
    protocol_revision: str = MCP_PROTOCOL_REVISION
    tool_catalog_revision: str = TOOL_CATALOG_REVISION

    def expected_call(self) -> FunctionCallPrediction:
        return FunctionCallPrediction(name=self.expected_tool, arguments=self.arguments)


@dataclass(frozen=True)
class EvalCaseResult:
    task_id: str
    category: str
    benchmark_schema_version: int
    protocol_revision: str
    tool_catalog_revision: str
    tool_correct: bool
    arguments_correct: bool
    schema_valid: bool
    execution_success: bool
    execution_outcome_success: bool
    outcome_valid: bool
    expected_result_keys_present: bool
    raw_model_output: str | None = None
    parsed_tool_call: dict[str, Any] | None = None
    execution_result: dict[str, Any] | None = None
    latency_ms: float | None = None
    error: str | None = None

    @property
    def passed(self) -> bool:
        return (
            self.tool_correct
            and self.arguments_correct
            and self.schema_valid
            and self.execution_success
            and self.execution_outcome_success
            and self.outcome_valid
            and self.expected_result_keys_present
        )


@dataclass(frozen=True)
class EvalSummary:
    cases: tuple[EvalCaseResult, ...]
    tool_precision: float
    tool_recall: float
    tool_f1: float
    argument_accuracy: float
    mode: BenchmarkMode = "registry-smoke"
    ablations: tuple[str, ...] = ()

    @property
    def pass_rate(self) -> float:
        if not self.cases:
            return 0.0
        return sum(1 for case in self.cases if case.passed) / len(self.cases)

    @property
    def schema_valid_rate(self) -> float:
        if not self.cases:
            return 0.0
        return sum(1 for case in self.cases if case.schema_valid) / len(self.cases)

    @property
    def tool_call_pass_rate(self) -> float:
        """Rate of case-aligned, schema-valid tool calls before backend outcome scoring."""

        if not self.cases:
            return 0.0
        return sum(
            1
            for case in self.cases
            if case.tool_correct and case.arguments_correct and case.schema_valid
        ) / len(self.cases)

    @property
    def semantic_outcome_rate(self) -> float:
        if not self.cases:
            return 0.0
        return sum(1 for case in self.cases if case.outcome_valid) / len(self.cases)

    def by_category(self) -> dict[str, float]:
        categories = sorted({case.category for case in self.cases})
        return {
            category: (
                sum(1 for case in self.cases if case.category == category and case.passed)
                / sum(1 for case in self.cases if case.category == category)
            )
            for category in categories
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "cases": [case.__dict__ for case in self.cases],
            "mode": self.mode,
            "ablations": list(self.ablations),
            "pass_rate": self.pass_rate,
            "tool_call_pass_rate": self.tool_call_pass_rate,
            "semantic_outcome_rate": self.semantic_outcome_rate,
            "schema_valid_rate": self.schema_valid_rate,
            "tool_precision": self.tool_precision,
            "tool_recall": self.tool_recall,
            "tool_f1": self.tool_f1,
            "argument_accuracy": self.argument_accuracy,
            "by_category": self.by_category(),
        }


class TaskBenchRunner:
    def __init__(self, registry: McpToolRegistry) -> None:
        self._registry = registry

    def run(
        self,
        tasks: list[McpTask],
        *,
        mode: BenchmarkMode = "registry-smoke",
        predictor: ToolCallPredictor | None = None,
        ablations: tuple[str, ...] = (),
    ) -> EvalSummary:
        if mode != "registry-smoke" and predictor is None:
            raise ValueError(f"{mode} requires a model-agent predictor")
        cases: list[EvalCaseResult] = []
        expected_calls: list[FunctionCallPrediction] = []
        predicted_calls: list[FunctionCallPrediction | None] = []
        for task in tasks:
            expected_calls.append(task.expected_call())
            case = (
                self._run_registry_smoke_case(task)
                if mode == "registry-smoke"
                else self._run_model_agent_case(task, predictor=predictor, ablations=ablations)
            )
            predicted_calls.append(
                (
                    FunctionCallPrediction(
                        name=case.parsed_tool_call["name"],
                        arguments=case.parsed_tool_call["arguments"],
                    )
                    if case.parsed_tool_call is not None
                    else None
                )
            )
            cases.append(case)
        metrics = score_function_calls(expected_calls, predicted_calls)
        return EvalSummary(
            cases=tuple(cases),
            tool_precision=metrics.precision,
            tool_recall=metrics.recall,
            tool_f1=metrics.f1,
            argument_accuracy=metrics.argument_accuracy,
            mode=mode,
            ablations=ablations,
        )

    def _run_registry_smoke_case(self, task: McpTask) -> EvalCaseResult:
        schema_valid = False
        execution_success = False
        outcome_valid = False
        keys_present = False
        error = None
        result: dict[str, Any] | None = None
        try:
            list_response = self._registry.handle(JsonRpcRequest(method="tools/list", id=f"{task.id}:list"))
            schema_valid = _tool_exists_in_schema(list_response.result or {}, task.expected_tool)
            response = self._registry.handle(
                JsonRpcRequest(
                    method="tools/call",
                    id=task.id,
                    params={"name": task.expected_tool, "arguments": task.arguments},
                )
            )
            execution_success = response.error is None if task.should_succeed else response.error is not None
            result = response.result or {}
            keys_present = all(key in result for key in task.expected_result_keys)
            outcome_valid = execution_success and _result_satisfies_task(task, result)
            if response.error is not None:
                error = response.error.message
        except Exception as exc:  # pragma: no cover - defensive harness boundary
            error = str(exc)
        return EvalCaseResult(
            task_id=task.id,
            category=task.category,
            benchmark_schema_version=task.benchmark_schema_version,
            protocol_revision=task.protocol_revision,
            tool_catalog_revision=task.tool_catalog_revision,
            tool_correct=True,
            arguments_correct=True,
            schema_valid=schema_valid,
            execution_success=execution_success,
            execution_outcome_success=outcome_valid,
            outcome_valid=outcome_valid,
            expected_result_keys_present=keys_present,
            parsed_tool_call={
                "name": task.expected_tool,
                "arguments": task.arguments,
            },
            execution_result=result if execution_success else None,
            error=error,
        )

    def _run_model_agent_case(
        self,
        task: McpTask,
        *,
        predictor: ToolCallPredictor | None,
        ablations: tuple[str, ...],
    ) -> EvalCaseResult:
        assert predictor is not None
        schema_valid = False
        keys_present = False
        error = None
        list_response = self._registry.handle(JsonRpcRequest(method="tools/list", id=f"{task.id}:list"))
        tools = filter_tools_for_ablations(list_response.result.get("tools", []) if list_response.result else [], ablations)
        schema_valid = _tool_exists_in_schema({"tools": tools}, task.expected_tool)
        prediction = run_model_agent_case(
            registry=self._registry,
            predictor=predictor,
            prompt=task.prompt,
            tools=tools,
            task_id=task.id,
        )
        parsed = prediction.parsed_tool_call
        execution_success = (
            prediction.execution_error is None if task.should_succeed else prediction.execution_error is not None
        )
        outcome_valid = execution_success and _result_satisfies_task(
            task,
            prediction.execution_result or {},
        )
        if prediction.execution_result is not None:
            keys_present = all(key in prediction.execution_result for key in task.expected_result_keys)
        if prediction.execution_error is not None:
            error = prediction.execution_error
        return EvalCaseResult(
            task_id=task.id,
            category=task.category,
            benchmark_schema_version=task.benchmark_schema_version,
            protocol_revision=task.protocol_revision,
            tool_catalog_revision=task.tool_catalog_revision,
            tool_correct=parsed is not None and parsed.name == task.expected_tool,
            arguments_correct=parsed is not None and parsed.arguments == task.arguments,
            schema_valid=schema_valid,
            execution_success=execution_success,
            execution_outcome_success=outcome_valid,
            outcome_valid=outcome_valid,
            expected_result_keys_present=keys_present,
            raw_model_output=prediction.raw_model_output,
            parsed_tool_call=(
                {"name": parsed.name, "arguments": parsed.arguments} if parsed is not None else None
            ),
            execution_result=prediction.execution_result,
            latency_ms=prediction.latency_ms,
            error=error,
        )


def load_tasks_jsonl(path: Path, *, expose_expected_arguments: bool = False) -> list[McpTask]:
    tasks = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            task = McpTask.model_validate_json(line)
            tasks.append(
                _make_prompt_self_contained(
                    task,
                    expose_expected_arguments=expose_expected_arguments,
                )
            )
    return tasks


def split_taskbench_tasks(
    tasks: list[McpTask],
    *,
    split: TaskBenchSplit,
    holdout_fraction: float = 0.2,
    seed: int = 17,
) -> list[McpTask]:
    """Select a deterministic, tool-stratified TaskBench partition.

    Repeated seed cases are grouped by their natural-language prompt before
    assignment so equivalent wording cannot appear in both partitions.
    """

    if split not in {"all", "train", "holdout"}:
        raise ValueError(f"Unsupported TaskBench split: {split}")
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be between 0 and 1")
    if split == "all":
        return list(tasks)

    groups_by_tool: dict[str, dict[str, list[McpTask]]] = {}
    for task in tasks:
        prompt_group = _taskbench_prompt_group(task)
        groups_by_tool.setdefault(task.expected_tool, {}).setdefault(prompt_group, []).append(task)

    holdout_groups: set[tuple[str, str]] = set()
    for tool_name, prompt_groups in groups_by_tool.items():
        ranked = sorted(
            prompt_groups,
            key=lambda prompt: hashlib.sha256(
                f"{seed}\0{tool_name}\0{prompt}".encode("utf-8")
            ).hexdigest(),
        )
        if len(ranked) == 1:
            continue
        target_count = round(sum(len(prompt_groups[prompt]) for prompt in ranked) * holdout_fraction)
        subsets: dict[int, tuple[str, ...]] = {0: ()}
        for prompt in ranked:
            group_size = len(prompt_groups[prompt])
            additions = {
                count + group_size: selected + (prompt,)
                for count, selected in subsets.items()
                if count + group_size not in subsets
            }
            subsets.update(additions)
        total_count = sum(len(group) for group in prompt_groups.values())
        valid_counts = [count for count in subsets if 0 < count < total_count]
        selected_count = min(valid_counts, key=lambda count: (abs(count - target_count), count))
        holdout_groups.update((tool_name, prompt) for prompt in subsets[selected_count])

    selected = []
    for task in tasks:
        is_holdout = (task.expected_tool, _taskbench_prompt_group(task)) in holdout_groups
        if (split == "holdout" and is_holdout) or (split == "train" and not is_holdout):
            selected.append(task)
    return selected


def write_summary(path: Path, summary: EvalSummary) -> None:
    path.write_text(json.dumps(summary.as_dict(), indent=2, sort_keys=True), encoding="utf-8")


def _tool_exists_in_schema(result: dict[str, Any], tool_name: str) -> bool:
    tools = result.get("tools", [])
    return any(isinstance(tool, dict) and tool.get("name") == tool_name for tool in tools)


def _result_satisfies_task(task: McpTask, result: dict[str, Any]) -> bool:
    if not task.should_succeed:
        return True
    if result.get("available") is False or result.get("found") is False:
        return False
    return all(result.get(key) == value for key, value in task.expected_result_values.items())


def _make_prompt_self_contained(
    task: McpTask,
    *,
    expose_expected_arguments: bool = False,
) -> McpTask:
    if "Task scope and identifiers as JSON:" in task.prompt:
        return task
    if "Task tenant/repository scope as JSON:" in task.prompt:
        return task
    if expose_expected_arguments:
        prompt_context = (
            "Task scope and identifiers as JSON:\n"
            f"{json.dumps(task.arguments, sort_keys=True)}"
        )
    else:
        scope = {
            key: task.arguments[key]
            for key in ("tenant_id", "repo_id")
            if key in task.arguments
        }
        if not scope:
            return task
        prompt_context = (
            "Task tenant/repository scope as JSON:\n"
            f"{json.dumps(scope, sort_keys=True)}"
        )
    return task.model_copy(
        update={
            "prompt": (
                f"{task.prompt}\n\n"
                f"{prompt_context}"
            )
        }
    )


def _taskbench_prompt_group(task: McpTask) -> str:
    prompt, _, _ = task.prompt.partition("\n\nTask scope and identifiers as JSON:")
    prompt, _, _ = prompt.partition("\n\nTask tenant/repository scope as JSON:")
    return prompt.strip()
