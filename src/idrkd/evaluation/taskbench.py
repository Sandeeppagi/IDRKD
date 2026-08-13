"""MCP-TaskBench runner over the real IDRKD MCP registry."""

from __future__ import annotations

from dataclasses import dataclass
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
    should_succeed: bool = True

    def expected_call(self) -> FunctionCallPrediction:
        return FunctionCallPrediction(name=self.expected_tool, arguments=self.arguments)


@dataclass(frozen=True)
class EvalCaseResult:
    task_id: str
    category: str
    tool_correct: bool
    arguments_correct: bool
    schema_valid: bool
    execution_success: bool
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
        predicted_calls: list[FunctionCallPrediction] = []
        for task in tasks:
            expected_calls.append(task.expected_call())
            case = (
                self._run_registry_smoke_case(task)
                if mode == "registry-smoke"
                else self._run_model_agent_case(task, predictor=predictor, ablations=ablations)
            )
            if case.parsed_tool_call is not None:
                predicted_calls.append(
                    FunctionCallPrediction(
                        name=case.parsed_tool_call["name"],
                        arguments=case.parsed_tool_call["arguments"],
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
            if response.error is not None:
                error = response.error.message
        except Exception as exc:  # pragma: no cover - defensive harness boundary
            error = str(exc)
        return EvalCaseResult(
            task_id=task.id,
            category=task.category,
            tool_correct=True,
            arguments_correct=True,
            schema_valid=schema_valid,
            execution_success=execution_success,
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
        if prediction.execution_result is not None:
            keys_present = all(key in prediction.execution_result for key in task.expected_result_keys)
        if prediction.execution_error is not None:
            error = prediction.execution_error
        return EvalCaseResult(
            task_id=task.id,
            category=task.category,
            tool_correct=parsed is not None and parsed.name == task.expected_tool,
            arguments_correct=parsed is not None and parsed.arguments == task.arguments,
            schema_valid=schema_valid,
            execution_success=execution_success,
            expected_result_keys_present=keys_present,
            raw_model_output=prediction.raw_model_output,
            parsed_tool_call=(
                {"name": parsed.name, "arguments": parsed.arguments} if parsed is not None else None
            ),
            execution_result=prediction.execution_result,
            latency_ms=prediction.latency_ms,
            error=error,
        )


def load_tasks_jsonl(path: Path) -> list[McpTask]:
    tasks = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            task = McpTask.model_validate_json(line)
            tasks.append(_make_prompt_self_contained(task))
    return tasks


def write_summary(path: Path, summary: EvalSummary) -> None:
    path.write_text(json.dumps(summary.as_dict(), indent=2, sort_keys=True), encoding="utf-8")


def _tool_exists_in_schema(result: dict[str, Any], tool_name: str) -> bool:
    tools = result.get("tools", [])
    return any(isinstance(tool, dict) and tool.get("name") == tool_name for tool in tools)


def _make_prompt_self_contained(task: McpTask) -> McpTask:
    if "Task scope and identifiers as JSON:" in task.prompt:
        return task
    return task.model_copy(
        update={
            "prompt": (
                f"{task.prompt}\n\n"
                "Task scope and identifiers as JSON:\n"
                f"{json.dumps(task.arguments, sort_keys=True)}"
            )
        }
    )
