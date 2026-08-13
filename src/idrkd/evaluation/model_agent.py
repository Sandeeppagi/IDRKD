"""Model-agent MCP tool prediction and execution for TaskBench."""

from __future__ import annotations

from dataclasses import dataclass
import json
import time
from typing import Any, Protocol
from urllib import request

from idrkd.evaluation.bfcl import FunctionCallPrediction
from idrkd.mcp.tools import JsonRpcRequest, JsonRpcResponse, McpToolRegistry


class ToolCallPredictor(Protocol):
    def predict_tool_call(
        self,
        *,
        prompt: str,
        tools: list[dict[str, Any]],
    ) -> tuple[str, FunctionCallPrediction | None]:
        ...


@dataclass(frozen=True)
class ModelAgentPredictionResult:
    raw_model_output: str
    parsed_tool_call: FunctionCallPrediction | None
    execution_result: dict[str, Any] | None
    execution_error: str | None
    latency_ms: float


@dataclass(frozen=True)
class OpenAICompatibleToolCallPredictor:
    base_url: str
    model: str
    api_key: str = "idrkd-local"
    timeout_seconds: float = 60.0
    temperature: float = 0.0
    max_tokens: int = 512

    def predict_tool_call(
        self,
        *,
        prompt: str,
        tools: list[dict[str, Any]],
    ) -> tuple[str, FunctionCallPrediction | None]:
        body = json.dumps(
            {
                "model": self.model,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Select exactly one MCP tool for the user task. "
                            "Return only JSON with keys: name, arguments."
                        ),
                    },
                    {"role": "user", "content": tool_selection_prompt(prompt=prompt, tools=tools)},
                ],
            }
        ).encode("utf-8")
        http_request = request.Request(
            f"{self.base_url.rstrip('/')}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with request.urlopen(http_request, timeout=self.timeout_seconds) as response:
            raw = json.loads(response.read().decode("utf-8"))
        choices = raw.get("choices", [])
        content = ""
        if choices:
            content = str(choices[0].get("message", {}).get("content", "")).strip()
        return content, parse_tool_call(content)


def run_model_agent_case(
    *,
    registry: McpToolRegistry,
    predictor: ToolCallPredictor,
    prompt: str,
    tools: list[dict[str, Any]],
    task_id: str,
) -> ModelAgentPredictionResult:
    started = time.perf_counter()
    raw_output = ""
    parsed: FunctionCallPrediction | None = None
    execution_result: dict[str, Any] | None = None
    execution_error: str | None = None
    try:
        raw_output, parsed = predictor.predict_tool_call(prompt=prompt, tools=tools)
        if parsed is None:
            execution_error = "model output did not contain a parseable tool call"
        else:
            response = registry.handle(
                JsonRpcRequest(
                    method="tools/call",
                    id=f"{task_id}:agent-call",
                    params={"name": parsed.name, "arguments": parsed.arguments},
                )
            )
            execution_result, execution_error = _response_payload(response)
    except Exception as exc:  # pragma: no cover - model/network boundary
        execution_error = str(exc)
    return ModelAgentPredictionResult(
        raw_model_output=raw_output,
        parsed_tool_call=parsed,
        execution_result=execution_result,
        execution_error=execution_error,
        latency_ms=(time.perf_counter() - started) * 1000.0,
    )


def parse_tool_call(raw_output: str) -> FunctionCallPrediction | None:
    payload = _extract_json_object(raw_output)
    if payload is None:
        return None
    name = payload.get("name") or payload.get("tool") or payload.get("tool_name")
    arguments = payload.get("arguments") or payload.get("args") or {}
    if not isinstance(name, str) or not isinstance(arguments, dict):
        return None
    return FunctionCallPrediction(name=name, arguments=arguments)


def filter_tools_for_ablations(
    tools: list[dict[str, Any]],
    ablations: tuple[str, ...],
) -> list[dict[str, Any]]:
    filtered = tools
    if "no_graph" in ablations:
        filtered = [
            tool
            for tool in filtered
            if tool.get("name") not in {"graph_bfs", "graph_path", "get_community"}
        ]
    return filtered


def tool_selection_prompt(*, prompt: str, tools: list[dict[str, Any]]) -> str:
    return (
        f"User task:\n{prompt}\n\n"
        "Available MCP tools as JSON schemas:\n"
        f"{json.dumps(tools, indent=2, sort_keys=True)}\n\n"
        "Tool decision rules:\n"
        "- code lookup or repository search -> search_code\n"
        "- fetch known graph entity -> get_entity\n"
        "- bounded dependency neighbors or BFS -> graph_bfs\n"
        "- path from source to downstream consumer -> graph_path\n"
        "- community members or Louvain/community assignment -> get_community\n"
        "- queue reindexing for a changed entity -> enqueue_reindex\n"
        "- compare schema versions or variants -> schema_diff\n"
        "- blast radius or downstream impact -> impact_analysis\n"
        "- reconciliation recommendation -> reconcile\n"
        "- persist final reconciliation decision -> resolve_conflict\n"
        "- centroid drift -> get_centroid_drift\n"
        "- inspect delegated reconciliation state -> get_conflict\n\n"
        "Argument rules:\n"
        "- Copy argument values exactly from the task scope JSON when present.\n"
        "- Return only arguments required for the selected tool and explicitly present in the task scope.\n"
        "- Do not invent default arguments such as limit, depth, or max_hops unless present in the task scope.\n\n"
        "Return only a JSON object like "
        '{"name":"search_code","arguments":{"tenant_id":"default","repo_id":"repo-a"}}.'
    )


def _extract_json_object(raw_output: str) -> dict[str, Any] | None:
    text = raw_output.strip()
    if not text:
        return None
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            decoded = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    if isinstance(decoded, dict):
        return decoded
    return None


def _response_payload(response: JsonRpcResponse) -> tuple[dict[str, Any] | None, str | None]:
    if response.error is not None:
        return None, response.error.message
    return response.result or {}, None
