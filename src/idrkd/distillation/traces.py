"""Teacher trace capture and SFT dataset shaping for Pillar 5."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


SYSTEM_PROMPT = (
    "You are the IDRKD student model. Answer with grounded reasoning and emit MCP tool "
    "calls only when the task requires repository evidence."
)


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class TraceStep:
    agent: str
    input_text: str
    output_text: str
    tool_calls: tuple[ToolCall, ...] = ()
    evidence_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class TeacherTrace:
    id: str
    tenant_id: str
    repo_id: str
    prompt: str
    answer: str
    steps: tuple[TraceStep, ...]
    faithfulness_score: float
    bfcl_category: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def uses_tool(self, tool_name: str) -> bool:
        return any(call.name == tool_name for step in self.steps for call in step.tool_calls)

    def evidence_ids(self) -> tuple[str, ...]:
        ids: list[str] = []
        for step in self.steps:
            ids.extend(step.evidence_ids)
        return tuple(dict.fromkeys(ids))


def sft_record(trace: TeacherTrace) -> dict[str, Any]:
    """Convert a grounded teacher trace into a chat SFT record."""

    tool_trace = [
        {"agent": step.agent, "tool_calls": [call.name for call in step.tool_calls]}
        for step in trace.steps
        if step.tool_calls
    ]
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": trace.prompt},
            {"role": "assistant", "content": trace.answer},
        ],
        "metadata": {
            "trace_id": trace.id,
            "tenant_id": trace.tenant_id,
            "repo_id": trace.repo_id,
            "bfcl_category": trace.bfcl_category,
            "faithfulness_score": trace.faithfulness_score,
            "evidence_ids": list(trace.evidence_ids()),
            "tool_trace": tool_trace,
        },
    }


def select_sft_traces(
    traces: list[TeacherTrace],
    *,
    min_faithfulness: float = 0.78,
    require_tool_use: bool = True,
) -> list[TeacherTrace]:
    """Keep only traces that are useful and safe enough for student SFT."""

    selected: list[TeacherTrace] = []
    for trace in traces:
        has_tool_use = any(step.tool_calls for step in trace.steps)
        if trace.faithfulness_score >= min_faithfulness and (has_tool_use or not require_tool_use):
            selected.append(trace)
    return selected
