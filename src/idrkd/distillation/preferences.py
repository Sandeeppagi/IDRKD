"""DPO preference-pair construction for SLM alignment."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from idrkd.distillation.traces import TeacherTrace, ToolCall, first_tool_call, tool_call_json


@dataclass(frozen=True)
class PreferencePair:
    prompt: str
    chosen: str
    rejected: str
    trace_id: str
    metadata: dict[str, Any]

    def to_dpo_record(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected,
            "metadata": self.metadata,
        }


def build_preference_pair(
    *,
    trace: TeacherTrace,
    rejected_answer: str | None = None,
    rejected_source: str = "sft_naive",
) -> PreferencePair:
    """Prefer the correct structured tool call over a wrong structured call."""

    target_call = first_tool_call(trace)
    if target_call is None:
        raise ValueError(f"Trace {trace.id} does not contain an MCP tool call")
    rejected_call = _rejected_tool_call_json(target_call, rejected_answer)
    return PreferencePair(
        prompt=trace.prompt,
        chosen=tool_call_json(target_call),
        rejected=rejected_call,
        trace_id=trace.id,
        metadata={
            "trace_id": trace.id,
            "tenant_id": trace.tenant_id,
            "repo_id": trace.repo_id,
            "rejected_source": rejected_source,
            "faithfulness_score": trace.faithfulness_score,
            "teacher_answer": trace.answer,
            "target_tool_call": {"name": target_call.name, "arguments": target_call.arguments},
            "rejected_answer_text": rejected_answer,
        },
    )


def _rejected_tool_call_json(target_call: ToolCall, rejected_answer: str | None) -> str:
    if rejected_answer is not None and _is_tool_call_json(rejected_answer):
        return rejected_answer
    wrong_arguments = dict(target_call.arguments)
    wrong_arguments["_idrkd_wrong_argument"] = True
    return tool_call_json(ToolCall(name=target_call.name, arguments=wrong_arguments))


def _is_tool_call_json(value: str) -> bool:
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return False
    return (
        isinstance(payload, dict)
        and isinstance(payload.get("name"), str)
        and isinstance(payload.get("arguments"), dict)
    )
