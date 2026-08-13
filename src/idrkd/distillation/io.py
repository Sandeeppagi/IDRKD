"""JSONL IO and dataset builders for distillation execution."""

from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from typing import Any

from idrkd.distillation.preferences import build_preference_pair
from idrkd.distillation.traces import TeacherTrace, ToolCall, TraceStep, select_sft_traces, sft_record


def teacher_trace_to_dict(trace: TeacherTrace) -> dict[str, Any]:
    return {
        "id": trace.id,
        "tenant_id": trace.tenant_id,
        "repo_id": trace.repo_id,
        "prompt": trace.prompt,
        "answer": trace.answer,
        "steps": [
            {
                "agent": step.agent,
                "input_text": step.input_text,
                "output_text": step.output_text,
                "tool_calls": [
                    {"name": call.name, "arguments": call.arguments} for call in step.tool_calls
                ],
                "evidence_ids": list(step.evidence_ids),
            }
            for step in trace.steps
        ],
        "faithfulness_score": trace.faithfulness_score,
        "bfcl_category": trace.bfcl_category,
        "created_at": trace.created_at.isoformat(),
    }


def teacher_trace_from_dict(payload: dict[str, Any]) -> TeacherTrace:
    steps = []
    for raw_step in _expect_list(payload.get("steps"), "steps"):
        step = _expect_dict(raw_step, "step")
        calls = tuple(
            ToolCall(
                name=str(_expect_dict(raw_call, "tool_call")["name"]),
                arguments=dict(_expect_dict(raw_call, "tool_call").get("arguments", {})),
            )
            for raw_call in _expect_list(step.get("tool_calls", []), "tool_calls")
        )
        steps.append(
            TraceStep(
                agent=str(step["agent"]),
                input_text=str(step["input_text"]),
                output_text=str(step["output_text"]),
                tool_calls=calls,
                evidence_ids=tuple(str(value) for value in step.get("evidence_ids", ())),
            )
        )

    created_at = payload.get("created_at")
    created = datetime.fromisoformat(str(created_at)) if created_at else datetime.now(UTC)
    return TeacherTrace(
        id=str(payload["id"]),
        tenant_id=str(payload["tenant_id"]),
        repo_id=str(payload["repo_id"]),
        prompt=str(payload["prompt"]),
        answer=str(payload["answer"]),
        steps=tuple(steps),
        faithfulness_score=float(payload["faithfulness_score"]),
        bfcl_category=str(payload["bfcl_category"]) if payload.get("bfcl_category") is not None else None,
        created_at=created,
    )


def load_teacher_traces(path: Path) -> list[TeacherTrace]:
    traces: list[TeacherTrace] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            traces.append(teacher_trace_from_dict(json.loads(line)))
    return traces


def write_jsonl_records(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = "\n".join(json.dumps(record, sort_keys=True) for record in records)
    path.write_text(f"{encoded}\n" if encoded else "", encoding="utf-8")


def read_jsonl_records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_sft_dataset_jsonl(
    *,
    traces_path: Path,
    out_path: Path,
    min_faithfulness: float = 0.78,
    require_tool_use: bool = True,
) -> list[dict[str, Any]]:
    traces = select_sft_traces(
        load_teacher_traces(traces_path),
        min_faithfulness=min_faithfulness,
        require_tool_use=require_tool_use,
    )
    records = [sft_record(trace) for trace in traces]
    write_jsonl_records(out_path, records)
    return records


def build_preference_dataset_jsonl(
    *,
    traces_path: Path,
    out_path: Path,
    rejected_answer: str | None = None,
    min_faithfulness: float = 0.78,
) -> list[dict[str, Any]]:
    traces = select_sft_traces(
        load_teacher_traces(traces_path),
        min_faithfulness=min_faithfulness,
        require_tool_use=True,
    )
    records = [
        build_preference_pair(trace=trace, rejected_answer=rejected_answer).to_dpo_record()
        for trace in traces
    ]
    write_jsonl_records(out_path, records)
    return records


def dataset_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _expect_list(value: Any, field_name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list")
    return value


def _expect_dict(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object")
    return value
