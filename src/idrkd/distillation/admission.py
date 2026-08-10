"""Teacher-trace admission logging for distillation datasets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

from idrkd.distillation.io import teacher_trace_to_dict, write_jsonl_records
from idrkd.distillation.traces import TeacherTrace
from idrkd.mcp.tools import TOOL_DEFINITIONS


@dataclass(frozen=True)
class TraceAdmissionPolicy:
    min_faithfulness: float = 0.78
    require_tool_use: bool = True
    generated_limit: int = 350
    admitted_limit: int = 320
    teacher_model_id: str = "frontier-teacher-staging-archive"
    validation_method: str = "mcp-schema-replay"


@dataclass(frozen=True)
class TraceAdmissionRecord:
    trace_id: str
    admitted: bool
    reasons: tuple[str, ...]
    replay_valid: bool
    teacher_model_id: str
    validation_method: str
    faithfulness_score: float
    bfcl_category: str | None
    tool_calls: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["reasons"] = list(self.reasons)
        payload["tool_calls"] = list(self.tool_calls)
        return payload


def admit_teacher_traces(
    traces: list[TeacherTrace],
    *,
    policy: TraceAdmissionPolicy = TraceAdmissionPolicy(),
) -> tuple[list[TeacherTrace], list[TraceAdmissionRecord]]:
    generated = traces[: policy.generated_limit]
    records: list[TraceAdmissionRecord] = []
    admitted: list[TeacherTrace] = []
    for trace in generated:
        reasons = _admission_reasons(trace, policy=policy)
        replay_valid = "replay_schema_invalid" not in reasons
        if not reasons and len(admitted) >= policy.admitted_limit:
            reasons = ("admission_cap_reached",)
        is_admitted = not reasons
        if is_admitted:
            admitted.append(trace)
        records.append(
            TraceAdmissionRecord(
                trace_id=trace.id,
                admitted=is_admitted,
                reasons=reasons,
                replay_valid=replay_valid,
                teacher_model_id=policy.teacher_model_id,
                validation_method=policy.validation_method,
                faithfulness_score=trace.faithfulness_score,
                bfcl_category=trace.bfcl_category,
                tool_calls=tuple(_trace_tool_names(trace)),
            )
        )
    return admitted, records


def write_admission_bundle(
    *,
    traces: list[TeacherTrace],
    output_dir: Path = Path("eval/distillation"),
    policy: TraceAdmissionPolicy = TraceAdmissionPolicy(),
) -> dict[str, Any]:
    admitted, records = admit_teacher_traces(traces, policy=policy)
    output_dir.mkdir(parents=True, exist_ok=True)
    admitted_path = output_dir / "frontier_admitted_teacher_traces.jsonl"
    log_path = output_dir / "frontier_trace_admission.jsonl"
    manifest_path = output_dir / "frontier_trace_admission_manifest.json"
    write_jsonl_records(admitted_path, [teacher_trace_to_dict(trace) for trace in admitted])
    write_jsonl_records(log_path, [record.as_dict() for record in records])
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "teacher_model_id": policy.teacher_model_id,
        "validation_method": policy.validation_method,
        "generated_count": len(records),
        "admitted_count": sum(1 for record in records if record.admitted),
        "rejected_count": sum(1 for record in records if not record.admitted),
        "replay_valid_count": sum(1 for record in records if record.replay_valid),
        "policy": asdict(policy),
        "admitted_traces_path": str(admitted_path),
        "admission_log_path": str(log_path),
        "rejection_reasons": _reason_counts(records),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _admission_reasons(trace: TeacherTrace, *, policy: TraceAdmissionPolicy) -> tuple[str, ...]:
    reasons: list[str] = []
    tool_calls = _trace_tool_names(trace)
    if trace.faithfulness_score < policy.min_faithfulness:
        reasons.append("faithfulness_below_threshold")
    if policy.require_tool_use and not tool_calls:
        reasons.append("no_tool_use")
    if not _trace_replay_valid(trace):
        reasons.append("replay_schema_invalid")
    return tuple(reasons)


def _trace_replay_valid(trace: TeacherTrace) -> bool:
    definitions = {definition.name: definition for definition in TOOL_DEFINITIONS}
    for step in trace.steps:
        for call in step.tool_calls:
            definition = definitions.get(call.name)
            if definition is None:
                return False
            try:
                definition.params_model.model_validate(call.arguments)
            except Exception:
                return False
    return True


def _trace_tool_names(trace: TeacherTrace) -> list[str]:
    return [call.name for step in trace.steps for call in step.tool_calls]


def _reason_counts(records: list[TraceAdmissionRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        for reason in record.reasons:
            counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items()))
