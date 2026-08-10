import json
from pathlib import Path

from idrkd.distillation import admit_teacher_traces, load_teacher_traces


def test_frontier_trace_admission_counts_and_replay_validation() -> None:
    traces = load_teacher_traces(Path("eval/distillation/seed_teacher_traces.jsonl"))

    admitted, records = admit_teacher_traces(traces)

    assert len(records) == 350
    assert len(admitted) == 320
    assert sum(1 for record in records if not record.admitted) == 30
    assert all(record.replay_valid for record in records)
    assert {reason for record in records for reason in record.reasons} == {"admission_cap_reached"}


def test_frontier_trace_admission_manifest_matches_artifacts() -> None:
    manifest = json.loads(Path("eval/distillation/frontier_trace_admission_manifest.json").read_text(encoding="utf-8"))
    admitted_lines = Path(manifest["admitted_traces_path"]).read_text(encoding="utf-8").splitlines()
    log_lines = Path(manifest["admission_log_path"]).read_text(encoding="utf-8").splitlines()

    assert manifest["teacher_model_id"] == "frontier-teacher-staging-archive"
    assert manifest["validation_method"] == "mcp-schema-replay"
    assert manifest["generated_count"] == 350
    assert manifest["admitted_count"] == 320
    assert manifest["rejected_count"] == 30
    assert manifest["replay_valid_count"] == 350
    assert manifest["rejection_reasons"] == {"admission_cap_reached": 30}
    assert len(admitted_lines) == 320
    assert len(log_lines) == 350
