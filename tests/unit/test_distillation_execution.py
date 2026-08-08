import json
from pathlib import Path

from idrkd.distillation import (
    DistillationRuntimeConfig,
    adapter_artifacts_written,
    build_preference_dataset_jsonl,
    build_sft_dataset_jsonl,
    load_teacher_traces,
    teacher_trace_to_dict,
    train_dpo,
    train_sft,
)
from idrkd.distillation.io import write_jsonl_records


def test_teacher_trace_jsonl_round_trip_and_dataset_builders(tmp_path: Path) -> None:
    seed_path = tmp_path / "traces.jsonl"
    traces = load_teacher_traces(Path("eval/distillation/seed_teacher_traces.jsonl"))
    write_jsonl_records(seed_path, [teacher_trace_to_dict(trace) for trace in traces])

    sft_path = tmp_path / "sft.jsonl"
    dpo_path = tmp_path / "dpo.jsonl"

    sft_records = build_sft_dataset_jsonl(traces_path=seed_path, out_path=sft_path)
    dpo_records = build_preference_dataset_jsonl(traces_path=seed_path, out_path=dpo_path)

    assert len(sft_records) == 2
    assert sft_records[0]["messages"][0]["role"] == "system"
    assert sft_records[0]["metadata"]["tool_trace"]
    assert len(dpo_records) == 2
    assert dpo_records[0]["chosen"]
    assert dpo_records[0]["rejected"]


def test_sft_and_dpo_training_dry_run_writes_reproducible_summary(tmp_path: Path) -> None:
    traces_path = tmp_path / "traces.jsonl"
    traces = load_teacher_traces(Path("eval/distillation/seed_teacher_traces.jsonl"))
    write_jsonl_records(traces_path, [teacher_trace_to_dict(trace) for trace in traces])
    sft_path = tmp_path / "sft.jsonl"
    dpo_path = tmp_path / "dpo.jsonl"
    build_sft_dataset_jsonl(traces_path=traces_path, out_path=sft_path)
    build_preference_dataset_jsonl(traces_path=traces_path, out_path=dpo_path)

    sft_result = train_sft(
        DistillationRuntimeConfig(
            dataset_path=sft_path,
            output_dir=tmp_path / "sft-out",
            base_model_id="local/tiny-student",
            dry_run=True,
        )
    )
    dpo_result = train_dpo(
        DistillationRuntimeConfig(
            dataset_path=dpo_path,
            output_dir=tmp_path / "dpo-out",
            base_model_id="local/tiny-student",
            dry_run=True,
        )
    )

    sft_summary = json.loads((tmp_path / "sft-out" / "sft-run-summary.json").read_text())
    dpo_summary = json.loads((tmp_path / "dpo-out" / "dpo-run-summary.json").read_text())
    assert sft_result.stage == "sft"
    assert sft_summary["record_count"] == 2
    assert sft_summary["dry_run"] is True
    assert dpo_result.stage == "dpo"
    assert dpo_summary["metrics"]["beta"] == 0.1


def test_adapter_artifact_check_requires_peft_config_and_weights(tmp_path: Path) -> None:
    output_dir = tmp_path / "adapter"
    output_dir.mkdir()

    assert adapter_artifacts_written(output_dir) is False

    (output_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    (output_dir / "adapter_model.safetensors").write_bytes(b"tiny")

    assert adapter_artifacts_written(output_dir) is True
