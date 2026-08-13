import json
from pathlib import Path

from idrkd.evaluation import MeasurementJob, build_measurement_bundle


def test_measurement_bundle_writes_three_seed_mode_outputs(tmp_path: Path) -> None:
    manifest = build_measurement_bundle(MeasurementJob(output_dir=tmp_path))

    assert manifest["seeds"] == [11, 23, 37]
    assert manifest["task_count_per_run"] == 440
    assert manifest["split"] == "all"
    assert len(manifest["runs"]) == 9
    assert set(manifest["aggregate"]) == {"registry-smoke", "student-agent", "teacher-agent"}

    for run in manifest["runs"]:
        summary = json.loads(Path(run["summary_path"]).read_text(encoding="utf-8"))
        promotion = json.loads(Path(run["promotion_decision_path"]).read_text(encoding="utf-8"))
        raw_outputs = json.loads(Path(run["raw_model_outputs_path"]).read_text(encoding="utf-8"))

        assert summary["seed"] in {11, 23, 37}
        assert summary["latency"]["p95_seconds"] >= 0.0
        assert summary["memory"]["peak_traced_mb"] >= 0.0
        assert promotion["promoted"] is False
        assert promotion["eligible_for_empirical_claims"] is False
        assert run["eligible_for_empirical_claims"] is False
        if run["mode"] == "registry-smoke":
            assert raw_outputs == []
        else:
            assert len(raw_outputs) == 440
            assert raw_outputs[0]["raw_model_output"]

    assert (tmp_path / "manifest.json").is_file()
    assert manifest["eligible_for_empirical_claims"] is False
