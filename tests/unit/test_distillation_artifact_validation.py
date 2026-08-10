import json
from pathlib import Path
import tarfile

import pytest

from idrkd.distillation.artifact_validation import (
    validate_distilled_adapter_artifact,
    validate_extracted_adapter_artifacts,
)


def test_validate_distilled_adapter_artifact_checks_sft_dpo_chain(tmp_path: Path) -> None:
    artifact_root = _write_artifact_tree(tmp_path / "artifact")
    archive_path = tmp_path / "phi4-mini-distilled-adapters.tar.gz"
    _tar_tree(archive_path, artifact_root)

    result = validate_distilled_adapter_artifact(archive_path, extract_dir=tmp_path / "extracted")

    assert result.sft.stage == "sft"
    assert result.dpo.stage == "dpo"
    assert result.sft.record_count == 500
    assert result.dpo.record_count == 500
    assert result.dpo_chained_from_sft is True
    assert result.generation_ran is False
    assert (tmp_path / "extracted" / "models" / "adapters" / "phi4-mini-dpo").is_dir()


def test_validate_extracted_adapter_artifacts_rejects_dpo_without_sft_source(tmp_path: Path) -> None:
    artifact_root = _write_artifact_tree(tmp_path / "artifact")
    dpo_summary_path = artifact_root / "models" / "adapters" / "phi4-mini-dpo" / "dpo-run-summary.json"
    dpo_summary = json.loads(dpo_summary_path.read_text(encoding="utf-8"))
    dpo_summary["sft_adapter_path"] = None
    dpo_summary_path.write_text(json.dumps(dpo_summary), encoding="utf-8")

    with pytest.raises(ValueError, match="SFT adapter path"):
        validate_extracted_adapter_artifacts(artifact_root)


def _write_artifact_tree(root: Path) -> Path:
    sft_dir = root / "models" / "adapters" / "phi4-mini-sft"
    dpo_dir = root / "models" / "adapters" / "phi4-mini-dpo"
    ref_dir = dpo_dir / "ref"
    for adapter_dir in (sft_dir, dpo_dir, ref_dir):
        adapter_dir.mkdir(parents=True)
        (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
        (adapter_dir / "adapter_model.safetensors").write_bytes(b"tiny")
    for adapter_dir in (sft_dir, dpo_dir):
        (adapter_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
        (adapter_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        (adapter_dir / "README.md").write_text("tiny adapter", encoding="utf-8")
    (sft_dir / "sft-run-summary.json").write_text(
        json.dumps(
            {
                "base_model_id": "microsoft/Phi-4-mini-instruct",
                "dry_run": False,
                "metrics": {"train_loss": 0.5},
                "record_count": 500,
                "sft_adapter_path": None,
                "stage": "sft",
            }
        ),
        encoding="utf-8",
    )
    (dpo_dir / "dpo-run-summary.json").write_text(
        json.dumps(
            {
                "base_model_id": "microsoft/Phi-4-mini-instruct",
                "dry_run": False,
                "metrics": {"beta": 0.1, "train_loss": 0.01},
                "record_count": 500,
                "sft_adapter_path": "/workspace/IDRKD/models/adapters/phi4-mini-sft",
                "stage": "dpo",
            }
        ),
        encoding="utf-8",
    )
    return root


def _tar_tree(archive_path: Path, root: Path) -> None:
    with tarfile.open(archive_path, mode="w:gz") as archive:
        for path in sorted(root.rglob("*")):
            archive.add(path, arcname=path.relative_to(root))
