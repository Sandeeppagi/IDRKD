"""Validation helpers for distilled PEFT adapter artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib
import json
from pathlib import Path
import tarfile
import tempfile
from typing import Any

from idrkd.distillation.execution import adapter_artifacts_written


REQUIRED_ADAPTER_FILES = (
    "adapter_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


@dataclass(frozen=True)
class AdapterStageValidation:
    stage: str
    path: str
    base_model_id: str
    dry_run: bool
    record_count: int
    adapter_written: bool
    summary_path: str
    metrics: dict[str, float]
    generation_text: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DistilledArtifactValidation:
    archive_path: str
    extracted_dir: str
    sft: AdapterStageValidation
    dpo: AdapterStageValidation
    dpo_chained_from_sft: bool
    generation_ran: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "archive_path": self.archive_path,
            "extracted_dir": self.extracted_dir,
            "sft": self.sft.as_dict(),
            "dpo": self.dpo.as_dict(),
            "dpo_chained_from_sft": self.dpo_chained_from_sft,
            "generation_ran": self.generation_ran,
        }


def validate_distilled_adapter_artifact(
    archive_path: Path,
    *,
    extract_dir: Path | None = None,
    run_generation: bool = False,
    local_files_only: bool = True,
    prompt: str = "Select the best MCP tool for repository search.",
    max_new_tokens: int = 16,
) -> DistilledArtifactValidation:
    """Extract and validate the committed Phi adapter artifact.

    The default validation is intentionally lightweight: it verifies tar safety,
    PEFT adapter files, tokenizer files, and SFT/DPO run summaries. Set
    ``run_generation=True`` to additionally load the DPO PEFT adapter on top of
    its base model and generate a tiny response.
    """

    if not archive_path.is_file():
        raise FileNotFoundError(f"Adapter artifact archive does not exist: {archive_path}")

    owned_temp: tempfile.TemporaryDirectory[str] | None = None
    if extract_dir is None:
        owned_temp = tempfile.TemporaryDirectory(prefix="idrkd-adapter-artifact-")
        target_dir = Path(owned_temp.name)
    else:
        target_dir = extract_dir
        target_dir.mkdir(parents=True, exist_ok=True)

    try:
        _safe_extract_tar_gz(archive_path, target_dir)
        models_dir = target_dir / "models" / "adapters"
        sft_dir = models_dir / "phi4-mini-sft"
        dpo_dir = models_dir / "phi4-mini-dpo"
        sft = _validate_stage(stage="sft", adapter_dir=sft_dir)
        dpo = _validate_stage(stage="dpo", adapter_dir=dpo_dir)
        if run_generation:
            dpo_generation = _run_generation_smoke(
                adapter_dir=dpo_dir,
                base_model_id=dpo.base_model_id,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                local_files_only=local_files_only,
            )
            dpo = AdapterStageValidation(
                **{
                    **dpo.as_dict(),
                    "generation_text": dpo_generation,
                }
            )

        dpo_summary = _read_json(dpo_dir / "dpo-run-summary.json")
        result = DistilledArtifactValidation(
            archive_path=str(archive_path),
            extracted_dir=str(target_dir),
            sft=sft,
            dpo=dpo,
            dpo_chained_from_sft=_dpo_summary_chains_sft(dpo_summary),
            generation_ran=run_generation,
        )
        if not result.dpo_chained_from_sft:
            raise ValueError("DPO summary does not record an SFT adapter path")
        return result
    finally:
        if owned_temp is not None:
            owned_temp.cleanup()


def validate_extracted_adapter_artifacts(
    root_dir: Path,
    *,
    run_generation: bool = False,
    local_files_only: bool = True,
    prompt: str = "Select the best MCP tool for repository search.",
    max_new_tokens: int = 16,
) -> DistilledArtifactValidation:
    """Validate an already extracted ``models/adapters`` artifact tree."""

    archive_label = str(root_dir)
    models_dir = root_dir / "models" / "adapters" if (root_dir / "models").exists() else root_dir / "adapters"
    sft_dir = models_dir / "phi4-mini-sft"
    dpo_dir = models_dir / "phi4-mini-dpo"
    sft = _validate_stage(stage="sft", adapter_dir=sft_dir)
    dpo = _validate_stage(stage="dpo", adapter_dir=dpo_dir)
    if run_generation:
        generation_text = _run_generation_smoke(
            adapter_dir=dpo_dir,
            base_model_id=dpo.base_model_id,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            local_files_only=local_files_only,
        )
        dpo = AdapterStageValidation(**{**dpo.as_dict(), "generation_text": generation_text})

    dpo_summary = _read_json(dpo_dir / "dpo-run-summary.json")
    result = DistilledArtifactValidation(
        archive_path=archive_label,
        extracted_dir=str(root_dir),
        sft=sft,
        dpo=dpo,
        dpo_chained_from_sft=_dpo_summary_chains_sft(dpo_summary),
        generation_ran=run_generation,
    )
    if not result.dpo_chained_from_sft:
        raise ValueError("DPO summary does not record an SFT adapter path")
    return result


def _validate_stage(*, stage: str, adapter_dir: Path) -> AdapterStageValidation:
    if not adapter_dir.is_dir():
        raise FileNotFoundError(f"{stage.upper()} adapter directory is missing: {adapter_dir}")
    for relative_path in REQUIRED_ADAPTER_FILES:
        candidate = adapter_dir / relative_path
        if not candidate.is_file():
            raise FileNotFoundError(f"{stage.upper()} adapter file is missing: {candidate}")
    if not adapter_artifacts_written(adapter_dir):
        raise FileNotFoundError(f"{stage.upper()} adapter weights are missing in {adapter_dir}")

    summary_path = adapter_dir / f"{stage}-run-summary.json"
    summary = _read_json(summary_path)
    if summary.get("stage") != stage:
        raise ValueError(f"{summary_path} has unexpected stage {summary.get('stage')!r}")
    if summary.get("dry_run") is not False:
        raise ValueError(f"{summary_path} must record dry_run=false")
    record_count = summary.get("record_count")
    if not isinstance(record_count, int) or record_count <= 0:
        raise ValueError(f"{summary_path} must record a positive record_count")
    base_model_id = summary.get("base_model_id")
    if not isinstance(base_model_id, str) or not base_model_id:
        raise ValueError(f"{summary_path} must record base_model_id")
    metrics = summary.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"{summary_path} must record metrics")

    return AdapterStageValidation(
        stage=stage,
        path=str(adapter_dir),
        base_model_id=base_model_id,
        dry_run=False,
        record_count=record_count,
        adapter_written=True,
        summary_path=str(summary_path),
        metrics={str(key): float(value) for key, value in metrics.items()},
    )


def _safe_extract_tar_gz(archive_path: Path, target_dir: Path) -> None:
    target_root = target_dir.resolve()
    with tarfile.open(archive_path, mode="r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            member_path = (target_dir / member.name).resolve()
            if target_root != member_path and target_root not in member_path.parents:
                raise ValueError(f"Refusing to extract unsafe tar member: {member.name}")
        archive.extractall(target_dir, members=members)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON file is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _dpo_summary_chains_sft(summary: dict[str, Any]) -> bool:
    sft_adapter_path = summary.get("sft_adapter_path")
    return isinstance(sft_adapter_path, str) and "phi4-mini-sft" in sft_adapter_path


def _run_generation_smoke(
    *,
    adapter_dir: Path,
    base_model_id: str,
    prompt: str,
    max_new_tokens: int,
    local_files_only: bool,
) -> str:
    try:
        transformers = importlib.import_module("transformers")
        peft = importlib.import_module("peft")
        torch = importlib.import_module("torch")
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError("Install ML dependencies with `uv sync --group dev --extra ml`.") from exc

    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            str(adapter_dir),
            local_files_only=local_files_only,
        )
        model = transformers.AutoModelForCausalLM.from_pretrained(
            base_model_id,
            local_files_only=local_files_only,
            device_map="auto",
        )
        model = peft.PeftModel.from_pretrained(
            model,
            str(adapter_dir),
            local_files_only=local_files_only,
            is_trainable=False,
        )
        model.eval()
        inputs = tokenizer(prompt, return_tensors="pt")
        model_device = next(model.parameters()).device
        inputs = {key: value.to(model_device) for key, value in inputs.items()}
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    except OSError as exc:
        cache_hint = "cached locally" if local_files_only else "downloadable or cached"
        raise RuntimeError(
            f"Generation smoke could not load base model {base_model_id!r}; "
            f"make sure it is {cache_hint} before using --run-generation."
        ) from exc
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    if not text.strip():
        raise RuntimeError("Generation smoke produced empty text")
    return str(text)
