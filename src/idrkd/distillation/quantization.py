"""AWQ quantisation metadata and model artifact manifests."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
import hashlib
import hmac
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any


@dataclass(frozen=True)
class AwqQuantizationConfig:
    bits: int = 4
    group_size: int = 128
    zero_point: bool = True
    backend: str = "awq"
    version: str = "GEMM"


@dataclass(frozen=True)
class ModelArtifactManifest:
    model_id: str
    adapter_path: str
    quantized_path: str
    base_model_id: str
    quantization: AwqQuantizationConfig = AwqQuantizationConfig()
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def payload(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "adapter_path": self.adapter_path,
            "quantized_path": self.quantized_path,
            "base_model_id": self.base_model_id,
            "quantization": asdict(self.quantization),
            "created_at": self.created_at.isoformat(),
        }

    def digest(self) -> str:
        encoded = json.dumps(self.payload(), sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def sign(self, secret: str) -> str:
        return hmac.new(secret.encode("utf-8"), self.digest().encode("utf-8"), hashlib.sha256).hexdigest()


@dataclass(frozen=True)
class AwqQuantizationJob:
    input_model_path: Path
    output_dir: Path
    model_id: str
    base_model_id: str
    adapter_path: Path | None = None
    calibration_path: Path | None = None
    quantization: AwqQuantizationConfig = AwqQuantizationConfig()
    local_files_only: bool = False
    trust_remote_code: bool = False
    max_calibration_samples: int = 128


def run_awq_quantization(job: AwqQuantizationJob) -> ModelArtifactManifest:
    """Quantize a merged model, or merge a PEFT adapter first, using AutoAWQ.

    AutoAWQ is intentionally imported lazily because it is a production GPU
    dependency and is usually installed only on Linux/CUDA builders.
    """

    modules = _load_quantization_modules()
    job.output_dir.mkdir(parents=True, exist_ok=True)

    if job.adapter_path is None:
        source_model = job.input_model_path
        return _quantize_merged_model(job=job, source_model=source_model, modules=modules)

    with TemporaryDirectory(prefix="idrkd-merged-model-") as tmpdir:
        merged_dir = Path(tmpdir) / "merged"
        _merge_adapter(job=job, output_dir=merged_dir, modules=modules)
        return _quantize_merged_model(job=job, source_model=merged_dir, modules=modules)


def _quantize_merged_model(
    *,
    job: AwqQuantizationJob,
    source_model: Path,
    modules: dict[str, Any],
) -> ModelArtifactManifest:
    tokenizer = modules["AutoTokenizer"].from_pretrained(
        str(source_model),
        local_files_only=job.local_files_only,
        trust_remote_code=job.trust_remote_code,
    )
    model = modules["AutoAWQForCausalLM"].from_pretrained(
        str(source_model),
        local_files_only=job.local_files_only,
        trust_remote_code=job.trust_remote_code,
        safetensors=True,
    )
    model.quantize(tokenizer, quant_config=_autoawq_config(job.quantization), calib_data=_calibration_data(job))
    model.save_quantized(str(job.output_dir), safetensors=True)
    tokenizer.save_pretrained(str(job.output_dir))
    manifest = ModelArtifactManifest(
        model_id=job.model_id,
        adapter_path=str(job.adapter_path or job.input_model_path),
        quantized_path=str(job.output_dir),
        base_model_id=job.base_model_id,
        quantization=job.quantization,
    )
    write_manifest(job.output_dir, manifest)
    return manifest


def _merge_adapter(
    *,
    job: AwqQuantizationJob,
    output_dir: Path,
    modules: dict[str, Any],
) -> None:
    tokenizer = modules["AutoTokenizer"].from_pretrained(
        job.base_model_id,
        local_files_only=job.local_files_only,
        trust_remote_code=job.trust_remote_code,
    )
    model = modules["AutoModelForCausalLM"].from_pretrained(
        job.base_model_id,
        local_files_only=job.local_files_only,
        trust_remote_code=job.trust_remote_code,
        device_map="auto",
    )
    peft_model = modules["PeftModel"].from_pretrained(model, str(job.adapter_path))
    merged = peft_model.merge_and_unload()
    output_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(output_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(output_dir))


def _autoawq_config(config: AwqQuantizationConfig) -> dict[str, object]:
    return {
        "zero_point": config.zero_point,
        "q_group_size": config.group_size,
        "w_bit": config.bits,
        "version": config.version,
    }


def _calibration_data(job: AwqQuantizationJob) -> list[str]:
    if job.calibration_path is None:
        return [
            "IDRKD reconciles repository evidence through MCP tools.",
            "Use grounded graph and vector context before answering.",
        ]
    samples = []
    for line in job.calibration_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        if isinstance(raw, str):
            samples.append(raw)
        elif isinstance(raw, dict):
            samples.append(str(raw.get("text") or raw.get("prompt") or raw.get("chosen") or raw))
        if len(samples) >= job.max_calibration_samples:
            break
    if not samples:
        raise ValueError(f"No calibration samples found in {job.calibration_path}")
    return samples


def write_manifest(output_dir: Path, manifest: ModelArtifactManifest) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "idrkd-model-manifest.json"
    payload = manifest.payload()
    payload["digest"] = manifest.digest()
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _load_quantization_modules() -> dict[str, Any]:
    try:
        from awq import AutoAWQForCausalLM  # type: ignore[import-not-found]
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - depends on CUDA production extras
        raise RuntimeError(
            "AutoAWQ quantization must run from an isolated legacy Linux/CUDA environment. "
            "Do not install AutoAWQ into the working IDRKD .venv; see the README Stage 12 "
            "AWQ runbook for the pinned torch/transformers/autoawq stack."
        ) from exc

    return {
        "AutoAWQForCausalLM": AutoAWQForCausalLM,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "PeftModel": PeftModel,
    }
