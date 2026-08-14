"""LLM Compressor AWQ quantization and model artifact manifests."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
import hashlib
import hmac
import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory
from typing import Any


_CALIBRATION_SYSTEM_PROMPT = (
    "You are the IDRKD student model. Select exactly one MCP tool for the user task. "
    'Return only JSON with keys: "name" and "arguments".'
)


@dataclass(frozen=True)
class AwqQuantizationConfig:
    bits: int = 4
    group_size: int = 128
    zero_point: bool = True
    backend: str = "llm-compressor"
    algorithm: str = "awq"
    scheme: str = "W4A16_ASYM"
    format: str = "compressed-tensors"

    def __post_init__(self) -> None:
        if (self.bits, self.group_size, self.zero_point) != (4, 128, True):
            raise ValueError(
                "IDRKD llm-compressor AWQ currently supports only asymmetric "
                "W4A16 with 4-bit weights and group size 128"
            )
        if self.scheme != "W4A16_ASYM":
            raise ValueError("IDRKD llm-compressor AWQ requires scheme W4A16_ASYM")


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
        return hmac.new(
            secret.encode("utf-8"),
            self.digest().encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()


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
    max_calibration_samples: int = 64
    max_sequence_length: int = 3072
    calibration_seed: int = 17


def run_awq_quantization(job: AwqQuantizationJob) -> ModelArtifactManifest:
    """Merge an optional PEFT adapter and apply llm-compressor AWQ."""

    _validate_job(job)
    modules = _load_quantization_modules()
    if job.adapter_path is None:
        return _quantize_merged_model(
            job=job,
            source_model=job.input_model_path,
            modules=modules,
        )

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
    model = modules["AutoModelForCausalLM"].from_pretrained(
        str(source_model),
        dtype="auto",
        device_map="auto",
        local_files_only=job.local_files_only,
        trust_remote_code=job.trust_remote_code,
    )
    samples = _calibration_data(job, tokenizer)
    dataset = modules["Dataset"].from_dict({"text": samples})
    recipe = [
        modules["AWQModifier"](),
        modules["QuantizationModifier"](
            targets=["Linear"],
            scheme=job.quantization.scheme,
            ignore=["lm_head"],
        ),
    ]
    modules["oneshot"](
        model=model,
        dataset=dataset,
        recipe=recipe,
        max_seq_length=job.max_sequence_length,
        num_calibration_samples=len(samples),
    )

    job.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(
        str(job.output_dir),
        save_compressed=True,
        safe_serialization=True,
    )
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
        dtype="auto",
        device_map="auto",
        local_files_only=job.local_files_only,
        trust_remote_code=job.trust_remote_code,
    )
    peft_model = modules["PeftModel"].from_pretrained(model, str(job.adapter_path))
    merged = peft_model.merge_and_unload()
    output_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(output_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(output_dir))


def _calibration_data(job: AwqQuantizationJob, tokenizer: Any) -> list[str]:
    if job.calibration_path is None:
        raise ValueError("llm-compressor AWQ requires a representative calibration JSONL file")

    samples: list[str] = []
    for line in job.calibration_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        sample = _calibration_sample(json.loads(line), tokenizer)
        if sample:
            samples.append(sample)
    if not samples:
        raise ValueError(f"No calibration samples found in {job.calibration_path}")
    if len(samples) <= job.max_calibration_samples:
        return samples
    return random.Random(job.calibration_seed).sample(samples, job.max_calibration_samples)


def _calibration_sample(raw: Any, tokenizer: Any) -> str:
    if isinstance(raw, str):
        return raw
    if not isinstance(raw, dict):
        return ""

    messages = raw.get("messages")
    if isinstance(messages, list) and messages:
        return _render_messages(tokenizer, messages)

    text = raw.get("text")
    if isinstance(text, str) and text.strip():
        return text

    prompt = raw.get("prompt")
    completion = raw.get("chosen") or _first_trace_tool_call(raw)
    if isinstance(prompt, str) and prompt.strip() and completion:
        return _render_messages(
            tokenizer,
            [
                {"role": "system", "content": _CALIBRATION_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": str(completion)},
            ],
        )
    return str(prompt or completion or "")


def _first_trace_tool_call(raw: dict[str, Any]) -> str:
    steps = raw.get("steps")
    if not isinstance(steps, list):
        return ""
    for step in steps:
        if not isinstance(step, dict):
            continue
        calls = step.get("tool_calls")
        if isinstance(calls, list) and calls and isinstance(calls[0], dict):
            return json.dumps(calls[0], separators=(",", ":"), sort_keys=True)
    return ""


def _render_messages(tokenizer: Any, messages: list[Any]) -> str:
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if apply_chat_template is None:
        return "\n".join(
            str(message.get("content", ""))
            for message in messages
            if isinstance(message, dict)
        )
    return str(
        apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    )


def _validate_job(job: AwqQuantizationJob) -> None:
    if job.max_calibration_samples <= 0:
        raise ValueError("max_calibration_samples must be positive")
    if job.max_sequence_length <= 0:
        raise ValueError("max_sequence_length must be positive")
    if job.adapter_path is None and not job.input_model_path.exists():
        raise ValueError(f"Merged input model does not exist: {job.input_model_path}")
    if job.adapter_path is not None and not job.adapter_path.exists():
        raise ValueError(f"PEFT adapter does not exist: {job.adapter_path}")
    if job.calibration_path is None or not job.calibration_path.is_file():
        raise ValueError(f"Calibration JSONL does not exist: {job.calibration_path}")


def write_manifest(output_dir: Path, manifest: ModelArtifactManifest) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "idrkd-model-manifest.json"
    payload = manifest.payload()
    payload["digest"] = manifest.digest()
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _load_quantization_modules() -> dict[str, Any]:
    try:
        from datasets import Dataset
        from llmcompressor import oneshot  # type: ignore[import-not-found]
        from llmcompressor.modifiers.quantization import (  # type: ignore[import-not-found]
            QuantizationModifier,
        )
        from llmcompressor.modifiers.transform.awq import (  # type: ignore[import-not-found]
            AWQModifier,
        )
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - production CUDA dependency
        raise RuntimeError(
            "llm-compressor AWQ must run from the isolated quantization environment. "
            "Install llmcompressor, transformers, peft, accelerate, and datasets; "
            "see the README Stage 12 runbook."
        ) from exc

    return {
        "AWQModifier": AWQModifier,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "Dataset": Dataset,
        "PeftModel": PeftModel,
        "QuantizationModifier": QuantizationModifier,
        "oneshot": oneshot,
    }
