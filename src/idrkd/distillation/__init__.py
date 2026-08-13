"""Pillar 5 student model distillation contracts.

Exports are loaded lazily so lightweight entry points such as AWQ quantization
do not require the training, serving, or MCP application dependency stacks.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORT_MODULES = {
    "AdapterStageValidation": "artifact_validation",
    "AwqQuantizationConfig": "quantization",
    "AwqQuantizationJob": "quantization",
    "BfclMetrics": "evaluation",
    "DistilledArtifactValidation": "artifact_validation",
    "DistillationGate": "evaluation",
    "DistillationRunResult": "execution",
    "DistillationRuntimeConfig": "execution",
    "DistillationSmokeResult": "execution",
    "DpoConfig": "training",
    "ModelArtifactManifest": "quantization",
    "OllamaServingConfig": "serving",
    "OpenAICompatibleStudentClient": "serving",
    "PreferencePair": "preferences",
    "QLoRAConfig": "training",
    "StudentModelClient": "serving",
    "TeacherTrace": "traces",
    "ToolCall": "traces",
    "TraceAdmissionPolicy": "admission",
    "TraceAdmissionRecord": "admission",
    "TraceStep": "traces",
    "TrainingPlan": "training",
    "VllmServingConfig": "serving",
    "adapter_artifacts_written": "execution",
    "admit_teacher_traces": "admission",
    "build_preference_dataset_jsonl": "io",
    "build_preference_pair": "preferences",
    "build_sft_dataset_jsonl": "io",
    "build_taskbench_preference_dataset_jsonl": "io",
    "build_taskbench_sft_dataset_jsonl": "io",
    "load_teacher_traces": "io",
    "run_awq_quantization": "quantization",
    "run_laptop_smoke_distillation": "execution",
    "select_sft_traces": "traces",
    "sft_record": "traces",
    "student_model_client_from_env": "serving",
    "teacher_trace_from_dict": "io",
    "teacher_trace_to_dict": "io",
    "train_dpo": "execution",
    "train_sft": "execution",
    "validate_distilled_adapter_artifact": "artifact_validation",
    "validate_extracted_adapter_artifacts": "artifact_validation",
    "write_admission_bundle": "admission",
    "write_manifest": "quantization",
}

__all__ = sorted(_EXPORT_MODULES)


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted([*globals(), *__all__])
