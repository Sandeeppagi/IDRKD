"""Pillar 5 student model distillation contracts."""

from idrkd.distillation.admission import (
    TraceAdmissionPolicy,
    TraceAdmissionRecord,
    admit_teacher_traces,
    write_admission_bundle,
)
from idrkd.distillation.evaluation import BfclMetrics, DistillationGate
from idrkd.distillation.execution import (
    DistillationRuntimeConfig,
    DistillationRunResult,
    DistillationSmokeResult,
    adapter_artifacts_written,
    run_laptop_smoke_distillation,
    train_dpo,
    train_sft,
)
from idrkd.distillation.io import (
    build_preference_dataset_jsonl,
    build_sft_dataset_jsonl,
    load_teacher_traces,
    teacher_trace_from_dict,
    teacher_trace_to_dict,
)
from idrkd.distillation.preferences import PreferencePair, build_preference_pair
from idrkd.distillation.quantization import (
    AwqQuantizationConfig,
    AwqQuantizationJob,
    ModelArtifactManifest,
    run_awq_quantization,
    write_manifest,
)
from idrkd.distillation.serving import (
    OllamaServingConfig,
    OpenAICompatibleStudentClient,
    StudentModelClient,
    VllmServingConfig,
    student_model_client_from_env,
)
from idrkd.distillation.traces import TeacherTrace, TraceStep, ToolCall, sft_record, select_sft_traces
from idrkd.distillation.training import DpoConfig, QLoRAConfig, TrainingPlan

__all__ = [
    "AwqQuantizationConfig",
    "AwqQuantizationJob",
    "BfclMetrics",
    "DistillationGate",
    "DistillationRuntimeConfig",
    "DistillationRunResult",
    "DistillationSmokeResult",
    "DpoConfig",
    "ModelArtifactManifest",
    "OllamaServingConfig",
    "OpenAICompatibleStudentClient",
    "PreferencePair",
    "QLoRAConfig",
    "StudentModelClient",
    "TeacherTrace",
    "TraceAdmissionPolicy",
    "TraceAdmissionRecord",
    "ToolCall",
    "TraceStep",
    "TrainingPlan",
    "VllmServingConfig",
    "admit_teacher_traces",
    "adapter_artifacts_written",
    "build_preference_dataset_jsonl",
    "build_preference_pair",
    "build_sft_dataset_jsonl",
    "load_teacher_traces",
    "run_laptop_smoke_distillation",
    "run_awq_quantization",
    "select_sft_traces",
    "sft_record",
    "student_model_client_from_env",
    "teacher_trace_from_dict",
    "teacher_trace_to_dict",
    "train_dpo",
    "train_sft",
    "write_manifest",
    "write_admission_bundle",
]
