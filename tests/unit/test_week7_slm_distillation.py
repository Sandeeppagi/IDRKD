from datetime import UTC, datetime
from pathlib import Path

from idrkd.distillation import (
    AwqQuantizationConfig,
    AwqQuantizationJob,
    BfclMetrics,
    DistillationGate,
    ModelArtifactManifest,
    OllamaServingConfig,
    OpenAICompatibleStudentClient,
    QLoRAConfig,
    TeacherTrace,
    ToolCall,
    TraceStep,
    TrainingPlan,
    VllmServingConfig,
    build_preference_pair,
    select_sft_traces,
    sft_record,
    student_model_client_from_env,
    write_manifest,
)


def _trace(*, score: float = 0.9) -> TeacherTrace:
    return TeacherTrace(
        id="trace-1",
        tenant_id="tenant-a",
        repo_id="repo-a",
        prompt="Find the customer API entrypoint",
        answer="Customer API is implemented by entity-a.",
        faithfulness_score=score,
        bfcl_category="tool_calling",
        steps=(
            TraceStep(
                agent="router",
                input_text="Find the customer API entrypoint",
                output_text="use search_code",
                tool_calls=(ToolCall(name="search_code", arguments={"query": "customer API"}),),
            ),
            TraceStep(
                agent="synthesis",
                input_text="entity-a",
                output_text="Customer API is implemented by entity-a.",
                evidence_ids=("entity-a",),
            ),
        ),
        created_at=datetime(2026, 6, 26, tzinfo=UTC),
    )


def test_teacher_trace_selects_grounded_tool_using_records_for_sft() -> None:
    selected = select_sft_traces([_trace(score=0.9), _trace(score=0.2)])

    assert [trace.id for trace in selected] == ["trace-1"]
    assert selected[0].uses_tool("search_code") is True


def test_sft_record_preserves_messages_evidence_and_tool_trace() -> None:
    record = sft_record(_trace())

    assert record["messages"][0]["role"] == "system"
    assert record["messages"][1]["content"] == "Find the customer API entrypoint"
    assert record["messages"][2]["content"] == '{"arguments":{"query":"customer API"},"name":"search_code"}'
    assert record["metadata"]["evidence_ids"] == ["entity-a"]
    assert record["metadata"]["tool_trace"] == [{"agent": "router", "tool_calls": ["search_code"]}]
    assert record["metadata"]["teacher_answer"] == "Customer API is implemented by entity-a."
    assert record["metadata"]["target_tool_call"] == {
        "name": "search_code",
        "arguments": {"query": "customer API"},
    }


def test_preference_pair_prefers_structured_tool_call() -> None:
    pair = build_preference_pair(trace=_trace(), rejected_answer="I am not sure.")

    assert pair.chosen == '{"arguments":{"query":"customer API"},"name":"search_code"}'
    assert pair.rejected == (
        '{"arguments":{"_idrkd_wrong_argument":true,"query":"customer API"},"name":"search_code"}'
    )
    assert pair.to_dpo_record()["metadata"]["rejected_source"] == "sft_naive"
    assert pair.to_dpo_record()["metadata"]["rejected_answer_text"] == "I am not sure."


def test_qlora_and_training_plan_match_pillar_5_lld_defaults() -> None:
    config = QLoRAConfig()
    plan = TrainingPlan()

    assert config.base_model_id == "microsoft/Phi-4-mini-4k-instruct"
    assert config.quantization_kwargs()["bnb_4bit_quant_type"] == "nf4"
    assert config.peft_kwargs()["r"] == 64
    assert config.peft_kwargs()["lora_alpha"] == 128
    assert plan.stage_order() == (
        "teacher_trace_export",
        "qlora_sft",
        "bfcl_eval",
        "dpo_alignment",
        "awq_quantize",
        "vllm_serve",
    )


def test_distillation_gates_enforce_bfcl_alignscore_and_ttft_targets() -> None:
    gate = DistillationGate()
    metrics = BfclMetrics(true_positives=82, false_positives=9, false_negatives=9)

    assert round(metrics.f1, 2) == 0.90
    assert gate.check_first_pass(metrics) is True
    assert gate.check_release(bfcl_metrics=metrics, align_score=0.8, ttft_seconds=1.1) is True
    assert gate.check_release(bfcl_metrics=metrics, align_score=0.8, ttft_seconds=1.3) is False


def test_manifest_digest_signature_awq_and_vllm_command_are_deterministic() -> None:
    manifest = ModelArtifactManifest(
        model_id="idrkd-phi4-mini",
        adapter_path="models/adapters/phi4-mini-dpo",
        quantized_path="models/checkpoints/phi4-mini-awq",
        base_model_id="microsoft/Phi-4-mini-4k-instruct",
        quantization=AwqQuantizationConfig(bits=4),
        created_at=datetime(2026, 6, 27, tzinfo=UTC),
    )
    serving = VllmServingConfig(model_path=manifest.quantized_path, host="127.0.0.1", port=8080)

    assert len(manifest.digest()) == 64
    assert manifest.sign("secret") == manifest.sign("secret")
    assert manifest.payload()["quantization"] == {
        "bits": 4,
        "group_size": 128,
        "zero_point": True,
        "backend": "awq",
        "version": "GEMM",
    }
    assert serving.openai_base_url() == "http://127.0.0.1:8080/v1"
    assert serving.command()[:3] == ("vllm", "serve", "models/checkpoints/phi4-mini-awq")


def test_awq_manifest_write_and_job_contract(tmp_path: Path) -> None:
    manifest = ModelArtifactManifest(
        model_id="idrkd-student-awq",
        adapter_path="models/adapters/local-smoke-dpo",
        quantized_path=str(tmp_path / "awq"),
        base_model_id="Qwen/Qwen2.5-0.5B-Instruct",
        created_at=datetime(2026, 8, 9, tzinfo=UTC),
    )
    job = AwqQuantizationJob(
        input_model_path=Path("models/checkpoints/merged-student"),
        output_dir=tmp_path / "awq",
        model_id=manifest.model_id,
        base_model_id=manifest.base_model_id,
        adapter_path=Path(manifest.adapter_path),
    )

    manifest_path = write_manifest(job.output_dir, manifest)

    assert job.quantization.bits == 4
    assert manifest_path.name == "idrkd-model-manifest.json"
    assert manifest_path.is_file()
    assert manifest.digest() in manifest_path.read_text(encoding="utf-8")


def test_openai_compatible_student_serving_configs_from_env() -> None:
    vllm = VllmServingConfig(model_path="/models/checkpoints/idrkd-student-awq")
    ollama = OllamaServingConfig(model_name="idrkd-student")
    client = student_model_client_from_env(
        {
            "IDRKD_STUDENT_MODEL_BASE_URL": "http://slm-server:8000/v1",
            "IDRKD_STUDENT_MODEL_ID": "idrkd-student-awq",
            "IDRKD_STUDENT_MODEL_API_KEY": "secret",
        }
    )

    assert vllm.openai_base_url() == "http://0.0.0.0:8000/v1"
    assert ollama.openai_base_url() == "http://0.0.0.0:11434/v1"
    assert isinstance(client, OpenAICompatibleStudentClient)
    assert client.base_url == "http://slm-server:8000/v1"
