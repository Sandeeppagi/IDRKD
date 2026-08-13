import json
from pathlib import Path

import idrkd.distillation.execution as execution
from idrkd.distillation import (
    DistillationRuntimeConfig,
    adapter_artifacts_written,
    build_preference_dataset_jsonl,
    build_sft_dataset_jsonl,
    build_taskbench_preference_dataset_jsonl,
    build_taskbench_sft_dataset_jsonl,
    load_teacher_traces,
    teacher_trace_to_dict,
    train_dpo,
    train_sft,
)
from idrkd.distillation.execution import render_sft_text
from idrkd.distillation.io import write_jsonl_records


def test_render_sft_text_uses_tokenizer_chat_template() -> None:
    record = {
        "messages": [
            {"role": "system", "content": "Return JSON."},
            {"role": "user", "content": "Retrieve customer record 42"},
            {"role": "assistant", "content": '{"name":"get_customer","arguments":{"customer_id":42}}'},
        ]
    }
    calls: dict[str, object] = {}

    class FakeTokenizer:
        def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: object) -> str:
            calls["messages"] = messages
            calls["kwargs"] = kwargs
            return "<chat-template-output>"

    assert render_sft_text(record, FakeTokenizer()) == "<chat-template-output>"
    assert calls["messages"] == record["messages"]
    assert calls["kwargs"] == {"tokenize": False, "add_generation_prompt": False}


def test_render_sft_text_fallback_terminates_each_message() -> None:
    record = {
        "messages": [
            {"role": "system", "content": "Return JSON."},
            {"role": "user", "content": "Retrieve customer record 42"},
            {"role": "assistant", "content": '{"name":"get_customer","arguments":{"customer_id":42}}'},
        ]
    }

    assert render_sft_text(record) == (
        '<|system|>Return JSON.<|end|>\n'
        '<|user|>Retrieve customer record 42<|end|>\n'
        '<|assistant|>{"name":"get_customer","arguments":{"customer_id":42}}<|end|>'
    )


def test_teacher_trace_jsonl_round_trip_and_dataset_builders(tmp_path: Path) -> None:
    seed_path = tmp_path / "traces.jsonl"
    traces = load_teacher_traces(Path("eval/distillation/seed_teacher_traces.jsonl"))
    write_jsonl_records(seed_path, [teacher_trace_to_dict(trace) for trace in traces])

    sft_path = tmp_path / "sft.jsonl"
    dpo_path = tmp_path / "dpo.jsonl"

    sft_records = build_sft_dataset_jsonl(traces_path=seed_path, out_path=sft_path)
    dpo_records = build_preference_dataset_jsonl(traces_path=seed_path, out_path=dpo_path)

    assert len(sft_records) >= 500
    assert sft_records[0]["messages"][0]["role"] == "system"
    assert json.loads(sft_records[0]["messages"][2]["content"]) == {
        "name": "search_code",
        "arguments": {
            "limit": 6,
            "query": "RAG orchestrator implementation details",
            "repo_id": "repo-seed",
            "tenant_id": "tenant-seed",
        },
    }
    assert sft_records[0]["metadata"]["tool_trace"]
    assert len(dpo_records) >= 500
    assert json.loads(dpo_records[0]["chosen"])["name"] == "search_code"
    assert json.loads(dpo_records[0]["rejected"])["arguments"]["_idrkd_wrong_argument"] is True


def test_taskbench_dataset_builders_match_evaluation_prompt_and_expected_call(tmp_path: Path) -> None:
    sft_path = tmp_path / "taskbench-sft.jsonl"
    dpo_path = tmp_path / "taskbench-dpo.jsonl"

    sft_records = build_taskbench_sft_dataset_jsonl(
        tasks_path=Path("eval/taskbench/seed_tasks.jsonl"),
        out_path=sft_path,
    )
    dpo_records = build_taskbench_preference_dataset_jsonl(
        tasks_path=Path("eval/taskbench/seed_tasks.jsonl"),
        out_path=dpo_path,
    )

    assert len(sft_records) == 360
    assert "Available MCP tools as JSON schemas" in sft_records[0]["messages"][1]["content"]
    assert json.loads(sft_records[0]["messages"][2]["content"]) == {
        "name": "search_code",
        "arguments": {
            "limit": 3,
            "query": "customer lookup",
            "repo_id": "repo-a",
            "tenant_id": "default",
        },
    }
    assert json.loads(dpo_records[0]["chosen"]) == json.loads(sft_records[0]["messages"][2]["content"])
    assert json.loads(dpo_records[0]["rejected"])["arguments"]["_idrkd_wrong_argument"] is True


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
    assert sft_summary["record_count"] >= 500
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


def test_dpo_loads_trainable_sft_adapter_when_provided(tmp_path: Path, monkeypatch) -> None:
    dpo_path = tmp_path / "dpo.jsonl"
    write_jsonl_records(dpo_path, [{"prompt": "Where is reconcile defined?", "chosen": "Use search_code.", "rejected": "Unknown."}])

    sft_adapter = tmp_path / "sft-adapter"
    sft_adapter.mkdir()
    (sft_adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (sft_adapter / "adapter_model.safetensors").write_bytes(b"tiny")

    calls: dict[str, object] = {"get_peft_model_called": False}

    class FakeTokenizer:
        pad_token = None
        eos_token = "<eos>"

        def save_pretrained(self, output_dir: str) -> None:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(base_model_id: str, **kwargs: object) -> FakeTokenizer:
            calls["tokenizer_base_model_id"] = base_model_id
            calls["tokenizer_kwargs"] = kwargs
            return FakeTokenizer()

    class FakeAutoModelForCausalLM:
        @staticmethod
        def from_pretrained(base_model_id: str, **kwargs: object) -> str:
            calls["model_base_model_id"] = base_model_id
            calls["model_kwargs"] = kwargs
            return "base-model"

    class FakePeftModel:
        @staticmethod
        def from_pretrained(model: object, adapter_path: str, *, is_trainable: bool) -> dict[str, object]:
            calls["peft_model"] = model
            calls["adapter_path"] = adapter_path
            calls["is_trainable"] = is_trainable
            return {"model": model, "adapter_path": adapter_path}

    class FakeDataset:
        @staticmethod
        def from_list(records: list[dict[str, object]]) -> list[dict[str, object]]:
            calls["records"] = records
            return records

    class FakeDPOConfig:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    class FakeDPOTrainer:
        def __init__(self, **kwargs: object) -> None:
            calls["trainer_kwargs"] = kwargs

        def train(self) -> object:
            return type("TrainOutput", (), {"training_loss": 0.25})()

        def save_model(self, output_dir: str) -> None:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            (output_path / "adapter_config.json").write_text("{}", encoding="utf-8")
            (output_path / "adapter_model.safetensors").write_bytes(b"tiny")

    def fail_get_peft_model(*args: object, **kwargs: object) -> None:
        calls["get_peft_model_called"] = True
        raise AssertionError("DPO should load the SFT adapter instead of creating a fresh LoRA adapter")

    monkeypatch.setattr(
        execution,
        "_load_ml_modules",
        lambda: {
            "AutoModelForCausalLM": FakeAutoModelForCausalLM,
            "AutoTokenizer": FakeAutoTokenizer,
            "BitsAndBytesConfig": lambda **kwargs: kwargs,
            "DataCollatorForLanguageModeling": object,
            "Trainer": object,
            "TrainingArguments": object,
            "Dataset": FakeDataset,
            "LoraConfig": lambda **kwargs: kwargs,
            "PeftModel": FakePeftModel,
            "get_peft_model": fail_get_peft_model,
            "prepare_model_for_kbit_training": lambda model: model,
            "DPOConfig": FakeDPOConfig,
            "DPOTrainer": FakeDPOTrainer,
        },
    )

    result = train_dpo(
        DistillationRuntimeConfig(
            dataset_path=dpo_path,
            output_dir=tmp_path / "dpo-out",
            base_model_id="local/tiny-student",
            sft_adapter_path=sft_adapter,
        )
    )

    assert calls["adapter_path"] == str(sft_adapter)
    assert calls["is_trainable"] is True
    assert calls["get_peft_model_called"] is False
    assert result.stage == "dpo"
    assert result.sft_adapter_path == str(sft_adapter)
