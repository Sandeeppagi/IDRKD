import json
from pathlib import Path
from typing import Any

import pytest

from idrkd.distillation import quantization
from idrkd.distillation.quantization import (
    AwqQuantizationConfig,
    AwqQuantizationJob,
    run_awq_quantization,
)


class _FakeTokenizer:
    loaded_from: list[str] = []

    @classmethod
    def from_pretrained(cls, path: str, **kwargs: Any) -> "_FakeTokenizer":
        del kwargs
        cls.loaded_from.append(path)
        return cls()

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is False
        return "<chat>" + "|".join(
            f"{message['role']}:{message['content']}" for message in messages
        )

    def save_pretrained(self, path: str) -> None:
        Path(path, "tokenizer.json").write_text("{}", encoding="utf-8")


class _FakeModel:
    def __init__(self) -> None:
        self.save_calls: list[dict[str, Any]] = []

    def save_pretrained(self, path: str, **kwargs: Any) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        Path(path, "model.safetensors").write_text("fake", encoding="utf-8")
        self.save_calls.append(kwargs)


class _FakeAutoModel:
    loaded_from: list[str] = []
    models: list[_FakeModel] = []

    @classmethod
    def from_pretrained(cls, path: str, **kwargs: Any) -> _FakeModel:
        assert kwargs["dtype"] == "auto"
        assert kwargs["device_map"] == "auto"
        cls.loaded_from.append(path)
        model = _FakeModel()
        cls.models.append(model)
        return model


class _FakePeftModel:
    @classmethod
    def from_pretrained(cls, model: _FakeModel, path: str) -> "_FakePeftModel":
        instance = cls()
        instance.model = model
        instance.path = path
        return instance

    def merge_and_unload(self) -> _FakeModel:
        return self.model


class _FakeDataset:
    @classmethod
    def from_dict(cls, records: dict[str, list[str]]) -> dict[str, list[str]]:
        return records


class _FakeAWQModifier:
    pass


class _FakeQuantizationModifier:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


def test_llm_compressor_awq_rejects_unsupported_quantization() -> None:
    with pytest.raises(ValueError, match="W4A16"):
        AwqQuantizationConfig(bits=8)


def test_calibration_data_renders_chat_and_teacher_trace(tmp_path: Path) -> None:
    calibration = tmp_path / "calibration.jsonl"
    calibration.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": "Find customer lookup"},
                            {
                                "role": "assistant",
                                "content": '{"name":"search_code","arguments":{}}',
                            },
                        ]
                    }
                ),
                json.dumps(
                    {
                        "prompt": "Fetch entity",
                        "steps": [
                            {
                                "tool_calls": [
                                    {
                                        "name": "get_entity",
                                        "arguments": {"entity_id": "entity-1"},
                                    }
                                ]
                            }
                        ],
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    job = AwqQuantizationJob(
        input_model_path=tmp_path,
        output_dir=tmp_path / "out",
        model_id="test",
        base_model_id="test/base",
        calibration_path=calibration,
    )

    samples = quantization._calibration_data(job, _FakeTokenizer())

    assert len(samples) == 2
    assert samples[0].startswith("<chat>user:Find customer lookup")
    assert "assistant:{\"arguments\":{\"entity_id\":\"entity-1\"}" in samples[1]


def test_quantization_merges_adapter_and_runs_llm_compressor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    calibration = tmp_path / "calibration.jsonl"
    calibration.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "Find customer lookup"},
                    {
                        "role": "assistant",
                        "content": '{"name":"search_code","arguments":{}}',
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    calls: list[dict[str, Any]] = []

    def fake_oneshot(**kwargs: Any) -> None:
        calls.append(kwargs)

    modules = {
        "AWQModifier": _FakeAWQModifier,
        "AutoModelForCausalLM": _FakeAutoModel,
        "AutoTokenizer": _FakeTokenizer,
        "Dataset": _FakeDataset,
        "PeftModel": _FakePeftModel,
        "QuantizationModifier": _FakeQuantizationModifier,
        "oneshot": fake_oneshot,
    }
    monkeypatch.setattr(quantization, "_load_quantization_modules", lambda: modules)
    output = tmp_path / "quantized"
    job = AwqQuantizationJob(
        input_model_path=tmp_path / "ignored-placeholder",
        output_dir=output,
        model_id="idrkd-test-awq",
        base_model_id="microsoft/Phi-4-mini-instruct",
        adapter_path=adapter,
        calibration_path=calibration,
        max_sequence_length=4096,
    )

    manifest = run_awq_quantization(job)

    assert len(calls) == 1
    assert calls[0]["max_seq_length"] == 4096
    assert calls[0]["num_calibration_samples"] == 1
    assert calls[0]["dataset"]["text"][0].startswith("<chat>user:")
    assert isinstance(calls[0]["recipe"][0], _FakeAWQModifier)
    assert calls[0]["recipe"][1].kwargs == {
        "targets": ["Linear"],
        "scheme": "W4A16_ASYM",
        "ignore": ["lm_head"],
    }
    assert _FakeAutoModel.models[-1].save_calls == [
        {"save_compressed": True, "safe_serialization": True}
    ]
    assert manifest.quantization.backend == "llm-compressor"
    assert manifest.quantization.format == "compressed-tensors"
    assert (output / "idrkd-model-manifest.json").is_file()
