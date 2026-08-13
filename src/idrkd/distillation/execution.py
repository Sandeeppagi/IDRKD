"""Executable SFT/DPO distillation runners.

The functions in this module are intentionally import-light: normal unit tests
do not import PyTorch or download model weights, while real runs load the ML
stack only after `dry_run=False`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
import importlib
import json
from pathlib import Path
from typing import Any, cast

from idrkd.distillation.io import dataset_digest, read_jsonl_records
from idrkd.distillation.traces import SYSTEM_PROMPT
from idrkd.distillation.training import DpoConfig, QLoRAConfig


@dataclass(frozen=True)
class DistillationRuntimeConfig:
    dataset_path: Path
    output_dir: Path
    base_model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    sft_adapter_path: Path | None = None
    max_seq_length: int = 1024
    epochs: float = 1.0
    learning_rate: float = 2e-4
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    use_4bit: bool = False
    device_map: str | None = "auto"
    local_files_only: bool = False
    max_steps: int = -1
    seed: int = 42
    dry_run: bool = False


@dataclass(frozen=True)
class DistillationRunResult:
    stage: str
    output_dir: str
    base_model_id: str
    sft_adapter_path: str | None
    dataset_path: str
    dataset_sha256: str
    record_count: int
    dry_run: bool
    metrics: dict[str, float]
    created_at: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DistillationSmokeResult:
    sft: DistillationRunResult
    dpo: DistillationRunResult
    sft_adapter_written: bool
    dpo_adapter_written: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "sft": self.sft.as_dict(),
            "dpo": self.dpo.as_dict(),
            "sft_adapter_written": self.sft_adapter_written,
            "dpo_adapter_written": self.dpo_adapter_written,
        }


def render_sft_text(record: dict[str, Any], tokenizer: Any | None = None) -> str:
    messages = record.get("messages", [])
    if not isinstance(messages, list):
        raise ValueError("SFT record messages must be a list")
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        return str(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        )
    chunks = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("SFT message must be an object")
        role = str(message["role"])
        content = str(message["content"])
        chunks.append(f"<|{role}|>{content}<|end|>")
    return "\n".join(chunks)


def render_sft_prompt_and_response(
    record: dict[str, Any],
    tokenizer: Any | None = None,
) -> tuple[str, str]:
    messages = record.get("messages", [])
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError("SFT record messages must contain prompt and assistant response")
    response = messages[-1]
    if not isinstance(response, dict) or response.get("role") != "assistant":
        raise ValueError("SFT record must end with an assistant message")
    prompt_messages = messages[:-1]
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        prompt_text = str(
            tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )
        full_text = str(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        )
        if full_text.startswith(prompt_text):
            return prompt_text, full_text[len(prompt_text) :]
        return prompt_text, str(response["content"])
    prompt_text = render_sft_text({"messages": prompt_messages})
    return f"{prompt_text}\n<|assistant|>", f"{response['content']}<|end|>"


def train_sft(
    config: DistillationRuntimeConfig,
    *,
    qlora: QLoRAConfig | None = None,
) -> DistillationRunResult:
    records = read_jsonl_records(config.dataset_path)
    if not records:
        raise ValueError("SFT dataset is empty")
    active_qlora = qlora or QLoRAConfig(
        base_model_id=config.base_model_id,
        load_in_4bit=config.use_4bit,
        max_seq_length=config.max_seq_length,
    )
    if config.dry_run:
        return _write_result(
            stage="sft",
            config=config,
            record_count=len(records),
            metrics={"train_loss": 0.0},
        )

    modules = _load_ml_modules()
    tokenizer = modules["AutoTokenizer"].from_pretrained(
        config.base_model_id,
        local_files_only=config.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {"local_files_only": config.local_files_only}
    if config.device_map is not None:
        model_kwargs["device_map"] = config.device_map
    if config.use_4bit:
        model_kwargs["quantization_config"] = modules["BitsAndBytesConfig"](
            **active_qlora.quantization_kwargs()
        )
    model = modules["AutoModelForCausalLM"].from_pretrained(config.base_model_id, **model_kwargs)
    if config.use_4bit:
        model = modules["prepare_model_for_kbit_training"](model)
    model = modules["get_peft_model"](model, modules["LoraConfig"](**active_qlora.peft_kwargs()))

    dataset = modules["Dataset"].from_list(
        [
            {
                "prompt_text": prompt_text,
                "response_text": response_text,
            }
            for prompt_text, response_text in (
                render_sft_prompt_and_response(record, tokenizer) for record in records
            )
        ]
    )
    tokenized = _tokenize_prompt_response_dataset(dataset, tokenizer, config.max_seq_length)
    args = modules["TrainingArguments"](
        output_dir=str(config.output_dir),
        num_train_epochs=config.epochs,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        max_steps=config.max_steps,
        logging_steps=1,
        save_strategy="epoch",
        report_to=[],
        seed=config.seed,
    )
    trainer = modules["Trainer"](
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=_sft_data_collator(tokenizer=tokenizer, torch=modules["torch"]),
    )
    train_output = trainer.train()
    trainer.save_model(str(config.output_dir))
    tokenizer.save_pretrained(str(config.output_dir))
    return _write_result(
        stage="sft",
        config=config,
        record_count=len(records),
        metrics={"train_loss": float(getattr(train_output, "training_loss", 0.0) or 0.0)},
    )


def train_dpo(
    config: DistillationRuntimeConfig,
    *,
    dpo: DpoConfig | None = None,
) -> DistillationRunResult:
    records = read_jsonl_records(config.dataset_path)
    if not records:
        raise ValueError("DPO dataset is empty")
    active_dpo = dpo or DpoConfig(epochs=int(config.epochs), learning_rate=config.learning_rate)
    if config.dry_run:
        return _write_result(
            stage="dpo",
            config=config,
            record_count=len(records),
            metrics={"train_loss": 0.0, "beta": active_dpo.beta},
        )

    sft_adapter_path = _validate_sft_adapter_path(config.sft_adapter_path)
    modules = _load_ml_modules()
    tokenizer = modules["AutoTokenizer"].from_pretrained(
        config.base_model_id,
        local_files_only=config.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rendered_records = _render_dpo_records(records, tokenizer)
    _ensure_dpo_sequences_fit(rendered_records, tokenizer, config.max_seq_length)

    model_kwargs: dict[str, Any] = {"local_files_only": config.local_files_only}
    if config.device_map is not None:
        model_kwargs["device_map"] = config.device_map
    if config.use_4bit:
        model_kwargs["quantization_config"] = modules["BitsAndBytesConfig"](
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    model = modules["AutoModelForCausalLM"].from_pretrained(config.base_model_id, **model_kwargs)
    if config.use_4bit:
        model = modules["prepare_model_for_kbit_training"](model)
    if sft_adapter_path is not None:
        # TRL >= 1.0 requires the trainable adapter to keep PEFT's default name
        # ("default"); DPOTrainer clones it into a frozen "ref" adapter itself.
        model = modules["PeftModel"].from_pretrained(
            model,
            str(sft_adapter_path),
            is_trainable=True,
        )
    else:
        active_qlora = QLoRAConfig(
            base_model_id=config.base_model_id,
            load_in_4bit=config.use_4bit,
            max_seq_length=config.max_seq_length,
        )
        model = modules["get_peft_model"](model, modules["LoraConfig"](**active_qlora.peft_kwargs()))
    _ensure_trainable_parameters(model)

    dataset = modules["Dataset"].from_list(rendered_records)
    args = modules["DPOConfig"](
        output_dir=str(config.output_dir),
        num_train_epochs=active_dpo.epochs,
        learning_rate=active_dpo.learning_rate,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        max_steps=config.max_steps,
        logging_steps=1,
        save_strategy="epoch",
        report_to=[],
        seed=config.seed,
        beta=active_dpo.beta,
        max_length=config.max_seq_length,
    )
    trainer = modules["DPOTrainer"](
        model=model,
        ref_model=None,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    train_output = trainer.train()
    trainer.save_model(str(config.output_dir))
    tokenizer.save_pretrained(str(config.output_dir))
    return _write_result(
        stage="dpo",
        config=config,
        record_count=len(records),
        metrics={"train_loss": float(getattr(train_output, "training_loss", 0.0) or 0.0), "beta": active_dpo.beta},
    )


def run_laptop_smoke_distillation(
    *,
    sft_config: DistillationRuntimeConfig,
    dpo_config: DistillationRuntimeConfig,
) -> DistillationSmokeResult:
    sft_result = train_sft(sft_config)
    sft_adapter_written = adapter_artifacts_written(sft_config.output_dir)
    if not sft_adapter_written:
        raise RuntimeError(f"SFT did not write PEFT adapter artifacts to {sft_config.output_dir}")

    active_dpo_config = dpo_config
    if active_dpo_config.sft_adapter_path is None:
        active_dpo_config = replace(active_dpo_config, sft_adapter_path=sft_config.output_dir)

    dpo_result = train_dpo(active_dpo_config)
    dpo_adapter_written = adapter_artifacts_written(active_dpo_config.output_dir)
    if not dpo_adapter_written:
        raise RuntimeError(f"DPO did not write PEFT adapter artifacts to {active_dpo_config.output_dir}")

    return DistillationSmokeResult(
        sft=sft_result,
        dpo=dpo_result,
        sft_adapter_written=sft_adapter_written,
        dpo_adapter_written=dpo_adapter_written,
    )


def adapter_artifacts_written(output_dir: Path) -> bool:
    return (output_dir / "adapter_config.json").is_file() and (
        (output_dir / "adapter_model.safetensors").is_file()
        or (output_dir / "adapter_model.bin").is_file()
    )


def _validate_sft_adapter_path(adapter_path: Path | None) -> Path | None:
    if adapter_path is None:
        return None
    if not adapter_path.is_dir():
        raise ValueError(f"SFT adapter directory does not exist: {adapter_path}")
    if not adapter_artifacts_written(adapter_path):
        raise ValueError(f"SFT adapter artifacts are incomplete: {adapter_path}")
    return adapter_path


def _ensure_dpo_sequences_fit(
    records: list[dict[str, Any]],
    tokenizer: Any,
    max_seq_length: int,
) -> None:
    """Fail before training when TRL's max_length truncation would drop completions.

    TRL truncates each prompt+completion sequence to max_length from the right, so a
    prompt at or beyond the limit leaves zero completion tokens and DPO silently
    trains on a constant ln(2) loss with zero gradients.
    """

    def token_count(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    required = 0
    for record in records:
        completion = max(token_count(record["chosen"]), token_count(record["rejected"]))
        required = max(required, token_count(record["prompt"]) + completion + 1)
    if required > max_seq_length:
        raise ValueError(
            f"DPO records need up to {required} tokens (prompt + completion + EOS) but "
            f"max_seq_length is {max_seq_length}; completions would be truncated away. "
            "Increase --max-seq-length."
        )


def _ensure_trainable_parameters(model: Any) -> None:
    named_parameters = getattr(model, "named_parameters", None)
    if named_parameters is None:
        return
    trainable_count = sum(param.numel() for _, param in named_parameters() if getattr(param, "requires_grad", False))
    if trainable_count == 0:
        raise RuntimeError("DPO model has no trainable parameters")


def _tokenize_text_dataset(dataset: Any, tokenizer: Any, max_seq_length: int) -> Any:
    def tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
        tokenized = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )
        tokenized["labels"] = [list(ids) for ids in tokenized["input_ids"]]
        return cast(dict[str, Any], tokenized)

    return dataset.map(tokenize, batched=True, remove_columns=["text"])


def _tokenize_prompt_response_dataset(dataset: Any, tokenizer: Any, max_seq_length: int) -> Any:
    def tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
        input_ids = []
        attention_mask = []
        labels = []
        for prompt_text, response_text in zip(batch["prompt_text"], batch["response_text"], strict=True):
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            response_ids = tokenizer(response_text, add_special_tokens=False)["input_ids"]
            ids = list(prompt_ids) + list(response_ids)
            row_labels = [-100] * len(prompt_ids) + list(response_ids)
            if len(ids) > max_seq_length:
                ids = ids[-max_seq_length:]
                row_labels = row_labels[-max_seq_length:]
            input_ids.append(ids)
            attention_mask.append([1] * len(ids))
            labels.append(row_labels)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    return dataset.map(tokenize, batched=True, remove_columns=["prompt_text", "response_text"])


def _render_dpo_records(records: list[dict[str, Any]], tokenizer: Any | None = None) -> list[dict[str, Any]]:
    rendered = []
    for record in records:
        prompt = str(record["prompt"])
        chosen = str(record["chosen"])
        rejected = str(record["rejected"])
        prompt_text, chosen_text = render_sft_prompt_and_response(
            _dpo_as_sft_record(prompt=prompt, response=chosen),
            tokenizer,
        )
        rejected_prompt_text, rejected_text = render_sft_prompt_and_response(
            _dpo_as_sft_record(prompt=prompt, response=rejected),
            tokenizer,
        )
        if rejected_prompt_text != prompt_text:
            raise ValueError("DPO chosen and rejected prompts rendered differently")
        rendered_record = {
            "prompt": prompt_text,
            "chosen": chosen_text,
            "rejected": rejected_text,
        }
        if "metadata" in record:
            rendered_record["metadata"] = record["metadata"]
        rendered.append(rendered_record)
    return rendered


def _dpo_as_sft_record(*, prompt: str, response: str) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
    }


def _sft_data_collator(*, tokenizer: Any, torch: Any) -> Any:
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    def collate(features: list[dict[str, list[int]]]) -> dict[str, Any]:
        max_length = max(len(feature["input_ids"]) for feature in features)
        input_ids = []
        attention_mask = []
        labels = []
        for feature in features:
            pad_length = max_length - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [pad_token_id] * pad_length)
            attention_mask.append(feature["attention_mask"] + [0] * pad_length)
            labels.append(feature["labels"] + [-100] * pad_length)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    return collate


def _write_result(
    *,
    stage: str,
    config: DistillationRuntimeConfig,
    record_count: int,
    metrics: dict[str, float],
) -> DistillationRunResult:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    result = DistillationRunResult(
        stage=stage,
        output_dir=str(config.output_dir),
        base_model_id=config.base_model_id,
        sft_adapter_path=str(config.sft_adapter_path) if config.sft_adapter_path is not None else None,
        dataset_path=str(config.dataset_path),
        dataset_sha256=dataset_digest(config.dataset_path),
        record_count=record_count,
        dry_run=config.dry_run,
        metrics=metrics,
        created_at=datetime.now(UTC).isoformat(),
    )
    (config.output_dir / f"{stage}-run-summary.json").write_text(
        json.dumps(result.as_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return result


def _load_ml_modules() -> dict[str, Any]:
    try:
        transformers = importlib.import_module("transformers")
        datasets = importlib.import_module("datasets")
        peft = importlib.import_module("peft")
        torch = importlib.import_module("torch")
        trl = importlib.import_module("trl")
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError("Install ML dependencies with `uv sync --extra ml --group dev`.") from exc

    return {
        "AutoModelForCausalLM": transformers.AutoModelForCausalLM,
        "AutoTokenizer": transformers.AutoTokenizer,
        "BitsAndBytesConfig": transformers.BitsAndBytesConfig,
        "DataCollatorForLanguageModeling": transformers.DataCollatorForLanguageModeling,
        "torch": torch,
        "Trainer": transformers.Trainer,
        "TrainingArguments": transformers.TrainingArguments,
        "Dataset": datasets.Dataset,
        "LoraConfig": peft.LoraConfig,
        "PeftModel": peft.PeftModel,
        "get_peft_model": peft.get_peft_model,
        "prepare_model_for_kbit_training": peft.prepare_model_for_kbit_training,
        "DPOConfig": trl.DPOConfig,
        "DPOTrainer": trl.DPOTrainer,
    }
