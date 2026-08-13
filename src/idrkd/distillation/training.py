"""QLoRA and DPO training plan contracts for Phi-4-mini."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QLoRAConfig:
    base_model_id: str = "microsoft/Phi-4-mini-instruct"
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = ("qkv_proj", "o_proj", "gate_up_proj", "down_proj")
    max_seq_length: int = 4096

    def peft_kwargs(self) -> dict[str, object]:
        return {
            "r": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": list(self.target_modules),
            "bias": "none",
            "task_type": "CAUSAL_LM",
        }

    def quantization_kwargs(self) -> dict[str, object]:
        return {
            "load_in_4bit": self.load_in_4bit,
            "bnb_4bit_quant_type": self.bnb_4bit_quant_type,
            "bnb_4bit_use_double_quant": True,
        }


@dataclass(frozen=True)
class DpoConfig:
    beta: float = 0.1
    epochs: int = 1
    target_pair_count: int = 500
    learning_rate: float = 5e-7


@dataclass(frozen=True)
class TrainingPlan:
    qlora: QLoRAConfig = QLoRAConfig()
    dpo: DpoConfig = DpoConfig()
    sft_output_dir: str = "models/adapters/phi4-mini-sft"
    dpo_output_dir: str = "models/adapters/phi4-mini-dpo"

    def stage_order(self) -> tuple[str, ...]:
        return ("teacher_trace_export", "qlora_sft", "bfcl_eval", "dpo_alignment", "awq_quantize", "vllm_serve")
