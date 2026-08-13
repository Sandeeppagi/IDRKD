# Spec: SLM Distillation LLD

## Goal

Implement the Pillar 5 low-level contracts for distilling the IDRKD teacher
workflow into a Phi-4-mini student model without making local unit tests depend
on GPU training or downloaded model weights.

## Contract

- `src/idrkd/distillation/traces.py`: `TeacherTrace` captures tenant/repo scoped
  teacher answers, agent steps, MCP tool calls, evidence IDs, BFCL category, and
  faithfulness score. `select_sft_traces(...)` keeps only grounded traces for
  SFT and `sft_record(...)` emits chat-format SFT records with audit metadata.
- `src/idrkd/distillation/io.py`: JSONL trace IO, SFT dataset export, DPO
  preference dataset export, and SHA-256 dataset digests for reproducible runs.
- `src/idrkd/distillation/training.py`: `QLoRAConfig` encodes the W7 Phi-4-mini
  SFT contract (`microsoft/Phi-4-mini-instruct`, 4-bit NF4, LoRA r=64,
  alpha=128, 4k context, and Phi target modules `qkv_proj`, `o_proj`,
  `gate_up_proj`, and `down_proj`). `TrainingPlan.stage_order()` preserves the LLD
  execution order: teacher traces, QLoRA SFT, BFCL eval, DPO, AWQ, vLLM.
- `src/idrkd/distillation/execution.py`: runnable SFT and DPO trainers. Dry-run
  mode validates datasets and writes run summaries without loading weights;
  normal mode imports `transformers`, `datasets`, `peft`, and `trl`, loads the
  configured base model, applies LoRA/QLoRA, trains, saves adapters/tokenizer,
  and writes reproducibility summaries.
- `src/idrkd/distillation/cli.py`: `idrkd-distill`/module CLI for building
  datasets and launching SFT/DPO execution.
- `src/idrkd/distillation/preferences.py`: `build_preference_pair(...)` turns a
  grounded teacher trace plus a weaker student answer into a DPO record where
  `chosen = teacher trace` and `rejected = SFT naive output`.
- `src/idrkd/distillation/evaluation.py`: `BfclMetrics` computes precision,
  recall, and F1; `DistillationGate` enforces the W7-W8 thresholds: first-pass
  BFCL F1 >= 0.75, post-DPO BFCL F1 >= 0.82, AlignScore >= 0.78, and TTFT <=
  1.2s.
- `src/idrkd/distillation/quantization.py`: `AwqQuantizationConfig` records 4-bit
  AWQ settings, `AwqQuantizationJob` runs AutoAWQ over a merged model or a
  PEFT adapter merged into its base model, and `ModelArtifactManifest` writes
  deterministic manifest, digest, and HMAC signature material for reproducible
  serving handoff.
- `src/idrkd/distillation/quantize_cli.py`: production CLI for Linux/CUDA AWQ
  quantization. It writes a quantized model directory plus
  `idrkd-model-manifest.json`.
- `src/idrkd/distillation/serving.py`: `VllmServingConfig`,
  `OllamaServingConfig`, and `OpenAICompatibleStudentClient` expose
  OpenAI-compatible `/v1` serving contracts for CUDA and laptop fallback.
- `src/idrkd/rag/orchestrator.py`: synthesis can call the served student model
  when `IDRKD_STUDENT_MODEL_BASE_URL` and `IDRKD_STUDENT_MODEL_ID` are set.

## Implementation

- `src/idrkd/distillation/__init__.py`
- `src/idrkd/distillation/traces.py`
- `src/idrkd/distillation/preferences.py`
- `src/idrkd/distillation/io.py`
- `src/idrkd/distillation/training.py`
- `src/idrkd/distillation/execution.py`
- `src/idrkd/distillation/cli.py`
- `src/idrkd/distillation/evaluation.py`
- `src/idrkd/distillation/quantization.py`
- `src/idrkd/distillation/quantize_cli.py`
- `src/idrkd/distillation/serving.py`
- `docker/docker-compose.yml`
- `eval/distillation/seed_teacher_traces.jsonl`
- `tests/unit/test_distillation_execution.py`
- `tests/unit/test_week7_slm_distillation.py`

## Acceptance Criteria

- Grounded teacher traces with MCP tool use are selected for SFT; low
  faithfulness traces are excluded.
- SFT records preserve system/user/assistant messages plus trace, evidence, and
  tenant/repo metadata.
- The SFT/DPO dataset builders read teacher-trace JSONL and write JSONL records
  that can be consumed by Hugging Face training.
- DPO records mark teacher output as `chosen` and naive student output as
  `rejected`.
- QLoRA defaults match the HLD/LLD Phi-4-mini 4-bit NF4, r=64, alpha=128 plan,
  including Phi module names instead of Llama-style split projection names.
- `train-sft` and `train-dpo` have both dry-run and real ML execution modes.
  Real mode uses optional ML dependencies and configured model IDs/local
  checkpoints; dry-run mode still writes dataset digests and run summaries.
- BFCL/AlignScore/TTFT gates reject releases that miss the Pillar 5 thresholds.
- AWQ manifest digests/signatures and vLLM serving commands are deterministic.
- The AWQ CLI is import-light until execution and fails clearly when AutoAWQ is
  absent from a non-CUDA/local environment.
- The `slm-server` Compose profile serves AWQ models with vLLM on CUDA hosts.
- The `slm-server-laptop` Compose profile exposes an Ollama fallback for Mac or
  non-CUDA laptop workflows.
- RAG synthesis can use any OpenAI-compatible student endpoint and falls back to
  deterministic evidence synthesis when no endpoint is configured.

## Verification

```bash
uv run pytest tests/unit/test_week7_slm_distillation.py tests/unit/test_distillation_execution.py
uv run python -m idrkd.distillation.cli build-sft --traces eval/distillation/seed_teacher_traces.jsonl --out /private/tmp/idrkd-sft.jsonl
uv run python -m idrkd.distillation.cli build-dpo --traces eval/distillation/seed_teacher_traces.jsonl --out /private/tmp/idrkd-dpo.jsonl
uv run python -m idrkd.distillation.cli train-sft --dataset /private/tmp/idrkd-sft.jsonl --out /private/tmp/idrkd-sft-out --base-model local/tiny-student --dry-run
uv run python -m idrkd.distillation.cli train-dpo --dataset /private/tmp/idrkd-dpo.jsonl --out /private/tmp/idrkd-dpo-out --base-model local/tiny-student --dry-run
```

Real local-smoke execution, after installing ML extras and making the selected
small model available locally or allowing Hugging Face download:

```bash
uv sync --group dev --extra ml
uv run python -m idrkd.distillation.cli local-smoke --base-model Qwen/Qwen2.5-0.5B-Instruct --max-steps 5 --max-seq-length 128 --device-map none
uv run python -m idrkd.distillation.cli train-sft --dataset /private/tmp/idrkd-sft.jsonl --out models/adapters/local-smoke-sft --base-model Qwen/Qwen2.5-0.5B-Instruct --max-steps 5 --max-seq-length 512
uv run python -m idrkd.distillation.cli train-dpo --dataset /private/tmp/idrkd-dpo.jsonl --out models/adapters/local-smoke-dpo --base-model Qwen/Qwen2.5-0.5B-Instruct --max-steps 5 --max-seq-length 512
```

Production quantization and serving, typically on Linux/CUDA:

```bash
uv run python -m idrkd.distillation.quantize_cli --input-model models/checkpoints/merged-student --adapter models/adapters/local-smoke-dpo --base-model Qwen/Qwen2.5-0.5B-Instruct --out models/checkpoints/idrkd-student-awq --model-id idrkd-student-awq --calibration /private/tmp/idrkd-dpo.jsonl
docker compose -f docker/docker-compose.yml --profile slm-server up -d slm-server
```

Laptop serving fallback:

```bash
docker compose -f docker/docker-compose.yml --profile slm-server-laptop up -d slm-server-laptop
export IDRKD_STUDENT_MODEL_BASE_URL=http://localhost:11434/v1
export IDRKD_STUDENT_MODEL_ID=idrkd-student
```
