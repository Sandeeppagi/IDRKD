#!/usr/bin/env bash
# Push IDRKD artifacts to Hugging Face Hub
# Run this on the RunPod instance where the real model files live.
#
# Usage:
#   export HF_TOKEN=hf_xxxxx
#   bash push_to_hf.sh [HF_USERNAME]
#
# Creates three HF repos:
#   1. {user}/idrkd-phi4-mini-awq           — quantized model checkpoint
#   2. {user}/idrkd-phi4-mini-adapters      — SFT + DPO LoRA adapters
#   3. {user}/idrkd-mcp-taskbench           — dataset + evaluation results
set -euo pipefail

HF_USER="${1:-Sandeeppagi}"

# Verify token
if [ -z "${HF_TOKEN:-}" ]; then
  echo "ERROR: Set HF_TOKEN first: export HF_TOKEN=hf_xxxxx"
  exit 1
fi

echo "Logging in to Hugging Face as ${HF_USER}..."
python3 -c "from huggingface_hub import HfApi; api=HfApi(token='${HF_TOKEN}'); print('Authenticated as:', api.whoami()['name'])"

# ──────────────────────────────────────────────────────────────
# 1. QUANTIZED MODEL CHECKPOINT
# ──────────────────────────────────────────────────────────────
REPO_MODEL="${HF_USER}/idrkd-phi4-mini-awq"
CHECKPOINT_DIR="/workspace/IDRKD/models/checkpoints/phi4-mini-dpo-tooljson-split-v2-llmc-awq"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  1/3  Pushing quantized AWQ model → ${REPO_MODEL}"
echo "═══════════════════════════════════════════════════════════"

# Create model card
cat > /tmp/idrkd-model-README.md << 'MODEL_CARD'
---
language:
  - en
license: apache-2.0
library_name: transformers
tags:
  - phi4
  - awq
  - quantized
  - tool-calling
  - mcp
  - vllm
  - knowledge-discovery
base_model: microsoft/Phi-4-mini-instruct
pipeline_tag: text-generation
---

# IDRKD Phi-4-mini AWQ (W4A16)

A distilled and quantized **Phi-4-mini-instruct** model fine-tuned for MCP (Model Context Protocol) tool-call routing in enterprise knowledge discovery systems.

## Model Details

| Property | Value |
|----------|-------|
| Base Model | `microsoft/Phi-4-mini-instruct` (3.8B params) |
| Fine-tuning | QLoRA SFT + DPO on MCP-TaskBench tool-call traces |
| Quantization | W4A16 AWQ via llm-compressor (compressed-tensors format) |
| Model Size | ~2.9 GB |
| Serving | vLLM with `--quantization compressed-tensors` |

## Training Pipeline

1. **SFT Stage**: QLoRA (rank 16, alpha 32) on 351 tool-call SFT records from MCP-TaskBench
2. **DPO Stage**: Direct Preference Optimization (β=0.1) on 351 preference pairs (correct JSON tool calls vs. rejected)
3. **Quantization**: W4A16 asymmetric AWQ with group_size=128, calibrated on 64 representative SFT records

## Evaluation Results

| Metric | Value |
|--------|-------|
| MCP-TaskBench Pass Rate | **1.0** (440/440 cases) |
| Holdout Tool F1 | **1.0** (89/89 unseen cases) |
| Argument Accuracy | **1.0** |
| Faithfulness (NLI) | **0.96** |
| TTFT p95 | **33 ms** |
| Latency p95 | **1.32 s** |

Evaluated on NVIDIA L40S (48 GB) with vLLM v0.27.1.

## Usage with vLLM

```bash
vllm serve Sandeeppagi/idrkd-phi4-mini-awq \
  --quantization compressed-tensors \
  --max-model-len 4096 \
  --host 0.0.0.0 --port 8000
```

## Tool-Call Format

The model outputs structured JSON tool calls:

```json
{"name": "search_code", "arguments": {"tenant_id": "t1", "repo_id": "r1", "query": "authentication"}}
```

## MCP Tool Catalog (14 tools)

`search_code`, `get_entity`, `graph_bfs`, `graph_path`, `get_community`, `enqueue_reindex`, `schema_diff`, `impact_analysis`, `reconcile`, `get_conflict`, `resolve_conflict`, `get_salience`, `get_centroid_drift`, `list_stale`

## Project

Part of the IDRKD (Intelligent Data Reconciliation and Knowledge Discovery) capstone research project.
- GitHub: [Sandeeppagi/IDRKD](https://github.com/Sandeeppagi/IDRKD)
MODEL_CARD

python3 << PYMODEL
from huggingface_hub import HfApi, create_repo
import os

api = HfApi(token=os.environ["HF_TOKEN"])
repo_id = "${REPO_MODEL}"

# Create repo (model type)
create_repo(repo_id, repo_type="model", exist_ok=True, token=os.environ["HF_TOKEN"])

# Upload model card
api.upload_file(
    path_or_fileobj="/tmp/idrkd-model-README.md",
    path_in_repo="README.md",
    repo_id=repo_id,
    repo_type="model",
)

# Upload checkpoint directory
api.upload_folder(
    folder_path="${CHECKPOINT_DIR}",
    repo_id=repo_id,
    repo_type="model",
    commit_message="Upload IDRKD Phi-4-mini AWQ W4A16 checkpoint",
)

print(f"✅ Model uploaded: https://huggingface.co/{repo_id}")
PYMODEL

# ──────────────────────────────────────────────────────────────
# 2. LORA ADAPTERS (SFT + DPO)
# ──────────────────────────────────────────────────────────────
REPO_ADAPTERS="${HF_USER}/idrkd-phi4-mini-adapters"
SFT_DIR="/workspace/IDRKD/models/adapters/phi4-mini-sft-tooljson-split-v2"
DPO_DIR="/workspace/IDRKD/models/adapters/phi4-mini-dpo-tooljson-split-v2"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  2/3  Pushing LoRA adapters → ${REPO_ADAPTERS}"
echo "═══════════════════════════════════════════════════════════"

cat > /tmp/idrkd-adapters-README.md << 'ADAPTER_CARD'
---
language:
  - en
license: apache-2.0
library_name: peft
tags:
  - phi4
  - lora
  - qlora
  - dpo
  - sft
  - tool-calling
  - mcp
base_model: microsoft/Phi-4-mini-instruct
---

# IDRKD Phi-4-mini LoRA Adapters (SFT + DPO)

QLoRA adapters for **Phi-4-mini-instruct** fine-tuned on MCP tool-call routing tasks.

## Contents

```
sft/    — Supervised Fine-Tuning adapter (QLoRA rank 16, alpha 32)
dpo/    — Direct Preference Optimization adapter (β=0.1)
```

## Training Details

### SFT Adapter
- **Dataset**: 351 MCP-TaskBench tool-call SFT records (train split, seed 17)
- **LoRA Config**: rank=16, alpha=32, targets: `qkv_proj`, `o_proj`, `gate_up_proj`, `down_proj`
- **Quantization**: 4-bit NF4 (QLoRA)

### DPO Adapter
- **Initialised from**: SFT adapter
- **Dataset**: 351 preference pairs (chosen = correct JSON tool call, rejected = wrong tool/args)
- **β**: 0.1

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("microsoft/Phi-4-mini-instruct")
model = PeftModel.from_pretrained(base, "Sandeeppagi/idrkd-phi4-mini-adapters/dpo")
```

## Evaluation

| Metric | SFT | DPO |
|--------|-----|-----|
| Holdout Tool F1 | 1.0 | 1.0 |
| Argument Accuracy | 1.0 | 1.0 |

## Project

- GitHub: [Sandeeppagi/IDRKD](https://github.com/Sandeeppagi/IDRKD)
ADAPTER_CARD

python3 << PYADAPTER
from huggingface_hub import HfApi, create_repo
import os

api = HfApi(token=os.environ["HF_TOKEN"])
repo_id = "${REPO_ADAPTERS}"

create_repo(repo_id, repo_type="model", exist_ok=True, token=os.environ["HF_TOKEN"])

# Upload README
api.upload_file(
    path_or_fileobj="/tmp/idrkd-adapters-README.md",
    path_in_repo="README.md",
    repo_id=repo_id,
    repo_type="model",
)

# Upload SFT adapter
api.upload_folder(
    folder_path="${SFT_DIR}",
    path_in_repo="sft",
    repo_id=repo_id,
    repo_type="model",
    commit_message="Upload SFT LoRA adapter",
)

# Upload DPO adapter
api.upload_folder(
    folder_path="${DPO_DIR}",
    path_in_repo="dpo",
    repo_id=repo_id,
    repo_type="model",
    commit_message="Upload DPO LoRA adapter",
)

print(f"✅ Adapters uploaded: https://huggingface.co/{repo_id}")
PYADAPTER

# ──────────────────────────────────────────────────────────────
# 3. DATASET + EVALUATION RESULTS
# ──────────────────────────────────────────────────────────────
REPO_DATASET="${HF_USER}/idrkd-mcp-taskbench"
IDRKD_ROOT="/workspace/IDRKD"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  3/3  Pushing dataset + eval → ${REPO_DATASET}"
echo "═══════════════════════════════════════════════════════════"

cat > /tmp/idrkd-dataset-README.md << 'DATASET_CARD'
---
language:
  - en
license: apache-2.0
tags:
  - mcp
  - tool-calling
  - benchmark
  - knowledge-discovery
  - function-calling
task_categories:
  - text-generation
size_categories:
  - n<1K
---

# IDRKD MCP-TaskBench

A benchmark dataset and evaluation suite for MCP (Model Context Protocol) tool-call routing in enterprise knowledge discovery systems.

## Dataset Contents

```
taskbench/
  seed_tasks.jsonl          — 361+ seed benchmark tasks across 6 categories
distillation/
  seed_teacher_traces.jsonl — Teacher model traces for SFT/DPO distillation
datasets/
  *.jsonl                   — Pre-built SFT and DPO training datasets
releases/
  phi4-mini-llmc-awq-v1/    — Complete release evaluation evidence
```

## Task Categories (6)

| Category | Description | Example Tools |
|----------|-------------|---------------|
| `tool_selection` | Semantic query → tool mapping | `search_code` |
| `schema_conformance` | Entity ID retrieval | `get_entity`, `schema_diff` |
| `multi_hop_planning` | Graph traversal queries | `graph_bfs`, `graph_path`, `get_community` |
| `conflict_resolution` | Version reconciliation | `reconcile`, `resolve_conflict` |
| `drift_trigger` | Stale entity reindexing | `enqueue_reindex`, `get_centroid_drift` |
| `a2a_delegation` | Multi-agent delegation | `get_conflict` |

## MCP Tool Catalog (14 tools)

`search_code`, `get_entity`, `graph_bfs`, `graph_path`, `get_community`, `enqueue_reindex`, `schema_diff`, `impact_analysis`, `reconcile`, `get_conflict`, `resolve_conflict`, `get_salience`, `get_centroid_drift`, `list_stale`

## Evaluation Results (Phi-4-mini AWQ)

| Metric | Value |
|--------|-------|
| Pass Rate | **1.0** (440/440) |
| Tool F1 | **1.0** |
| Argument Accuracy | **1.0** |
| TTFT p95 | **33 ms** |

## Usage

```python
import json

tasks = [json.loads(line) for line in open("taskbench/seed_tasks.jsonl")]
print(f"Tasks: {len(tasks)}, Categories: {set(t['category'] for t in tasks)}")
```

## Project

- GitHub: [Sandeeppagi/IDRKD](https://github.com/Sandeeppagi/IDRKD)
DATASET_CARD

python3 << PYDATASET
from huggingface_hub import HfApi, create_repo
import os

api = HfApi(token=os.environ["HF_TOKEN"])
repo_id = "${REPO_DATASET}"

create_repo(repo_id, repo_type="dataset", exist_ok=True, token=os.environ["HF_TOKEN"])

# Upload README
api.upload_file(
    path_or_fileobj="/tmp/idrkd-dataset-README.md",
    path_in_repo="README.md",
    repo_id=repo_id,
    repo_type="dataset",
)

# Upload TaskBench seed tasks
api.upload_folder(
    folder_path="${IDRKD_ROOT}/eval/taskbench",
    path_in_repo="taskbench",
    repo_id=repo_id,
    repo_type="dataset",
    commit_message="Upload MCP-TaskBench seed tasks",
)

# Upload teacher traces
api.upload_folder(
    folder_path="${IDRKD_ROOT}/eval/distillation",
    path_in_repo="distillation",
    repo_id=repo_id,
    repo_type="dataset",
    commit_message="Upload teacher traces for distillation",
)

# Upload pre-built datasets if they exist
datasets_dir = "${IDRKD_ROOT}/artifacts/datasets"
if os.path.isdir(datasets_dir):
    api.upload_folder(
        folder_path=datasets_dir,
        path_in_repo="datasets",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Upload pre-built SFT/DPO datasets",
    )

# Upload release evidence
api.upload_folder(
    folder_path="${IDRKD_ROOT}/eval/releases/phi4-mini-llmc-awq-v1",
    path_in_repo="releases/phi4-mini-llmc-awq-v1",
    repo_id=repo_id,
    repo_type="dataset",
    commit_message="Upload release evaluation evidence",
)

print(f"✅ Dataset uploaded: https://huggingface.co/datasets/{repo_id}")
PYDATASET

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  All done! Your Hugging Face repos:"
echo ""
echo "  🏆 Model:    https://huggingface.co/${REPO_MODEL}"
echo "  🔧 Adapters: https://huggingface.co/${REPO_ADAPTERS}"
echo "  📊 Dataset:  https://huggingface.co/datasets/${REPO_DATASET}"
echo "═══════════════════════════════════════════════════════════"
