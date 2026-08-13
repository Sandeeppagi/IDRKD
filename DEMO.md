# IDRKD Demo Script

This demo shows the local, reproducible path for IDRKD without requiring Docker,
GPU training, vLLM, or external model downloads. It proves the core contracts:
MCP tool evaluation works, SFT/DPO datasets are generated, and distillation
targets now match the JSON tool-call format and prompt shape expected by
TaskBench.

## Quick Run

```bash
uv sync --group dev
./scripts/demo_local.sh
```

The script writes demo artifacts to `/private/tmp/idrkd-demo` by default. To use
another directory:

```bash
IDRKD_DEMO_OUT=/private/tmp/idrkd-demo-run ./scripts/demo_local.sh
```

## Presenter Flow

### 1. Open With The Problem

IDRKD reconciles enterprise knowledge across code, schemas, documents, graph
records, vector search, MCP tools, and a distilled student model.

The key engineering question is:

```text
Can the system pick the right tool, with the right arguments, from grounded
repository context?
```

### 2. Show The Local Smoke Demo

Run:

```bash
./scripts/demo_local.sh
```

Narrate the six sections printed by the script:

```text
1. Verify the local code path
2. Build JSON tool-call SFT and DPO datasets
3. Show that training targets match TaskBench parser contract
4. Run dry-run SFT and DPO training summaries
5. Run MCP-TaskBench registry smoke evaluation
6. Print demo artifacts
```

Expected headline output:

```text
SFT records: 500
DPO records: 500
TaskBench SFT records: 416
TaskBench DPO records: 416
TaskBench pass rate: 1.0
TaskBench tool F1: 1.0
```

### 3. Explain The Recent Format Fix

Before the fix, the teacher traces trained the student to answer in prose:

```text
Use `search_code` for scoped semantic repository lookup...
```

TaskBench expects a machine-readable tool call:

```json
{"name":"search_code","arguments":{"tenant_id":"tenant-seed","repo_id":"repo-seed"}}
```

The demo script prints the first SFT target, DPO chosen target, and DPO rejected
target, plus the first TaskBench-aligned SFT target. It parses each with the
same `parse_tool_call()` function used by evaluation. The TaskBench-aligned
records also use the same tool-schema user prompt as live model-agent
evaluation.

### 4. Show The Generated Artifacts

After the script completes, inspect:

```bash
ls -lah /private/tmp/idrkd-demo
jq . /private/tmp/idrkd-demo/sft-dry-run/sft-run-summary.json
jq . /private/tmp/idrkd-demo/dpo-dry-run/dpo-run-summary.json
jq '{pass_rate, tool_f1, argument_accuracy, schema_valid_rate}' \
  /private/tmp/idrkd-demo/registry-smoke-summary.json
```

Useful files:

```text
/private/tmp/idrkd-demo/idrkd-sft.jsonl
/private/tmp/idrkd-demo/idrkd-dpo.jsonl
/private/tmp/idrkd-demo/idrkd-taskbench-sft.jsonl
/private/tmp/idrkd-demo/idrkd-taskbench-dpo.jsonl
/private/tmp/idrkd-demo/sft-dry-run/sft-run-summary.json
/private/tmp/idrkd-demo/dpo-dry-run/dpo-run-summary.json
/private/tmp/idrkd-demo/registry-smoke-summary.json
```

### 5. Optional Live MCP Demo

If Docker services are available:

```bash
docker compose -f docker/docker-compose.yml up -d --build mcp-server
uv run idrkd-mcp-smoke --tenant-id tenant-live --repo-id week5-e2e
```

This verifies the running MCP server with `tools/list`, `search_code`,
`graph_bfs`, and `graph_path`.

### 6. Optional Model-Agent Demo

If an OpenAI-compatible local model endpoint is available:

```bash
uv run python -m idrkd.evaluation.cli \
  --mode student-agent \
  --model-base-url http://localhost:11434/v1 \
  --model-id qwen2.5:0.5b \
  --tasks eval/taskbench/seed_tasks.jsonl \
  --out /private/tmp/idrkd-demo/student-agent-summary.json
```

For the corrected Phi student, run this only after retraining with the JSON
SFT/DPO targets.

### 7. Optional GPU Quantization Demo

Run AWQ only on a Linux/CUDA builder using the isolated legacy environment from
the README. Do not install AutoAWQ into the working IDRKD `.venv`.

The expected final artifact is:

```text
models/checkpoints/phi4-mini-awq/idrkd-model-manifest.json
```

## Reset

```bash
rm -rf /private/tmp/idrkd-demo
```
