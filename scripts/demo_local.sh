#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${IDRKD_DEMO_OUT:-/private/tmp/idrkd-demo}"
TASKS_PATH="${IDRKD_DEMO_TASKS:-eval/taskbench/seed_tasks.jsonl}"
TRACES_PATH="${IDRKD_DEMO_TRACES:-eval/distillation/seed_teacher_traces.jsonl}"
export OUT_DIR

cd "$ROOT_DIR"
mkdir -p "$OUT_DIR"

section() {
  printf "\n==> %s\n" "$1"
}

section "1. Verify the local code path"
uv run pytest tests/unit/test_week7_slm_distillation.py tests/unit/test_distillation_execution.py tests/evaluation/test_taskbench_harness.py

section "2. Build JSON tool-call SFT and DPO datasets"
uv run python -m idrkd.distillation.cli build-sft \
  --traces "$TRACES_PATH" \
  --out "$OUT_DIR/idrkd-sft.jsonl"
uv run python -m idrkd.distillation.cli build-dpo \
  --traces "$TRACES_PATH" \
  --out "$OUT_DIR/idrkd-dpo.jsonl"
uv run python -m idrkd.distillation.cli build-taskbench-sft \
  --tasks "$TASKS_PATH" \
  --include-synthetic-schemas \
  --split train \
  --out "$OUT_DIR/idrkd-taskbench-sft.jsonl"
uv run python -m idrkd.distillation.cli build-taskbench-dpo \
  --tasks "$TASKS_PATH" \
  --include-synthetic-schemas \
  --split train \
  --out "$OUT_DIR/idrkd-taskbench-dpo.jsonl"

section "3. Show that training targets match TaskBench parser contract"
uv run python - <<'PY'
import json
import os
from pathlib import Path

from idrkd.evaluation.model_agent import parse_tool_call

out_dir = Path(os.environ["OUT_DIR"])
sft = json.loads((out_dir / "idrkd-sft.jsonl").read_text(encoding="utf-8").splitlines()[0])
dpo = json.loads((out_dir / "idrkd-dpo.jsonl").read_text(encoding="utf-8").splitlines()[0])
taskbench_sft = json.loads((out_dir / "idrkd-taskbench-sft.jsonl").read_text(encoding="utf-8").splitlines()[0])

for label, raw in (
    ("SFT assistant target", sft["messages"][2]["content"]),
    ("DPO chosen target", dpo["chosen"]),
    ("DPO rejected target", dpo["rejected"]),
    ("TaskBench SFT target", taskbench_sft["messages"][2]["content"]),
):
    print(f"{label}: {raw}")
    print(f"Parsed: {parse_tool_call(raw)}")
PY

section "4. Run dry-run SFT and DPO training summaries"
uv run python -m idrkd.distillation.cli train-sft \
  --dataset "$OUT_DIR/idrkd-sft.jsonl" \
  --out "$OUT_DIR/sft-dry-run" \
  --base-model local/tiny-student \
  --dry-run
uv run python -m idrkd.distillation.cli train-dpo \
  --dataset "$OUT_DIR/idrkd-dpo.jsonl" \
  --out "$OUT_DIR/dpo-dry-run" \
  --base-model local/tiny-student \
  --dry-run

section "5. Run MCP-TaskBench registry smoke evaluation"
uv run python -m idrkd.evaluation.cli \
  --mode registry-smoke \
  --tasks "$TASKS_PATH" \
  --include-synthetic-schemas \
  --out "$OUT_DIR/registry-smoke-summary.json"

section "6. Print demo artifacts"
uv run python - <<'PY'
import json
import os
from pathlib import Path

out_dir = Path(os.environ["OUT_DIR"])
summary = json.loads((out_dir / "registry-smoke-summary.json").read_text(encoding="utf-8"))
sft_summary = json.loads((out_dir / "sft-dry-run" / "sft-run-summary.json").read_text(encoding="utf-8"))
dpo_summary = json.loads((out_dir / "dpo-dry-run" / "dpo-run-summary.json").read_text(encoding="utf-8"))
taskbench_sft_count = len((out_dir / "idrkd-taskbench-sft.jsonl").read_text(encoding="utf-8").splitlines())
taskbench_dpo_count = len((out_dir / "idrkd-taskbench-dpo.jsonl").read_text(encoding="utf-8").splitlines())

print("Demo output directory:", out_dir)
print("SFT records:", sft_summary["record_count"])
print("DPO records:", dpo_summary["record_count"])
print("TaskBench SFT records:", taskbench_sft_count)
print("TaskBench DPO records:", taskbench_dpo_count)
print("TaskBench cases:", len(summary["cases"]))
print("TaskBench pass rate:", summary["pass_rate"])
print("TaskBench tool F1:", summary["tool_f1"])
print("TaskBench summary:", out_dir / "registry-smoke-summary.json")
PY

section "Demo complete"
