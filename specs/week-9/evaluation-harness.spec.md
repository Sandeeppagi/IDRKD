# Spec: Evaluation Harness

## Goal

Replace empty evaluation folders with a runnable MCP-TaskBench/BFCL-style
evaluation harness that can run deterministic registry smoke tests or prompt
real model agents to choose and execute MCP tools.

## Contract

- `eval/taskbench/seed_tasks.jsonl` stores JSONL `McpTask` records with
  category, prompt, expected tool, expected arguments, and expected result keys.
- `TaskBenchRunner.run(..., mode="registry-smoke")` preserves deterministic
  JSON-RPC `tools/list` + expected `tools/call` execution against
  `McpToolRegistry`.
- `student-agent` and `teacher-agent` modes prompt an OpenAI-compatible model
  with MCP tool schemas, parse the selected tool call, execute the selected
  tool, and compare model-selected tool/arguments against expected values.
- `ablation` mode uses the same model-agent path with schema/tool prompt
  changes such as `no_graph`.
- Each case records tool correctness, argument correctness, schema validity,
  execution success, expected result-key presence, raw model output, parsed
  tool call, execution result, latency, and error text.
- `EvalSummary` reports pass rate, schema-valid rate, category pass rates,
  benchmark mode, ablations, and BFCL-style tool precision/recall/F1/argument
  accuracy.
- `evaluate_promotion(...)` gates student promotion on tool F1, faithfulness,
  tenant/security test status, latency/TTFT, and acceptable regression versus a
  previous artifact.
- `python -m idrkd.evaluation.cli --tasks ... --out ...` runs the harness and
  writes a JSON summary.

## Implementation

- `src/idrkd/evaluation/__init__.py`
- `src/idrkd/evaluation/bfcl.py`
- `src/idrkd/evaluation/model_agent.py`
- `src/idrkd/evaluation/promotion.py`
- `src/idrkd/evaluation/taskbench.py`
- `src/idrkd/evaluation/cli.py`
- `eval/taskbench/seed_tasks.jsonl`
- `tests/evaluation/test_taskbench_harness.py`
- `tests/unit/test_evaluation_imports.py`

## Acceptance Criteria

- Seed tasks load from JSONL through the Pydantic `McpTask` model.
- `registry-smoke` executes tasks through the real registry, not a mock
  evaluator.
- Model-agent modes do not use expected calls as predictions; they store raw
  model output, parsed selected call, tool execution result, and latency.
- Tool F1 and argument accuracy are computed using BFCL-style matching.
- Case-level pass/fail includes JSON Schema exposure and expected result keys.
- CLI supports `registry-smoke`, `student-agent`, `teacher-agent`, and
  `ablation` modes and writes a summary JSON file.
- Promotion gate blocks artifacts that miss BFCL/tool F1, faithfulness,
  tenant/security, latency/TTFT, or regression thresholds.

## Verification

```bash
uv run pytest tests/evaluation/test_taskbench_harness.py tests/unit/test_evaluation_imports.py
uv run python -m idrkd.evaluation.cli --tasks eval/taskbench/seed_tasks.jsonl --out /tmp/idrkd-eval-summary.json
uv run python -m idrkd.evaluation.cli --mode student-agent --model-base-url http://localhost:11434/v1 --model-id qwen2.5:0.5b --tasks eval/taskbench/seed_tasks.jsonl --out /tmp/idrkd-student-agent-summary.json
uv run python -m idrkd.evaluation.cli --mode ablation --ablation no_graph --model-base-url http://localhost:11434/v1 --model-id qwen2.5:0.5b --tasks eval/taskbench/seed_tasks.jsonl --out /tmp/idrkd-ablation-summary.json
```
