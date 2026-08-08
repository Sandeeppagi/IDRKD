# Spec: Evaluation Harness

## Goal

Replace empty evaluation folders with a runnable MCP-TaskBench/BFCL-style
evaluation harness that exercises the actual IDRKD MCP registry.

## Contract

- `eval/taskbench/seed_tasks.jsonl` stores JSONL `McpTask` records with
  category, prompt, expected tool, expected arguments, and expected result keys.
- `TaskBenchRunner.run(...)` executes each task as real JSON-RPC
  `tools/list` + `tools/call` requests against `McpToolRegistry`.
- Each case records tool correctness, argument correctness, schema validity,
  execution success, expected result-key presence, and error text.
- `EvalSummary` reports pass rate, schema-valid rate, category pass rates,
  and BFCL-style tool precision/recall/F1/argument accuracy.
- `python -m idrkd.evaluation.cli --tasks ... --out ...` runs the harness and
  writes a JSON summary.

## Implementation

- `src/idrkd/evaluation/__init__.py`
- `src/idrkd/evaluation/bfcl.py`
- `src/idrkd/evaluation/taskbench.py`
- `src/idrkd/evaluation/cli.py`
- `eval/taskbench/seed_tasks.jsonl`
- `tests/evaluation/test_taskbench_harness.py`
- `tests/unit/test_evaluation_imports.py`

## Acceptance Criteria

- Seed tasks load from JSONL through the Pydantic `McpTask` model.
- The runner executes tasks through the real registry, not a mock evaluator.
- Tool F1 and argument accuracy are computed using BFCL-style matching.
- Case-level pass/fail includes JSON Schema exposure and expected result keys.
- CLI writes a summary JSON file.

## Verification

```bash
uv run pytest tests/evaluation/test_taskbench_harness.py tests/unit/test_evaluation_imports.py
uv run python -m idrkd.evaluation.cli --tasks eval/taskbench/seed_tasks.jsonl --out /tmp/idrkd-eval-summary.json
```
