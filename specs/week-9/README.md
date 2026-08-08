# Week 9 Specs

Week 9 implements the evaluation harness:

- MCP-TaskBench JSONL task loading.
- JSON-RPC execution against the real `McpToolRegistry`.
- True model-agent tool selection for student, teacher, and ablation modes.
- Raw model output, parsed tool call, execution result, and latency capture.
- Tool-selection, schema-conformance, execution, argument, and result-key
  grading.
- BFCL-style function-call precision/recall/F1 primitives.
- Promotion gates for tool F1, faithfulness, tenant/security, latency/TTFT, and
  artifact regression.

## Spec Index

| Spec | Status | Primary Verification |
|---|---:|---|
| [Evaluation Harness](evaluation-harness.spec.md) | Implemented | `tests/evaluation/test_taskbench_harness.py` |
