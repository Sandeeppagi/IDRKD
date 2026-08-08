# Week 7 Specs

Week 7 starts Pillar 5 SLM Distillation from the HLD/LLD W7-W8 plan:
teacher trace export, Phi-4-mini QLoRA SFT, first-pass BFCL evaluation,
DPO preference construction, AWQ artifact metadata, and vLLM serving.

## Spec Index

| Spec | Status | Primary Verification |
|---|---:|---|
| [SLM Distillation LLD](slm-distillation.spec.md) | Implemented | `tests/unit/test_week7_slm_distillation.py tests/unit/test_distillation_execution.py` |
