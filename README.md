# IDRKD - Intelligent Data Reconciliation and Knowledge Discovery

IDRKD is a capstone research project for building an on-premise, MCP-native knowledge discovery system over technical enterprise assets such as source code, schemas, documents, and API specifications.

The project combines structural ingestion, a Neo4j knowledge graph, pgvector-backed retrieval, LangGraph/AutoGen orchestration, a distilled Phi-4-mini student model, and drift-triggered re-indexing. The current repository is scaffolded around the HLD/LLD v3 design and the May-July 2026 completion plan.

## Current Artifacts

- `docs/design/IDRKD_HLD_LLD_v3_updated.docx` - updated high-level and low-level design document.
- `docs/project-plan/IDRKD_Project_Plan_v2_updated.html` - updated project completion plan, dated 16 May to 5 July 2026.
- `tools/` - helper scripts used to regenerate or update the planning/design documents.

## Repository Layout

```text
.
├── configs/                 # Application, model, eval, and environment config templates
├── data/                    # Local data only; raw and processed directories are git-ignored
├── docker/                  # Docker Compose and container build assets
├── docs/
│   ├── design/              # HLD/LLD, ADRs, diagrams, threat model
│   ├── project-plan/        # Timeline, milestone, and project tracking artifacts
│   └── references/          # Research notes and reference bibliography material
├── eval/
│   ├── fixtures/            # Small synthetic corpora for deterministic evaluation
│   └── taskbench/           # MCP-TaskBench tasks, graders, and manifests
├── k8s/                     # Kubernetes deployment manifests for stretch deployment
├── models/                  # Local model outputs only; checkpoints and adapters are git-ignored
├── notebooks/               # Exploratory analysis notebooks
├── scripts/                 # Operational scripts and one-off repo automation
├── src/idrkd/
│   ├── a2a/                 # Agent-to-Agent bridge and AutoGen integration
│   ├── common/              # Shared schemas, settings, IDs, logging, and utilities
│   ├── distillation/        # Teacher traces, QLoRA, DPO, quantisation, publishing
│   ├── drift/               # Entity and cluster drift scoring plus re-index orchestration
│   ├── graph/               # Neo4j schema, graph writes, graph analytics, traversal
│   ├── ingestion/           # Webhook, parser workers, Tree-sitter, document/schema extraction
│   ├── mcp/                 # MCP gateway, tool catalog, JSON-RPC contracts
│   ├── observability/       # Metrics, traces, dashboards, runbooks
│   ├── rag/                 # Hybrid vector/graph retrieval, reranking, critic loop, synthesis
│   └── security/            # Threat-model controls, tenancy, prompt-injection containment
└── tests/
    ├── evaluation/          # Benchmark and metric tests
    ├── integration/         # Cross-service tests
    └── unit/                # Component-level tests
```

## Planned Milestones

1. Foundation and ingestion MVP: Docker stack, Tree-sitter parsing, Neo4j writes, pgvector setup, and OTel tracing.
2. Knowledge graph and agentic RAG skeleton: hybrid retrieval, graph traversal, reranking, and bounded critic loop.
3. MCP and A2A orchestration: MCP tool catalog, A2A bridge, security gates, and MCP-TaskBench seed tasks.
4. SLM distillation: teacher traces, Phi-4-mini QLoRA, DPO alignment, quantisation, and vLLM serving.
5. Drift detection and re-indexing: entity-level cosine drift, cluster centroid drift, Celery orchestration, and SLO dashboards.
6. Evaluation and viva wrap: MCP-TaskBench expansion, ablations, reproducibility manifest, and presentation material.

## Implementation Status Through 3 August 2026

Implemented foundation, June 26 MVP, Week 4 agentic RAG, and Week 5 MCP/A2A/security pieces:

- Docker Compose stack for Neo4j, PostgreSQL + pgvector/HNSW, Kafka, Redis, MinIO, Prometheus, Grafana, OTel Collector, and the standalone MCP FastAPI server.
- Graphify Docker bootstrap against a public Telstra Python repo with an IDRKD Neo4j importer bridge.
- Stable SHA-256 content hashing and deterministic entity/relation IDs.
- Tree-sitter Python and JavaScript extraction into typed records.
- JSON/CSV schema extraction and Markdown/document NER with optional transformer pipeline.
- Commit webhook serialization, Kafka event contracts, and commit-event parser pipeline with automatic entity embeddings into pgvector.
- OpenTelemetry correlation helper, ingestion SLO gate, and Lamport clock stamping.
- Neo4j typed labels, temporal properties, tenant scoping, and idempotent graph writer.
- BGE-M3 embedding adapter with optional real model inference, automatic pgvector embedding upserts, pgvector search SQL, hybrid RRF retrieval, MiniLM reranker with optional cross-encoder inference, and graph analytics primitives.
- MCP JSON-RPC 2.0 tool registry with the 14-tool W5-W6 suite, each tool backed by a real Pydantic request model (not a hand-rolled schema), concrete default handlers, and a Dockerized FastAPI MCP server exposing `/mcp` and `/healthz`.
- Neo4j/pgvector-backed MCP business tools for keyword/semantic search, entity reads, schema diff, impact analysis, salience, centroid drift, and stale entity listing, plus Redis-persistent re-index queue and conflict-resolution state when `REDIS_URL` is configured.
- A2A bridge rebuilt on the official `a2a-sdk` v1.0 package (real `AgentCard`/`AgentExecutor`/JSON-RPC server and client, legacy 0.3 compatibility), with this repo's own tested HMAC agent-card signing preserved underneath.
- A2A cancellation state store with submitted/running cancellation, running-task tracking, and terminal-state rejection.
- Real, read-only Cypher for BFS neighbourhood expand, shortestPath, and community subgraph, wired into the `graph_bfs`/`graph_path`/`get_community` MCP tools.
- Security gates: a 5-layer prompt-injection containment chain (marker denylist, role-impersonation detection, read-only Cypher, Cypher-escape hardening, output-side quarantine/secret-leakage scan), plus an mTLS transport config contract for A2A calls.
- STRIDE threat model covering 6 trust boundaries (`docs/design/threat-model.md`).
- 5-agent LangGraph-style RAG orchestrator (Router, VectorRetriever, GraphTraversal, Synthesis, Critic) with a bounded, critic-gated re-retrieve loop.
- DeBERTa-v3-large NLI faithfulness critic with optional transformer pipeline gating synthesis output against an AlignScore-style threshold.
- Query-path SLO gate (p50/p95 latency, bounded re-retrieve rounds) and HotpotQA-lite top-10 recall helper.
- Pillar 5 SLM distillation LLD contracts: tenant-scoped teacher traces, SFT record shaping, Phi-4-mini QLoRA defaults, DPO preference pairs, BFCL/AlignScore/TTFT gates, AWQ artifact manifests, and vLLM serving config.
- Drift/re-index workers: entity cosine drift over pgvector, community centroid drift over Neo4j communities, Redis-compatible re-index queue, and bounded graph-neighbourhood re-embedding.
- Evaluation harness: MCP-TaskBench JSONL tasks, BFCL-style function-call metrics, JSON-RPC registry execution, category pass rates, and CLI summary output.
- Prometheus and OpenTelemetry instrumentation across MCP tool execution, graph traversal, pgvector search/upserts, ingestion embedding upserts, and drift/re-index workers.
- Specs for Week 1 through Week 5 plus Week 7 SLM distillation, Week 8 drift workers, and Week 9 evaluation harness under `specs/`.

Current local verification:

```bash
uv run pytest tests/unit
uv run ruff check src tests/unit
uv run mypy src
```

## Development Setup

This repository uses `uv` for Python environment and dependency management.

```bash
uv sync --group dev
uv run pytest
uv run ruff check .
uv run mypy src
```

For model training/distillation dependencies:

```bash
uv sync --group dev --extra ml
```

Build and smoke-run the distillation execution path:

```bash
uv run python -m idrkd.distillation.cli build-sft --traces eval/distillation/seed_teacher_traces.jsonl --out /private/tmp/idrkd-sft.jsonl
uv run python -m idrkd.distillation.cli build-dpo --traces eval/distillation/seed_teacher_traces.jsonl --out /private/tmp/idrkd-dpo.jsonl
uv run python -m idrkd.distillation.cli build-taskbench-sft --include-synthetic-schemas --split train --out /private/tmp/idrkd-taskbench-sft.jsonl
uv run python -m idrkd.distillation.cli build-taskbench-dpo --include-synthetic-schemas --split train --out /private/tmp/idrkd-taskbench-dpo.jsonl
uv run python -m idrkd.distillation.cli train-sft --dataset /private/tmp/idrkd-sft.jsonl --out /private/tmp/idrkd-sft-out --base-model local/tiny-student --dry-run
uv run python -m idrkd.distillation.cli train-dpo --dataset /private/tmp/idrkd-dpo.jsonl --out /private/tmp/idrkd-dpo-out --base-model local/tiny-student --dry-run
```

Run a bounded real laptop smoke train that writes PEFT adapters for both SFT
and DPO:

```bash
uv run python -m idrkd.distillation.cli local-smoke --base-model Qwen/Qwen2.5-0.5B-Instruct --max-steps 5 --max-seq-length 128 --device-map none
```

For Phi-4-mini tool-call SFT smoke runs, start from a clean adapter directory
after any LoRA target-module change. The project targets Phi's fused modules
(`qkv_proj`, `o_proj`, `gate_up_proj`, `down_proj`), not Llama-style split
projection names:

```bash
rm -rf models/adapters/phi4-mini-sft-tooljson-smoke
uv run python -m idrkd.distillation.cli build-taskbench-sft \
  --include-synthetic-schemas \
  --split train \
  --out artifacts/datasets/idrkd-taskbench-tooljson-v2.jsonl
uv run python -m idrkd.distillation.cli train-sft \
  --dataset artifacts/datasets/idrkd-taskbench-tooljson-v2.jsonl \
  --out models/adapters/phi4-mini-sft-tooljson-smoke \
  --base-model microsoft/Phi-4-mini-instruct \
  --max-steps 20 \
  --max-seq-length 4096 \
  --learning-rate 2e-4 \
  --batch-size 1 \
  --gradient-accumulation-steps 4 \
  --use-4bit
uv run python scripts/probe_taskbench_adapter.py \
  --adapter models/adapters/phi4-mini-sft-tooljson-smoke \
  --base-model microsoft/Phi-4-mini-instruct \
  --include-synthetic-schemas \
  --split holdout \
  --limit 5 \
  --use-4bit
```

TaskBench datasets use a deterministic, tool-stratified 80/20 split. Repeated
natural-language prompts remain together, so equivalent wording cannot leak
between training and holdout. With synthetic schemas enabled, the current split
contains 351 training records and 89 holdout records. The synthetic conflict
tasks also balance `reconcile` and `get_conflict` in both partitions.

Use only the training partition for both SFT and DPO, then evaluate the adapters
against the matching holdout partition and seed:

```bash
uv run python -m idrkd.distillation.cli build-taskbench-dpo \
  --include-synthetic-schemas \
  --split train \
  --split-seed 17 \
  --out artifacts/datasets/idrkd-taskbench-dpo-hardneg-train-v2.jsonl
uv run python scripts/probe_taskbench_adapter.py \
  --adapter models/adapters/phi4-mini-dpo-tooljson-smoke \
  --base-model microsoft/Phi-4-mini-instruct \
  --include-synthetic-schemas \
  --split holdout \
  --split-seed 17 \
  --limit 1000 \
  --use-4bit
```

Adapters trained from the former unsplit dataset must be retrained before their
holdout score is valid, including the SFT adapter used to initialise DPO.

Production AWQ quantization uses
[vLLM llm-compressor](https://github.com/vllm-project/llm-compressor) and writes
a compressed-tensors checkpoint for vLLM. Run Stage 12 from a separate
Linux/CUDA environment instead of changing the working IDRKD `.venv`.

Create the quantization environment on the CUDA builder:

```bash
cd /workspace/IDRKD
uv run python -m idrkd.distillation.cli build-taskbench-sft \
  --include-synthetic-schemas \
  --split train \
  --split-seed 17 \
  --out artifacts/datasets/idrkd-taskbench-sft-train-v3.jsonl
uv venv /workspace/.venv-quantization --python 3.11
source /workspace/.venv-quantization/bin/activate
uv pip install llmcompressor peft accelerate datasets
uv pip install --no-deps -e /workspace/IDRKD
```

Do not install vLLM into this environment. Quantization uses llm-compressor's
Transformers-based `oneshot` entry point and does not require vLLM. Use a
separate compatible vLLM environment or the `vllm/vllm-openai` container for
the post-quantization release gate.

Verify the isolated stack and the representative SFT calibration records:

```bash
python - <<'PY'
from importlib.metadata import version

import torch
import transformers
import llmcompressor

print("Torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("Transformers:", transformers.__version__)
print("llm-compressor:", version("llmcompressor"))
PY
wc -l /workspace/IDRKD/artifacts/datasets/idrkd-taskbench-sft-train-v3.jsonl
```

Then quantize. The `--input-model` value may be a placeholder when `--adapter`
is supplied, because the CLI merges the base model and PEFT adapter first. The
calibration loader applies the model chat template to SFT `messages` records:

```bash
TOKENIZERS_PARALLELISM=false \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
idrkd-quantize \
  --input-model /workspace/IDRKD/models/checkpoints/merged-placeholder \
  --base-model microsoft/Phi-4-mini-instruct \
  --adapter /workspace/IDRKD/models/adapters/phi4-mini-dpo-tooljson-split-v2 \
  --out /workspace/IDRKD/models/checkpoints/phi4-mini-dpo-tooljson-split-v2-llmc-awq \
  --model-id idrkd-phi4-mini-dpo-tooljson-split-v2-llmc-awq \
  --calibration /workspace/IDRKD/artifacts/datasets/idrkd-taskbench-sft-train-v3.jsonl \
  --max-calibration-samples 128 \
  --max-sequence-length 4096 \
  --sequential-target Linear
```

`Linear` sequential targets keep AWQ calibration bounded to one dense module at
a time. The expandable CUDA allocator also reduces fragmentation during AWQ
smoothing on 48 GB builders.

Inspect the checkpoint and manifest:

```bash
ls -lah /workspace/IDRKD/models/checkpoints/phi4-mini-dpo-tooljson-split-v2-llmc-awq
jq . /workspace/IDRKD/models/checkpoints/phi4-mini-dpo-tooljson-split-v2-llmc-awq/idrkd-model-manifest.json
```

Serve the compressed checkpoint with vLLM:

```bash
vllm serve \
  /workspace/IDRKD/models/checkpoints/phi4-mini-dpo-tooljson-split-v2-llmc-awq \
  --served-model-name idrkd-phi4-mini-dpo-tooljson-split-v2-llmc-awq \
  --quantization compressed-tensors \
  --host 0.0.0.0 \
  --port 8000
```

In another terminal, run the same deterministic 89-case holdout gate used for
the adapters. `pass_rate`, `tool_f1`, and `argument_accuracy` must all remain
`1.0` before promotion:

```bash
cd /workspace/IDRKD
uv run python -m idrkd.evaluation.cli \
  --mode student-agent \
  --model-base-url http://127.0.0.1:8000/v1 \
  --model-id idrkd-phi4-mini-dpo-tooljson-split-v2-llmc-awq \
  --include-synthetic-schemas \
  --split holdout \
  --split-seed 17 \
  --out /workspace/llmc-awq-holdout.json
jq '{cases: (.cases | length), pass_rate, tool_f1, argument_accuracy}' \
  /workspace/llmc-awq-holdout.json
```

Stop vLLM and run `deactivate` when the release gate finishes.

`idrkd-quantize-awq` remains available as a compatibility alias. Both commands
now use llm-compressor; AutoAWQ is no longer imported or supported.

The expected local service stack is:

- Python 3.11 or newer
- Docker or Docker Desktop
- Neo4j Community
- PostgreSQL with pgvector
- Kafka
- Redis
- MinIO
- Prometheus, Grafana, and OpenTelemetry Collector

Run Python commands through `uv run` so they use the locked project environment rather than the system Python.

Run the Dockerized MCP server with the local stack:

```bash
docker compose -f docker/docker-compose.yml up -d --build mcp-server
```

Serve the quantized student model through a CUDA/vLLM OpenAI-compatible API:

```bash
docker compose -f docker/docker-compose.yml --profile slm-server up -d slm-server
```

On a laptop or Mac, use the Ollama fallback profile and configure IDRKD to call
Ollama's OpenAI-compatible endpoint:

```bash
docker compose -f docker/docker-compose.yml --profile slm-server-laptop up -d slm-server-laptop
export IDRKD_STUDENT_MODEL_BASE_URL=http://localhost:11434/v1
export IDRKD_STUDENT_MODEL_ID=idrkd-student
```

The service is published on `http://localhost:8080`, uses Redis keys
`idrkd:mcp:reindex` and `idrkd:mcp:conflict:<conflict_id>` for persistent
queue/conflict state, and connects to the Compose Neo4j/PostgreSQL services by
default. Compose reads `TENANT_ID` and `REPO_ID`, defaulting to the live smoke
data scope `tenant-live` / `week5-e2e`.

Run the live tenant-aligned MCP smoke after the stack is up:

```bash
uv run idrkd-mcp-smoke --tenant-id tenant-live --repo-id week5-e2e
```

That smoke verifies `tools/list`, `search_code`, `graph_bfs`, and `graph_path`
against the running MCP server and live Neo4j/pgvector data.

Run benchmark modes:

```bash
uv run python -m idrkd.evaluation.cli --mode registry-smoke --split all --tasks eval/taskbench/seed_tasks.jsonl --out /private/tmp/idrkd-registry-smoke.json
uv run python -m idrkd.evaluation.cli --mode student-agent --split holdout --model-base-url http://localhost:11434/v1 --model-id qwen2.5:0.5b --tasks eval/taskbench/seed_tasks.jsonl --out /private/tmp/idrkd-student-agent.json
uv run python -m idrkd.evaluation.cli --mode teacher-agent --split holdout --model-base-url http://localhost:11434/v1 --teacher-model-id qwen2.5:7b --tasks eval/taskbench/seed_tasks.jsonl --out /private/tmp/idrkd-teacher-agent.json
uv run python -m idrkd.evaluation.cli --mode ablation --split holdout --ablation no_graph --model-base-url http://localhost:11434/v1 --model-id qwen2.5:0.5b --tasks eval/taskbench/seed_tasks.jsonl --out /private/tmp/idrkd-ablation-no-graph.json
```

Run the Dockerized re-index worker and verify MCP enqueue consumption:

```bash
docker compose -f docker/docker-compose.yml up -d --build reindex-worker
uv run idrkd-drift-worker live-smoke --external-worker --mcp-base-url http://localhost:8080 --tenant-id tenant-live --repo-id week5-e2e
```

The `reindex-worker` consumes Redis queue `idrkd:mcp:reindex`, re-embeds the
requested graph neighbourhood into pgvector, and clears Neo4j `stale` flags.
The optional `drift-worker` profile consumes `idrkd:reindex` for drift-generated
requests.

Scrape runtime metrics:

```bash
curl http://localhost:8080/metrics
curl http://localhost:9101/metrics
```

Prometheus is configured to scrape `mcp-server:8080`, `reindex-worker:9101`,
and profile-based `drift-worker:9102`. The exported IDRKD metric families are
`idrkd_mcp_tool_calls_total`, `idrkd_mcp_tool_latency_seconds`,
`idrkd_reindex_jobs_total`, `idrkd_embedding_upserts_total`, and
`idrkd_graph_traversal_latency_seconds`.

Run repeatable Graphify bootstrap on demand:

```bash
docker compose --profile graphify up graphify
```

The profile aligns `TENANT_ID`, `REPO_ID`, and `SOURCE_LABEL`, imports the
generated `graph.json` into Neo4j, and then smoke-verifies imported nodes,
relationships, and orphan relationship count. Graphify stays profile-based; it
does not need to run continuously for production unless you want continuous repo
ingestion.

Current local verification:

```bash
uv run python -m compileall -q src tests
uv run pytest tests/unit
```

## Reference Materials

The expanded project plan includes references for:

- Tree-sitter, Graphify, Neo4j Graph Data Science, Kafka, and OpenTelemetry
- Model Context Protocol and Agent-to-Agent Protocol v1.0
- LangGraph, AutoGen, pgvector, BGE-M3, CRAG, and RAG literature
- Phi-4-mini, QLoRA, DPO, AWQ, vLLM, and PEFT
- BFCL, AgentBench, AlignScore, DeBERTa, bootstrap confidence intervals, and Wilcoxon signed-rank testing

## GitHub Remote

The local repository remote is configured as:

```text
origin https://github.com/Sandeeppagi/IDRKD.git
```
