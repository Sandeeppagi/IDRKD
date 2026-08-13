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
uv run python -m idrkd.distillation.cli train-sft --dataset /private/tmp/idrkd-sft.jsonl --out /private/tmp/idrkd-sft-out --base-model local/tiny-student --dry-run
uv run python -m idrkd.distillation.cli train-dpo --dataset /private/tmp/idrkd-dpo.jsonl --out /private/tmp/idrkd-dpo-out --base-model local/tiny-student --dry-run
```

Run a bounded real laptop smoke train that writes PEFT adapters for both SFT
and DPO:

```bash
uv run python -m idrkd.distillation.cli local-smoke --base-model Qwen/Qwen2.5-0.5B-Instruct --max-steps 5 --max-seq-length 128 --device-map none
```

Production AWQ quantization is intended for a Linux/CUDA builder. AutoAWQ is
deprecated, so run Stage 12 from a separate pinned environment instead of
installing it into the working IDRKD `.venv`.

Create the legacy AWQ environment on the CUDA builder:

```bash
cd /workspace/IDRKD
uv venv /workspace/.venv-awq --python 3.11
source /workspace/.venv-awq/bin/activate
uv pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126
uv pip install "autoawq==0.2.9" "transformers==4.51.3" "peft>=0.11" "accelerate" "datasets>=2.20"
uv pip install --no-deps -e /workspace/IDRKD
```

Verify the isolated stack and calibration traces:

```bash
python - <<'PY'
import torch
import transformers
from awq import AutoAWQForCausalLM

print("Torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("Transformers:", transformers.__version__)
PY
wc -l /workspace/IDRKD/eval/distillation/frontier_admitted_teacher_traces.jsonl
```

Then quantize. The `--input-model` value may be a placeholder when `--adapter`
is supplied, because the CLI merges the base model and PEFT adapter before AWQ:

```bash
idrkd-quantize-awq \
  --input-model /workspace/IDRKD/models/checkpoints/merged-placeholder \
  --base-model microsoft/Phi-4-mini-instruct \
  --adapter /workspace/IDRKD/models/adapters/phi4-mini-dpo \
  --out /workspace/IDRKD/models/checkpoints/phi4-mini-awq \
  --model-id idrkd-phi4-mini-awq \
  --calibration /workspace/IDRKD/eval/distillation/frontier_admitted_teacher_traces.jsonl \
  --max-calibration-samples 128
ls -lah /workspace/IDRKD/models/checkpoints/phi4-mini-awq
cat /workspace/IDRKD/models/checkpoints/phi4-mini-awq/idrkd-model-manifest.json | jq .
deactivate
```

If AutoAWQ fails because Phi-4-mini is unsupported by the archived package, stop
there and leave the working training environment unchanged. The appropriate next
code change is to migrate quantization to AutoAWQ's recommended successor,
vLLM's `llm-compressor`.

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
uv run python -m idrkd.evaluation.cli --mode registry-smoke --tasks eval/taskbench/seed_tasks.jsonl --out /private/tmp/idrkd-registry-smoke.json
uv run python -m idrkd.evaluation.cli --mode student-agent --model-base-url http://localhost:11434/v1 --model-id qwen2.5:0.5b --tasks eval/taskbench/seed_tasks.jsonl --out /private/tmp/idrkd-student-agent.json
uv run python -m idrkd.evaluation.cli --mode teacher-agent --model-base-url http://localhost:11434/v1 --teacher-model-id qwen2.5:7b --tasks eval/taskbench/seed_tasks.jsonl --out /private/tmp/idrkd-teacher-agent.json
uv run python -m idrkd.evaluation.cli --mode ablation --ablation no_graph --model-base-url http://localhost:11434/v1 --model-id qwen2.5:0.5b --tasks eval/taskbench/seed_tasks.jsonl --out /private/tmp/idrkd-ablation-no-graph.json
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
