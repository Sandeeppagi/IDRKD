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

## Implementation Status Through Week 3

Implemented foundation and Week 2-3 MVP pieces:

- Docker Compose stack for Neo4j, PostgreSQL + pgvector/HNSW, Kafka, Redis, MinIO, Prometheus, Grafana, and OTel Collector.
- Graphify Docker bootstrap against a public Telstra Python repo with an IDRKD Neo4j importer bridge.
- Stable SHA-256 content hashing and deterministic entity/relation IDs.
- Tree-sitter Python and JavaScript extraction into typed records.
- JSON/CSV schema extraction and Markdown/document NER facade.
- Commit webhook serialization, Kafka event contracts, and commit-event parser pipeline.
- OpenTelemetry correlation helper, ingestion SLO gate, and Lamport clock stamping.
- Neo4j typed labels, temporal properties, tenant scoping, and idempotent graph writer.
- BGE-M3 embedding adapter contract, pgvector search SQL, hybrid RRF retrieval, MiniLM reranker facade, and graph analytics primitives.
- Specs for Week 1, Week 2, and Week 3 under `specs/`.

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
