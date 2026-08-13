# Live Model Promotion

`idrkd-release` generates measured release evidence and a final promotion record. It never fills
faithfulness, security, or TTFT fields with defaults.

## 1. Capture the serving runtime

Run this inside the isolated vLLM environment:

```bash
source /workspace/.venv-vllm/bin/activate
python /workspace/IDRKD/scripts/capture_release_runtime.py \
  --out /workspace/release-evidence/runtime.json
```

This records the exact Python, vLLM, PyTorch, CUDA, and GPU versions used by the server.

## 2. Curate live repository cases

Start from `eval/rag/live_repository_queries.example.jsonl`. Each line requires `id`, `query`,
`tenant_id`, and `repo_id`. Add stable `expected_entity_ids` after inspecting the ingested graph;
these produce retrieval recall evidence but do not replace the NLI faithfulness gate.

The repositories in the case file must already be ingested into Neo4j and pgvector under the same
tenant and repository IDs. The vLLM OpenAI-compatible server must be listening before the run.

For a native Runpod installation, ingest the committed five-repository corpus after PostgreSQL,
pgvector, and Neo4j are running. Stop vLLM first if it owns most GPU memory, then run:

```bash
idrkd-ingest-corpus \
  --tenant-id default \
  --postgres-dsn "${POSTGRES_DSN:-postgresql://idrkd:idrkd@localhost:5432/idrkd}" \
  --neo4j-uri "${NEO4J_URI:-bolt://localhost:7687}" \
  --neo4j-user "${NEO4J_USER:-neo4j}" \
  --neo4j-password "${NEO4J_PASSWORD:-change-me}" \
  --embedding-device cuda \
  --fetch-missing \
  --out /workspace/release-evidence/corpus-ingestion.json
```

The command is idempotent and captures individual parser failures instead of discarding completed
repositories. Use repeated `--repo-id` options to ingest or retry a subset. At release time the RAG
command defaults query embeddings and reranking to CPU so vLLM can retain GPU memory; set
`--embedding-device cuda` or `--reranker-device cuda` only when sufficient GPU memory remains.

## 3. Run all release gates

Run this from the normal IDRKD environment, which owns the database, sentence-transformer, NLI, and
test dependencies:

```bash
cd /workspace/IDRKD
source .venv/bin/activate

idrkd-release run \
  --model-base-url http://127.0.0.1:8000/v1 \
  --model-id idrkd-phi4-mini-dpo-tooljson-split-v2-llmc-awq \
  --rag-cases eval/rag/live_repository_queries.jsonl \
  --postgres-dsn "${POSTGRES_DSN:-postgresql://idrkd:idrkd@localhost:5432/idrkd}" \
  --neo4j-uri "${NEO4J_URI:-bolt://localhost:7687}" \
  --neo4j-user "${NEO4J_USER:-neo4j}" \
  --neo4j-password "${NEO4J_PASSWORD:-change-me}" \
  --runtime /workspace/release-evidence/runtime.json \
  --holdout eval/distillation/llmc-awq-holdout.json \
  --checkpoint artifacts/models/checkpoints/phi4-mini-dpo-tooljson-split-v2-llmc-awq \
  --performance-samples 20 \
  --performance-warmups 2 \
  --out-dir /workspace/release-evidence/final
```

The command writes `taskbench-all.json`, `live-rag.json`, `streaming-performance.json`,
`security.json`, and `promotion-record.json`. The full TaskBench artifact always uses `split=all`
and includes all 360 seed cases plus the 80 synthetic schema and conflict cases. It is a live-model
tool-call conformance regression with fixture execution and has `generalization_claim=false`. The
89-case prompt-group holdout is the internal generalisation gate. Neither run is official BFCL.
The record is promoted only when:

- all expected 89 holdout calls conform, argument accuracy is 1.0, and tool F1 is at least 0.82;
- all expected 440 full TaskBench calls complete without harness/model errors, tool F1 is at
  least 0.82, and semantic outcome rate is inspected separately from JSON-RPC transport success;
- every live RAG case executes, at least one case has non-empty expected entity IDs for retrieval
  recall, and its transformer-NLI score is at least 0.78;
- both tenant and agent security suites pass;
- every streaming sample succeeds, p95 TTFT is at most 1.2 seconds, and p95 completion latency is
  at most 8 seconds;
- runtime versions, manifest digest, artifact Git commit, and checkpoint Git LFS OIDs are present.

The record includes the path, size, and SHA-256 digest of every input evidence file. Its final
`record_digest` is SHA-256 over the canonical promotion record before that digest field is added. A
rejected run still writes all evidence and explicit rejection reasons.

## 4. Run only the full TaskBench gate

When the other release evidence is already valid, rerun only the deployment-level no-split suite:

```bash
cd /workspace/IDRKD
source .venv/bin/activate

idrkd-release taskbench \
  --model-base-url http://127.0.0.1:8000/v1 \
  --model-id idrkd-phi4-mini-dpo-tooljson-split-v2-llmc-awq \
  --out /workspace/release-evidence/final-v5/taskbench-all.json
```

Verify the complete result before regenerating the promotion record:

```bash
jq '{split, evaluation_scope, generalization_claim, case_count, error_count, pass_rate, tool_call_pass_rate, semantic_outcome_rate, schema_valid_rate, tool_f1, argument_accuracy, by_category}' \
  /workspace/release-evidence/final-v5/taskbench-all.json
```

The expected `split` is `all` and the expected `case_count` is `440`.

Bind the new artifact to an existing evidence set with the `record` command's `--taskbench`
argument. This produces a schema-v2 promotion record and includes the TaskBench artifact's path,
size, and SHA-256 digest:

```bash
idrkd-release record \
  --repo-root /workspace/IDRKD \
  --checkpoint artifacts/models/checkpoints/phi4-mini-dpo-tooljson-split-v2-llmc-awq \
  --runtime /workspace/release-evidence/runtime.json \
  --holdout eval/distillation/llmc-awq-holdout.json \
  --taskbench /workspace/release-evidence/final-v5/taskbench-all.json \
  --rag /workspace/release-evidence/final-v3/live-rag.json \
  --performance /workspace/release-evidence/final-v4/streaming-performance.json \
  --security /workspace/release-evidence/final-v3/security.json \
  --out /workspace/release-evidence/final-v5/promotion-record.json
```
