# IDRKD Threat Model (STRIDE)

Week 5 companion to the HLD/LLD (`docs/design/IDRKD_HLD_LLD_v3_updated.docx`).
Six trust boundaries are identified below; each is STRIDE-categorized with
the concrete control that mitigates it and the code that implements the
control. This document tracks the code — when a control changes, update
this file in the same change.

## 1. Client/User <-> MCP Gateway

The boundary where an external caller (human, agent, or another service)
submits a JSON-RPC `tools/call` request.

| STRIDE | Risk | Control | Implementation |
|---|---|---|---|
| Spoofing | Caller claims a `tenant_id` it doesn't own | `validate_tenant_scope` rejects any request where `tenant_id != principal_tenant_id` | `src/idrkd/security/gates.py` |
| Tampering | Caller sends unexpected/extra arguments to smuggle a differently-shaped payload | Per-tool Pydantic models with `extra="forbid"`; every field validated by type | `src/idrkd/mcp/tools.py` (`ToolParams` subclasses) |
| Repudiation | No record of which tool was called with what arguments | OTel `traced_span` correlation IDs on ingestion; A2A tasks carry `traceparent` end to end | `src/idrkd/observability/tracing.py`, `src/idrkd/a2a/bridge.py` |
| Information Disclosure | Tool schema leaks internal-only fields | JSON Schema is generated from the same Pydantic model that validates input — no separate, driftable schema | `src/idrkd/mcp/tools.py` (`ToolDefinition.schema`) |
| Denial of Service | Unbounded `limit`/`depth` arguments trigger an expensive graph scan | `GraphBfsParams.depth`/`limit`, `GraphPathParams.max_hops` are bounded, defaulted fields validated before any Cypher runs | `src/idrkd/mcp/tools.py`, `src/idrkd/graph/cypher.py` |
| Elevation of Privilege | Caller-supplied keys reach a handler outside its declared contract | Layer 5 of prompt-injection containment: strict Pydantic allowlisting (`validate_tool_argument_allowlist` as the reusable primitive) | `src/idrkd/security/gates.py` |

## 2. MCP Gateway <-> Neo4j Cypher execution

| STRIDE | Risk | Control | Implementation |
|---|---|---|---|
| Tampering | A write/delete statement reaches Neo4j through a "read" tool | `validate_read_only_cypher` denylists `CREATE/DELETE/DETACH/DROP/LOAD CSV/MERGE/REMOVE/SET` and multi-statement queries | `src/idrkd/security/gates.py` |
| Tampering | Cypher injection via string-concatenated values | Cypher-escape hardening: every Week 5 traversal template binds dynamic values only through `$param`; `assert_no_inline_literals` rejects any query containing a quoted string literal | `src/idrkd/graph/cypher.py`, `src/idrkd/security/gates.py` |
| Denial of Service | Unbounded variable-length path expansion | BFS/shortestPath range bounds (`depth`, `max_hops`) are validated positive integers baked into the query template, never left unbounded | `src/idrkd/graph/cypher.py` (`bfs_neighbors_query`, `shortest_path_query`) |
| Information Disclosure | A traversal crosses tenant/repo boundaries | Every traversal query filters on `tenant_id`/`repo_id` at each `MATCH` clause, not just the seed node | `src/idrkd/graph/cypher.py` |

## 3. MCP Gateway <-> pgvector/Postgres

| STRIDE | Risk | Control | Implementation |
|---|---|---|---|
| Tampering | SQL injection via query text | `PGVECTOR_SEARCH_SQL` is a static parameterized statement; embeddings and IDs are always bound, never interpolated | `src/idrkd/rag/vector_store.py` |
| Information Disclosure | A search returns another tenant's vectors | `WHERE tenant_id = %(tenant_id)s AND repo_id = %(repo_id)s` scopes every search | `src/idrkd/rag/vector_store.py` |

## 4. Agent <-> Agent via A2A

The boundary between two independently-deployed IDRKD (or peer) agent
processes exchanging tasks over the network.

| STRIDE | Risk | Control | Implementation |
|---|---|---|---|
| Spoofing | A malicious process presents itself as a trusted peer agent | Agent cards are HMAC-signed over their canonical JSON payload with a shared secret; `verify_remote_card` rejects any tampered or unsigned card | `src/idrkd/a2a/bridge.py` (`sign_agent_card`, `SignedAgentCard.verify`) |
| Tampering | A message is altered in transit | Real `a2a-sdk` v1.0 JSON-RPC transport carries structured, typed `Message`/`Task` protobufs, not free-form strings; combined with mTLS (below) for transport integrity | `src/idrkd/a2a/executor.py`, `src/idrkd/a2a/server.py` |
| Repudiation | No traceability across an A2A hop | `A2AMessage.traceparent` propagates the W3C trace-context header across the bridge | `src/idrkd/a2a/bridge.py` |
| Information Disclosure | Cross-process calls transmitted in cleartext | `TransportSecurityConfig`/`build_ssl_context` build a `ssl.SSLContext` requiring a client certificate (`CERT_REQUIRED`) for A2A calls when `require_mtls=True`, wired into `IdrkdA2AClient`'s `httpx.AsyncClient` | `src/idrkd/security/transport.py`, `src/idrkd/a2a/client.py` |
| Elevation of Privilege | A task payload is used to invoke an MCP tool the caller shouldn't reach | `IdrkdAgentExecutor` dispatches every A2A task through `McpToolRegistry.call_tool`, so the same tenant-scoping and Pydantic-allowlist gates from boundary 1 apply — there is no separate, weaker A2A-only code path | `src/idrkd/a2a/executor.py` |
| Compatibility downgrade | A legacy 0.3 peer is used to bypass 1.0-era validation | `enable_v0_3_compat` is explicit and additive (both interfaces are advertised on the signed card); it does not disable any gate in `McpToolRegistry` | `src/idrkd/a2a/agent_card.py`, `src/idrkd/a2a/server.py` |

## 5. Ingestion webhook <-> Kafka producer

| STRIDE | Risk | Control | Implementation |
|---|---|---|---|
| Spoofing | An unrelated system posts a fabricated commit event | `tenant_id`/`repo_id` are required on every `CommitWebhookPayload`; downstream consumers re-validate tenant scope before writing | `src/idrkd/ingestion/webhook.py` |
| Repudiation | A commit event can't be traced back to its webhook delivery | `x_correlation_id` header is threaded through to the Kafka event | `src/idrkd/ingestion/webhook.py` |
| Denial of Service | A single webhook triggers an unbounded ingestion fan-out | `IngestionSlo.max_files` bounds the file count per ingestion run | `src/idrkd/ingestion/slo.py` |

## 6. Orchestrator/Critic <-> retrieved tool/document content

The boundary where text pulled from the graph, vector store, or another
agent's tool result re-enters the orchestrator as *data* that must never be
treated as *instructions*.

| STRIDE | Risk | Control | Implementation |
|---|---|---|---|
| Elevation of Privilege | Retrieved content contains an embedded instruction ("ignore previous instructions...") | Layer 1: marker denylist on inbound text | `detect_prompt_injection`, `src/idrkd/security/gates.py` |
| Elevation of Privilege | Retrieved content impersonates a system/developer role turn | Layer 2: role-impersonation detection | `detect_role_impersonation`, `src/idrkd/security/gates.py` |
| Tampering | Retrieved content tries to escape a quarantine fence to merge with trusted instructions | Layer 3: every tool result is wrapped with an explicit untrusted-data delimiter before further use; a fence-breakout attempt is rejected | `quarantine_untrusted_text`, `detect_quarantine_breakout`, `src/idrkd/security/gates.py` |
| Information Disclosure | A tool result inadvertently echoes a secret (shared HMAC secret, credentials) back to the caller | Layer 4: output-side secret-leakage scan against known secret values before a result is returned | `scan_for_secret_leakage`, `screen_tool_response`, `src/idrkd/security/gates.py`, wired into `McpToolRegistry.call_tool` |
| Tampering | An ungrounded synthesized answer is presented as fact | The Week 4 `CriticAgent`/`FaithfulnessCritic` gates synthesis on an AlignScore-style entailment threshold before acceptance | `src/idrkd/rag/critic.py`, `src/idrkd/rag/orchestrator.py` |

## Non-goals

- Real certificate issuance/rotation infrastructure for mTLS is out of
  scope for this repo; `TransportSecurityConfig` is the config contract a
  deployment wires to its actual PKI.
- The SDK's own `[signing]` (JWS-based) agent-card signing is not adopted;
  see `src/idrkd/a2a/bridge.py` for the HMAC rationale.
