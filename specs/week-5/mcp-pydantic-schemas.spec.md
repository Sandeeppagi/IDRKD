# Spec: MCP Pydantic Schemas

## Goal

Close the gap between the plan's "MCP server expanded to 14 tools (full
Pydantic schema suite per LLD Sec3.5)" and the hand-rolled all-string JSON
Schema the June 26 milestone actually shipped.

## Contract

- Every one of the 14 `TOOL_DEFINITIONS` carries a `params_model: type[ToolParams]`
  — a real Pydantic model, not a generic `{field: {"type": "string"}}` dict.
- `ToolParams` (`src/idrkd/mcp/tools.py`) is the shared base: `tenant_id`,
  `repo_id`, and `model_config = ConfigDict(extra="forbid")`.
- `ToolDefinition.schema()` derives its JSON Schema from
  `params_model.model_json_schema()` — the schema a caller sees and the
  model that validates a call can never drift apart.
- `McpToolRegistry.call_tool` validates arguments by constructing
  `definition.params_model.model_validate(params)`; a `pydantic.ValidationError`
  (missing field, wrong type, or an unexpected key) becomes a `ValueError`,
  which `McpToolRegistry.handle` turns into a JSON-RPC `-32602` error —
  unchanged from the existing error-code contract.
- `extra="forbid"` on every model *is* Week 5 security-hardening layer 5
  (tool-argument allowlisting): a caller cannot smuggle an extra key past
  a tool's declared contract.

## Implementation

- `src/idrkd/mcp/backends.py`
- `src/idrkd/mcp/server.py`
- `src/idrkd/mcp/tools.py`
- `docker/idrkd.Dockerfile`
- `docker/docker-compose.yml`
- `tests/unit/test_week5_mcp_schemas.py`
- `tests/unit/test_mcp_server_and_real_adapters.py`

## Acceptance Criteria

- All 14 tools still round-trip through `tools/list` (`test_mcp_registry_exposes_fourteen_json_rpc_tools`
  in `tests/unit/test_week6_mcp_a2a_security.py` is unchanged and still
  passes).
- An unexpected extra argument raises before any handler runs.
- Default values (`limit`, `depth`, `max_hops`) are enforced by the model,
  not by handler-side `params.get(...)` guessing.
- Default handlers no longer return generic `{"status": "accepted"}`
  placeholders. Neo4j/pgvector-backed handlers serve search, entity, schema
  diff, impact, salience, centroid drift, and stale listing when configured;
  re-index and conflict tools keep explicit state.
- `create_mcp_app(...)` exposes the registry through a standalone FastAPI
  `/mcp` JSON-RPC endpoint plus `/healthz`.
- Docker Compose exposes `mcp-server` on port `8080` and injects
  `NEO4J_URI`, `POSTGRES_DSN`, `REDIS_URL`, `TENANT_ID`, and `REPO_ID`.
- `TENANT_ID` and `REPO_ID` are configurable via environment variables and
  default to `tenant-live` / `week5-e2e`, matching the live graph/vector smoke
  data rather than the empty `default` tenant.
- When `REDIS_URL` is set, `enqueue_reindex`, `reconcile`, `get_conflict`,
  and `resolve_conflict` use Redis-backed state instead of process-local
  memory. Queue items are stored at `idrkd:mcp:reindex`; conflicts are stored at
  `idrkd:mcp:conflict:<conflict_id>` and survive an MCP server restart.
- `idrkd-mcp-smoke` performs a live JSON-RPC smoke against the running MCP
  server and validates `search_code`, `graph_bfs`, and `graph_path` for the
  configured tenant/repo.

## Verification

```bash
uv run pytest tests/unit/test_week5_mcp_schemas.py tests/unit/test_mcp_server_and_real_adapters.py tests/unit/test_week6_mcp_a2a_security.py
docker compose -f docker/docker-compose.yml config
docker compose -f docker/docker-compose.yml up -d --build mcp-server
uv run idrkd-mcp-smoke --tenant-id tenant-live --repo-id week5-e2e
```

Live Docker evidence from 2026-08-04:

- `docker-mcp-server-1` reached `Up ... (healthy)` on `0.0.0.0:8080`.
- JSON-RPC `enqueue_reindex` returned `queued: true` and wrote one Redis list
  item to `idrkd:mcp:reindex`.
- JSON-RPC `reconcile` then `resolve_conflict` wrote resolved conflict JSON to
  `idrkd:mcp:conflict:conflict-docker-1`.
- After `docker compose -f docker/docker-compose.yml restart mcp-server`,
  JSON-RPC `get_conflict` returned the same resolved conflict with
  `found: true`.
- After tenant alignment, `uv run idrkd-mcp-smoke --tenant-id tenant-live
  --repo-id week5-e2e` returned 14 tools, non-empty `search_code` hits, 4 BFS
  neighbors, and a 2-hop `graph_path`.
