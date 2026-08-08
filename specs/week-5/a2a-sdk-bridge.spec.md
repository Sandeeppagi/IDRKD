# Spec: A2A SDK Bridge

## Goal

Replace the bespoke, dependency-free A2A facade with the official
`a2a-sdk` v1.0 package (`a2a-sdk[http-server]`), while preserving this
repo's own tested HMAC agent-card signing contract.

## Contract

- `src/idrkd/a2a/agent_card.py`: `build_idrkd_agent_card(...)` builds a
  real `a2a.types.AgentCard` with one `AgentInterface` per protocol
  version it supports; passing `legacy_endpoint` adds a `protocol_version="0.3"`
  interface alongside the `"1.0"` one — the backward-compat layer for
  legacy 0.3 peers, in addition to `create_jsonrpc_routes(..., enable_v0_3_compat=True)`
  on the server side.
- `src/idrkd/a2a/bridge.py`: `agent_card_payload(card) -> dict` (via
  `google.protobuf.json_format.MessageToDict`, snake_case field names) is
  what `sign_payload`/`sign_agent_card`/`SignedAgentCard.verify` sign and
  verify — same HMAC contract as before, retargeted from the old
  dataclass `AgentCard` to the real SDK type. `A2AMessage` (sender,
  recipient, task, payload, `traceparent`) is unchanged and still the
  internal trace-context envelope built before SDK handoff.
- `src/idrkd/a2a/executor.py`: `IdrkdAgentExecutor(AgentExecutor)` reads a
  `{"name": ..., "arguments": {...}}` JSON payload from the incoming
  message text and dispatches it through `McpToolRegistry.call_tool` —
  the same tenant-scoping, Pydantic-allowlist, and prompt-injection gates
  that apply to a direct MCP call apply identically over A2A. A missing
  tool or a `PermissionError`/`ValueError` from the registry fails the
  task (`TaskState.TASK_STATE_FAILED`) rather than raising through the
  transport.
- `src/idrkd/a2a/server.py`: `build_a2a_app(agent_card, *, registry, ...) -> Starlette`
  assembles `create_agent_card_routes` + `create_jsonrpc_routes` around a
  `DefaultRequestHandler` — no module-level `uvicorn.run`, so it is
  directly testable in-process.
- `src/idrkd/a2a/client.py`: `IdrkdA2AClient` wraps `a2a.client.create_client`/`ClientConfig`,
  accepting an optional `TransportSecurityConfig` (Sec `security-hardening.spec.md`)
  to build an mTLS-enabled `httpx.AsyncClient`.

## Implementation

- `src/idrkd/a2a/agent_card.py`, `bridge.py`, `executor.py`, `server.py`, `client.py`
- `pyproject.toml` (`a2a-sdk[http-server]>=1.1` dependency)
- `tests/unit/test_week5_a2a_bridge.py`
- `tests/unit/test_week6_mcp_a2a_security.py` (updated for the new `AgentCard` shape)

## Acceptance Criteria

- A signed card round-trips: `verify_remote_card` accepts a card signed
  with the correct secret and rejects a tampered one.
- The agent-card HTTP route (`/.well-known/agent-card.json`) serves a
  card whose identity matches the signed card.
- A `message/send` JSON-RPC call for a known tool round-trips through
  `IdrkdAgentExecutor` to a `TASK_STATE_COMPLETED` task carrying the tool
  result as a JSON artifact.
- A `message/send` call for an unknown tool ends in `TASK_STATE_FAILED`
  rather than an unhandled exception.
- The card advertises both a `"1.0"` and a `"0.3"` `AgentInterface` when
  built with `legacy_endpoint`.

## Verification

```bash
uv run pytest tests/unit/test_week5_a2a_bridge.py tests/unit/test_week6_mcp_a2a_security.py
```
