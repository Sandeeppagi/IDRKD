# Week 5 Specs

Week 5 closes out the HLD/LLD `W5-W6` MCP + Graph Traversal + A2A Bridge
phase (`w:'W5'` items only — MCP-TaskBench, BFCL eval, and the BITS
mid-semester report are `w:'W6'` and out of scope here):

- Real, read-only Cypher for the Graph Traversal Agent (BFS neighbourhood
  expand, shortestPath, community subgraph), wired into the `graph_bfs`,
  `graph_path`, and `get_community` MCP tools.
- Real per-tool Pydantic request models for the full 14-tool MCP suite,
  replacing the hand-rolled all-string JSON Schema.
- Concrete default MCP handlers plus a standalone FastAPI MCP JSON-RPC
  server; Neo4j/pgvector-backed tools use live adapters when configured, and
  queue/conflict tools keep explicit state instead of generic accepted stubs.
- The A2A bridge rebuilt on the official `a2a-sdk` v1.0 package (real
  `AgentCard`/`AgentExecutor`/JSON-RPC server and client), with a legacy
  0.3 compatibility layer, while keeping this repo's own tested HMAC
  agent-card signing.
- Security hardening: a 5-layer prompt-injection containment chain,
  Cypher-escape hardening, and an mTLS transport config contract.
- A STRIDE threat model covering 6 trust boundaries
  (`docs/design/threat-model.md`).

## Spec Index

| Spec | Status | Primary Verification |
|---|---:|---|
| [Graph Traversal](graph-traversal.spec.md) | Implemented | `tests/unit/test_week5_graph_traversal.py` |
| [MCP Pydantic Schemas](mcp-pydantic-schemas.spec.md) | Implemented | `tests/unit/test_week5_mcp_schemas.py`, `tests/unit/test_mcp_server_and_real_adapters.py` |
| [A2A SDK Bridge](a2a-sdk-bridge.spec.md) | Implemented | `tests/unit/test_week5_a2a_bridge.py` |
| [Security Hardening](security-hardening.spec.md) | Implemented | `tests/unit/test_week5_security_hardening.py` |
