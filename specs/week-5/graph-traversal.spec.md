# Spec: Graph Traversal

## Goal

Provide the Week 5 Graph Traversal Agent as real, read-only, parameterized
Cypher — BFS neighbourhood expand, shortestPath, and community subgraph —
and wire it into the `graph_bfs`, `graph_path`, and `get_community` MCP
tools.

## Contract

Query builders (`src/idrkd/graph/cypher.py`):

- `bfs_neighbors_query(depth: int) -> str` / `bfs_neighbors_params(...)`:
  bounded `[:RELATES_TO*1..depth]` expansion from a tenant/repo-scoped
  seed entity, ordered by hop distance.
- `shortest_path_query(max_hops: int) -> str` / `shortest_path_params(...)`:
  `shortestPath(...)` between two tenant/repo-scoped entities, bounded by
  `max_hops`.
- `community_subgraph_params(...)` / `community_for_entity_params(...)`:
  all entities sharing a `community_id` property, either given directly or
  resolved from a seed entity in the same query.
- `depth`/`max_hops` are the only values ever spliced into query text
  (as validated positive integers — Cypher does not allow parameters in a
  variable-length range bound); every other value is a bound `$param`.

Runtime (`src/idrkd/graph/traversal.py`):

- `Neo4jGraphTraversal`: `bfs_neighbors`, `shortest_path`,
  `community_subgraph`, `community_for_entity`, each running the
  corresponding query through a Neo4j driver session and returning a typed
  dataclass (`BfsNeighbor`, `ShortestPath`, `CommunityMember`).
- `CypherGraphSearch`: adapts `Neo4jGraphTraversal` to the `GraphSearch`
  protocol (`src/idrkd/rag/retrieval.py`) so it is a drop-in alternative to
  `KeywordGraphSearch` for `HybridRetriever` (Week 3) and
  `GraphTraversalAgent` (Week 4) — free-text queries resolve to seed
  entities by term overlap, then each seed is BFS-expanded through real
  Cypher.

MCP wiring (`src/idrkd/mcp/tools.py`):

- `McpToolRegistry(..., graph_traversal=Neo4jGraphTraversal(...))` wires
  `graph_bfs`/`graph_path`/`get_community` to the real traversal; omitting
  `graph_traversal` keeps the existing stub-handler behavior so no
  existing test or caller breaks.

## Implementation

- `src/idrkd/graph/cypher.py`
- `src/idrkd/graph/traversal.py`
- `src/idrkd/mcp/tools.py`
- `tests/unit/test_week5_graph_traversal.py`
- `tests/unit/test_week5_mcp_schemas.py`

## Acceptance Criteria

- Every generated query passes `validate_read_only_cypher` and
  `assert_no_inline_literals`.
- `bfs_neighbors_query`/`shortest_path_query` reject a non-positive
  `depth`/`max_hops`.
- All dynamic identifiers (tenant, repo, entity, community IDs) are
  tenant/repo-scoped in every `MATCH` clause, not just the seed node.
- `graph_bfs`/`graph_path`/`get_community` return real traversal results
  when a `Neo4jGraphTraversal` is supplied, and fall back to the existing
  stub behavior when it is not.

## Verification

```bash
uv run pytest tests/unit/test_week5_graph_traversal.py tests/unit/test_week5_mcp_schemas.py
```
