import pytest

from idrkd.graph.traversal import BfsNeighbor, CommunityMember, ShortestPath
from idrkd.mcp.backends import InMemoryMcpStateStore
from idrkd.mcp.tools import GraphBfsParams, McpToolRegistry, SearchCodeParams, TOOL_DEFINITIONS


def test_every_tool_definition_exposes_a_real_pydantic_json_schema() -> None:
    for definition in TOOL_DEFINITIONS:
        schema = definition.schema()

        assert schema["inputSchema"]["type"] == "object"
        assert "tenant_id" in schema["inputSchema"]["properties"]
        assert schema["inputSchema"]["additionalProperties"] is False


def test_search_code_params_rejects_unexpected_arguments() -> None:
    with pytest.raises(Exception):
        SearchCodeParams(tenant_id="t", repo_id="r", query="q", unexpected="x")


def test_graph_bfs_params_default_depth_and_limit() -> None:
    params = GraphBfsParams(tenant_id="t", repo_id="r", entity_id="e")

    assert params.depth == 2
    assert params.limit == 10


def test_registry_rejects_extra_argument_via_pydantic_allowlist() -> None:
    registry = McpToolRegistry(principal_tenant_id="tenant-a")

    with pytest.raises(ValueError, match="invalid arguments"):
        registry.call_tool(
            "search_code",
            {"tenant_id": "tenant-a", "repo_id": "repo", "query": "customer api", "extra_field": "nope"},
        )


class _StubTraversal:
    def bfs_neighbors(self, **_kwargs):
        return [BfsNeighbor(entity_id="entity-b", kind="Function", distance=1)]

    def shortest_path(self, **_kwargs):
        return ShortestPath(node_ids=("entity-a", "entity-b"), hop_count=1)

    def community_for_entity(self, **_kwargs):
        return [CommunityMember(entity_id="entity-a", kind="Function", name="run")]


def test_graph_tools_are_wired_to_real_traversal_when_supplied() -> None:
    registry = McpToolRegistry(principal_tenant_id="tenant-a", graph_traversal=_StubTraversal())

    bfs_result = registry.call_tool("graph_bfs", {"tenant_id": "tenant-a", "repo_id": "repo", "entity_id": "e"})
    path_result = registry.call_tool(
        "graph_path",
        {"tenant_id": "tenant-a", "repo_id": "repo", "source_id": "a", "target_id": "b"},
    )
    community_result = registry.call_tool(
        "get_community", {"tenant_id": "tenant-a", "repo_id": "repo", "entity_id": "e"}
    )

    assert bfs_result["neighbors"][0]["entity_id"] == "entity-b"
    assert path_result["path"]["hop_count"] == 1
    assert community_result["members"][0]["name"] == "run"


def test_graph_tools_fall_back_to_stub_handler_without_traversal() -> None:
    registry = McpToolRegistry(principal_tenant_id="tenant-a")

    result = registry.call_tool("graph_bfs", {"tenant_id": "tenant-a", "repo_id": "repo", "entity_id": "e"})

    assert result["neighbors"] == []
    assert result["available"] is False


def test_default_handlers_are_stateful_not_generic_accepted_stubs() -> None:
    registry = McpToolRegistry(principal_tenant_id="tenant-a")

    search = registry.call_tool(
        "search_code",
        {"tenant_id": "tenant-a", "repo_id": "repo", "query": "customer api"},
    )
    queued = registry.call_tool(
        "enqueue_reindex",
        {"tenant_id": "tenant-a", "repo_id": "repo", "entity_id": "entity-a"},
    )
    conflict = registry.call_tool(
        "reconcile",
        {"tenant_id": "tenant-a", "repo_id": "repo", "conflict_id": "conflict-a"},
    )
    resolved = registry.call_tool(
        "resolve_conflict",
        {
            "tenant_id": "tenant-a",
            "repo_id": "repo",
            "conflict_id": "conflict-a",
            "resolution": "use_latest",
        },
    )

    assert search == {"hits": [], "sources": {"neo4j_keyword": False, "pgvector": False}}
    assert queued["queued"] is True
    assert queued["item"]["queue_position"] == 1
    assert conflict["conflict"]["status"] == "resolved" or conflict["conflict"]["status"] == "open"
    assert resolved["conflict"]["resolution"] == "use_latest"


def test_queue_and_conflict_state_can_persist_across_registry_instances() -> None:
    state = InMemoryMcpStateStore()
    first = McpToolRegistry(principal_tenant_id="tenant-a", runtime_state=state)
    second = McpToolRegistry(principal_tenant_id="tenant-a", runtime_state=state)

    first.call_tool(
        "enqueue_reindex",
        {"tenant_id": "tenant-a", "repo_id": "repo", "entity_id": "entity-a"},
    )
    first.call_tool(
        "resolve_conflict",
        {
            "tenant_id": "tenant-a",
            "repo_id": "repo",
            "conflict_id": "conflict-a",
            "resolution": "use_latest",
        },
    )

    queued = second.call_tool(
        "enqueue_reindex",
        {"tenant_id": "tenant-a", "repo_id": "repo", "entity_id": "entity-b"},
    )
    conflict = second.call_tool(
        "get_conflict",
        {"tenant_id": "tenant-a", "repo_id": "repo", "conflict_id": "conflict-a"},
    )

    assert queued["item"]["queue_position"] == 2
    assert conflict["found"] is True
    assert conflict["conflict"]["resolution"] == "use_latest"
