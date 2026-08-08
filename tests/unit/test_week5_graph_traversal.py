from idrkd.graph.cypher import (
    bfs_neighbors_params,
    bfs_neighbors_query,
    community_for_entity_params,
    community_subgraph_params,
    shortest_path_params,
    shortest_path_query,
)
from idrkd.graph.traversal import CypherGraphSearch, Neo4jGraphTraversal
from idrkd.security.gates import assert_no_inline_literals, validate_read_only_cypher


def test_bfs_neighbors_query_is_bounded_and_read_only() -> None:
    query = bfs_neighbors_query(depth=3)
    params = bfs_neighbors_params(tenant_id="tenant-a", repo_id="repo-a", entity_id="entity-a", limit=5)

    assert "*1..3" in query
    assert validate_read_only_cypher(query).allowed
    assert assert_no_inline_literals(query).allowed
    assert params == {"tenant_id": "tenant-a", "repo_id": "repo-a", "entity_id": "entity-a", "limit": 5}


def test_bfs_neighbors_query_rejects_non_positive_depth() -> None:
    try:
        bfs_neighbors_query(depth=0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for depth=0")


def test_shortest_path_query_is_bounded_and_read_only() -> None:
    query = shortest_path_query(max_hops=6)
    params = shortest_path_params(tenant_id="tenant-a", repo_id="repo-a", source_id="a", target_id="b")

    assert "shortestPath" in query
    assert "*..6" in query
    assert validate_read_only_cypher(query).allowed
    assert assert_no_inline_literals(query).allowed
    assert params["source_id"] == "a"
    assert params["target_id"] == "b"


def test_community_subgraph_and_community_for_entity_params_are_tenant_scoped() -> None:
    subgraph_params = community_subgraph_params(tenant_id="tenant-a", repo_id="repo-a", community_id=2)
    for_entity_params = community_for_entity_params(tenant_id="tenant-a", repo_id="repo-a", entity_id="entity-a")

    assert subgraph_params["tenant_id"] == "tenant-a"
    assert for_entity_params["entity_id"] == "entity-a"


def test_assert_no_inline_literals_flags_string_concatenated_cypher() -> None:
    unsafe_query = "MATCH (n {id: 'entity-a'}) RETURN n"

    decision = assert_no_inline_literals(unsafe_query)

    assert not decision.allowed
    assert "inline string literal" in decision.reasons[0]


class _FakeRunResult:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = rows

    def __iter__(self):
        return iter(_FakeRecord(row) for row in self._rows)


class _FakeRecord:
    def __init__(self, row: dict[str, object]) -> None:
        self._row = row

    def data(self) -> dict[str, object]:
        return self._row


class _FakeTx:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = rows

    def run(self, _statement: str, _params: dict[str, object]) -> _FakeRunResult:
        return _FakeRunResult(self._rows)


class _FakeSession:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = rows

    def __enter__(self) -> "_FakeSession":
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def execute_read(self, fn, statement, params):
        return fn(_FakeTx(self._rows), statement, params)


class _FakeDriver:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = rows

    def session(self) -> _FakeSession:
        return _FakeSession(self._rows)


def _traversal_with_rows(rows: list[dict[str, object]]) -> Neo4jGraphTraversal:
    traversal = Neo4jGraphTraversal.__new__(Neo4jGraphTraversal)
    traversal._driver = _FakeDriver(rows)  # noqa: SLF001
    return traversal


def test_neo4j_graph_traversal_bfs_neighbors_maps_records() -> None:
    traversal = _traversal_with_rows([{"id": "entity-b", "kind": "Function", "distance": 1}])

    neighbors = traversal.bfs_neighbors(tenant_id="tenant-a", repo_id="repo-a", entity_id="entity-a")

    assert neighbors[0].entity_id == "entity-b"
    assert neighbors[0].distance == 1


def test_cypher_graph_search_resolves_seeds_and_expands_bfs() -> None:
    traversal = _traversal_with_rows([{"id": "entity-c", "kind": "Function", "distance": 1}])
    search = CypherGraphSearch(
        traversal,
        tenant_id="tenant-a",
        repo_id="repo-a",
        labels_by_entity={"entity-a": "Customer API", "entity-b": "Billing Worker"},
    )

    hits = search.bfs("customer API")

    assert hits[0].entity_id == "entity-c"
    assert hits[0].source == "graph"
