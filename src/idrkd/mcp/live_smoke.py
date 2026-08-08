"""Live MCP smoke test for tenant-aligned Docker deployments."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from typing import Any
from urllib import request


@dataclass(frozen=True)
class LiveSmokeConfig:
    base_url: str = "http://localhost:8080"
    tenant_id: str = "tenant-live"
    repo_id: str = "week5-e2e"
    search_query: str = "example Worker run"
    source_id: str | None = None
    target_id: str | None = None
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "change-me"


@dataclass(frozen=True)
class LiveSmokeResult:
    tenant_id: str
    repo_id: str
    tools_count: int
    search_hits: int
    bfs_neighbors: int
    path_hops: int
    source_id: str
    target_id: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_live_smoke(config: LiveSmokeConfig) -> LiveSmokeResult:
    health = _get_json(f"{config.base_url.rstrip('/')}/healthz")
    if health.get("status") != "ok":
        raise RuntimeError(f"MCP health check failed: {health}")

    tools_response = _json_rpc(config.base_url, {"jsonrpc": "2.0", "id": "tools", "method": "tools/list", "params": {}})
    tools = (tools_response.get("result") or {}).get("tools", [])
    if not isinstance(tools, list) or len(tools) != 14:
        raise RuntimeError(f"expected 14 MCP tools, got {len(tools) if isinstance(tools, list) else 'invalid'}")

    source_id = config.source_id
    target_id = config.target_id
    if source_id is None or target_id is None:
        source_id, target_id = discover_graph_path(
            tenant_id=config.tenant_id,
            repo_id=config.repo_id,
            neo4j_uri=config.neo4j_uri,
            neo4j_user=config.neo4j_user,
            neo4j_password=config.neo4j_password,
        )

    search = call_tool(
        config.base_url,
        "search_code",
        {
            "tenant_id": config.tenant_id,
            "repo_id": config.repo_id,
            "query": config.search_query,
            "limit": 5,
        },
    )
    hits = search.get("hits", [])
    if not isinstance(hits, list) or not hits:
        raise RuntimeError(
            f"search_code returned no hits for tenant={config.tenant_id} repo={config.repo_id}; "
            "check TENANT_ID/REPO_ID or seed live data"
        )

    bfs = call_tool(
        config.base_url,
        "graph_bfs",
        {
            "tenant_id": config.tenant_id,
            "repo_id": config.repo_id,
            "entity_id": source_id,
            "depth": 2,
            "limit": 5,
        },
    )
    neighbors = bfs.get("neighbors", [])
    if not isinstance(neighbors, list) or not neighbors:
        raise RuntimeError(f"graph_bfs returned no neighbors for source_id={source_id}")

    path = call_tool(
        config.base_url,
        "graph_path",
        {
            "tenant_id": config.tenant_id,
            "repo_id": config.repo_id,
            "source_id": source_id,
            "target_id": target_id,
            "max_hops": 6,
        },
    ).get("path")
    if not isinstance(path, dict) or int(path.get("hop_count", 0)) < 1:
        raise RuntimeError(f"graph_path returned no path for source_id={source_id} target_id={target_id}")

    return LiveSmokeResult(
        tenant_id=config.tenant_id,
        repo_id=config.repo_id,
        tools_count=len(tools),
        search_hits=len(hits),
        bfs_neighbors=len(neighbors),
        path_hops=int(path["hop_count"]),
        source_id=source_id,
        target_id=target_id,
    )


def call_tool(base_url: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    response = _json_rpc(
        base_url,
        {
            "jsonrpc": "2.0",
            "id": name,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        },
    )
    error = response.get("error")
    if error is not None:
        raise RuntimeError(f"{name} failed: {error}")
    result = response.get("result")
    if not isinstance(result, dict):
        raise RuntimeError(f"{name} returned invalid result: {result}")
    return result


def discover_graph_path(
    *,
    tenant_id: str,
    repo_id: str,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
) -> tuple[str, str]:
    from neo4j import GraphDatabase

    query = """
    MATCH path = (source:CodeEntity {tenant_id: $tenant_id, repo_id: $repo_id})
      -[:RELATES_TO*1..6]->
      (target:CodeEntity {tenant_id: $tenant_id, repo_id: $repo_id})
    RETURN source.id AS source_id, target.id AS target_id, length(path) AS hops
    ORDER BY hops DESC, source_id ASC, target_id ASC
    LIMIT 1
    """
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    try:
        with driver.session() as session:
            record = session.run(query, {"tenant_id": tenant_id, "repo_id": repo_id}).single()
    finally:
        driver.close()
    if record is None:
        raise RuntimeError(f"no graph path found for tenant={tenant_id} repo={repo_id}")
    return str(record["source_id"]), str(record["target_id"])


def config_from_env() -> LiveSmokeConfig:
    return LiveSmokeConfig(
        base_url=os.getenv("MCP_BASE_URL", "http://localhost:8080"),
        tenant_id=os.getenv("TENANT_ID", os.getenv("IDRKD_TENANT_ID", "tenant-live")),
        repo_id=os.getenv("REPO_ID", "week5-e2e"),
        search_query=os.getenv("MCP_SMOKE_QUERY", "example Worker run"),
        source_id=os.getenv("MCP_SMOKE_SOURCE_ID") or None,
        target_id=os.getenv("MCP_SMOKE_TARGET_ID") or None,
        neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
        neo4j_password=os.getenv("NEO4J_PASSWORD", "change-me"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run live MCP search/traversal smoke checks.")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--tenant-id", default=None)
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--search-query", default=None)
    parser.add_argument("--source-id", default=None)
    parser.add_argument("--target-id", default=None)
    parser.add_argument("--neo4j-uri", default=None)
    parser.add_argument("--neo4j-user", default=None)
    parser.add_argument("--neo4j-password", default=None)
    args = parser.parse_args()
    env_config = config_from_env()
    config = LiveSmokeConfig(
        base_url=args.base_url or env_config.base_url,
        tenant_id=args.tenant_id or env_config.tenant_id,
        repo_id=args.repo_id or env_config.repo_id,
        search_query=args.search_query or env_config.search_query,
        source_id=args.source_id or env_config.source_id,
        target_id=args.target_id or env_config.target_id,
        neo4j_uri=args.neo4j_uri or env_config.neo4j_uri,
        neo4j_user=args.neo4j_user or env_config.neo4j_user,
        neo4j_password=args.neo4j_password or env_config.neo4j_password,
    )
    print(json.dumps(run_live_smoke(config).as_dict(), sort_keys=True))


def _json_rpc(base_url: str, payload: dict[str, Any]) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        f"{base_url.rstrip('/')}/mcp",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=10) as response:
        loaded = json.loads(response.read().decode("utf-8"))
    if not isinstance(loaded, dict):
        raise RuntimeError(f"invalid JSON-RPC response: {loaded}")
    return loaded


def _get_json(url: str) -> dict[str, Any]:
    with request.urlopen(url, timeout=10) as response:
        loaded = json.loads(response.read().decode("utf-8"))
    if not isinstance(loaded, dict):
        raise RuntimeError(f"invalid JSON response: {loaded}")
    return loaded


if __name__ == "__main__":
    main()
