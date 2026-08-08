"""Standalone FastAPI JSON-RPC server for the IDRKD MCP registry."""

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.responses import Response

from idrkd.graph.traversal import Neo4jGraphTraversal
from idrkd.mcp.backends import Neo4jMcpBackend, PgvectorSearchBackend, RedisMcpStateStore
from idrkd.mcp.tools import JsonRpcRequest, JsonRpcResponse, McpToolRegistry
from idrkd.observability.metrics import metrics_response


def create_mcp_app(registry: McpToolRegistry | None = None) -> FastAPI:
    app = FastAPI(title="IDRKD MCP JSON-RPC Server")
    active_registry = registry or build_registry_from_env()

    @app.post("/mcp")
    def handle_json_rpc(request: JsonRpcRequest) -> JsonRpcResponse:
        return active_registry.handle(request)

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics")
    def metrics() -> Response:
        body, content_type = metrics_response()
        return Response(content=body, media_type=content_type)

    return app


def build_registry_from_env() -> McpToolRegistry:
    tenant_id = os.getenv("TENANT_ID", "default")
    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "change-me")
    postgres_dsn = os.getenv("POSTGRES_DSN")
    redis_url = os.getenv("REDIS_URL")

    graph_traversal = None
    graph_backend = None
    if neo4j_uri:
        graph_traversal = Neo4jGraphTraversal(neo4j_uri, neo4j_user, neo4j_password)
        graph_backend = Neo4jMcpBackend(neo4j_uri, neo4j_user, neo4j_password)

    vector_backend = PgvectorSearchBackend(postgres_dsn) if postgres_dsn else None
    runtime_state = RedisMcpStateStore(redis_url) if redis_url else None
    return McpToolRegistry(
        principal_tenant_id=tenant_id,
        graph_traversal=graph_traversal,
        graph_backend=graph_backend,
        vector_backend=vector_backend,
        runtime_state=runtime_state,
    )


app = create_mcp_app()
