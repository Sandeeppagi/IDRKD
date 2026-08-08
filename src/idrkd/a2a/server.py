"""Builds the IDRKD A2A server as a reusable Starlette app (no module-level uvicorn.run)."""

from __future__ import annotations

from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore, TaskStore
from a2a.types import AgentCard
from starlette.applications import Starlette

from idrkd.a2a.executor import IdrkdAgentExecutor
from idrkd.a2a.task_state import A2ATaskStateStore
from idrkd.mcp.tools import McpToolRegistry


def build_a2a_app(
    agent_card: AgentCard,
    *,
    registry: McpToolRegistry,
    task_store: TaskStore | None = None,
    task_state_store: A2ATaskStateStore | None = None,
    rpc_url: str = "/",
    enable_v0_3_compat: bool = True,
) -> Starlette:
    """Assembles the agent-card discovery route and JSON-RPC task routes
    around `IdrkdAgentExecutor`. `enable_v0_3_compat=True` is the
    backward-compat layer for legacy 0.3 A2A clients/servers."""
    handler = DefaultRequestHandler(
        agent_executor=IdrkdAgentExecutor(registry, task_state_store),
        task_store=task_store or InMemoryTaskStore(),
        agent_card=agent_card,
    )
    routes = [
        *create_agent_card_routes(agent_card),
        *create_jsonrpc_routes(handler, rpc_url, enable_v0_3_compat=enable_v0_3_compat),
    ]
    return Starlette(routes=routes)
