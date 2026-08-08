"""Model Context Protocol JSON-RPC tool contracts."""

from idrkd.mcp.server import build_registry_from_env, create_mcp_app
from idrkd.mcp.tools import JsonRpcError, JsonRpcRequest, JsonRpcResponse, McpToolRegistry, TOOL_DEFINITIONS

__all__ = [
    "JsonRpcError",
    "JsonRpcRequest",
    "JsonRpcResponse",
    "McpToolRegistry",
    "TOOL_DEFINITIONS",
    "build_registry_from_env",
    "create_mcp_app",
]
