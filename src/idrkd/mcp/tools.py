"""JSON-RPC 2.0 MCP tool contracts for the June 26 milestone."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from idrkd.security.gates import validate_tool_payload


JsonRpcId = str | int | None
ToolHandler = Callable[[dict[str, Any]], dict[str, Any]]


class JsonRpcRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    jsonrpc: Literal["2.0"] = "2.0"
    method: str
    params: dict[str, Any] = Field(default_factory=dict)
    id: JsonRpcId = None


class JsonRpcError(BaseModel):
    code: int
    message: str
    data: dict[str, Any] | None = None


class JsonRpcResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    jsonrpc: Literal["2.0"] = "2.0"
    id: JsonRpcId = None
    result: dict[str, Any] | None = None
    error: JsonRpcError | None = None


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    required: tuple[str, ...]

    def schema(self) -> dict[str, Any]:
        properties = {field: {"type": "string"} for field in self.required}
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": {
                "type": "object",
                "properties": properties,
                "required": list(self.required),
                "additionalProperties": True,
            },
        }


TOOL_DEFINITIONS: tuple[ToolDefinition, ...] = (
    ToolDefinition("search_code", "Hybrid semantic and graph code search.", ("tenant_id", "repo_id", "query")),
    ToolDefinition("get_entity", "Fetch an entity by stable ID.", ("tenant_id", "repo_id", "entity_id")),
    ToolDefinition("graph_bfs", "Return bounded BFS neighbors.", ("tenant_id", "repo_id", "entity_id")),
    ToolDefinition("graph_path", "Return a path between two entities.", ("tenant_id", "repo_id", "source_id", "target_id")),
    ToolDefinition("get_community", "Return Louvain/community assignment.", ("tenant_id", "repo_id", "entity_id")),
    ToolDefinition("enqueue_reindex", "Queue a scoped re-index request.", ("tenant_id", "repo_id", "entity_id")),
    ToolDefinition("schema_diff", "Compare two schema fingerprints.", ("tenant_id", "repo_id", "left_id", "right_id")),
    ToolDefinition("impact_analysis", "Estimate downstream change impact.", ("tenant_id", "repo_id", "entity_id")),
    ToolDefinition("reconcile", "Produce reconciliation recommendations.", ("tenant_id", "repo_id", "conflict_id")),
    ToolDefinition("get_conflict", "Fetch a reconciliation conflict.", ("tenant_id", "repo_id", "conflict_id")),
    ToolDefinition("resolve_conflict", "Mark a conflict resolution decision.", ("tenant_id", "repo_id", "conflict_id")),
    ToolDefinition("get_salience", "Return entity salience score.", ("tenant_id", "repo_id", "entity_id")),
    ToolDefinition("get_centroid_drift", "Return community centroid drift.", ("tenant_id", "repo_id", "community_id")),
    ToolDefinition("list_stale", "List stale entities needing verification.", ("tenant_id", "repo_id")),
)


def _default_handler(tool_name: str) -> ToolHandler:
    def handler(params: dict[str, Any]) -> dict[str, Any]:
        return {"tool": tool_name, "status": "accepted", "params": params}

    return handler


class McpToolRegistry:
    def __init__(self, *, principal_tenant_id: str, handlers: dict[str, ToolHandler] | None = None) -> None:
        self._principal_tenant_id = principal_tenant_id
        self._definitions = {definition.name: definition for definition in TOOL_DEFINITIONS}
        self._handlers = {name: _default_handler(name) for name in self._definitions}
        if handlers:
            self._handlers.update(handlers)

    def list_tools(self) -> list[dict[str, Any]]:
        return [definition.schema() for definition in self._definitions.values()]

    def call_tool(self, name: str, params: dict[str, Any]) -> dict[str, Any]:
        definition = self._definitions.get(name)
        if definition is None:
            raise ValueError(f"unknown MCP tool: {name}")
        missing = [field for field in definition.required if field not in params]
        if missing:
            raise ValueError(f"missing required field(s): {', '.join(missing)}")
        tenant_id = str(params["tenant_id"])
        query_text = str(params.get("query", "")) if "query" in params else None
        cypher = str(params.get("cypher", "")) if "cypher" in params else None
        validate_tool_payload(
            tenant_id=tenant_id,
            principal_tenant_id=self._principal_tenant_id,
            query_text=query_text,
            cypher=cypher,
        ).require_allowed()
        return self._handlers[name](params)

    def handle(self, request: JsonRpcRequest) -> JsonRpcResponse:
        try:
            if request.method == "tools/list":
                return JsonRpcResponse(id=request.id, result={"tools": self.list_tools()})
            if request.method == "tools/call":
                name = str(request.params.get("name", ""))
                params = request.params.get("arguments", {})
                if not isinstance(params, dict):
                    raise ValueError("tools/call arguments must be an object")
                return JsonRpcResponse(id=request.id, result=self.call_tool(name, params))
            raise ValueError(f"unsupported JSON-RPC method: {request.method}")
        except PermissionError as exc:
            return JsonRpcResponse(id=request.id, error=JsonRpcError(code=-32001, message=str(exc)))
        except ValueError as exc:
            return JsonRpcResponse(id=request.id, error=JsonRpcError(code=-32602, message=str(exc)))
