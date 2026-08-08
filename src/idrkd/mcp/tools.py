"""JSON-RPC 2.0 MCP tool contracts.

Week 5 replaces the hand-rolled all-string JSON Schema with a real
per-tool Pydantic request model (LLD Sec3.5's "full Pydantic schema
suite"), and wires `graph_bfs`/`graph_path`/`get_community` to the real
Cypher traversal in `idrkd.graph.traversal` when a `Neo4jGraphTraversal`
is supplied.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
import json
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from idrkd.mcp.backends import EntityRecord, InMemoryMcpStateStore, McpStateStore, SearchResult
from idrkd.observability.metrics import MCP_TOOL_CALLS, MCP_TOOL_LATENCY, observe_histogram
from idrkd.observability.tracing import traced_span
from idrkd.security.gates import screen_tool_response, validate_tool_payload


if TYPE_CHECKING:
    from idrkd.graph.traversal import Neo4jGraphTraversal
    from idrkd.mcp.backends import Neo4jMcpBackend, PgvectorSearchBackend


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


class ToolParams(BaseModel):
    """Base contract for every tool's arguments: strict, tenant-scoped."""

    model_config = ConfigDict(extra="forbid")

    tenant_id: str
    repo_id: str


class SearchCodeParams(ToolParams):
    query: str
    limit: int = 10


class GetEntityParams(ToolParams):
    entity_id: str


class GraphBfsParams(ToolParams):
    entity_id: str
    depth: int = 2
    limit: int = 10


class GraphPathParams(ToolParams):
    source_id: str
    target_id: str
    max_hops: int = 6


class GetCommunityParams(ToolParams):
    entity_id: str
    limit: int = 25


class EnqueueReindexParams(ToolParams):
    entity_id: str


class SchemaDiffParams(ToolParams):
    left_id: str
    right_id: str


class ImpactAnalysisParams(ToolParams):
    entity_id: str


class ReconcileParams(ToolParams):
    conflict_id: str


class GetConflictParams(ToolParams):
    conflict_id: str


class ResolveConflictParams(ToolParams):
    conflict_id: str
    resolution: str = "pending"


class GetSalienceParams(ToolParams):
    entity_id: str


class GetCentroidDriftParams(ToolParams):
    community_id: str


class ListStaleParams(ToolParams):
    pass


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    params_model: type[ToolParams]

    @property
    def required(self) -> tuple[str, ...]:
        return tuple(
            field_name
            for field_name, field_info in self.params_model.model_fields.items()
            if field_info.is_required()
        )

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.params_model.model_json_schema(),
        }


TOOL_DEFINITIONS: tuple[ToolDefinition, ...] = (
    ToolDefinition("search_code", "Hybrid semantic and graph code search.", SearchCodeParams),
    ToolDefinition("get_entity", "Fetch an entity by stable ID.", GetEntityParams),
    ToolDefinition("graph_bfs", "Return bounded BFS neighbors.", GraphBfsParams),
    ToolDefinition("graph_path", "Return a path between two entities.", GraphPathParams),
    ToolDefinition("get_community", "Return Louvain/community assignment.", GetCommunityParams),
    ToolDefinition("enqueue_reindex", "Queue a scoped re-index request.", EnqueueReindexParams),
    ToolDefinition("schema_diff", "Compare two schema fingerprints.", SchemaDiffParams),
    ToolDefinition("impact_analysis", "Estimate downstream change impact.", ImpactAnalysisParams),
    ToolDefinition("reconcile", "Produce reconciliation recommendations.", ReconcileParams),
    ToolDefinition("get_conflict", "Fetch a reconciliation conflict.", GetConflictParams),
    ToolDefinition("resolve_conflict", "Mark a conflict resolution decision.", ResolveConflictParams),
    ToolDefinition("get_salience", "Return entity salience score.", GetSalienceParams),
    ToolDefinition("get_centroid_drift", "Return community centroid drift.", GetCentroidDriftParams),
    ToolDefinition("list_stale", "List stale entities needing verification.", ListStaleParams),
)


def _entity_payload(entity: EntityRecord) -> dict[str, Any]:
    return {
        "id": entity.id,
        "kind": entity.kind,
        "name": entity.name,
        "qualified_name": entity.qualified_name,
        "path": entity.path,
        "content_hash": entity.content_hash,
        "properties": entity.properties,
    }


def _search_payload(hit: SearchResult) -> dict[str, Any]:
    return {
        "entity_id": hit.entity_id,
        "score": hit.score,
        "source": hit.source,
        "name": hit.name,
        "kind": hit.kind,
    }


def _merge_ranked_hits(*rankings: list[SearchResult], limit: int) -> list[dict[str, Any]]:
    by_entity: dict[str, SearchResult] = {}
    for ranking in rankings:
        for hit in ranking:
            existing = by_entity.get(hit.entity_id)
            if existing is None or hit.score > existing.score:
                by_entity[hit.entity_id] = hit
    ordered = sorted(by_entity.values(), key=lambda hit: hit.score, reverse=True)[:limit]
    return [_search_payload(hit) for hit in ordered]


def _search_code_handler(
    *,
    graph_backend: Neo4jMcpBackend | None,
    vector_backend: PgvectorSearchBackend | None,
) -> ToolHandler:
    def handler(params: dict[str, Any]) -> dict[str, Any]:
        graph_hits = (
            graph_backend.search_code(
                tenant_id=params["tenant_id"],
                repo_id=params["repo_id"],
                query_text=params["query"],
                limit=params.get("limit", 10),
            )
            if graph_backend is not None
            else []
        )
        vector_hits = (
            vector_backend.search_code(
                tenant_id=params["tenant_id"],
                repo_id=params["repo_id"],
                query_text=params["query"],
                limit=params.get("limit", 10),
            )
            if vector_backend is not None
            else []
        )
        return {
            "hits": _merge_ranked_hits(graph_hits, vector_hits, limit=params.get("limit", 10)),
            "sources": {
                "neo4j_keyword": graph_backend is not None,
                "pgvector": vector_backend is not None,
            },
        }

    return handler


def _get_entity_handler(graph_backend: Neo4jMcpBackend | None) -> ToolHandler:
    def handler(params: dict[str, Any]) -> dict[str, Any]:
        entity = (
            graph_backend.get_entity(
                tenant_id=params["tenant_id"], repo_id=params["repo_id"], entity_id=params["entity_id"]
            )
            if graph_backend is not None
            else None
        )
        return {"found": entity is not None, "entity": _entity_payload(entity) if entity is not None else None}

    return handler


def _enqueue_reindex_handler(state: McpStateStore) -> ToolHandler:
    def handler(params: dict[str, Any]) -> dict[str, Any]:
        item = state.enqueue_reindex(
            {
                "tenant_id": params["tenant_id"],
                "repo_id": params["repo_id"],
                "entity_id": params["entity_id"],
            }
        )
        return {"queued": True, "item": item}

    return handler


def _schema_diff_handler(graph_backend: Neo4jMcpBackend | None) -> ToolHandler:
    def handler(params: dict[str, Any]) -> dict[str, Any]:
        left = (
            graph_backend.get_entity(
                tenant_id=params["tenant_id"], repo_id=params["repo_id"], entity_id=params["left_id"]
            )
            if graph_backend is not None
            else None
        )
        right = (
            graph_backend.get_entity(
                tenant_id=params["tenant_id"], repo_id=params["repo_id"], entity_id=params["right_id"]
            )
            if graph_backend is not None
            else None
        )
        left_fields = set(_schema_fields(left))
        right_fields = set(_schema_fields(right))
        return {
            "left_found": left is not None,
            "right_found": right is not None,
            "added": sorted(right_fields - left_fields),
            "removed": sorted(left_fields - right_fields),
            "unchanged": sorted(left_fields & right_fields),
        }

    return handler


def _impact_analysis_handler(
    *,
    graph_backend: Neo4jMcpBackend | None,
    graph_traversal: Neo4jGraphTraversal | None,
) -> ToolHandler:
    def handler(params: dict[str, Any]) -> dict[str, Any]:
        downstream = (
            graph_backend.downstream_impact(
                tenant_id=params["tenant_id"], repo_id=params["repo_id"], entity_id=params["entity_id"]
            )
            if graph_backend is not None
            else []
        )
        neighbors = (
            graph_traversal.bfs_neighbors(
                tenant_id=params["tenant_id"],
                repo_id=params["repo_id"],
                entity_id=params["entity_id"],
                depth=2,
                limit=25,
            )
            if graph_traversal is not None
            else []
        )
        return {
            "entity_id": params["entity_id"],
            "downstream": [_search_payload(hit) for hit in downstream],
            "neighbors": [asdict(neighbor) for neighbor in neighbors],
            "impact_count": len(downstream) + len(neighbors),
        }

    return handler


def _reconcile_handler(state: McpStateStore) -> ToolHandler:
    def handler(params: dict[str, Any]) -> dict[str, Any]:
        conflict = state.ensure_conflict(
            {
                "conflict_id": params["conflict_id"],
                "tenant_id": params["tenant_id"],
                "repo_id": params["repo_id"],
                "status": "open",
                "resolution": None,
                "recommendation": "review_latest_entity_version",
            },
        )
        return {"conflict": conflict, "recommendation": conflict["recommendation"]}

    return handler


def _get_conflict_handler(state: McpStateStore) -> ToolHandler:
    def handler(params: dict[str, Any]) -> dict[str, Any]:
        conflict = state.get_conflict(params["conflict_id"])
        return {"found": conflict is not None, "conflict": conflict}

    return handler


def _resolve_conflict_handler(state: McpStateStore) -> ToolHandler:
    def handler(params: dict[str, Any]) -> dict[str, Any]:
        conflict = state.resolve_conflict(
            {
                "conflict_id": params["conflict_id"],
                "tenant_id": params["tenant_id"],
                "repo_id": params["repo_id"],
                "recommendation": "manual_resolution_recorded",
            },
            params["resolution"],
        )
        return {"resolved": True, "conflict": conflict}

    return handler


def _get_salience_handler(graph_backend: Neo4jMcpBackend | None) -> ToolHandler:
    def handler(params: dict[str, Any]) -> dict[str, Any]:
        score = (
            graph_backend.salience(
                tenant_id=params["tenant_id"], repo_id=params["repo_id"], entity_id=params["entity_id"]
            )
            if graph_backend is not None
            else None
        )
        return {"entity_id": params["entity_id"], "salience": score, "available": score is not None}

    return handler


def _get_centroid_drift_handler(graph_backend: Neo4jMcpBackend | None) -> ToolHandler:
    def handler(params: dict[str, Any]) -> dict[str, Any]:
        drift = (
            graph_backend.centroid_drift(
                tenant_id=params["tenant_id"], repo_id=params["repo_id"], community_id=params["community_id"]
            )
            if graph_backend is not None
            else None
        )
        return {"community_id": params["community_id"], "centroid_drift": drift, "available": drift is not None}

    return handler


def _list_stale_handler(graph_backend: Neo4jMcpBackend | None) -> ToolHandler:
    def handler(params: dict[str, Any]) -> dict[str, Any]:
        stale = (
            graph_backend.stale_entities(tenant_id=params["tenant_id"], repo_id=params["repo_id"])
            if graph_backend is not None
            else []
        )
        return {"entities": [_entity_payload(entity) for entity in stale], "count": len(stale)}

    return handler


def _schema_fields(entity: EntityRecord | None) -> tuple[str, ...]:
    if entity is None:
        return ()
    fields = entity.properties.get("fields", ())
    if not isinstance(fields, list):
        return ()
    return tuple(str(field) for field in fields)


def _graph_bfs_handler(traversal: Neo4jGraphTraversal) -> ToolHandler:
    def handler(params: dict[str, Any]) -> dict[str, Any]:
        neighbors = traversal.bfs_neighbors(
            tenant_id=params["tenant_id"],
            repo_id=params["repo_id"],
            entity_id=params["entity_id"],
            depth=params.get("depth", 2),
            limit=params.get("limit", 10),
        )
        return {"neighbors": [asdict(neighbor) for neighbor in neighbors]}

    return handler


def _graph_bfs_unavailable_handler() -> ToolHandler:
    def handler(params: dict[str, Any]) -> dict[str, Any]:
        return {
            "neighbors": [],
            "available": False,
            "reason": "graph traversal backend is not configured",
            "entity_id": params["entity_id"],
        }

    return handler


def _graph_path_handler(traversal: Neo4jGraphTraversal) -> ToolHandler:
    def handler(params: dict[str, Any]) -> dict[str, Any]:
        path = traversal.shortest_path(
            tenant_id=params["tenant_id"],
            repo_id=params["repo_id"],
            source_id=params["source_id"],
            target_id=params["target_id"],
            max_hops=params.get("max_hops", 6),
        )
        return {"path": asdict(path) if path is not None else None}

    return handler


def _graph_path_unavailable_handler() -> ToolHandler:
    def handler(params: dict[str, Any]) -> dict[str, Any]:
        return {
            "path": None,
            "available": False,
            "reason": "graph traversal backend is not configured",
            "source_id": params["source_id"],
            "target_id": params["target_id"],
        }

    return handler


def _get_community_handler(traversal: Neo4jGraphTraversal) -> ToolHandler:
    def handler(params: dict[str, Any]) -> dict[str, Any]:
        members = traversal.community_for_entity(
            tenant_id=params["tenant_id"],
            repo_id=params["repo_id"],
            entity_id=params["entity_id"],
            limit=params.get("limit", 25),
        )
        return {"members": [asdict(member) for member in members]}

    return handler


def _get_community_unavailable_handler() -> ToolHandler:
    def handler(params: dict[str, Any]) -> dict[str, Any]:
        return {
            "members": [],
            "available": False,
            "reason": "graph traversal backend is not configured",
            "entity_id": params["entity_id"],
        }

    return handler


class McpToolRegistry:
    def __init__(
        self,
        *,
        principal_tenant_id: str,
        handlers: dict[str, ToolHandler] | None = None,
        graph_traversal: Neo4jGraphTraversal | None = None,
        graph_backend: Neo4jMcpBackend | None = None,
        vector_backend: PgvectorSearchBackend | None = None,
        runtime_state: McpStateStore | None = None,
        known_secrets: tuple[str, ...] = (),
    ) -> None:
        self._principal_tenant_id = principal_tenant_id
        self._known_secrets = known_secrets
        self._runtime_state = runtime_state or InMemoryMcpStateStore()
        self._definitions = {definition.name: definition for definition in TOOL_DEFINITIONS}
        self._handlers = {
            "search_code": _search_code_handler(graph_backend=graph_backend, vector_backend=vector_backend),
            "get_entity": _get_entity_handler(graph_backend),
            "graph_bfs": _graph_bfs_unavailable_handler(),
            "graph_path": _graph_path_unavailable_handler(),
            "get_community": _get_community_unavailable_handler(),
            "enqueue_reindex": _enqueue_reindex_handler(self._runtime_state),
            "schema_diff": _schema_diff_handler(graph_backend),
            "impact_analysis": _impact_analysis_handler(
                graph_backend=graph_backend,
                graph_traversal=graph_traversal,
            ),
            "reconcile": _reconcile_handler(self._runtime_state),
            "get_conflict": _get_conflict_handler(self._runtime_state),
            "resolve_conflict": _resolve_conflict_handler(self._runtime_state),
            "get_salience": _get_salience_handler(graph_backend),
            "get_centroid_drift": _get_centroid_drift_handler(graph_backend),
            "list_stale": _list_stale_handler(graph_backend),
        }
        if graph_traversal is not None:
            self._handlers["graph_bfs"] = _graph_bfs_handler(graph_traversal)
            self._handlers["graph_path"] = _graph_path_handler(graph_traversal)
            self._handlers["get_community"] = _get_community_handler(graph_traversal)
        if handlers:
            self._handlers.update(handlers)

    def list_tools(self) -> list[dict[str, Any]]:
        return [definition.schema() for definition in self._definitions.values()]

    def call_tool(self, name: str, params: dict[str, Any]) -> dict[str, Any]:
        status = "success"
        try:
            with observe_histogram(MCP_TOOL_LATENCY, name), traced_span(
                "mcp.tool.execute",
                correlation_id=str(params.get("correlation_id", "")),
                tool_name=name,
                tenant_id=params.get("tenant_id"),
                repo_id=params.get("repo_id"),
            ):
                definition = self._definitions.get(name)
                if definition is None:
                    raise ValueError(f"unknown MCP tool: {name}")
                try:
                    validated = definition.params_model.model_validate(params)
                except ValidationError as exc:
                    raise ValueError(f"invalid arguments for tool {name}: {exc}") from exc

                query_text = getattr(validated, "query", None)
                validate_tool_payload(
                    tenant_id=validated.tenant_id,
                    principal_tenant_id=self._principal_tenant_id,
                    query_text=query_text,
                ).require_allowed()

                result = self._handlers[name](validated.model_dump())
                _quarantined, decision = screen_tool_response(
                    json.dumps(result, sort_keys=True, default=str), known_secrets=self._known_secrets
                )
                decision.require_allowed()
                return result
        except Exception:
            status = "error"
            raise
        finally:
            MCP_TOOL_CALLS.labels(name, status).inc()

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
