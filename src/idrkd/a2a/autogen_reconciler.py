"""AutoGen reconciliation agent exposed through the IDRKD A2A boundary."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
import json
from typing import Any, Protocol

from a2a.helpers import get_message_text, new_task_from_user_message, new_text_message, new_text_part
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import TaskState
from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import BaseChatMessage, TextMessage
from autogen_core import CancellationToken

from idrkd.a2a.bridge import A2ABridge
from idrkd.a2a.client import IdrkdA2AClient
from idrkd.a2a.task_state import A2ATaskStateStore
from idrkd.mcp.tools import McpToolRegistry


@dataclass(frozen=True)
class ReconciliationRequest:
    tenant_id: str
    repo_id: str
    conflict_id: str
    query: str
    evidence: tuple[str, ...] = ()
    traceparent: str = ""


@dataclass(frozen=True)
class ReconciliationResult:
    conflict_id: str
    recommendation: str
    details: dict[str, Any]
    framework: str = "autogen"

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> ReconciliationResult:
        return cls(
            conflict_id=str(payload["conflict_id"]),
            recommendation=str(payload["recommendation"]),
            details=dict(payload.get("details", {})),
            framework=str(payload.get("framework", "autogen")),
        )


class ReconciliationDelegate(Protocol):
    async def reconcile(self, request: ReconciliationRequest) -> ReconciliationResult:
        ...


class AutoGenReconciliationAgent(BaseChatAgent):
    """A deterministic AutoGen agent that invokes the governed MCP tool."""

    def __init__(self, registry: McpToolRegistry) -> None:
        super().__init__(
            name="idrkd_autogen_reconciler",
            description="Reconciles repository conflicts through tenant-scoped MCP tools.",
        )
        self._registry = registry

    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        return (TextMessage,)

    async def on_messages(
        self,
        messages: Sequence[BaseChatMessage],
        cancellation_token: CancellationToken,
    ) -> Response:
        if not messages:
            raise ValueError("AutoGen reconciliation requires an input message")
        message = messages[-1]
        if not isinstance(message, TextMessage):
            raise TypeError("AutoGen reconciliation accepts text JSON messages only")
        payload = _reconciliation_payload(message.content)
        _raise_if_cancelled(cancellation_token)
        result = await asyncio.to_thread(
            self._registry.call_tool,
            "reconcile",
            {
                "tenant_id": str(payload["tenant_id"]),
                "repo_id": str(payload["repo_id"]),
                "conflict_id": str(payload["conflict_id"]),
            },
        )
        _raise_if_cancelled(cancellation_token)
        response = {
            "conflict_id": str(payload["conflict_id"]),
            "recommendation": str(result.get("recommendation", "manual_review_required")),
            "details": result,
            "framework": "autogen",
            "agent": self.name,
        }
        return Response(
            chat_message=TextMessage(
                content=json.dumps(response, sort_keys=True),
                source=self.name,
            )
        )

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        _raise_if_cancelled(cancellation_token)


class AutoGenReconciliationExecutor(AgentExecutor):
    """Runs an AutoGen reconciliation agent for each A2A task."""

    def __init__(
        self,
        agent: AutoGenReconciliationAgent,
        task_states: A2ATaskStateStore | None = None,
    ) -> None:
        self._agent = agent
        self._task_states = task_states or A2ATaskStateStore()
        self._cancellation_tokens: dict[str, CancellationToken] = {}

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        message = context.message
        if message is None:
            raise ValueError("A2A request is missing a message")
        task = context.current_task
        if task is None:
            task = new_task_from_user_message(message)
            await event_queue.enqueue_event(task)

        await self._task_states.submit(task.id)
        started = await self._task_states.start(task.id)
        if started.cancellation_requested:
            return
        updater = TaskUpdater(event_queue=event_queue, task_id=task.id, context_id=task.context_id)
        await updater.update_status(
            state=TaskState.TASK_STATE_WORKING,
            message=new_text_message("AutoGen reconciliation is running."),
        )
        token = CancellationToken()
        self._cancellation_tokens[task.id] = token
        try:
            result = await self._agent.run(
                task=get_message_text(message) or "{}",
                cancellation_token=token,
            )
            output = result.messages[-1]
            if not isinstance(output, TextMessage):
                raise TypeError("AutoGen reconciler returned a non-text response")
            if await self._task_states.is_canceled(task.id):
                return
            await updater.add_artifact(
                parts=[new_text_part(text=output.content, media_type="application/json")]
            )
            await self._task_states.complete(task.id)
            await updater.update_status(
                state=TaskState.TASK_STATE_COMPLETED,
                message=new_text_message("AutoGen reconciliation completed."),
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError, PermissionError) as exc:
            if await self._task_states.is_canceled(task.id):
                return
            await self._task_states.fail(task.id)
            await updater.update_status(
                state=TaskState.TASK_STATE_FAILED,
                message=new_text_message(str(exc)),
            )
        finally:
            self._cancellation_tokens.pop(task.id, None)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        task = context.current_task
        task_id = task.id if task is not None else context.task_id
        context_id = task.context_id if task is not None else context.context_id
        if task_id is None or context_id is None:
            raise ValueError("A2A cancellation requires a task id and context id")
        if not await self._task_states.request_cancel(task_id):
            raise ValueError(f"task {task_id} cannot be canceled")
        token = self._cancellation_tokens.get(task_id)
        if token is not None:
            token.cancel()
        updater = TaskUpdater(event_queue=event_queue, task_id=task_id, context_id=context_id)
        await updater.cancel(message=new_text_message("AutoGen reconciliation canceled."))


class A2AReconciliationClient(ReconciliationDelegate):
    """LangGraph-facing adapter over the official A2A SDK client."""

    def __init__(self, *, client: IdrkdA2AClient, bridge: A2ABridge) -> None:
        self._client = client
        self._bridge = bridge

    async def reconcile(self, request: ReconciliationRequest) -> ReconciliationResult:
        message = self._bridge.build_message(
            recipient="idrkd_autogen_reconciler",
            task="reconcile",
            payload={
                "tenant_id": request.tenant_id,
                "repo_id": request.repo_id,
                "conflict_id": request.conflict_id,
                "query": request.query,
                "evidence": list(request.evidence),
            },
            traceparent=request.traceparent,
        )
        completed_task: Any | None = None
        async for chunk in self._client.send_message(json.dumps(message.to_payload(), sort_keys=True)):
            task = getattr(chunk, "task", None)
            if task is not None:
                completed_task = task
        if completed_task is None:
            raise RuntimeError("A2A reconciler returned no task")
        if completed_task.status.state != TaskState.TASK_STATE_COMPLETED:
            raise RuntimeError(f"A2A reconciliation failed: {completed_task.status.state}")
        if not completed_task.artifacts or not completed_task.artifacts[-1].parts:
            raise RuntimeError("A2A reconciler returned no artifact")
        text = completed_task.artifacts[-1].parts[-1].text
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise TypeError("A2A reconciliation artifact must be a JSON object")
        return ReconciliationResult.from_payload(payload)

    async def close(self) -> None:
        await self._client.close()


def _reconciliation_payload(content: str) -> dict[str, Any]:
    value = json.loads(content)
    if not isinstance(value, dict):
        raise TypeError("reconciliation message must be a JSON object")
    if value.get("task") == "reconcile" and isinstance(value.get("payload"), dict):
        value = dict(value["payload"])
    required = ("tenant_id", "repo_id", "conflict_id")
    missing = [key for key in required if not str(value.get(key, "")).strip()]
    if missing:
        raise ValueError(f"reconciliation message is missing: {', '.join(missing)}")
    return value


def _raise_if_cancelled(token: CancellationToken) -> None:
    if token.is_cancelled():
        raise asyncio.CancelledError
