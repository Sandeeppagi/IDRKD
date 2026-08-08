"""Dispatches A2A tasks into the IDRKD MCP tool registry."""

from __future__ import annotations

import asyncio
import json

from a2a.helpers import get_message_text, new_task_from_user_message, new_text_message, new_text_part
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import TaskState

from idrkd.a2a.task_state import A2ATaskStateStore
from idrkd.mcp.tools import McpToolRegistry


class IdrkdAgentExecutor(AgentExecutor):
    """Executes an A2A task by dispatching a `{"name", "arguments"}` JSON
    tool-call payload, carried as the message text, to the MCP registry."""

    def __init__(self, registry: McpToolRegistry, task_states: A2ATaskStateStore | None = None) -> None:
        self._registry = registry
        self._task_states = task_states or A2ATaskStateStore()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        message = context.message
        if message is None:
            raise ValueError("A2A request is missing a message")

        task = context.current_task
        if task is None:
            task = new_task_from_user_message(message)
            await event_queue.enqueue_event(task)

        await self._task_states.submit(task.id)
        updater = TaskUpdater(event_queue=event_queue, task_id=task.id, context_id=task.context_id)
        started = await self._task_states.start(task.id)
        if started.cancellation_requested or await self._task_states.is_canceled(task.id):
            return
        await updater.update_status(
            state=TaskState.TASK_STATE_WORKING,
            message=new_text_message("Processing IDRKD tool call..."),
        )

        query = get_message_text(message)
        try:
            call = json.loads(query) if query else {}
            tool_name = str(call["name"])
            arguments = dict(call["arguments"])
            if await self._task_states.is_canceled(task.id):
                return
            result = await asyncio.to_thread(self._registry.call_tool, tool_name, arguments)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError, PermissionError) as exc:
            if await self._task_states.is_canceled(task.id):
                return
            await self._task_states.fail(task.id)
            await updater.update_status(
                state=TaskState.TASK_STATE_FAILED,
                message=new_text_message(str(exc)),
            )
            return

        if await self._task_states.is_canceled(task.id):
            return
        await updater.add_artifact(
            parts=[new_text_part(text=json.dumps(result), media_type="application/json")]
        )
        await self._task_states.complete(task.id)
        await updater.update_status(
            state=TaskState.TASK_STATE_COMPLETED,
            message=new_text_message("IDRKD tool call completed."),
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        task = context.current_task
        task_id = task.id if task is not None else context.task_id
        context_id = task.context_id if task is not None else context.context_id
        if task_id is None or context_id is None:
            raise ValueError("A2A cancellation requires a task id and context id")
        if not await self._task_states.request_cancel(task_id):
            raise ValueError(f"task {task_id} cannot be canceled")
        updater = TaskUpdater(event_queue=event_queue, task_id=task_id, context_id=context_id)
        await updater.cancel(message=new_text_message("IDRKD task canceled."))

    @property
    def task_states(self) -> A2ATaskStateStore:
        return self._task_states
