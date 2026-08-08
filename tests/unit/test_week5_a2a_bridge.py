import asyncio
import json
import threading
from dataclasses import dataclass

import httpx
from a2a.client import ClientConfig, create_client
from a2a.helpers import new_task_from_user_message, new_text_message
from a2a.server.events import EventQueue
from a2a.types import Role, SendMessageRequest, TaskState

from idrkd.a2a import (
    A2ABridge,
    A2ATaskStateStore,
    IdrkdAgentExecutor,
    IdrkdTaskState,
    agent_card_payload,
    build_a2a_app,
    build_idrkd_agent_card,
    sign_agent_card,
)
from idrkd.mcp.tools import McpToolRegistry


def _build_card():
    return build_idrkd_agent_card(
        name="IDRKD Reconciler",
        description="Executes IDRKD MCP tools over A2A.",
        version="0.1.0",
        endpoint="http://testserver/",
        capabilities=("mcp.delegate",),
        legacy_endpoint="http://testserver/legacy",
    )


class _EventQueue(EventQueue):
    def __init__(self) -> None:
        self.events = []

    async def enqueue_event(self, event) -> None:  # noqa: ANN001
        self.events.append(event)


@dataclass
class _Context:
    message: object | None
    current_task: object | None

    @property
    def task_id(self) -> str | None:
        return self.current_task.id if self.current_task is not None else None

    @property
    def context_id(self) -> str | None:
        return self.current_task.context_id if self.current_task is not None else None


def _status_states(events) -> list[TaskState]:  # noqa: ANN001
    return [event.status.state for event in events if hasattr(event, "status")]


def test_agent_card_advertises_both_v1_and_legacy_v0_3_interfaces() -> None:
    card = _build_card()

    versions = {iface.protocol_version for iface in card.supported_interfaces}

    assert versions == {"1.0", "0.3"}


async def test_agent_card_route_serves_the_signed_card() -> None:
    card = _build_card()
    bridge = A2ABridge(local_card=card, shared_secret="shared-secret")
    signed = bridge.signed_card()
    registry = McpToolRegistry(principal_tenant_id="tenant-a")
    app = build_a2a_app(card, registry=registry)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as http_client:
        response = await http_client.get("/.well-known/agent-card.json")

    assert response.status_code == 200
    served = response.json()
    assert served["name"] == "IDRKD Reconciler"
    # The wire route uses protobuf's camelCase JSON convention; our HMAC
    # signature covers the snake_case payload independently (verified below).
    assert served["supportedInterfaces"][0]["protocolVersion"] == "1.0"
    assert sign_agent_card(card, "shared-secret").signature == signed.signature
    assert agent_card_payload(card)["name"] == served["name"]


async def test_message_send_round_trips_through_executor_to_completed_task() -> None:
    card = _build_card()
    registry = McpToolRegistry(
        principal_tenant_id="tenant-a",
        handlers={"search_code": lambda params: {"hits": [params["query"]]}},
    )
    app = build_a2a_app(card, registry=registry)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as http_client:
        client = await create_client(agent=card, client_config=ClientConfig(streaming=False, httpx_client=http_client))
        call = json.dumps(
            {"name": "search_code", "arguments": {"tenant_id": "tenant-a", "repo_id": "repo", "query": "customer api"}}
        )
        request = SendMessageRequest(message=new_text_message(call, role=Role.ROLE_USER))

        chunks = [chunk async for chunk in client.send_message(request)]
        await client.close()

    assert len(chunks) == 1
    task = chunks[0].task
    assert task.status.state == TaskState.TASK_STATE_COMPLETED
    artifact_text = task.artifacts[0].parts[0].text
    assert json.loads(artifact_text) == {"hits": ["customer api"]}


async def test_message_send_with_unknown_tool_fails_the_task() -> None:
    card = _build_card()
    registry = McpToolRegistry(principal_tenant_id="tenant-a")
    app = build_a2a_app(card, registry=registry)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as http_client:
        client = await create_client(agent=card, client_config=ClientConfig(streaming=False, httpx_client=http_client))
        call = json.dumps({"name": "no_such_tool", "arguments": {}})
        request = SendMessageRequest(message=new_text_message(call, role=Role.ROLE_USER))

        chunks = [chunk async for chunk in client.send_message(request)]
        await client.close()

    assert chunks[0].task.status.state == TaskState.TASK_STATE_FAILED


async def test_cancel_pending_task_transitions_to_canceled() -> None:
    registry = McpToolRegistry(principal_tenant_id="tenant-a")
    state_store = A2ATaskStateStore()
    executor = IdrkdAgentExecutor(registry, state_store)
    message = new_text_message("{}", role=Role.ROLE_USER)
    task = new_task_from_user_message(message)
    await state_store.submit(task.id)
    queue = _EventQueue()

    await executor.cancel(_Context(message=message, current_task=task), queue)  # type: ignore[arg-type]

    record = await state_store.get(task.id)
    assert record is not None
    assert record.state == IdrkdTaskState.CANCELED
    assert _status_states(queue.events) == [TaskState.TASK_STATE_CANCELED]


async def test_cancel_running_task_requests_cancel_and_skips_completion() -> None:
    started = threading.Event()
    release = threading.Event()

    def blocking_handler(params):  # noqa: ANN001
        started.set()
        assert release.wait(timeout=5)
        return {"hits": [params["query"]]}

    registry = McpToolRegistry(
        principal_tenant_id="tenant-a",
        handlers={"search_code": blocking_handler},
    )
    state_store = A2ATaskStateStore()
    executor = IdrkdAgentExecutor(registry, state_store)
    call = json.dumps(
        {"name": "search_code", "arguments": {"tenant_id": "tenant-a", "repo_id": "repo", "query": "customer api"}}
    )
    message = new_text_message(call, role=Role.ROLE_USER)
    task = new_task_from_user_message(message)
    queue = _EventQueue()
    context = _Context(message=message, current_task=task)

    execute_task = asyncio.create_task(executor.execute(context, queue))  # type: ignore[arg-type]
    assert await asyncio.to_thread(started.wait, 5)
    assert task.id in await state_store.running_task_ids()

    await executor.cancel(context, queue)  # type: ignore[arg-type]
    release.set()
    await execute_task

    record = await state_store.get(task.id)
    assert record is not None
    assert record.state == IdrkdTaskState.CANCELED
    assert TaskState.TASK_STATE_CANCELED in _status_states(queue.events)
    assert TaskState.TASK_STATE_COMPLETED not in _status_states(queue.events)
    assert not any(hasattr(event, "artifact") for event in queue.events)


async def test_cancel_completed_task_is_rejected() -> None:
    registry = McpToolRegistry(principal_tenant_id="tenant-a")
    state_store = A2ATaskStateStore()
    executor = IdrkdAgentExecutor(registry, state_store)
    message = new_text_message("{}", role=Role.ROLE_USER)
    task = new_task_from_user_message(message)
    await state_store.submit(task.id)
    assert await state_store.complete(task.id) is True

    try:
        await executor.cancel(_Context(message=message, current_task=task), _EventQueue())  # type: ignore[arg-type]
    except ValueError as exc:
        assert "cannot be canceled" in str(exc)
    else:
        raise AssertionError("completed task cancellation should fail")
