"""Re-index queue adapters."""

from __future__ import annotations

from collections import deque
import json
from typing import Any, cast
from typing import Protocol

import redis

from idrkd.drift.events import ReindexRequest


class ReindexQueue(Protocol):
    def enqueue(self, request: ReindexRequest) -> int:
        ...

    def dequeue(self) -> ReindexRequest | None:
        ...


class InMemoryReindexQueue:
    def __init__(self) -> None:
        self._items: deque[ReindexRequest] = deque()

    def enqueue(self, request: ReindexRequest) -> int:
        self._items.append(request)
        return len(self._items)

    def dequeue(self) -> ReindexRequest | None:
        return self._items.popleft() if self._items else None


class RedisReindexQueue:
    def __init__(self, redis_url: str, *, key: str = "idrkd:reindex") -> None:
        self._client = redis.Redis.from_url(redis_url)
        self._key = key

    def enqueue(self, request: ReindexRequest) -> int:
        return cast(int, self._client.lpush(self._key, json.dumps(request.payload(), sort_keys=True)))

    def dequeue(self) -> ReindexRequest | None:
        raw = self._client.rpop(self._key)
        if raw is None:
            return None
        payload = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else str(raw))
        return reindex_request_from_payload(payload)


class RedisMcpReindexQueue(RedisReindexQueue):
    """Consume MCP `enqueue_reindex` queue items as `ReindexRequest`s."""

    def __init__(self, redis_url: str, *, key: str = "idrkd:mcp:reindex") -> None:
        super().__init__(redis_url, key=key)

    def enqueue(self, request: ReindexRequest) -> int:
        payload = request.payload()
        payload["status"] = "queued"
        return cast(int, self._client.rpush(self._key, json.dumps(payload, sort_keys=True)))


def reindex_request_from_payload(payload: dict[str, Any]) -> ReindexRequest:
    return ReindexRequest(
        tenant_id=str(payload["tenant_id"]),
        repo_id=str(payload["repo_id"]),
        entity_id=str(payload["entity_id"]),
        reason=str(payload.get("reason", "mcp_enqueue_reindex")),
        depth=int(payload.get("depth", 2)),
    )
