"""A2A task-state tracking and cancellation decisions."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import StrEnum


class IdrkdTaskState(StrEnum):
    SUBMITTED = "submitted"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


@dataclass
class IdrkdTaskRecord:
    task_id: str
    state: IdrkdTaskState
    cancellation_requested: bool = False


class A2ATaskStateStore:
    """Small async-safe state store for executor-level cancellation."""

    def __init__(self) -> None:
        self._records: dict[str, IdrkdTaskRecord] = {}
        self._running: set[str] = set()
        self._lock = asyncio.Lock()

    async def submit(self, task_id: str) -> IdrkdTaskRecord:
        async with self._lock:
            record = self._records.get(task_id)
            if record is None:
                record = IdrkdTaskRecord(task_id=task_id, state=IdrkdTaskState.SUBMITTED)
                self._records[task_id] = record
            return IdrkdTaskRecord(**record.__dict__)

    async def start(self, task_id: str) -> IdrkdTaskRecord:
        async with self._lock:
            record = self._records.setdefault(
                task_id,
                IdrkdTaskRecord(task_id=task_id, state=IdrkdTaskState.SUBMITTED),
            )
            if record.state == IdrkdTaskState.CANCELED:
                return IdrkdTaskRecord(**record.__dict__)
            if record.state in {IdrkdTaskState.COMPLETED, IdrkdTaskState.FAILED}:
                return IdrkdTaskRecord(**record.__dict__)
            record.state = IdrkdTaskState.RUNNING
            self._running.add(task_id)
            return IdrkdTaskRecord(**record.__dict__)

    async def complete(self, task_id: str) -> bool:
        return await self._terminal(task_id, IdrkdTaskState.COMPLETED)

    async def fail(self, task_id: str) -> bool:
        return await self._terminal(task_id, IdrkdTaskState.FAILED)

    async def request_cancel(self, task_id: str) -> bool:
        async with self._lock:
            record = self._records.setdefault(
                task_id,
                IdrkdTaskRecord(task_id=task_id, state=IdrkdTaskState.SUBMITTED),
            )
            if record.state in {IdrkdTaskState.COMPLETED, IdrkdTaskState.FAILED, IdrkdTaskState.CANCELED}:
                return False
            record.cancellation_requested = True
            record.state = IdrkdTaskState.CANCELED
            self._running.discard(task_id)
            return True

    async def is_canceled(self, task_id: str) -> bool:
        async with self._lock:
            record = self._records.get(task_id)
            return bool(record and record.state == IdrkdTaskState.CANCELED)

    async def get(self, task_id: str) -> IdrkdTaskRecord | None:
        async with self._lock:
            record = self._records.get(task_id)
            return IdrkdTaskRecord(**record.__dict__) if record is not None else None

    async def running_task_ids(self) -> tuple[str, ...]:
        async with self._lock:
            return tuple(sorted(self._running))

    async def _terminal(self, task_id: str, state: IdrkdTaskState) -> bool:
        async with self._lock:
            record = self._records.setdefault(
                task_id,
                IdrkdTaskRecord(task_id=task_id, state=IdrkdTaskState.SUBMITTED),
            )
            if record.state == IdrkdTaskState.CANCELED:
                return False
            record.state = state
            self._running.discard(task_id)
            return True
