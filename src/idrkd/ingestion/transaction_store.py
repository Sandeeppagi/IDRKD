"""Postgres-backed ingestion saga journal and transactional outbox."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

import psycopg
from psycopg.rows import dict_row


TERMINAL_STATUSES = frozenset({"committed", "failed", "rolled_back"})

INGESTION_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS ingestion_transactions (
  event_id text PRIMARY KEY,
  correlation_id text NOT NULL,
  status text NOT NULL CHECK (
    status IN ('processing', 'committed', 'rolled_back', 'failed', 'repair_required')
  ),
  stage text NOT NULL,
  event_payload jsonb NOT NULL,
  staged_plan jsonb NOT NULL DEFAULT '{}'::jsonb,
  error text,
  attempts integer NOT NULL DEFAULT 1,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  completed_at timestamptz
);
CREATE INDEX IF NOT EXISTS ingestion_transactions_repair_idx
ON ingestion_transactions (updated_at) WHERE status = 'repair_required';
CREATE TABLE IF NOT EXISTS ingestion_outbox (
  id text PRIMARY KEY,
  event_id text NOT NULL REFERENCES ingestion_transactions(event_id) ON DELETE CASCADE,
  topic text NOT NULL,
  message_key text NOT NULL,
  payload jsonb NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  published_at timestamptz
);
CREATE INDEX IF NOT EXISTS ingestion_outbox_pending_idx
ON ingestion_outbox (created_at) WHERE published_at IS NULL;
"""


@dataclass(frozen=True)
class TransactionRecord:
    event_id: str
    status: str
    stage: str
    payload: dict[str, Any]
    plan: dict[str, Any]
    error: str | None


@dataclass(frozen=True)
class OutboxMessage:
    id: str
    event_id: str
    topic: str
    message_key: str
    payload: dict[str, Any]


class PostgresIngestionStore:
    def __init__(self, dsn: str) -> None:
        self._dsn = dsn

    def apply_schema(self) -> None:
        with psycopg.connect(self._dsn) as conn:
            with conn.cursor() as cursor:
                cursor.execute(INGESTION_SCHEMA_SQL)

    def begin(self, *, event_id: str, correlation_id: str, payload: dict[str, Any]) -> TransactionRecord:
        sql = """
        INSERT INTO ingestion_transactions (
          event_id, correlation_id, status, stage, event_payload, attempts
        ) VALUES (%(event_id)s, %(correlation_id)s, 'processing', 'received', %(payload)s::jsonb, 1)
        ON CONFLICT (event_id) DO UPDATE SET
          attempts = ingestion_transactions.attempts + 1,
          updated_at = now()
        RETURNING event_id, status, stage, event_payload AS payload,
                  staged_plan AS plan, error
        """
        with psycopg.connect(self._dsn, row_factory=dict_row) as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    sql,
                    {
                        "event_id": event_id,
                        "correlation_id": correlation_id,
                        "payload": json.dumps(payload, sort_keys=True),
                    },
                )
                row = cursor.fetchone()
        if row is None:
            raise RuntimeError("Postgres did not return the ingestion transaction")
        return TransactionRecord(**row)

    def stage(self, event_id: str, *, stage: str, plan: dict[str, Any]) -> None:
        self._update(event_id, status="processing", stage=stage, plan=plan)

    def mark_stage(self, event_id: str, stage: str) -> None:
        self._update(event_id, status="processing", stage=stage)

    def finish(
        self,
        event_id: str,
        *,
        status: str,
        stage: str,
        error: str | None = None,
        outbox: list[tuple[str, str, dict[str, Any]]] | None = None,
    ) -> None:
        with psycopg.connect(self._dsn) as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE ingestion_transactions
                    SET status = %(status)s, stage = %(stage)s, error = %(error)s,
                        completed_at = CASE WHEN %(terminal)s THEN now() ELSE completed_at END,
                        updated_at = now()
                    WHERE event_id = %(event_id)s
                    """,
                    {
                        "event_id": event_id,
                        "status": status,
                        "stage": stage,
                        "error": error,
                        "terminal": status in TERMINAL_STATUSES,
                    },
                )
                for topic, message_key, payload in outbox or []:
                    message_id = _outbox_id(event_id, topic, message_key)
                    cursor.execute(
                        """
                        INSERT INTO ingestion_outbox (id, event_id, topic, message_key, payload)
                        VALUES (%(id)s, %(event_id)s, %(topic)s, %(message_key)s, %(payload)s::jsonb)
                        ON CONFLICT (id) DO NOTHING
                        """,
                        {
                            "id": message_id,
                            "event_id": event_id,
                            "topic": topic,
                            "message_key": message_key,
                            "payload": json.dumps(payload, sort_keys=True),
                        },
                    )

    def get(self, event_id: str) -> TransactionRecord | None:
        with psycopg.connect(self._dsn, row_factory=dict_row) as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT event_id, status, stage, event_payload AS payload,
                           staged_plan AS plan, error
                    FROM ingestion_transactions WHERE event_id = %s
                    """,
                    (event_id,),
                )
                row = cursor.fetchone()
        return TransactionRecord(**row) if row else None

    def repair_required(self, *, limit: int = 25) -> list[TransactionRecord]:
        with psycopg.connect(self._dsn, row_factory=dict_row) as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT event_id, status, stage, event_payload AS payload,
                           staged_plan AS plan, error
                    FROM ingestion_transactions
                    WHERE status = 'repair_required'
                    ORDER BY updated_at ASC LIMIT %s
                    """,
                    (limit,),
                )
                rows = cursor.fetchall()
        return [TransactionRecord(**row) for row in rows]

    def pending_outbox(self, event_id: str, *, limit: int = 100) -> list[OutboxMessage]:
        with psycopg.connect(self._dsn, row_factory=dict_row) as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT id, event_id, topic, message_key, payload
                    FROM ingestion_outbox
                    WHERE event_id = %s AND published_at IS NULL
                    ORDER BY created_at ASC LIMIT %s
                    """,
                    (event_id, limit),
                )
                rows = cursor.fetchall()
        return [OutboxMessage(**row) for row in rows]

    def mark_published(self, message_id: str) -> None:
        with psycopg.connect(self._dsn) as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "UPDATE ingestion_outbox SET published_at = now() WHERE id = %s",
                    (message_id,),
                )

    def _update(
        self,
        event_id: str,
        *,
        status: str,
        stage: str,
        plan: dict[str, Any] | None = None,
    ) -> None:
        with psycopg.connect(self._dsn) as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE ingestion_transactions
                    SET status = %(status)s, stage = %(stage)s,
                        staged_plan = COALESCE(%(plan)s::jsonb, staged_plan), updated_at = now()
                    WHERE event_id = %(event_id)s
                    """,
                    {
                        "event_id": event_id,
                        "status": status,
                        "stage": stage,
                        "plan": json.dumps(plan, sort_keys=True) if plan is not None else None,
                    },
                )


def _outbox_id(event_id: str, topic: str, message_key: str) -> str:
    digest = hashlib.sha256(f"{event_id}|{topic}|{message_key}".encode()).hexdigest()[:32]
    return f"out_{digest}"
