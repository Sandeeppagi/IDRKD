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
ON ingestion_transactions (updated_at)
WHERE status = 'repair_required';

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
ON ingestion_outbox (created_at)
WHERE published_at IS NULL;
