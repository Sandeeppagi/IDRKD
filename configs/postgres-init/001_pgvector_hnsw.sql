CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS knowledge_embeddings (
  id text PRIMARY KEY,
  tenant_id text NOT NULL,
  repo_id text NOT NULL,
  entity_id text,
  source text NOT NULL,
  content_hash text NOT NULL,
  embedding_model text NOT NULL,
  embedding vector(1536) NOT NULL,
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS knowledge_embeddings_tenant_repo_idx
ON knowledge_embeddings (tenant_id, repo_id);

CREATE INDEX IF NOT EXISTS knowledge_embeddings_entity_idx
ON knowledge_embeddings (entity_id);

CREATE INDEX IF NOT EXISTS knowledge_embeddings_hnsw_cosine_idx
ON knowledge_embeddings
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
