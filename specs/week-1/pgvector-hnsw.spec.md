# Spec: pgvector HNSW

## Goal

Provide a PostgreSQL vector storage surface for early RAG integration testing,
with pgvector installed and an HNSW index ready for approximate nearest-neighbor
search.

## Contract

Table: `knowledge_embeddings`

Required columns:

- `id text PRIMARY KEY`
- `tenant_id text NOT NULL`
- `repo_id text NOT NULL`
- `entity_id text`
- `source text NOT NULL`
- `content_hash text NOT NULL`
- `embedding_model text NOT NULL`
- `embedding vector(1536) NOT NULL`
- `metadata jsonb NOT NULL`
- `created_at timestamptz NOT NULL`
- `updated_at timestamptz NOT NULL`

Indexes:

- `knowledge_embeddings_tenant_repo_idx`
- `knowledge_embeddings_entity_idx`
- `knowledge_embeddings_hnsw_cosine_idx`

HNSW contract:

```sql
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64)
```

## Implementation

- `configs/postgres-init/001_pgvector_hnsw.sql`
- `docker/docker-compose.yml` Postgres init mount

## Acceptance Criteria

- `vector` extension exists.
- `knowledge_embeddings` table exists.
- HNSW cosine index exists.
- Init SQL is mounted into fresh Postgres containers.

## Verification

```bash
docker exec docker-postgres-1 psql -U idrkd -d idrkd -tAc \
  "SELECT extname FROM pg_extension WHERE extname='vector';
   SELECT indexname, indexdef
   FROM pg_indexes
   WHERE tablename='knowledge_embeddings'
     AND indexdef ILIKE '%hnsw%';"
```

Expected:

```text
vector
knowledge_embeddings_hnsw_cosine_idx | USING hnsw (embedding vector_cosine_ops)
```
