# Spec: Graphify Bootstrap

## Goal

Run Graphify against a public Telstra Python repository and pre-populate Neo4j
with a code graph for early integration testing before the custom IDRKD
pipeline is fully live.

## Scope

In scope:

- Build a Docker image with Graphify installed.
- Run Graphify against `https://github.com/telstra/MessagingAPI-SDK-python.git`.
- Persist Graphify `graph.json` into Neo4j.
- Verify node and edge counts.

Out of scope:

- Semantic LLM extraction.
- Replacing the custom IDRKD ingestion pipeline.
- Treating Graphify as the long-term graph writer.

## Contract

Input repository:

- Remote: `https://github.com/telstra/MessagingAPI-SDK-python.git`
- Local ignored path: `data/raw/telstra-messaging-api-sdk-python`

Generated output:

- `data/processed/graphify-out/telstra-messaging-api-sdk-python/graph.json`

Neo4j graph contract:

- Nodes use label `:GraphifyNode`.
- Relationships use type `:GRAPHIFY_RELATION`.
- Every node has `tenant_id`, `repo_id`, `graphify_id`, `created_at`,
  `updated_at`, and `lamport_clock`.
- Every relationship has `tenant_id`, `repo_id`, `relation`, `created_at`,
  `updated_at`, and `lamport_clock`.
- Missing external endpoints are preserved as stub nodes so all Graphify edges
  are importable.

## Implementation

- `docker/graphify.Dockerfile`
- `docker/docker-compose.yml` service `graphify`
- `scripts/bootstrap_graphify_telstra.sh`
- `src/idrkd/graph/graphify_importer.py`
- `tests/unit/test_graphify_importer.py`
- `docs/references/graphify-telstra-bootstrap.md`

## Acceptance Criteria

- Graphify Docker image builds successfully.
- Graphify runs against the Telstra Python repo.
- Neo4j is populated from Graphify output.
- The importer is idempotent via Neo4j `MERGE`.
- Node and edge counts are recorded.

## Verification

Run:

```bash
scripts/bootstrap_graphify_telstra.sh
```

Verify Neo4j counts:

```cypher
MATCH (n:GraphifyNode {repo_id: 'telstra-messaging-api-sdk-python'})
RETURN count(n) AS telstra_nodes;

MATCH (:GraphifyNode {repo_id: 'telstra-messaging-api-sdk-python'})
      -[r:GRAPHIFY_RELATION]->
      (:GraphifyNode {repo_id: 'telstra-messaging-api-sdk-python'})
RETURN count(r) AS telstra_edges;
```

Expected verified counts:

- `telstra_nodes`: `402`
- `telstra_edges`: `505`

## Notes

The installed `graphifyy` CLI does not expose the planned
`--neo4j-push bolt://localhost:7687` option. IDRKD provides an equivalent bridge
by running `graphify update` and then importing `graph.json` through
`idrkd.graph.graphify_importer`.
