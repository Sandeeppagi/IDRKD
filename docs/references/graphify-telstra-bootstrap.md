# Graphify Bootstrap - Telstra Messaging API SDK Python

Date: 2026-08-05

Input repository:

- `https://github.com/telstra/MessagingAPI-SDK-python.git`
- Local ignored path: `data/raw/telstra-messaging-api-sdk-python`

Run command:

```bash
docker compose --profile graphify up graphify
```

`scripts/bootstrap_graphify_telstra.sh` runs the same profile command.

Observed Graphify output:

```text
Rebuilt (no clustering): 376 nodes, 764 edges
```

Observed Neo4j import output:

```text
Imported Graphify graph into Neo4j: 400 Neo4j nodes (376 unique Graphify nodes from 376 entries + 24 stubs), 764 relationships (764 Graphify edges).
{"expected_nodes": 400, "expected_relationships": 764, "imported_nodes": 400, "imported_relationships": 764, "orphan_relationships": 0, "passed": true, "repo_id": "telstra-messaging-api-sdk-python", "source_label": "telstra-messaging-api-sdk-python", "tenant_id": "tenant-live"}
```

Verification query:

```cypher
MATCH (n:GraphifyNode {
  tenant_id: 'tenant-live',
  repo_id: 'telstra-messaging-api-sdk-python',
  source_label: 'telstra-messaging-api-sdk-python'
})
RETURN count(n) AS telstra_nodes;

MATCH (:GraphifyNode {
        tenant_id: 'tenant-live',
        repo_id: 'telstra-messaging-api-sdk-python',
        source_label: 'telstra-messaging-api-sdk-python'
      })
      -[r:GRAPHIFY_RELATION {
        tenant_id: 'tenant-live',
        repo_id: 'telstra-messaging-api-sdk-python'
      }]->
      (:GraphifyNode {
        tenant_id: 'tenant-live',
        repo_id: 'telstra-messaging-api-sdk-python',
        source_label: 'telstra-messaging-api-sdk-python'
      })
RETURN count(r) AS telstra_edges;
```

Verified counts:

- `telstra_nodes`: `400`
- `telstra_edges`: `764`
- `orphan_relationships`: `0`

Notes:

- The installed `graphifyy` CLI does not expose `--neo4j-push bolt://localhost:7687`.
- IDRKD bridges that gap by importing the generated `graph.json` into Neo4j via
  `python -m idrkd.graph.graphify_importer import`.
- The Docker profile immediately runs
  `python -m idrkd.graph.graphify_importer smoke` after import.
- Stub nodes preserve all Graphify edges whose source or target references an
  import/external id not present in the Graphify `nodes` list.
