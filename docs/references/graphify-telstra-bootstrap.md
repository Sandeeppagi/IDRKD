# Graphify Bootstrap - Telstra Messaging API SDK Python

Date: 2026-05-16

Input repository:

- `https://github.com/telstra/MessagingAPI-SDK-python.git`
- Local ignored path: `data/raw/telstra-messaging-api-sdk-python`

Run command:

```bash
scripts/bootstrap_graphify_telstra.sh
```

Observed Graphify output:

```text
Rebuilt (no clustering): 364 nodes, 505 edges
```

Observed Neo4j import output:

```text
Imported Graphify graph into Neo4j: 402 Neo4j nodes (362 unique Graphify nodes from 364 entries + 40 stubs), 505 relationships (505 Graphify edges).
```

Verification query:

```cypher
MATCH (n:GraphifyNode {repo_id: 'telstra-messaging-api-sdk-python'})
RETURN count(n) AS telstra_nodes;

MATCH (:GraphifyNode {repo_id: 'telstra-messaging-api-sdk-python'})
      -[r:GRAPHIFY_RELATION]->
      (:GraphifyNode {repo_id: 'telstra-messaging-api-sdk-python'})
RETURN count(r) AS telstra_edges;
```

Verified counts:

- `telstra_nodes`: `402`
- `telstra_edges`: `505`

Notes:

- The installed `graphifyy` CLI does not expose `--neo4j-push bolt://localhost:7687`.
- IDRKD bridges that gap by importing the generated `graph.json` into Neo4j via
  `python -m idrkd.graph.graphify_importer`.
- Stub nodes preserve all Graphify edges whose source or target references an
  import/external id not present in the Graphify `nodes` list.
