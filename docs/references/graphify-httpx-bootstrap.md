# Graphify Bootstrap - httpx

Date: 2026-05-16

Graphify was installed in Docker using the PyPI package `graphifyy` in `docker/graphify.Dockerfile`.

Input repository:

- `https://github.com/encode/httpx.git`
- Local ignored path: `data/raw/httpx`

Command used:

```bash
GRAPHIFY_SOURCE_DIR=httpx GRAPHIFY_OUTPUT_NAME=httpx REPO_ID=httpx \
  docker compose --profile graphify up graphify
```

Observed output:

```text
AST extraction: 100/100 files (100%) [11 workers]
Rebuilt (no clustering): 2158 nodes, 3477 edges
graph.json updated in /tmp/httpx/graphify-out
```

Generated local output:

- `data/processed/graphify-out/httpx/graph.json`

Notes:

- The installed Graphify CLI does not expose the originally planned `--neo4j-push bolt://localhost:7687` flag.
- For the Docker-verified path, the project uses `graphify update` to produce a local `graph.json`.
- IDRKD provides `idrkd.graph.graphify_importer` as the missing bridge from
  Graphify `graph.json` to Neo4j. The importer creates `:GraphifyNode` nodes
  and `:GRAPHIFY_RELATION` relationships with idempotent `MERGE` writes.
- The profile-based service now runs import and smoke verification in the same
  repeatable container command.
