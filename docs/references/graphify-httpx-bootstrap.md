# Graphify Bootstrap - httpx

Date: 2026-05-16

Graphify was installed in Docker using the PyPI package `graphifyy` in `docker/graphify.Dockerfile`.

Input repository:

- `https://github.com/encode/httpx.git`
- Local ignored path: `data/raw/httpx`

Command used:

```bash
docker compose -f docker/docker-compose.yml --profile graphify run --rm --no-deps --entrypoint /bin/sh graphify -lc 'rm -rf /tmp/httpx /work/graphify-out/httpx && cp -R /work/raw/httpx /tmp/httpx && graphify update /tmp/httpx --force --no-cluster && mkdir -p /work/graphify-out && cp -R /tmp/httpx/graphify-out /work/graphify-out/httpx && find /work/graphify-out/httpx -maxdepth 2 -type f -print'
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
- Loading that graph into Neo4j remains a follow-up integration task.
