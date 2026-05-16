#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ ! -d "data/raw/httpx/.git" ]; then
  git clone --depth 1 https://github.com/encode/httpx.git data/raw/httpx
fi

docker compose -f docker/docker-compose.yml up -d neo4j
docker compose -f docker/docker-compose.yml --profile graphify run --rm graphify

echo "Graphify output: data/processed/graphify-out/httpx"
