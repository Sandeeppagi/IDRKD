#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ ! -d "data/raw/httpx/.git" ]; then
  git clone --depth 1 https://github.com/encode/httpx.git data/raw/httpx
fi

docker compose -f docker/docker-compose.yml up -d neo4j
GRAPHIFY_SOURCE_DIR=httpx \
GRAPHIFY_OUTPUT_NAME=httpx \
GRAPHIFY_GIT_URL=https://github.com/encode/httpx.git \
REPO_ID=httpx \
docker compose -f docker/docker-compose.yml --profile graphify up graphify

echo "Graphify output: data/processed/graphify-out/httpx"
