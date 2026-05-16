#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

repo_dir="data/raw/telstra-messaging-api-sdk-python"
if [ ! -d "$repo_dir/.git" ]; then
  git clone --depth 1 https://github.com/telstra/MessagingAPI-SDK-python.git "$repo_dir"
fi

docker compose -f docker/docker-compose.yml up -d neo4j
GRAPHIFY_SOURCE_DIR=telstra-messaging-api-sdk-python \
GRAPHIFY_OUTPUT_NAME=telstra-messaging-api-sdk-python \
REPO_ID=telstra-messaging-api-sdk-python \
docker compose -f docker/docker-compose.yml --profile graphify run --rm graphify

echo "Graphify output: data/processed/graphify-out/telstra-messaging-api-sdk-python"
