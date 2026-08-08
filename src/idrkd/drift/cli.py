"""Docker worker entrypoints for drift and re-index jobs."""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
import json
import logging
import os
import time
from typing import Any, cast
from urllib import request

import psycopg
import redis
from prometheus_client import start_http_server

from idrkd.drift.queue import RedisMcpReindexQueue, RedisReindexQueue
from idrkd.drift.store import Neo4jDriftStore
from idrkd.drift.workers import ReindexWorker
from idrkd.graph.traversal import Neo4jGraphTraversal
from idrkd.mcp.backends import Neo4jMcpBackend
from idrkd.observability.metrics import REINDEX_JOBS
from idrkd.observability.tracing import traced_span
from idrkd.rag.vector_store import PostgresVectorStore


LOGGER = logging.getLogger("idrkd.drift.worker")


class NoWorkProcessed(RuntimeError):
    pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Run IDRKD drift/re-index workers.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    loop = subparsers.add_parser("reindex-loop", help="Continuously consume re-index requests.")
    _add_common_args(loop)
    loop.add_argument("--idle-sleep", type=float, default=float(os.getenv("WORKER_IDLE_SLEEP_SECONDS", "2")))
    loop.add_argument("--max-jobs", type=int, default=int(os.getenv("WORKER_MAX_JOBS", "0")))

    once = subparsers.add_parser("reindex-once", help="Consume one re-index request and exit.")
    _add_common_args(once)
    once.add_argument("--require-work", action="store_true")

    health = subparsers.add_parser("healthcheck", help="Verify Redis/Postgres/Neo4j connectivity.")
    _add_common_args(health)

    smoke = subparsers.add_parser("live-smoke", help="Verify MCP enqueue -> worker consume -> clear stale.")
    _add_common_args(smoke)
    smoke.add_argument("--mcp-base-url", default=os.getenv("MCP_BASE_URL", "http://localhost:8080"))
    smoke.add_argument("--tenant-id", default=os.getenv("TENANT_ID", "tenant-live"))
    smoke.add_argument("--repo-id", default=os.getenv("REPO_ID", "week5-e2e"))
    smoke.add_argument("--entity-id", default=os.getenv("REINDEX_SMOKE_ENTITY_ID"))
    smoke.add_argument("--timeout", type=float, default=30.0)
    smoke.add_argument("--external-worker", action="store_true")

    args = parser.parse_args()
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    if args.command == "healthcheck":
        run_healthcheck(args)
        print(json.dumps({"status": "ok"}, sort_keys=True))
    elif args.command == "reindex-once":
        result = process_one(args, require_work=args.require_work)
        print(json.dumps(_jsonable(result), sort_keys=True))
    elif args.command == "reindex-loop":
        run_reindex_loop(args)
    elif args.command == "live-smoke":
        print(json.dumps(run_live_smoke(args), sort_keys=True))


def run_reindex_loop(args: argparse.Namespace) -> None:
    _start_metrics_server(args.metrics_port)
    LOGGER.info("starting reindex loop queue=%s", args.queue_key)
    processed = 0
    while True:
        result = process_one(args, require_work=False)
        if result is None:
            time.sleep(args.idle_sleep)
        else:
            processed += 1
            LOGGER.info("processed reindex request result=%s", json.dumps(_jsonable(result), sort_keys=True))
        if args.max_jobs and processed >= args.max_jobs:
            LOGGER.info("stopping reindex loop after max_jobs=%s", args.max_jobs)
            return


def process_one(args: argparse.Namespace, *, require_work: bool) -> Any:
    if getattr(args, "metrics_port", 0) and require_work:
        _start_metrics_server(args.metrics_port)
    queue = _queue(args)
    request_item = queue.dequeue()
    if request_item is None:
        LOGGER.info("no reindex work available queue=%s", args.queue_key)
        REINDEX_JOBS.labels(args.queue_kind, "no_work").inc()
        if require_work:
            raise NoWorkProcessed(f"no work available in {args.queue_key}")
        return None
    worker = _reindex_worker(args)
    try:
        LOGGER.info(
            "processing reindex tenant=%s repo=%s entity=%s reason=%s",
            request_item.tenant_id,
            request_item.repo_id,
            request_item.entity_id,
            request_item.reason,
        )
        with traced_span(
            "drift.reindex_job",
            correlation_id="",
            tenant_id=request_item.tenant_id,
            repo_id=request_item.repo_id,
            entity_id=request_item.entity_id,
            reason=request_item.reason,
            queue_kind=args.queue_kind,
        ):
            result = worker.process(request_item)
        REINDEX_JOBS.labels(args.queue_kind, "success").inc()
        return result
    except Exception:
        REINDEX_JOBS.labels(args.queue_kind, "error").inc()
        raise
    finally:
        worker._graph_backend.close()
        worker._traversal.close()
        worker._drift_store.close()


def run_healthcheck(args: argparse.Namespace) -> None:
    redis.Redis.from_url(args.redis_url).ping()
    with psycopg.connect(args.postgres_dsn) as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT 1")
            cursor.fetchone()
    store = Neo4jDriftStore(args.neo4j_uri, args.neo4j_user, args.neo4j_password)
    try:
        store._write("RETURN 1", {})
    finally:
        store.close()


def run_live_smoke(args: argparse.Namespace) -> dict[str, Any]:
    entity_id = args.entity_id or _discover_source_entity(args)
    _mark_stale(args, entity_id)
    _enqueue_via_mcp(args.mcp_base_url, args.tenant_id, args.repo_id, entity_id)

    deadline = time.monotonic() + args.timeout
    last_result: Any = None
    while time.monotonic() < deadline:
        if not args.external_worker:
            last_result = process_one(args, require_work=False)
        if _is_reindexed(args, entity_id):
            queue_depth = cast(int, redis.Redis.from_url(args.redis_url).llen(args.queue_key))
            return {
                "status": "ok",
                "tenant_id": args.tenant_id,
                "repo_id": args.repo_id,
                "entity_id": entity_id,
                "queue_depth": int(queue_depth),
                "external_worker": bool(args.external_worker),
                "last_result": _jsonable(last_result),
            }
        time.sleep(1)
    raise TimeoutError(f"reindex smoke did not clear stale for {entity_id} within {args.timeout}s")


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--redis-url", default=os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    parser.add_argument("--queue-kind", choices=("mcp", "drift"), default=os.getenv("REINDEX_QUEUE_KIND", "mcp"))
    parser.add_argument("--queue-key", default=os.getenv("REINDEX_QUEUE_KEY", "idrkd:mcp:reindex"))
    parser.add_argument("--neo4j-uri", default=os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    parser.add_argument("--neo4j-user", default=os.getenv("NEO4J_USER", "neo4j"))
    parser.add_argument("--neo4j-password", default=os.getenv("NEO4J_PASSWORD", "change-me"))
    parser.add_argument("--postgres-dsn", default=os.getenv("POSTGRES_DSN", "postgresql://idrkd:idrkd@localhost:5432/idrkd"))
    parser.add_argument("--metrics-port", type=int, default=int(os.getenv("WORKER_METRICS_PORT", "0")))


_METRICS_SERVER_STARTED = False


def _start_metrics_server(port: int) -> None:
    global _METRICS_SERVER_STARTED
    if port <= 0 or _METRICS_SERVER_STARTED:
        return
    start_http_server(port)
    _METRICS_SERVER_STARTED = True
    LOGGER.info("started worker metrics server port=%s", port)


def _queue(args: argparse.Namespace) -> RedisMcpReindexQueue | RedisReindexQueue:
    if args.queue_kind == "mcp":
        return RedisMcpReindexQueue(args.redis_url, key=args.queue_key)
    return RedisReindexQueue(args.redis_url, key=args.queue_key)


def _reindex_worker(args: argparse.Namespace) -> ReindexWorker:
    return ReindexWorker(
        graph_backend=Neo4jMcpBackend(args.neo4j_uri, args.neo4j_user, args.neo4j_password),
        traversal=Neo4jGraphTraversal(args.neo4j_uri, args.neo4j_user, args.neo4j_password),
        vector_store=PostgresVectorStore(args.postgres_dsn),
        drift_store=Neo4jDriftStore(args.neo4j_uri, args.neo4j_user, args.neo4j_password),
    )


def _discover_source_entity(args: argparse.Namespace) -> str:
    from neo4j import GraphDatabase

    query = """
    MATCH (source:CodeEntity {tenant_id: $tenant_id, repo_id: $repo_id})-[:RELATES_TO]->()
    RETURN source.id AS entity_id
    ORDER BY entity_id ASC
    LIMIT 1
    """
    driver = GraphDatabase.driver(args.neo4j_uri, auth=(args.neo4j_user, args.neo4j_password))
    try:
        with driver.session() as session:
            record = session.run(query, {"tenant_id": args.tenant_id, "repo_id": args.repo_id}).single()
    finally:
        driver.close()
    if record is None:
        raise RuntimeError(f"no source entity with outgoing edges for tenant={args.tenant_id} repo={args.repo_id}")
    return str(record["entity_id"])


def _mark_stale(args: argparse.Namespace, entity_id: str) -> None:
    store = Neo4jDriftStore(args.neo4j_uri, args.neo4j_user, args.neo4j_password)
    try:
        store.update_entity_drift(
            tenant_id=args.tenant_id,
            repo_id=args.repo_id,
            entity_id=entity_id,
            drift_score=1.0,
            stale=True,
        )
    finally:
        store.close()


def _is_reindexed(args: argparse.Namespace, entity_id: str) -> bool:
    from neo4j import GraphDatabase

    query = """
    MATCH (n:CodeEntity {tenant_id: $tenant_id, repo_id: $repo_id, id: $entity_id})
    RETURN coalesce(n.stale, false) AS stale, n.reindexed_at IS NOT NULL AS reindexed
    LIMIT 1
    """
    driver = GraphDatabase.driver(args.neo4j_uri, auth=(args.neo4j_user, args.neo4j_password))
    try:
        with driver.session() as session:
            record = session.run(
                query,
                {"tenant_id": args.tenant_id, "repo_id": args.repo_id, "entity_id": entity_id},
            ).single()
    finally:
        driver.close()
    if record is None:
        return False
    return bool(record["reindexed"]) and not bool(record["stale"]) and _has_reindex_embedding(args, entity_id)


def _has_reindex_embedding(args: argparse.Namespace, entity_id: str) -> bool:
    sql = """
    SELECT 1
    FROM knowledge_embeddings
    WHERE tenant_id = %(tenant_id)s
      AND repo_id = %(repo_id)s
      AND entity_id = %(entity_id)s
      AND source = 'reindex'
    LIMIT 1
    """
    with psycopg.connect(args.postgres_dsn) as conn:
        with conn.cursor() as cursor:
            cursor.execute(sql, {"tenant_id": args.tenant_id, "repo_id": args.repo_id, "entity_id": entity_id})
            return cursor.fetchone() is not None


def _enqueue_via_mcp(base_url: str, tenant_id: str, repo_id: str, entity_id: str) -> None:
    payload = {
        "jsonrpc": "2.0",
        "id": "drift-smoke-enqueue",
        "method": "tools/call",
        "params": {
            "name": "enqueue_reindex",
            "arguments": {"tenant_id": tenant_id, "repo_id": repo_id, "entity_id": entity_id},
        },
    }
    req = request.Request(
        f"{base_url.rstrip('/')}/mcp",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=10) as response:
        body = json.loads(response.read().decode("utf-8"))
    if body.get("error") is not None:
        raise RuntimeError(f"MCP enqueue_reindex failed: {body['error']}")


def _jsonable(value: Any) -> Any:
    if value is None:
        return None
    if is_dataclass(value) and not isinstance(value, type):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, datetime | date):
        return value.isoformat()
    return value


if __name__ == "__main__":
    main()
