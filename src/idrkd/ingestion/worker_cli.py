"""Runtime entrypoint for Kafka-driven ingestion and repair."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import time
from typing import cast

from kafka import KafkaConsumer, KafkaProducer  # type: ignore[import-untyped]
from minio import Minio

from idrkd.graph.writer import Neo4jCodeGraphWriter
from idrkd.ingestion.consumer import CommitEventConsumer, OutboxPublisher
from idrkd.ingestion.events import CommitEvent
from idrkd.ingestion.object_store import MinioSourceArchive
from idrkd.ingestion.pipeline import CommitIngestionPipeline
from idrkd.ingestion.saga import EventIngestionSaga
from idrkd.ingestion.transaction_store import PostgresIngestionStore
from idrkd.rag.embeddings import BgeM3EmbeddingAdapter
from idrkd.rag.vector_store import PostgresVectorStore


LOGGER = logging.getLogger("idrkd.ingestion.worker")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run event-driven IDRKD ingestion.")
    parser.add_argument(
        "command",
        choices=("consume", "repair", "repair-loop", "healthcheck"),
        nargs="?",
        default="consume",
    )
    parser.add_argument("--bootstrap-servers", default=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"))
    parser.add_argument("--topic", default=os.getenv("INGESTION_COMMIT_TOPIC", "commit-events"))
    parser.add_argument("--group-id", default=os.getenv("INGESTION_CONSUMER_GROUP", "idrkd-ingestion-v1"))
    parser.add_argument("--repository-root", type=Path, default=Path(os.getenv("REPOSITORY_ROOT", "data/raw")))
    parser.add_argument("--postgres-dsn", default=os.getenv("POSTGRES_DSN", "postgresql://idrkd:idrkd@localhost:5432/idrkd"))
    parser.add_argument("--neo4j-uri", default=os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    parser.add_argument("--neo4j-user", default=os.getenv("NEO4J_USER", "neo4j"))
    parser.add_argument("--neo4j-password", default=os.getenv("NEO4J_PASSWORD", "change-me"))
    parser.add_argument("--minio-endpoint", default=os.getenv("MINIO_ENDPOINT", "localhost:9000"))
    parser.add_argument("--minio-access-key", default=os.getenv("MINIO_ACCESS_KEY", "idrkd"))
    parser.add_argument("--minio-secret-key", default=os.getenv("MINIO_SECRET_KEY", "change-me-now"))
    parser.add_argument("--minio-secure", action="store_true", default=_env_bool("MINIO_SECURE"))
    parser.add_argument("--archive-bucket", default=os.getenv("MINIO_ARCHIVE_BUCKET", "idrkd-source-archive"))
    parser.add_argument("--max-file-bytes", type=int, default=int(os.getenv("INGESTION_MAX_FILE_BYTES", "2000000")))
    parser.add_argument("--embedding-model", default=os.getenv("EMBEDDING_MODEL", "bge-m3"))
    parser.add_argument("--embedding-dimensions", type=int, default=int(os.getenv("EMBEDDING_DIMENSIONS", "1536")))
    parser.add_argument("--max-messages", type=int, default=0)
    parser.add_argument("--repair-limit", type=int, default=25)
    parser.add_argument("--repair-interval", type=float, default=30.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    journal = PostgresIngestionStore(args.postgres_dsn)
    graph = Neo4jCodeGraphWriter(args.neo4j_uri, args.neo4j_user, args.neo4j_password)
    vectors = PostgresVectorStore(args.postgres_dsn)
    archive = MinioSourceArchive(
        Minio(
            args.minio_endpoint,
            access_key=args.minio_access_key,
            secret_key=args.minio_secret_key,
            secure=args.minio_secure,
        ),
        bucket=args.archive_bucket,
        max_file_bytes=args.max_file_bytes,
    )
    embeddings = BgeM3EmbeddingAdapter(dimensions=args.embedding_dimensions)

    def repo_root(event: CommitEvent) -> Path:
        return cast(Path, args.repository_root) / event.repo_id

    def pipeline(root: Path) -> CommitIngestionPipeline:
        return CommitIngestionPipeline(
            repo_root=root,
            writer=graph,
            embedding_sink=vectors,
            embeddings=embeddings,
            embedding_model_name=args.embedding_model,
        )

    saga = EventIngestionSaga(
        archive=archive,
        journal=journal,
        graph=graph,
        vectors=vectors,
        repo_root=repo_root,
        pipeline=pipeline,
    )
    try:
        journal.apply_schema()
        if args.command == "repair":
            print(json.dumps([result.__dict__ for result in saga.repair_pending(limit=args.repair_limit)]))
            return
        if args.command == "repair-loop":
            while True:
                results = saga.repair_pending(limit=args.repair_limit)
                for result in results:
                    LOGGER.info("repair completed event_id=%s status=%s", result.event_id, result.status)
                time.sleep(args.repair_interval)
        if args.command == "healthcheck":
            _healthcheck(args, graph, journal)
            print(json.dumps({"status": "ok"}))
            return
        consumer = KafkaConsumer(
            args.topic,
            bootstrap_servers=args.bootstrap_servers,
            group_id=args.group_id,
            enable_auto_commit=False,
            auto_offset_reset="earliest",
            value_deserializer=None,
        )
        producer = KafkaProducer(bootstrap_servers=args.bootstrap_servers)
        worker = CommitEventConsumer(consumer, saga, OutboxPublisher(journal, producer))
        LOGGER.info("starting commit consumer topic=%s group=%s", args.topic, args.group_id)
        worker.run(max_messages=args.max_messages)
    finally:
        graph.close()


def _healthcheck(
    args: argparse.Namespace,
    graph: Neo4jCodeGraphWriter,
    journal: PostgresIngestionStore,
) -> None:
    journal.apply_schema()
    graph.apply_schema()
    journal.repair_required(limit=1)
    client = Minio(
        args.minio_endpoint,
        access_key=args.minio_access_key,
        secret_key=args.minio_secret_key,
        secure=args.minio_secure,
    )
    client.bucket_exists(args.archive_bucket)


def _env_bool(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


if __name__ == "__main__":
    main()
