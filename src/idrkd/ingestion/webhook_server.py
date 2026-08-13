"""Executable webhook service backed by Kafka."""

from __future__ import annotations

import argparse
import os

from kafka import KafkaProducer  # type: ignore[import-untyped]
import uvicorn

from idrkd.ingestion.webhook import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the signed IDRKD commit webhook.")
    parser.add_argument("--host", default=os.getenv("WEBHOOK_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("WEBHOOK_PORT", "8081")))
    parser.add_argument("--bootstrap-servers", default=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"))
    args = parser.parse_args()
    secret = os.getenv("IDRKD_WEBHOOK_SECRET")
    if not secret:
        raise RuntimeError("IDRKD_WEBHOOK_SECRET must be configured")
    producer = KafkaProducer(bootstrap_servers=args.bootstrap_servers)
    uvicorn.run(create_app(producer, webhook_secret=secret), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
