# Spec: Docker Compose Stack

## Goal

Provide a local developer stack for Week 1 integration testing.

## Services

The stack must include:

- Neo4j Community
- PostgreSQL with pgvector
- Kafka
- Redis
- MinIO
- Prometheus
- Grafana
- OpenTelemetry Collector

## Port Contract

| Service | Ports |
|---|---|
| Neo4j | `7474`, `7687` |
| PostgreSQL | `5432` |
| Kafka | `9092` |
| Redis | `6379` |
| MinIO | `9000`, `9001` |
| Prometheus | `9090` |
| Grafana | `3000` |
| OTel Collector | `4317`, `4318`, `8889` |

## Implementation

- `docker/docker-compose.yml`
- `configs/prometheus.yml`
- `configs/otel-collector.yml`

## Acceptance Criteria

- Docker Compose config is valid.
- All services start.
- Services expose the expected ports.
- Neo4j, PostgreSQL, Redis, and MinIO health checks pass.
- Kafka responds to broker API requests.
- Prometheus, Grafana, MinIO, and OTel endpoints respond from the Docker
  network.

## Verification

Start stack:

```bash
docker compose -f docker/docker-compose.yml up -d neo4j postgres redis kafka minio prometheus grafana otel-collector
```

Check service status:

```bash
docker compose -f docker/docker-compose.yml ps --all
```

Kafka broker check:

```bash
docker exec docker-kafka-1 /opt/kafka/bin/kafka-broker-api-versions.sh --bootstrap-server localhost:9092
```

HTTP readiness checks:

```bash
docker run --rm --network docker_default curlimages/curl:8.11.0 -fsS http://prometheus:9090/-/ready
docker run --rm --network docker_default curlimages/curl:8.11.0 -fsS http://grafana:3000/api/health
docker run --rm --network docker_default curlimages/curl:8.11.0 -fsS http://minio:9000/minio/health/live
docker run --rm --network docker_default curlimages/curl:8.11.0 -fsS http://otel-collector:8889/metrics
```
