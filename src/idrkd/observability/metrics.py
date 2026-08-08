"""Prometheus metrics for IDRKD runtime paths."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import time

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from prometheus_client.registry import REGISTRY


def _counter(name: str, documentation: str, labelnames: tuple[str, ...]) -> Counter:
    existing = REGISTRY._names_to_collectors.get(name)  # noqa: SLF001
    if isinstance(existing, Counter):
        return existing
    return Counter(name, documentation, labelnames)


def _histogram(name: str, documentation: str, labelnames: tuple[str, ...]) -> Histogram:
    existing = REGISTRY._names_to_collectors.get(name)  # noqa: SLF001
    if isinstance(existing, Histogram):
        return existing
    return Histogram(name, documentation, labelnames)


MCP_TOOL_CALLS = _counter(
    "idrkd_mcp_tool_calls_total",
    "Total MCP tool calls handled by IDRKD.",
    ("tool", "status"),
)
MCP_TOOL_LATENCY = _histogram(
    "idrkd_mcp_tool_latency_seconds",
    "MCP tool execution latency in seconds.",
    ("tool",),
)
REINDEX_JOBS = _counter(
    "idrkd_reindex_jobs_total",
    "Total re-index jobs processed by IDRKD workers.",
    ("worker", "status"),
)
EMBEDDING_UPSERTS = _counter(
    "idrkd_embedding_upserts_total",
    "Total embedding records upserted.",
    ("source",),
)
GRAPH_TRAVERSAL_LATENCY = _histogram(
    "idrkd_graph_traversal_latency_seconds",
    "Graph traversal latency in seconds.",
    ("operation",),
)


@contextmanager
def observe_histogram(histogram: Histogram, *labelvalues: str) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        histogram.labels(*labelvalues).observe(time.perf_counter() - start)


def metrics_response() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST
