"""True SSE time-to-first-token and end-to-end latency measurement."""

from __future__ import annotations

from datetime import UTC, datetime
import json
import statistics
import time
from typing import Any, Protocol
from urllib import error, request

from idrkd.distillation.serving import grounded_chat_messages


class OpenRequest(Protocol):
    def __call__(self, url: request.Request, *, timeout: float) -> Any:
        ...


def stream_chat_completion(
    *,
    base_url: str,
    model: str,
    query: str,
    evidence: list[str],
    api_key: str = "idrkd-local",
    max_tokens: int = 256,
    timeout_seconds: float = 60.0,
    opener: OpenRequest | None = None,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": True,
        "messages": grounded_chat_messages(query=query, evidence=evidence),
    }
    http_request = request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    open_request = opener or request.urlopen
    started = time.perf_counter()
    first_content_at: float | None = None
    chunks: list[str] = []
    try:
        with open_request(http_request, timeout=timeout_seconds) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                event = json.loads(data)
                choices = event.get("choices", [])
                if not choices:
                    continue
                content = choices[0].get("delta", {}).get("content")
                if isinstance(content, str) and content:
                    if first_content_at is None:
                        first_content_at = time.perf_counter()
                    chunks.append(content)
    except error.HTTPError as exc:
        response_body = exc.read().decode("utf-8", errors="replace").strip()
        detail = f": {response_body}" if response_body else ""
        raise RuntimeError(f"Streaming endpoint returned HTTP {exc.code}{detail}") from exc
    completed = time.perf_counter()
    if first_content_at is None:
        raise RuntimeError("Streaming response completed without a content token")
    return {
        "ttft_seconds": first_content_at - started,
        "latency_seconds": completed - started,
        "output": "".join(chunks),
    }


def run_streaming_benchmark(
    rag_artifact: dict[str, Any],
    *,
    base_url: str,
    model: str,
    samples: int = 20,
    warmups: int = 2,
    api_key: str = "idrkd-local",
    max_tokens: int = 256,
    evidence_limit: int = 3,
    timeout_seconds: float = 60.0,
    opener: OpenRequest | None = None,
) -> dict[str, Any]:
    prompts = [
        (
            str(case["query"]),
            [
                str(item)
                for item in case.get("synthesis_evidence", case.get("evidence", []))
            ][:evidence_limit],
        )
        for case in rag_artifact.get("cases", [])
        if case.get("error") is None
        and case.get("synthesis_evidence", case.get("evidence", []))
    ]
    if not prompts:
        raise ValueError("RAG artifact has no successful grounded prompts for streaming measurement")
    if samples < 1 or warmups < 0 or evidence_limit < 1:
        raise ValueError(
            "samples and evidence_limit must be positive and warmups must be non-negative"
        )

    for index in range(warmups):
        query, evidence = prompts[index % len(prompts)]
        stream_chat_completion(
            base_url=base_url,
            model=model,
            query=query,
            evidence=evidence,
            api_key=api_key,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
            opener=opener,
        )

    measurements: list[dict[str, Any]] = []
    for index in range(samples):
        query, evidence = prompts[index % len(prompts)]
        try:
            result = stream_chat_completion(
                base_url=base_url,
                model=model,
                query=query,
                evidence=evidence,
                api_key=api_key,
                max_tokens=max_tokens,
                timeout_seconds=timeout_seconds,
                opener=opener,
            )
            measurements.append({"sample": index + 1, "query": query, **result, "error": None})
        except Exception as exc:
            measurements.append(
                {
                    "sample": index + 1,
                    "query": query,
                    "ttft_seconds": None,
                    "latency_seconds": None,
                    "output": "",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    successful = [measurement for measurement in measurements if measurement["error"] is None]
    ttfts = [float(measurement["ttft_seconds"]) for measurement in successful]
    latencies = [float(measurement["latency_seconds"]) for measurement in successful]
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "benchmark": "openai-sse-streaming",
        "model_id": model,
        "warmup_count": warmups,
        "evidence_limit": evidence_limit,
        "sample_count": samples,
        "success_count": len(successful),
        "error_count": samples - len(successful),
        "ttft": _distribution(ttfts),
        "latency": _distribution(latencies),
        "measurements": measurements,
    }


def _distribution(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min_seconds": 0.0, "mean_seconds": 0.0, "p50_seconds": 0.0, "p95_seconds": 0.0}
    return {
        "min_seconds": min(values),
        "mean_seconds": statistics.fmean(values),
        "p50_seconds": _percentile(values, 50),
        "p95_seconds": _percentile(values, 95),
    }


def _percentile(values: list[float], percentile: int) -> float:
    ordered = sorted(values)
    rank = (len(ordered) - 1) * (percentile / 100)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight
