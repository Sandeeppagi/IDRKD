"""Tracing helpers for correlation-aware ingestion flows."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from opentelemetry import trace
from opentelemetry.util.types import AttributeValue


TRACER_NAME = "idrkd"


def tracer() -> trace.Tracer:
    return trace.get_tracer(TRACER_NAME)


@contextmanager
def traced_span(name: str, *, correlation_id: str, **attributes: object) -> Iterator[None]:
    with tracer().start_as_current_span(name) as span:
        span.set_attribute("correlation_id", correlation_id)
        for key, value in attributes.items():
            attribute_value = _attribute_value(value)
            if attribute_value is not None:
                span.set_attribute(key, attribute_value)
        yield


def _attribute_value(value: object) -> AttributeValue | None:
    if value is None or isinstance(value, str | bool | int | float):
        return value
    if isinstance(value, list | tuple) and all(isinstance(item, str | bool | int | float) for item in value):
        return value
    return str(value)
