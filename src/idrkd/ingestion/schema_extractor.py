"""JSON and CSV schema extraction for Week 2."""

from __future__ import annotations

import csv
import json
from io import StringIO
from typing import Any

from idrkd.common.fingerprints import entity_id, normalise_repo_path, sha256_text
from idrkd.common.models import CodeEntity, EntityKind, ParsedFile, SourceLocation


def parse_schema_file(
    *,
    tenant_id: str,
    repo_id: str,
    path: str,
    source: str,
) -> ParsedFile:
    repo_path = normalise_repo_path(path)
    content_hash = sha256_text(source)
    fields = _json_fields(source) if repo_path.endswith(".json") else _csv_fields(source)
    entity = CodeEntity(
        id=entity_id(tenant_id, repo_id, repo_path, EntityKind.SCHEMA.value, repo_path),
        tenant_id=tenant_id,
        repo_id=repo_id,
        kind=EntityKind.SCHEMA,
        name=repo_path.rsplit("/", 1)[-1],
        qualified_name=repo_path,
        location=SourceLocation(path=repo_path, start_line=1, end_line=max(1, len(source.splitlines()))),
        content_hash=content_hash,
        language="schema",
        properties={"fields": fields, "format": "json" if repo_path.endswith(".json") else "csv"},
    )
    return ParsedFile(
        tenant_id=tenant_id,
        repo_id=repo_id,
        path=repo_path,
        language="schema",
        content_hash=content_hash,
        entities=(entity,),
        relations=(),
    )


def _json_fields(source: str) -> list[str]:
    data = json.loads(source)
    fields: set[str] = set()

    def visit(value: Any, prefix: str = "") -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                name = f"{prefix}.{key}" if prefix else str(key)
                fields.add(name)
                visit(child, name)
        elif isinstance(value, list):
            for child in value[:5]:
                visit(child, prefix)

    visit(data)
    return sorted(fields)


def _csv_fields(source: str) -> list[str]:
    reader = csv.reader(StringIO(source))
    return next(reader, [])
