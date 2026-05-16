"""Core ingestion records shared across pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class EntityKind(StrEnum):
    FILE = "file"
    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    IMPORT = "import"
    SCHEMA = "schema"
    DOCUMENT = "document"


class RelationType(StrEnum):
    CONTAINS = "CONTAINS"
    IMPORTS = "IMPORTS"
    DEFINES = "DEFINES"
    REFERENCES = "REFERENCES"


@dataclass(frozen=True)
class SourceLocation:
    path: str
    start_line: int
    end_line: int


@dataclass(frozen=True)
class CodeEntity:
    id: str
    tenant_id: str
    repo_id: str
    kind: EntityKind
    name: str
    qualified_name: str
    location: SourceLocation
    content_hash: str
    language: str
    properties: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    lamport_clock: int = 0


@dataclass(frozen=True)
class CodeRelation:
    id: str
    tenant_id: str
    repo_id: str
    source_id: str
    target_id: str
    relation_type: RelationType
    properties: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    lamport_clock: int = 0


@dataclass(frozen=True)
class ParsedFile:
    tenant_id: str
    repo_id: str
    path: str
    language: str
    content_hash: str
    entities: tuple[CodeEntity, ...]
    relations: tuple[CodeRelation, ...]
