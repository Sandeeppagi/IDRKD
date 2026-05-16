"""Stable identifiers and fingerprints for ingested assets."""

from __future__ import annotations

import hashlib
from pathlib import PurePosixPath


def normalise_repo_path(path: str) -> str:
    """Return a stable POSIX-style path without leading slash noise."""
    normalised = PurePosixPath(path.replace("\\", "/")).as_posix()
    return normalised.lstrip("./")


def sha256_text(text: str) -> str:
    """Hash text content using UTF-8 with deterministic line endings."""
    normalised = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def entity_id(tenant_id: str, repo_id: str, path: str, kind: str, qualified_name: str) -> str:
    """Build a stable entity id suitable for Neo4j MERGE operations."""
    source = "|".join(
        [
            tenant_id.strip(),
            repo_id.strip(),
            normalise_repo_path(path),
            kind.strip().lower(),
            qualified_name.strip(),
        ]
    )
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:24]
    return f"ent_{digest}"


def relation_id(
    tenant_id: str,
    repo_id: str,
    source_id: str,
    relation_type: str,
    target_id: str,
) -> str:
    """Build a stable relation id for idempotent graph writes."""
    source = "|".join(
        [
            tenant_id.strip(),
            repo_id.strip(),
            source_id.strip(),
            relation_type.strip().upper(),
            target_id.strip(),
        ]
    )
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:24]
    return f"rel_{digest}"

