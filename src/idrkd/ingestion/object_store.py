"""Immutable source archive for commit ingestion."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Protocol
from urllib.parse import quote

from idrkd.ingestion.events import CommitEvent


class ObjectStoreClient(Protocol):
    def bucket_exists(self, bucket_name: str) -> bool: ...

    def make_bucket(self, bucket_name: str) -> None: ...

    def stat_object(self, bucket_name: str, object_name: str) -> object: ...

    def put_object(
        self,
        bucket_name: str,
        object_name: str,
        data: BinaryIO,
        length: int,
        content_type: str = "application/octet-stream",
        metadata: dict[str, str | list[str] | tuple[str]] | None = None,
    ) -> object: ...


@dataclass(frozen=True)
class ArchivedFile:
    path: str
    object_key: str | None
    sha256: str | None
    size: int
    deleted: bool


@dataclass(frozen=True)
class ArchiveResult:
    bucket: str
    manifest_key: str
    files: tuple[ArchivedFile, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "bucket": self.bucket,
            "manifest_key": self.manifest_key,
            "files": [asdict(file) for file in self.files],
        }


class MinioSourceArchive:
    """Archive content-addressed blobs and one immutable manifest per event."""

    def __init__(
        self,
        client: ObjectStoreClient,
        *,
        bucket: str = "idrkd-source-archive",
        max_file_bytes: int = 2_000_000,
    ) -> None:
        self._client = client
        self._bucket = bucket
        self._max_file_bytes = max_file_bytes

    def archive(self, event: CommitEvent, *, repo_root: Path, event_id: str) -> ArchiveResult:
        self._ensure_bucket()
        files: list[ArchivedFile] = []
        for raw_path in event.changed_paths:
            path = _safe_relative_path(raw_path)
            source_path = (repo_root / path).resolve()
            if not source_path.is_relative_to(repo_root.resolve()):
                raise ValueError(f"Changed path escapes repository root: {raw_path}")
            if not source_path.is_file():
                files.append(ArchivedFile(path, None, None, 0, True))
                continue
            size = source_path.stat().st_size
            if size > self._max_file_bytes:
                raise ValueError(
                    f"Changed file exceeds {self._max_file_bytes} byte archive limit: {path}"
                )
            body = source_path.read_bytes()
            digest = hashlib.sha256(body).hexdigest()
            object_key = f"blobs/sha256/{digest[:2]}/{digest}"
            self._put_immutable(object_key, body, content_type="application/octet-stream")
            files.append(ArchivedFile(path, object_key, digest, len(body), False))

        prefix = "/".join(
            quote(value, safe="") for value in (event.tenant_id, event.repo_id, event.commit_sha)
        )
        manifest_key = f"commits/{prefix}/{event_id}.json"
        manifest = {
            "schema_version": 1,
            "event_id": event_id,
            "tenant_id": event.tenant_id,
            "repo_id": event.repo_id,
            "commit_sha": event.commit_sha,
            "received_at": event.received_at.isoformat(),
            "files": [asdict(file) for file in files],
        }
        body = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
        self._put_immutable(manifest_key, body, content_type="application/json")
        return ArchiveResult(self._bucket, manifest_key, tuple(files))

    def _ensure_bucket(self) -> None:
        if not self._client.bucket_exists(self._bucket):
            self._client.make_bucket(self._bucket)

    def _put_immutable(self, key: str, body: bytes, *, content_type: str) -> None:
        digest = hashlib.sha256(body).hexdigest()
        try:
            stat = self._client.stat_object(self._bucket, key)
            metadata = getattr(stat, "metadata", {})
            existing = metadata.get("x-amz-meta-sha256") or metadata.get("sha256")
            if existing != digest:
                raise RuntimeError(f"Immutable archive object hash mismatch: {key}")
            return
        except Exception as exc:
            if not _is_missing_object(exc):
                raise
        self._client.put_object(
            self._bucket,
            key,
            BytesIO(body),
            len(body),
            content_type,
            metadata={"sha256": digest, "immutable": "true"},
        )


def _safe_relative_path(value: str) -> str:
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Changed path must be repository-relative: {value}")
    return path.as_posix()


def _is_missing_object(exc: Exception) -> bool:
    code = getattr(exc, "code", None)
    return isinstance(exc, KeyError) or code in {"NoSuchKey", "NoSuchObject", "NoSuchBucket"}
