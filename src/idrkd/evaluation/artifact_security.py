"""Cosign-backed model release descriptors and fail-closed verification."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


ARTIFACT_TYPE = "idrkd-model-release"
DESCRIPTOR_SCHEMA_VERSION = 1
CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


class ArtifactVerificationError(RuntimeError):
    """Raised when signed release material cannot be trusted."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_release_descriptor(
    *,
    checkpoint_dir: Path,
    promotion_record_path: Path,
) -> dict[str, Any]:
    checkpoint = checkpoint_dir.resolve()
    promotion_path = promotion_record_path.resolve()
    if not checkpoint.is_dir():
        raise ArtifactVerificationError(f"Checkpoint directory does not exist: {checkpoint}")
    if not promotion_path.is_file():
        raise ArtifactVerificationError(f"Promotion record does not exist: {promotion_path}")

    manifest_path = checkpoint / "idrkd-model-manifest.json"
    manifest = _read_json(manifest_path, label="model manifest")
    _verify_manifest_digest(manifest)
    promotion = _read_json(promotion_path, label="promotion record")
    _verify_promotion_record(promotion)
    _verify_release_links(manifest=manifest, promotion=promotion)

    files = _checkpoint_files(checkpoint)
    _verify_lfs_links(promotion=promotion, files=files)
    return {
        "artifact_type": ARTIFACT_TYPE,
        "schema_version": DESCRIPTOR_SCHEMA_VERSION,
        "model_id": manifest.get("model_id"),
        "checkpoint": {
            "file_count": len(files),
            "files": files,
            "manifest_path": manifest_path.relative_to(checkpoint).as_posix(),
            "manifest_digest": manifest["digest"],
        },
        "promotion": {
            "file_name": promotion_path.name,
            "sha256": sha256_file(promotion_path),
            "record_digest": promotion["record_digest"],
            "status": promotion["decision"]["status"],
        },
    }


def write_release_descriptor(path: Path, descriptor: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(descriptor, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def sign_release(
    *,
    checkpoint_dir: Path,
    promotion_record_path: Path,
    descriptor_path: Path,
    bundle_path: Path,
    key: str,
    cosign: str = "cosign",
    runner: CommandRunner = subprocess.run,
) -> dict[str, Any]:
    descriptor = build_release_descriptor(
        checkpoint_dir=checkpoint_dir,
        promotion_record_path=promotion_record_path,
    )
    write_release_descriptor(descriptor_path, descriptor)
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    bundle_path.unlink(missing_ok=True)
    _run_cosign(
        [
            cosign,
            "sign-blob",
            "--yes",
            "--key",
            key,
            "--bundle",
            str(bundle_path),
            str(descriptor_path),
        ],
        runner=runner,
        action="sign",
    )
    if not bundle_path.is_file():
        raise ArtifactVerificationError(f"Cosign did not write a bundle: {bundle_path}")
    return {
        "descriptor": str(descriptor_path),
        "descriptor_sha256": sha256_file(descriptor_path),
        "bundle": str(bundle_path),
        "model_id": descriptor["model_id"],
    }


def verify_release(
    *,
    checkpoint_dir: Path,
    promotion_record_path: Path,
    descriptor_path: Path,
    bundle_path: Path,
    public_key: str,
    cosign: str = "cosign",
    runner: CommandRunner = subprocess.run,
) -> dict[str, Any]:
    for label, path in (
        ("descriptor", descriptor_path),
        ("Cosign bundle", bundle_path),
    ):
        if not path.is_file():
            raise ArtifactVerificationError(f"{label} does not exist: {path}")

    _run_cosign(
        [
            cosign,
            "verify-blob",
            "--offline",
            "--key",
            public_key,
            "--bundle",
            str(bundle_path),
            str(descriptor_path),
        ],
        runner=runner,
        action="verify",
    )

    signed_descriptor = _read_json(descriptor_path, label="release descriptor")
    expected_descriptor = build_release_descriptor(
        checkpoint_dir=checkpoint_dir,
        promotion_record_path=promotion_record_path,
    )
    if signed_descriptor != expected_descriptor:
        raise ArtifactVerificationError(
            "Signed descriptor does not match the checkpoint and promotion evidence"
        )
    if signed_descriptor.get("artifact_type") != ARTIFACT_TYPE:
        raise ArtifactVerificationError("Unsupported release descriptor artifact type")
    if signed_descriptor.get("schema_version") != DESCRIPTOR_SCHEMA_VERSION:
        raise ArtifactVerificationError("Unsupported release descriptor schema version")
    return {
        "verified": True,
        "model_id": signed_descriptor["model_id"],
        "descriptor_sha256": sha256_file(descriptor_path),
        "checkpoint_files": signed_descriptor["checkpoint"]["file_count"],
        "promotion_record_digest": signed_descriptor["promotion"]["record_digest"],
    }


def _checkpoint_files(checkpoint: Path) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for path in sorted(checkpoint.rglob("*")):
        if path.is_symlink():
            raise ArtifactVerificationError(f"Checkpoint contains a symbolic link: {path}")
        if not path.is_file():
            continue
        files.append(
            {
                "path": path.relative_to(checkpoint).as_posix(),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    if not files:
        raise ArtifactVerificationError(f"Checkpoint has no files: {checkpoint}")
    return files


def _verify_manifest_digest(manifest: dict[str, Any]) -> None:
    recorded = manifest.get("digest")
    payload = {key: value for key, value in manifest.items() if key != "digest"}
    calculated = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    if not isinstance(recorded, str) or recorded != calculated:
        raise ArtifactVerificationError("Model manifest digest is invalid")


def _verify_promotion_record(record: dict[str, Any]) -> None:
    decision = record.get("decision")
    if not isinstance(decision, dict) or decision.get("status") != "promoted":
        raise ArtifactVerificationError("Promotion record is not promoted")
    recorded = record.get("record_digest")
    unsigned = {key: value for key, value in record.items() if key != "record_digest"}
    calculated = "sha256:" + hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if not isinstance(recorded, str) or recorded != calculated:
        raise ArtifactVerificationError("Promotion record digest is invalid")


def _verify_release_links(*, manifest: dict[str, Any], promotion: dict[str, Any]) -> None:
    model = promotion.get("model")
    if not isinstance(model, dict):
        raise ArtifactVerificationError("Promotion record has no model provenance")
    if model.get("model_id") != manifest.get("model_id"):
        raise ArtifactVerificationError("Manifest and promotion record model IDs differ")
    if model.get("manifest_digest") != manifest.get("digest"):
        raise ArtifactVerificationError("Manifest and promotion record digests differ")


def _verify_lfs_links(
    *,
    promotion: dict[str, Any],
    files: list[dict[str, Any]],
) -> None:
    model = promotion["model"]
    checkpoint_path = model.get("checkpoint_path")
    lfs_objects = model.get("lfs_objects")
    if not isinstance(checkpoint_path, str) or not isinstance(lfs_objects, list) or not lfs_objects:
        raise ArtifactVerificationError("Promotion record has no Git LFS checkpoint provenance")
    prefix = checkpoint_path.rstrip("/") + "/"
    actual = {item["path"]: item["sha256"] for item in files}
    for item in lfs_objects:
        if not isinstance(item, dict):
            raise ArtifactVerificationError("Promotion record has invalid Git LFS provenance")
        path = item.get("path")
        oid = item.get("oid")
        if not isinstance(path, str) or not path.startswith(prefix):
            raise ArtifactVerificationError("Git LFS path is outside the promoted checkpoint")
        relative_path = path[len(prefix) :]
        if actual.get(relative_path) != str(oid).removeprefix("sha256:"):
            raise ArtifactVerificationError(
                f"Git LFS object does not match checkpoint file: {relative_path}"
            )


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ArtifactVerificationError(f"Cannot read {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ArtifactVerificationError(f"{label.capitalize()} is not a JSON object: {path}")
    return value


def _run_cosign(
    command: Sequence[str],
    *,
    runner: CommandRunner,
    action: str,
) -> None:
    try:
        completed = runner(
            list(command),
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise ArtifactVerificationError("Cosign executable was not found") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise ArtifactVerificationError(f"Cosign {action} failed: {detail}")
