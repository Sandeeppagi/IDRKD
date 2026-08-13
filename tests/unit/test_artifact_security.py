from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from idrkd.distillation.secure_serving import build_parser as build_serve_parser
from idrkd.distillation.secure_serving import secured_environment, vllm_command
from idrkd.evaluation.artifact_security import (
    ArtifactVerificationError,
    build_release_descriptor,
    sign_release,
    verify_release,
)
from idrkd.evaluation.release_cli import build_parser as build_release_parser


def _write_release(tmp_path: Path) -> tuple[Path, Path]:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"trusted weights")
    manifest = {
        "adapter_path": "models/adapter",
        "base_model_id": "microsoft/Phi-4-mini-instruct",
        "created_at": "2026-08-13T00:00:00+00:00",
        "model_id": "idrkd-test-awq",
        "quantization": {"backend": "llm-compressor", "bits": 4},
        "quantized_path": str(checkpoint),
    }
    manifest["digest"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode("utf-8")
    ).hexdigest()
    (checkpoint / "idrkd-model-manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    promotion = {
        "schema_version": 2,
        "model": {
            "model_id": manifest["model_id"],
            "manifest_digest": manifest["digest"],
            "checkpoint_path": "artifacts/models/checkpoints/idrkd-test-awq",
            "lfs_objects": [
                {
                    "path": (
                        "artifacts/models/checkpoints/idrkd-test-awq/"
                        "idrkd-model-manifest.json"
                    ),
                    "oid": "sha256:"
                    + hashlib.sha256(json.dumps(manifest).encode("utf-8")).hexdigest(),
                },
                {
                    "path": "artifacts/models/checkpoints/idrkd-test-awq/model.safetensors",
                    "oid": "sha256:"
                    + hashlib.sha256(b"trusted weights").hexdigest(),
                },
            ],
        },
        "decision": {"status": "promoted", "reasons": []},
    }
    promotion["record_digest"] = "sha256:" + hashlib.sha256(
        json.dumps(promotion, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    promotion_path = tmp_path / "promotion-record.json"
    promotion_path.write_text(json.dumps(promotion), encoding="utf-8")
    return checkpoint, promotion_path


def test_release_descriptor_is_deterministic_and_covers_every_file(tmp_path: Path) -> None:
    checkpoint, promotion = _write_release(tmp_path)

    first = build_release_descriptor(
        checkpoint_dir=checkpoint,
        promotion_record_path=promotion,
    )
    second = build_release_descriptor(
        checkpoint_dir=checkpoint,
        promotion_record_path=promotion,
    )

    assert first == second
    assert first["artifact_type"] == "idrkd-model-release"
    assert first["checkpoint"]["file_count"] == 2
    assert {item["path"] for item in first["checkpoint"]["files"]} == {
        "idrkd-model-manifest.json",
        "model.safetensors",
    }
    assert first["promotion"]["status"] == "promoted"


def test_cosign_sign_and_offline_verify_round_trip(tmp_path: Path) -> None:
    checkpoint, promotion = _write_release(tmp_path)
    descriptor = tmp_path / "model-release.json"
    bundle = tmp_path / "model-release.sigstore.json"
    commands: list[list[str]] = []

    def runner(command: list[str], **_kwargs) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        if command[1] == "sign-blob":
            Path(command[command.index("--bundle") + 1]).write_text("{}", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="Verified OK", stderr="")

    sign_release(
        checkpoint_dir=checkpoint,
        promotion_record_path=promotion,
        descriptor_path=descriptor,
        bundle_path=bundle,
        key="kms://signing-key",
        runner=runner,
    )
    result = verify_release(
        checkpoint_dir=checkpoint,
        promotion_record_path=promotion,
        descriptor_path=descriptor,
        bundle_path=bundle,
        public_key="/trust/cosign.pub",
        runner=runner,
    )

    assert result["verified"] is True
    assert commands[0][:2] == ["cosign", "sign-blob"]
    assert "--yes" in commands[0]
    assert commands[1][:2] == ["cosign", "verify-blob"]
    assert "--offline" in commands[1]


def test_verification_refuses_tampered_weights_after_valid_signature(tmp_path: Path) -> None:
    checkpoint, promotion = _write_release(tmp_path)
    descriptor = tmp_path / "model-release.json"
    bundle = tmp_path / "model-release.sigstore.json"
    descriptor.write_text(
        json.dumps(
            build_release_descriptor(
                checkpoint_dir=checkpoint,
                promotion_record_path=promotion,
            )
        ),
        encoding="utf-8",
    )
    bundle.write_text("{}", encoding="utf-8")
    (checkpoint / "model.safetensors").write_bytes(b"tampered weights")

    with pytest.raises(ArtifactVerificationError, match="does not match"):
        verify_release(
            checkpoint_dir=checkpoint,
            promotion_record_path=promotion,
            descriptor_path=descriptor,
            bundle_path=bundle,
            public_key="cosign.pub",
            runner=lambda command, **kwargs: subprocess.CompletedProcess(command, 0),
        )


def test_verification_refuses_invalid_cosign_signature(tmp_path: Path) -> None:
    checkpoint, promotion = _write_release(tmp_path)
    descriptor = tmp_path / "model-release.json"
    descriptor.write_text("{}", encoding="utf-8")
    bundle = tmp_path / "model-release.sigstore.json"
    bundle.write_text("{}", encoding="utf-8")

    with pytest.raises(ArtifactVerificationError, match="Cosign verify failed"):
        verify_release(
            checkpoint_dir=checkpoint,
            promotion_record_path=promotion,
            descriptor_path=descriptor,
            bundle_path=bundle,
            public_key="cosign.pub",
            runner=lambda command, **kwargs: subprocess.CompletedProcess(
                command, 1, stdout="", stderr="invalid signature"
            ),
        )


def test_descriptor_refuses_rejected_or_modified_promotion(tmp_path: Path) -> None:
    checkpoint, promotion = _write_release(tmp_path)
    record = json.loads(promotion.read_text(encoding="utf-8"))
    record["decision"]["status"] = "rejected"
    promotion.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ArtifactVerificationError, match="not promoted"):
        build_release_descriptor(
            checkpoint_dir=checkpoint,
            promotion_record_path=promotion,
        )


def test_secure_serving_requires_api_key_and_enforces_offline_environment() -> None:
    with pytest.raises(RuntimeError, match="at least 16"):
        secured_environment({"IDRKD_VLLM_API_KEY": "change-me"})

    environment = secured_environment({"IDRKD_VLLM_API_KEY": "a-secure-api-key-value"})

    assert environment["VLLM_API_KEY"] == "a-secure-api-key-value"
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert environment["TRANSFORMERS_OFFLINE"] == "1"
    assert environment["HF_DATASETS_OFFLINE"] == "1"


def test_secure_serving_reads_compose_api_key_file(tmp_path: Path) -> None:
    key_file = tmp_path / "vllm-api-key"
    key_file.write_text("a-file-mounted-api-key-value\n", encoding="utf-8")

    environment = secured_environment({"IDRKD_VLLM_API_KEY_FILE": str(key_file)})

    assert environment["VLLM_API_KEY"] == "a-file-mounted-api-key-value"
    assert "IDRKD_VLLM_API_KEY" not in environment


def test_secure_serving_command_uses_verified_local_checkpoint() -> None:
    args = build_serve_parser().parse_args(
        [
            "--checkpoint",
            "/models/checkpoint",
            "--promotion-record",
            "/release/promotion-record.json",
            "--descriptor",
            "/release/model-release.json",
            "--bundle",
            "/release/model-release.sigstore.json",
            "--public-key",
            "/keys/cosign.pub",
            "--served-model-name",
            "idrkd-model",
        ]
    )

    command = vllm_command(args)

    assert command[:3] == ["vllm", "serve", "/models/checkpoint"]
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert "--trust-remote-code" not in command


def test_release_cli_exposes_sign_and_verify_commands() -> None:
    sign = build_release_parser().parse_args(
        [
            "sign",
            "--promotion-record",
            "promotion.json",
            "--key",
            "cosign.key",
            "--descriptor",
            "release.json",
            "--bundle",
            "release.sigstore.json",
        ]
    )
    verify = build_release_parser().parse_args(
        [
            "verify",
            "--promotion-record",
            "promotion.json",
            "--public-key",
            "cosign.pub",
            "--descriptor",
            "release.json",
            "--bundle",
            "release.sigstore.json",
        ]
    )

    assert sign.command == "sign"
    assert verify.command == "verify"


def test_compose_vllm_profile_is_hardened() -> None:
    compose = Path("docker/docker-compose.yml").read_text(encoding="utf-8")

    assert "vllm/vllm-openai:latest" not in compose
    assert "docker/vllm-secure.Dockerfile" in compose
    assert 'read_only: true' in compose
    assert 'no-new-privileges:true' in compose
    assert 'HF_HUB_OFFLINE: "1"' in compose
    assert 'IDRKD_VLLM_API_KEY_FILE: /run/secrets/vllm_api_key' in compose
    assert '127.0.0.1:${IDRKD_SLM_PORT:-8000}:8000' in compose
