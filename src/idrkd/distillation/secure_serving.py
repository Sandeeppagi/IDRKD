"""Verify a signed release before replacing this process with vLLM."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

from idrkd.evaluation.artifact_security import verify_release


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify signed IDRKD model material, then start vLLM.",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--promotion-record", type=Path, required=True)
    parser.add_argument("--descriptor", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--public-key", required=True, help="Cosign public key path or KMS URI.")
    parser.add_argument("--cosign", default="cosign")
    parser.add_argument("--vllm", default="vllm")
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    return parser


def vllm_command(args: argparse.Namespace) -> list[str]:
    return [
        args.vllm,
        "serve",
        str(args.checkpoint),
        "--served-model-name",
        args.served_model_name,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--generation-config",
        "vllm",
    ]


def secured_environment(environ: dict[str, str] | None = None) -> dict[str, str]:
    values = dict(environ or os.environ)
    api_key = values.get("IDRKD_VLLM_API_KEY", "").strip()
    api_key_file = values.get("IDRKD_VLLM_API_KEY_FILE", "").strip()
    if not api_key and api_key_file:
        try:
            api_key = Path(api_key_file).read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise RuntimeError(f"Cannot read IDRKD_VLLM_API_KEY_FILE: {exc}") from exc
    if len(api_key) < 16 or api_key.lower() in {"change-me", "idrkd-local"}:
        raise RuntimeError(
            "IDRKD_VLLM_API_KEY or IDRKD_VLLM_API_KEY_FILE must provide a non-default "
            "secret of at least 16 characters"
        )
    values.pop("IDRKD_VLLM_API_KEY", None)
    values["VLLM_API_KEY"] = api_key
    values["HF_HUB_OFFLINE"] = "1"
    values["TRANSFORMERS_OFFLINE"] = "1"
    values["HF_DATASETS_OFFLINE"] = "1"
    values.setdefault("DO_NOT_TRACK", "1")
    return values


def main() -> None:
    args = build_parser().parse_args()
    try:
        result = verify_release(
            checkpoint_dir=args.checkpoint,
            promotion_record_path=args.promotion_record,
            descriptor_path=args.descriptor,
            bundle_path=args.bundle,
            public_key=args.public_key,
            cosign=args.cosign,
        )
        if result["model_id"] != args.served_model_name:
            raise RuntimeError(
                "Served model name must match the cryptographically signed model ID"
            )
        environment = secured_environment()
    except Exception as exc:
        raise SystemExit(f"secure vLLM startup refused: {exc}") from exc
    print(
        f"verified model={result['model_id']} "
        f"descriptor=sha256:{result['descriptor_sha256']}",
        file=sys.stderr,
        flush=True,
    )
    command = vllm_command(args)
    os.execvpe(command[0], command, environment)


if __name__ == "__main__":
    main()
