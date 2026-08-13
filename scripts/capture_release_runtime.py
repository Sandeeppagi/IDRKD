#!/usr/bin/env python3
"""Capture serving runtime versions without importing the IDRKD package."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import importlib.metadata
import json
from pathlib import Path
import platform

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "vllm": importlib.metadata.version("vllm"),
        "torch": str(torch.__version__),
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
