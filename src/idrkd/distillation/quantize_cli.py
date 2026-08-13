"""Production llm-compressor AWQ CLI for distilled IDRKD student models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from idrkd.distillation.quantization import (
    AwqQuantizationConfig,
    AwqQuantizationJob,
    run_awq_quantization,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Quantize an IDRKD student model with llm-compressor AWQ and write a "
            "compressed-tensors manifest."
        )
    )
    parser.add_argument(
        "--input-model",
        type=Path,
        required=True,
        help="Merged model directory, or a placeholder path when --adapter is supplied.",
    )
    parser.add_argument("--adapter", type=Path, help="Optional PEFT adapter directory to merge before AWQ.")
    parser.add_argument("--base-model", required=True, help="Base model id/path used for manifest and adapter merge.")
    parser.add_argument("--out", type=Path, required=True, help="Quantized model output directory.")
    parser.add_argument("--model-id", default="idrkd-student-awq")
    parser.add_argument(
        "--calibration",
        type=Path,
        required=True,
        help="Representative JSONL calibration data with messages, text, or prompt/chosen fields.",
    )
    parser.add_argument("--max-calibration-samples", type=int, default=128)
    parser.add_argument("--max-sequence-length", type=int, default=4096)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")

    args = parser.parse_args()
    manifest = run_awq_quantization(
        AwqQuantizationJob(
            input_model_path=args.input_model,
            output_dir=args.out,
            model_id=args.model_id,
            base_model_id=args.base_model,
            adapter_path=args.adapter,
            calibration_path=args.calibration,
            quantization=AwqQuantizationConfig(
                bits=4,
                group_size=128,
                zero_point=True,
            ),
            local_files_only=args.local_files_only,
            trust_remote_code=args.trust_remote_code,
            max_calibration_samples=args.max_calibration_samples,
            max_sequence_length=args.max_sequence_length,
        )
    )
    payload = manifest.payload()
    payload["digest"] = manifest.digest()
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
