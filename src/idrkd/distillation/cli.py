"""CLI for actual distillation dataset creation and training execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from idrkd.distillation.artifact_validation import (
    validate_distilled_adapter_artifact,
    validate_extracted_adapter_artifacts,
)
from idrkd.distillation.execution import (
    DistillationRuntimeConfig,
    run_laptop_smoke_distillation,
    train_dpo,
    train_sft,
)
from idrkd.distillation.io import (
    build_preference_dataset_jsonl,
    build_sft_dataset_jsonl,
    build_taskbench_preference_dataset_jsonl,
    build_taskbench_sft_dataset_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run IDRKD SLM distillation stages.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sft_dataset = subparsers.add_parser("build-sft", help="Build chat SFT JSONL from teacher traces.")
    sft_dataset.add_argument("--traces", type=Path, required=True)
    sft_dataset.add_argument("--out", type=Path, required=True)
    sft_dataset.add_argument("--min-faithfulness", type=float, default=0.78)
    sft_dataset.add_argument("--allow-no-tool", action="store_true")

    pref_dataset = subparsers.add_parser("build-dpo", help="Build DPO preference JSONL from traces.")
    pref_dataset.add_argument("--traces", type=Path, required=True)
    pref_dataset.add_argument("--out", type=Path, required=True)
    pref_dataset.add_argument(
        "--rejected-answer",
        default=None,
        help="Optional rejected JSON tool call. Prose values are kept only as metadata.",
    )
    pref_dataset.add_argument("--min-faithfulness", type=float, default=0.78)

    taskbench_sft = subparsers.add_parser(
        "build-taskbench-sft",
        help="Build eval-aligned SFT JSONL from MCP-TaskBench expected calls.",
    )
    _add_taskbench_dataset_args(taskbench_sft)

    taskbench_dpo = subparsers.add_parser(
        "build-taskbench-dpo",
        help="Build eval-aligned DPO JSONL from MCP-TaskBench expected calls.",
    )
    _add_taskbench_dataset_args(taskbench_dpo)

    train_sft_parser = subparsers.add_parser("train-sft", help="Run LoRA/QLoRA SFT training.")
    _add_training_args(train_sft_parser, default_output=Path("models/adapters/local-smoke-sft"))

    train_dpo_parser = subparsers.add_parser("train-dpo", help="Run DPO preference training.")
    _add_training_args(train_dpo_parser, default_output=Path("models/adapters/local-smoke-dpo"))
    train_dpo_parser.add_argument(
        "--sft-adapter",
        type=Path,
        default=None,
        help="Optional PEFT SFT adapter directory to load before DPO.",
    )

    smoke_parser = subparsers.add_parser(
        "local-smoke",
        help="Build seed datasets and run bounded real SFT+DPO adapter smoke training.",
    )
    smoke_parser.add_argument("--traces", type=Path, default=Path("eval/distillation/seed_teacher_traces.jsonl"))
    smoke_parser.add_argument("--work-dir", type=Path, default=Path("/private/tmp/idrkd-distill-smoke"))
    smoke_parser.add_argument("--sft-out", type=Path, default=Path("models/adapters/local-smoke-sft"))
    smoke_parser.add_argument("--dpo-out", type=Path, default=Path("models/adapters/local-smoke-dpo"))
    smoke_parser.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    smoke_parser.add_argument("--max-steps", type=int, default=5)
    smoke_parser.add_argument("--max-seq-length", type=int, default=512)
    smoke_parser.add_argument("--epochs", type=float, default=1.0)
    smoke_parser.add_argument("--learning-rate", type=float, default=2e-4)
    smoke_parser.add_argument("--batch-size", type=int, default=1)
    smoke_parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    smoke_parser.add_argument("--use-4bit", action="store_true")
    smoke_parser.add_argument("--device-map", default="auto")
    smoke_parser.add_argument("--local-files-only", action="store_true")

    artifact_parser = subparsers.add_parser(
        "validate-artifact",
        help="Validate a distilled SFT+DPO PEFT adapter artifact archive or extracted tree.",
    )
    artifact_input = artifact_parser.add_mutually_exclusive_group(required=True)
    artifact_input.add_argument("--archive", type=Path, help="Path to phi4-mini adapter tar.gz.")
    artifact_input.add_argument("--extracted-dir", type=Path, help="Path to extracted artifact root.")
    artifact_parser.add_argument(
        "--extract-dir",
        type=Path,
        default=None,
        help="Directory to extract archive into before validation.",
    )
    artifact_parser.add_argument("--run-generation", action="store_true")
    artifact_parser.add_argument("--allow-downloads", action="store_true")
    artifact_parser.add_argument(
        "--prompt",
        default="Select the best MCP tool for repository search.",
        help="Tiny prompt used when --run-generation is enabled.",
    )
    artifact_parser.add_argument("--max-new-tokens", type=int, default=16)

    args = parser.parse_args()
    if args.command == "build-sft":
        records = build_sft_dataset_jsonl(
            traces_path=args.traces,
            out_path=args.out,
            min_faithfulness=args.min_faithfulness,
            require_tool_use=not args.allow_no_tool,
        )
        print(json.dumps({"records": len(records), "out": str(args.out)}, sort_keys=True))
        return
    if args.command == "build-dpo":
        records = build_preference_dataset_jsonl(
            traces_path=args.traces,
            out_path=args.out,
            rejected_answer=args.rejected_answer,
            min_faithfulness=args.min_faithfulness,
        )
        print(json.dumps({"records": len(records), "out": str(args.out)}, sort_keys=True))
        return
    if args.command == "build-taskbench-sft":
        records = build_taskbench_sft_dataset_jsonl(
            tasks_path=args.tasks,
            out_path=args.out,
            include_synthetic_schemas=args.include_synthetic_schemas,
            synthetic_schemas_path=args.synthetic_schemas,
            synthetic_conflicts_path=args.synthetic_conflicts,
            split=args.split,
            holdout_fraction=args.holdout_fraction,
            split_seed=args.split_seed,
        )
        print(
            json.dumps(
                {
                    "holdout_fraction": args.holdout_fraction,
                    "out": str(args.out),
                    "records": len(records),
                    "split": args.split,
                    "split_seed": args.split_seed,
                },
                sort_keys=True,
            )
        )
        return
    if args.command == "build-taskbench-dpo":
        records = build_taskbench_preference_dataset_jsonl(
            tasks_path=args.tasks,
            out_path=args.out,
            include_synthetic_schemas=args.include_synthetic_schemas,
            synthetic_schemas_path=args.synthetic_schemas,
            synthetic_conflicts_path=args.synthetic_conflicts,
            split=args.split,
            holdout_fraction=args.holdout_fraction,
            split_seed=args.split_seed,
        )
        print(
            json.dumps(
                {
                    "holdout_fraction": args.holdout_fraction,
                    "out": str(args.out),
                    "records": len(records),
                    "split": args.split,
                    "split_seed": args.split_seed,
                },
                sort_keys=True,
            )
        )
        return

    if args.command == "local-smoke":
        args.work_dir.mkdir(parents=True, exist_ok=True)
        sft_path = args.work_dir / "idrkd-sft.jsonl"
        dpo_path = args.work_dir / "idrkd-dpo.jsonl"
        sft_records = build_sft_dataset_jsonl(traces_path=args.traces, out_path=sft_path)
        dpo_records = build_preference_dataset_jsonl(traces_path=args.traces, out_path=dpo_path)
        common = {
            "base_model_id": args.base_model,
            "max_seq_length": args.max_seq_length,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "per_device_train_batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "use_4bit": args.use_4bit,
            "device_map": _normalise_device_map(args.device_map),
            "local_files_only": args.local_files_only,
            "max_steps": args.max_steps,
        }
        smoke_result = run_laptop_smoke_distillation(
            sft_config=DistillationRuntimeConfig(
                dataset_path=sft_path,
                output_dir=args.sft_out,
                **common,
            ),
            dpo_config=DistillationRuntimeConfig(
                dataset_path=dpo_path,
                output_dir=args.dpo_out,
                sft_adapter_path=args.sft_out,
                **common,
            ),
        )
        payload = smoke_result.as_dict()
        payload["datasets"] = {
            "sft": {"path": str(sft_path), "records": len(sft_records)},
            "dpo": {"path": str(dpo_path), "records": len(dpo_records)},
        }
        print(json.dumps(payload, sort_keys=True))
        return

    if args.command == "validate-artifact":
        try:
            if args.archive is not None:
                artifact_result = validate_distilled_adapter_artifact(
                    args.archive,
                    extract_dir=args.extract_dir,
                    run_generation=args.run_generation,
                    local_files_only=not args.allow_downloads,
                    prompt=args.prompt,
                    max_new_tokens=args.max_new_tokens,
                )
            else:
                artifact_result = validate_extracted_adapter_artifacts(
                    args.extracted_dir,
                    run_generation=args.run_generation,
                    local_files_only=not args.allow_downloads,
                    prompt=args.prompt,
                    max_new_tokens=args.max_new_tokens,
                )
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            parser.exit(1, f"validate-artifact failed: {exc}\n")
        print(json.dumps(artifact_result.as_dict(), sort_keys=True))
        return

    config = DistillationRuntimeConfig(
        dataset_path=args.dataset,
        output_dir=args.out,
        base_model_id=args.base_model,
        sft_adapter_path=getattr(args, "sft_adapter", None),
        max_seq_length=args.max_seq_length,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        use_4bit=args.use_4bit,
        device_map=_normalise_device_map(args.device_map),
        local_files_only=args.local_files_only,
        max_steps=args.max_steps,
        dry_run=args.dry_run,
    )
    train_result = train_sft(config) if args.command == "train-sft" else train_dpo(config)
    print(json.dumps(train_result.as_dict(), sort_keys=True))


def _add_training_args(parser: argparse.ArgumentParser, *, default_output: Path) -> None:
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=default_output)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--use-4bit", action="store_true")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--dry-run", action="store_true")


def _add_taskbench_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tasks", type=Path, default=Path("eval/taskbench/seed_tasks.jsonl"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--include-synthetic-schemas", action="store_true")
    parser.add_argument("--synthetic-schemas", type=Path, default=Path("eval/synthetic_schemas/schemas.jsonl"))
    parser.add_argument("--synthetic-conflicts", type=Path, default=Path("eval/synthetic_schemas/conflicts.jsonl"))
    parser.add_argument("--split", choices=("train", "holdout", "all"), default="train")
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=17)


def _normalise_device_map(value: str | None) -> str | None:
    if value is None or value.lower() in {"none", "null", "off"}:
        return None
    return value


if __name__ == "__main__":
    main()
