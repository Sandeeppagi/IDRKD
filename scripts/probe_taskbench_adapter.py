#!/usr/bin/env python
"""Probe a PEFT adapter against TaskBench tool-call targets."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from idrkd.distillation.traces import SYSTEM_PROMPT
from idrkd.evaluation.model_agent import parse_tool_call, tool_selection_prompt
from idrkd.evaluation.synthetic_schemas import (
    build_synthetic_schema_registry,
    build_synthetic_schema_tasks,
    load_synthetic_schema_corpus,
)
from idrkd.evaluation.taskbench import McpTask, load_tasks_jsonl, split_taskbench_tasks
from idrkd.mcp.tools import McpToolRegistry


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe a PEFT adapter on MCP-TaskBench prompts.")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--base-model", default="microsoft/Phi-4-mini-instruct")
    parser.add_argument("--tasks", type=Path, default=Path("eval/taskbench/seed_tasks.jsonl"))
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--use-4bit", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--include-synthetic-schemas", action="store_true")
    parser.add_argument("--split", choices=("train", "holdout", "all"), default="holdout")
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=17)
    parser.add_argument("--synthetic-schemas", type=Path, default=Path("eval/synthetic_schemas/schemas.jsonl"))
    parser.add_argument("--synthetic-conflicts", type=Path, default=Path("eval/synthetic_schemas/conflicts.jsonl"))
    args = parser.parse_args()

    transformers, peft, torch = _load_ml_modules()
    tasks, tools = _tasks_and_tools(
        tasks_path=args.tasks,
        include_synthetic_schemas=args.include_synthetic_schemas,
        synthetic_schemas_path=args.synthetic_schemas,
        synthetic_conflicts_path=args.synthetic_conflicts,
        split=args.split,
        holdout_fraction=args.holdout_fraction,
        split_seed=args.split_seed,
    )
    selected_tasks = tasks[: args.limit]
    print(
        f"TaskBench split: {args.split} "
        f"(seed={args.split_seed}, holdout_fraction={args.holdout_fraction}, "
        f"selected={len(selected_tasks)}/{len(tasks)})"
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(args.adapter),
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "local_files_only": args.local_files_only,
        "device_map": "auto",
    }
    if args.use_4bit:
        model_kwargs["quantization_config"] = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    model = transformers.AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    model = peft.PeftModel.from_pretrained(
        model,
        str(args.adapter),
        local_files_only=args.local_files_only,
        is_trainable=False,
    )
    model.eval()

    json_matches = 0
    tool_matches = 0
    argument_matches = 0
    exact_matches = 0
    for index, task in enumerate(selected_tasks, start=1):
        raw_output = _generate(
            model=model,
            tokenizer=tokenizer,
            torch=torch,
            prompt=tool_selection_prompt(prompt=task.prompt, tools=tools),
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        parsed = parse_tool_call(raw_output)
        expected = task.expected_call()
        json_matches += parsed is not None
        tool_matches += parsed is not None and parsed.name == expected.name
        argument_matches += parsed is not None and parsed.arguments == expected.arguments
        exact_matches += (
            parsed is not None
            and parsed.name == expected.name
            and parsed.arguments == expected.arguments
        )
        print(f"\n===== CASE {index} ({task.id}) =====")
        print("TARGET:", {"name": expected.name, "arguments": expected.arguments})
        print("OUTPUT:", raw_output)
        print(
            "PARSED:",
            {"name": parsed.name, "arguments": parsed.arguments} if parsed is not None else None,
        )

    total = len(selected_tasks)
    print("\n===== SUMMARY =====")
    print(f"Split: {args.split} (seed={args.split_seed}, holdout_fraction={args.holdout_fraction})")
    print(f"JSON parse: {json_matches} / {total}")
    print(f"Tool match: {tool_matches} / {total}")
    print(f"Argument match: {argument_matches} / {total}")
    print(f"Exact call match: {exact_matches} / {total}")


def _generate(
    *,
    model: Any,
    tokenizer: Any,
    torch: Any,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
    generation_kwargs: dict[str, Any] = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0:
        generation_kwargs["temperature"] = temperature
    with torch.inference_mode():
        output_ids = model.generate(**generation_kwargs)
    new_tokens = output_ids[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def _tasks_and_tools(
    *,
    tasks_path: Path,
    include_synthetic_schemas: bool,
    synthetic_schemas_path: Path,
    synthetic_conflicts_path: Path,
    split: str,
    holdout_fraction: float,
    split_seed: int,
) -> tuple[list[McpTask], list[dict[str, Any]]]:
    tasks = load_tasks_jsonl(tasks_path)
    if include_synthetic_schemas:
        corpus = load_synthetic_schema_corpus(
            schemas_path=synthetic_schemas_path,
            conflicts_path=synthetic_conflicts_path,
        )
        tasks.extend(build_synthetic_schema_tasks(corpus))
        tools = build_synthetic_schema_registry(corpus).list_tools()
    else:
        tools = McpToolRegistry(principal_tenant_id="default").list_tools()
    selected = split_taskbench_tasks(
        tasks,
        split=split,
        holdout_fraction=holdout_fraction,
        seed=split_seed,
    )
    return selected, tools


def _load_ml_modules() -> tuple[Any, Any, Any]:
    try:
        import peft
        import torch
        import transformers
    except ImportError as exc:
        raise RuntimeError("Install ML dependencies with `uv sync --group dev --extra ml`.") from exc
    return transformers, peft, torch


if __name__ == "__main__":
    main()
