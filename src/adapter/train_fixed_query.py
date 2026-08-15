from __future__ import annotations

import argparse
import gc
import os
import random
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from src.adapter.train_query import (
    _checkpoint_weight,
    _load_task_examples,
    _setup_distributed,
    train_query_partition,
    validate_query_training_protocol,
)
from src.common import config_sha256, load_yaml, tokenizer_sha256
from src.data.format_tasks import load_manifest
from src.modeling.partition import MoiraiPartition, fixed_kimi_partition


FIXED_TASK = "fixed"
FIXED_SOURCE_TASKS = ("math", "multihop")


def validate_fixed_adapter_config(config: dict[str, Any]) -> None:
    validate_query_training_protocol(config, stage_name="Fixed query")
    expected = {
        "num_transformer_blocks": 28,
        "source_tasks": list(FIXED_SOURCE_TASKS),
        "training_cases_per_task": 100,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(
                f"Fixed query config mismatch for {key}: "
                f"expected {value!r}, got {config.get(key)!r}"
            )
    for key in {
        "base_checkpoint",
        "data_manifest",
        "data_config",
        "output_root",
    }:
        if key not in config:
            raise ValueError(f"Fixed query config is missing {key}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage3_fixed_adapter.yaml")
    parser.add_argument("--resume", nargs="?", const="auto", default="")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--data-manifest", default="")
    parser.add_argument("--output-root", default="")
    return parser.parse_args()


def _combined_examples(
    *,
    split_name: str,
    records: list[dict[str, Any]],
    data_config: dict[str, Any],
    tokenizer,
    expected_count: int | None,
):
    combined = []
    for source_task in FIXED_SOURCE_TASKS:
        examples, _ = _load_task_examples(
            task=source_task,
            split_name=split_name,
            records=records,
            data_config=data_config,
            tokenizer=tokenizer,
            expected_count=None,
        )
        if expected_count is not None:
            if len(examples) < expected_count:
                raise RuntimeError(
                    f"Fixed query {source_task}/{split_name} requires at least "
                    f"{expected_count} cases, found {len(examples)}"
                )
            examples = examples[:expected_count]
        combined.extend(examples)
    return tuple(combined)


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    validate_fixed_adapter_config(config)
    run_config_hash = config_sha256(args.config)
    checkpoint = Path(args.checkpoint or config["base_checkpoint"])
    _, base_checkpoint_hash, checkpoint_manifest = _checkpoint_weight(checkpoint)
    output_dir = Path(args.output_root or config["output_root"]) / FIXED_TASK
    data_manifest_path = Path(args.data_manifest or config["data_manifest"])
    if not data_manifest_path.is_file():
        raise FileNotFoundError("FAILED: Fixed query data manifest is missing")

    partition = fixed_kimi_partition(
        task=FIXED_TASK,
        num_transformer_blocks=int(config["num_transformer_blocks"]),
    )
    data_config = load_yaml(config["data_config"])
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        local_files_only=True,
        use_fast=True,
    )
    if tokenizer_sha256(tokenizer) != checkpoint_manifest.get("tokenizer_sha256"):
        raise ValueError("Fixed query checkpoint tokenizer hash mismatch")
    records = load_manifest(data_manifest_path)
    train_examples = _combined_examples(
        split_name="stage3_adapter_train",
        records=records,
        data_config=data_config,
        tokenizer=tokenizer,
        expected_count=int(config["training_cases_per_task"]),
    )
    validation_examples = _combined_examples(
        split_name="stage3_adapter_val",
        records=records,
        data_config=data_config,
        tokenizer=tokenizer,
        expected_count=None,
    )

    rank, world_size, _, device = _setup_distributed()
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/moiraiblock-xdg-cache")
    os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/moiraiblock-triton-cache")
    os.environ.setdefault("DS_SKIP_CUDA_CHECK", "1")
    try:
        import deepspeed
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Formal fixed query training requires DeepSpeed; install requirements.txt"
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    partition_path = output_dir / "partition.json"
    if partition_path.is_file():
        existing = MoiraiPartition.from_json(partition_path)
        if existing.sha256 != partition.sha256:
            raise ValueError("Existing fixed partition does not use four-layer blocks")
    elif rank == 0:
        partition.save_json(partition_path)
    dist.barrier()
    if MoiraiPartition.from_json(partition_path).sha256 != partition.sha256:
        raise ValueError("Fixed partition changed during distributed setup")

    random.seed(int(config["seed"]) + rank)
    torch.manual_seed(int(config["seed"]) + rank)
    torch.cuda.manual_seed_all(int(config["seed"]) + rank)
    state_path = output_dir / "last_state.pt"
    if args.resume == "auto":
        resume = state_path.is_file()
    elif args.resume:
        requested = Path(args.resume).resolve()
        resume = requested in {output_dir.resolve(), state_path.resolve()}
    else:
        resume = False
    train_query_partition(
        task=FIXED_TASK,
        partition=partition,
        output_dir=output_dir,
        checkpoint=checkpoint,
        base_checkpoint_hash=base_checkpoint_hash,
        expected_q_full_hash=checkpoint_manifest["q_full_sha256"],
        expected_q_full_names=checkpoint_manifest["q_full_parameter_names"],
        config=config,
        run_config_hash=run_config_hash,
        train_examples=train_examples,
        validation_examples=validation_examples,
        tokenizer=tokenizer,
        rank=rank,
        world_size=world_size,
        device=device,
        resume=resume,
        deepspeed_module=deepspeed,
    )
    dist.barrier()
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
