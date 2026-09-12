from __future__ import annotations

import argparse
import gc
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist
import yaml

from src.common import load_yaml, sha256_file, sha256_json
from src.data.format_tasks import (
    PromptOnlyExample,
    TargetCausalExample,
    format_task_target,
    nested_value,
)
from src.data.leakage_audit import audit_manifest
from src.discovery.fixed_policy import resolve_fixed_num_blocks
from src.discovery.ordinary_residual import (
    formal_discovery_status,
    require_defined_residual_cost,
)
from src.formal.checkpoint import converted_config_sha256, load_joint_checkpoint, save_joint_checkpoint
from src.formal.config import (
    FORMAL_TASKS,
    load_formal_config,
    resolve_base_model_path,
)
from src.formal.data import (
    audit_formal_source_provenance,
    build_formal_examples,
    formal_data_manifest_sha256,
    load_formal_records,
    load_record_row,
    records_by_task_and_stage,
    validate_formal_source_policy,
)
from src.formal.probe import load_formal_probe_head, train_formal_probe
from src.formal.runtime import identity_test, parameter_hash, trainability_audit
from src.formal.task_banks import TaskBank
from src.formal.train_joint import train_token_budget_mixture
from src.evaluation.task_metrics import mean_metrics, task_score
from src.modeling.partition import MoiraiPartition


def _path(raw: str) -> Path:
    return Path(os.path.expandvars(raw))


def _pipeline_paths(config: dict[str, Any]) -> dict[str, Path]:
    values = config["pipeline"]
    return {key: _path(str(value)) for key, value in values.items() if key != "max_sequence_length"}


def _native_weight_hash(checkpoint: Path) -> str:
    weights = sorted(checkpoint.glob("model*.safetensors"))
    if not weights:
        raise FileNotFoundError(f"Native Qwen3 checkpoint has no model*.safetensors: {checkpoint}")
    return sha256_json({path.name: sha256_file(path) for path in weights})


def _native_config(checkpoint: Path):
    config_path = checkpoint / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Native Qwen3 config is missing: {config_path}")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    config = SimpleNamespace(**payload)
    if getattr(config, "model_type", None) != "qwen3":
        raise ValueError("Formal pipeline requires a native Qwen3 checkpoint")
    if any(hasattr(config, name) for name in ("attnres_execution", "moirai_partition")):
        raise ValueError("Formal pipeline cannot use a converted AttnRes checkpoint as Discovery input")
    return config


def _device() -> torch.device:
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank)
    return torch.device("cpu")


def _write_json(path: Path, payload: Any) -> None:
    if not dist.is_initialized() or dist.get_rank() == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if dist.is_initialized():
        dist.barrier()


def _require_single_process(stage: str) -> None:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        raise RuntimeError(
            f"Formal {stage} stage requires one process; use a rank-0 single-process "
            "launch after the distributed training checkpoint is complete"
        )


def _formal_worker_count(config: dict[str, Any]) -> int:
    value = int(config["distributed"]["formal_processes_per_node"])
    if value <= 0:
        raise ValueError("Formal stage worker count must be positive")
    return value


def _formal_master_port(config: dict[str, Any]) -> int:
    value = int(
        os.environ.get(
            "QWEN3_FORMAL_MASTER_PORT",
            config["distributed"]["formal_master_port"],
        )
    )
    if not 1024 <= value <= 65535:
        raise ValueError("QWEN3_FORMAL_MASTER_PORT must be in [1024, 65535]")
    return value


def _launch_formal_stage(
    config: dict[str, Any],
    checkpoint: Path,
    *,
    config_path: str,
    stage: str,
    max_steps: int | None = None,
) -> dict[str, Any]:
    """Run one FSDP formal stage from a single coordinator process."""
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise RuntimeError("Formal stage coordinator cannot run inside an existing worker group")
    if stage not in {"train", "probe", "evaluate"}:
        raise ValueError(f"Unsupported distributed formal stage: {stage}")
    paths = _pipeline_paths(config)
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--rdzv_backend",
        "static",
        "--master_addr",
        "127.0.0.1",
        "--master_port",
        str(_formal_master_port(config)),
        "--nproc_per_node",
        str(_formal_worker_count(config)),
        "-m",
        "src.formal.pipeline",
        "--config",
        config_path,
        "--stage",
        stage,
    ]
    if max_steps is not None:
        command.extend(("--max-steps", str(max_steps)))
    subprocess.run(command, check=True)
    summary_path = {
        "train": paths["joint_checkpoint_output"].parent / "formal_training_summary.json",
        "probe": paths["probe_output"] / "probe_manifest.json",
        "evaluate": paths["evaluation_output"] / "evaluation_results.json",
    }[stage]
    if not summary_path.is_file():
        raise RuntimeError(f"Formal {stage} stage did not write {summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _data_context(config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], str, dict[str, Any]]:
    paths = _pipeline_paths(config)
    data_config = load_yaml(paths["data_config"])
    records = load_formal_records(paths["data_manifest"])
    audit_formal_source_provenance(records, data_config=data_config)
    validate_formal_source_policy(config, data_config)
    allowed = data_config.get("allowed_cross_stage_reuse", ())
    leakage = audit_manifest(paths["data_manifest"], allowed_cross_stage_reuse=allowed)
    if leakage["status"] != "PASS":
        raise RuntimeError("DATA_LEAKAGE")
    manifest_report_path = paths["data_manifest"].with_name("data_manifest.json")
    if manifest_report_path.is_file():
        report = json.loads(manifest_report_path.read_text(encoding="utf-8"))
        declared_hash = report.get("manifest_sha256")
        actual_hash = formal_data_manifest_sha256(paths["data_manifest"])
        if declared_hash is not None and declared_hash != actual_hash:
            raise RuntimeError("Formal data manifest report hash mismatch")
    return records, data_config, formal_data_manifest_sha256(paths["data_manifest"]), leakage


def _required_counts(data_config: dict[str, Any], stage: str) -> dict[str, int]:
    from src.data.format_tasks import task_split_count

    return {
        task: task_split_count(data_config, task=task, split_name=stage)
        for task in FORMAL_TASKS
    }


def _formal_discovery_counts(config: dict[str, Any]) -> dict[str, int]:
    values = config["discovery"]["cases"]
    return {task: int(values[task]) for task in FORMAL_TASKS}


def _validate_data_for_stage(
    config: dict[str, Any],
    *,
    stage: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], str, dict[str, tuple[dict[str, Any], ...]], dict[str, int]]:
    records, data_config, data_hash, leakage = _data_context(config)
    expected = (
        _formal_discovery_counts(config)
        if stage == "stage2_discovery"
        else _required_counts(data_config, stage)
    )
    grouped = records_by_task_and_stage(records, stage=stage, expected_counts=expected)
    return records, data_config, data_hash, grouped, expected


def _load_partitions(
    config: dict[str, Any],
    checkpoint: Path,
) -> tuple[dict[str, MoiraiPartition], int, str]:
    native_config = _native_config(checkpoint)
    layers = int(native_config.num_hidden_layers)
    partition_config = config["discovery"]["partition"]
    num_blocks = resolve_fixed_num_blocks(
        layers,
        fixed_block_size=int(partition_config["fixed_block_size"]),
        policy=str(partition_config["num_blocks_policy"]),
    )
    root = _pipeline_paths(config)["discovery_output"]
    partitions: dict[str, MoiraiPartition] = {}
    base_hash = _native_weight_hash(checkpoint)
    for task in FORMAL_TASKS:
        path = root / task / "partition.json"
        if not path.is_file():
            raise FileNotFoundError(f"Formal Discovery partition is missing: {path}")
        partition = MoiraiPartition.from_json(path)
        if partition.task != task or len(partition.blocks) != num_blocks:
            raise ValueError(f"Formal partition N mismatch for {task}")
        if partition.num_transformer_blocks != layers:
            raise ValueError(f"Formal partition depth mismatch for {task}")
        if partition.min_block_length != int(partition_config["min_block_length"]):
            raise ValueError(f"Formal partition minimum length mismatch for {task}")
        if partition.max_block_length != int(partition_config["max_block_length"]):
            raise ValueError(f"Formal partition maximum length mismatch for {task}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("discovery_checkpoint_sha256") != base_hash:
            raise ValueError(f"Formal partition base checkpoint mismatch for {task}")
        partitions[task] = partition
    return partitions, num_blocks, base_hash


def _build_task_banks(
    model,
    partitions: dict[str, MoiraiPartition],
) -> dict[str, TaskBank]:
    banks: dict[str, TaskBank] = {}
    for task in FORMAL_TASKS:
        partition = partitions[task]
        model.config.moirai_partition = list(partition.lengths)
        model.config.moirai_task = task
        model.config.attnres_execution = "formal"
        bank = TaskBank.from_model(model, task=task, partition_sha256=partition.sha256)
        bank.partition_lengths = partition.lengths
        banks[task] = bank
    banks["math"].activate(model)
    return banks


def _load_formal_runtime(
    config: dict[str, Any],
    *,
    checkpoint: Path,
    partitions: dict[str, MoiraiPartition],
    device: torch.device,
    identity_examples: Mapping[str, Sequence[TargetCausalExample]] | Sequence[TargetCausalExample] = (),
    load_checkpoint_dir: Path | None = None,
    distributed_context=None,
    expected_base_checkpoint_sha256: str | None = None,
    expected_data_manifest_sha256: str | None = None,
    expected_config_sha256: str | None = None,
):
    from src.formal.conversion import convert_qwen3_checkpoint

    original, formal = convert_qwen3_checkpoint(
        checkpoint,
        partition=list(partitions["math"].lengths),
        task="math",
        min_block_length=partitions["math"].min_block_length,
        max_block_length=partitions["math"].max_block_length,
        no_adjacent_singletons=partitions["math"].no_adjacent_singletons,
        dtype=torch.bfloat16,
    )
    banks = _build_task_banks(formal, partitions)
    identity = None
    run_identity = identity_examples and (
        distributed_context is None or distributed_context.is_rank0
    )
    if run_identity:
        if isinstance(identity_examples, Mapping):
            examples_by_task = {
                task: tuple(identity_examples.get(task, ()))
                for task in FORMAL_TASKS
            }
        else:
            examples_by_task = {
                "math": tuple(identity_examples),
                "multihop": (),
                "code": (),
            }
        if any(not examples_by_task[task] for task in FORMAL_TASKS):
            raise RuntimeError(
                "FORMAL_IDENTITY_TEST_MISSING: one identity example is required "
                "for each formal task partition"
            )
        if distributed_context is not None and distributed_context.distributed:
            # A 14B CPU forward is unnecessarily slow on the coordinator node.
            # Rank 0 owns the validation work and can use its assigned GPU;
            # return both models to CPU before FSDP wraps the formal model.
            original.to(device)
            formal.to(device)
        per_task_identity: dict[str, dict[str, Any]] = {}
        for task in FORMAL_TASKS:
            banks[task].activate(formal)
            example = examples_by_task[task][0]
            ids = example.input_ids.unsqueeze(0)
            mask = example.attention_mask.unsqueeze(0)
            if distributed_context is not None and distributed_context.distributed:
                ids = ids.to(device)
                mask = mask.to(device)
            task_identity = identity_test(original, formal, ids, mask)
            if task_identity["status"] != "PASS":
                raise RuntimeError(
                    f"IDENTITY_CONVERSION_FAILED for formal task {task}"
                )
            per_task_identity[task] = task_identity
        banks["math"].activate(formal)
        identity = {
            "status": "PASS",
            "max_abs_logit_diff": max(
                float(value["max_abs_logit_diff"])
                for value in per_task_identity.values()
            ),
            "mean_abs_logit_diff": max(
                float(value["mean_abs_logit_diff"])
                for value in per_task_identity.values()
            ),
            "per_task": per_task_identity,
        }
        if distributed_context is not None and distributed_context.distributed:
            formal.cpu()
            original.cpu()
    if distributed_context is not None and distributed_context.distributed:
        from src.distributed.fsdp_utils import broadcast_object

        identity = broadcast_object(identity, distributed_context)
    if distributed_context is not None and distributed_context.distributed:
        from src.distributed.fsdp_utils import wrap_qwen3_fsdp
        from src.modeling.full_attnres import MoiraiQwen3DecoderLayer

        formal = wrap_qwen3_fsdp(
            formal,
            distributed_context,
            decoder_layer_classes=(MoiraiQwen3DecoderLayer,),
            device_id=device,
        )
    else:
        formal.to(device)
    if distributed_context is not None and distributed_context.distributed:
        # FSDP synchronizes model parameters, but the CPU task-bank snapshots
        # were created before wrapping and must be synchronized separately.
        for bank in banks.values():
            bank.capture(formal, distributed_context=distributed_context)
    if load_checkpoint_dir is not None:
        load_joint_checkpoint(
            load_checkpoint_dir,
            model=formal,
            banks=banks,
            active_task="math",
            restore_rng=False,
            expected_base_checkpoint_sha256=expected_base_checkpoint_sha256,
            expected_data_manifest_sha256=expected_data_manifest_sha256,
            expected_config_sha256=expected_config_sha256,
        )
    formal_root = formal.module if distributed_context is not None and distributed_context.distributed else formal
    formal_root.config.use_cache = False
    if bool(config["model"]["gradient_checkpointing"]):
        enable_checkpointing = getattr(formal_root, "gradient_checkpointing_enable", None)
        if enable_checkpointing is None:
            raise RuntimeError("Formal model does not expose gradient_checkpointing_enable")
        enable_checkpointing()
        if not bool(getattr(formal_root.model, "gradient_checkpointing", False)):
            raise RuntimeError("Formal gradient checkpointing was not enabled")
    else:
        disable_checkpointing = getattr(formal_root, "gradient_checkpointing_disable", None)
        if disable_checkpointing is not None:
            disable_checkpointing()
    formal.eval()
    return original, formal, banks, identity


def _run_formal_discovery(config: dict[str, Any], checkpoint: Path) -> dict[str, Any]:
    records, data_config, _, grouped, _ = _validate_data_for_stage(
        config, stage="stage2_discovery"
    )
    del records, data_config, grouped
    require_defined_residual_cost(config["discovery"])
    paths = _pipeline_paths(config)
    worker_count = int(config["distributed"]["discovery_processes_per_node"])
    master_port = int(os.environ.get(
        "QWEN3_DISCOVERY_MASTER_PORT",
        config["distributed"]["discovery_master_port"],
    ))
    if not 1024 <= master_port <= 65535:
        raise ValueError("QWEN3_DISCOVERY_MASTER_PORT must be in [1024, 65535]")
    runtime_config = {
        "seed": int(config["experiment"]["seed"]),
        "model_mode": "original_residual_only",
        "formal_discovery": True,
        "ordinary_residual_cost_defined": True,
        "base_checkpoint": str(checkpoint),
        "data_manifest": str(paths["data_manifest"]),
        "data_config": str(paths["data_config"]),
        "tasks": list(FORMAL_TASKS),
        "discovery_cases_per_task": _formal_discovery_counts(config),
        "fixed_block_size": int(config["discovery"]["partition"]["fixed_block_size"]),
        "num_blocks_policy": str(config["discovery"]["partition"]["num_blocks_policy"]),
        "min_block_length": int(config["discovery"]["partition"]["min_block_length"]),
        "max_block_length": int(config["discovery"]["partition"]["max_block_length"]),
        "no_adjacent_singletons": bool(config["discovery"]["partition"]["no_adjacent_singletons"]),
        "discovery_processes_per_node": worker_count,
        "discovery_master_port": master_port,
        "output_dir": str(paths["discovery_output"]),
    }
    paths["discovery_output"].mkdir(parents=True, exist_ok=True)
    runtime_config_path = paths["discovery_output"] / "formal_discovery_runtime.yaml"
    runtime_config_path.write_text(yaml.safe_dump(runtime_config, sort_keys=False), encoding="utf-8")
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise RuntimeError(
            "Formal Discovery stage is a single coordinator; launch it without "
            "an outer torchrun so the stage can create its configured worker group"
        )
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--rdzv_backend",
        "static",
        "--master_addr",
        "127.0.0.1",
        "--master_port",
        str(master_port),
        "--nproc_per_node",
        str(worker_count),
        "-m",
        "src.discovery.run_all",
        "--config",
        str(runtime_config_path),
        "--checkpoint",
        str(checkpoint),
        "--data-manifest",
        str(paths["data_manifest"]),
        "--output-dir",
        str(paths["discovery_output"]),
        "--resume",
        str(paths["discovery_output"]),
    ]
    subprocess.run(command, check=True)
    partitions, num_blocks, base_hash = _load_partitions(config, checkpoint)
    result = {
        "status": "PASS",
        "base_checkpoint_sha256": base_hash,
        "num_transformer_blocks": int(_native_config(checkpoint).num_hidden_layers),
        "num_moirai_blocks": num_blocks,
        "tasks": {
            task: {
                "lengths": list(partitions[task].lengths),
                "partition_sha256": partitions[task].sha256,
            }
            for task in FORMAL_TASKS
        },
    }
    _write_json(paths["discovery_output"] / "formal_pipeline_discovery_summary.json", result)
    return result


def _run_train(
    config: dict[str, Any],
    checkpoint: Path,
    *,
    max_steps: int | None = None,
) -> dict[str, Any]:
    _, data_config, data_hash, grouped, expected = _validate_data_for_stage(
        config, stage="stage3_adapter_train"
    )
    partitions, _, base_hash = _load_partitions(config, checkpoint)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, use_fast=True)
    max_length = int(config["pipeline"]["max_sequence_length"])
    examples = build_formal_examples(
        grouped,
        data_config=data_config,
        tokenizer=tokenizer,
        target=True,
        max_length=max_length,
    )
    device = _device()
    distributed_context = None
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        from src.distributed.fsdp_utils import init_distributed

        distributed_context = init_distributed(require_cuda=True)
        device = distributed_context.device
    original, model, banks, identity = _load_formal_runtime(
        config,
        checkpoint=checkpoint,
        partitions=partitions,
        device=device,
        identity_examples=examples,
        distributed_context=distributed_context,
    )
    del original
    paths = _pipeline_paths(config)
    if identity is None:
        raise RuntimeError("FORMAL_IDENTITY_TEST_MISSING")
    identity_record = {
        **identity,
        "base_checkpoint_sha256": base_hash,
        "converted_config_sha256": converted_config_sha256(model),
    }
    _write_json(paths["joint_checkpoint_output"].parent / "identity_test.json", identity_record)
    train_audit = trainability_audit(model)
    backbone_names = train_audit["backbone"]
    query_names = train_audit["query"]
    alpha_names = train_audit["alpha"]
    before_backbone_hash = parameter_hash(
        model,
        backbone_names,
        distributed_context=distributed_context,
    )
    before_bank_hashes = {
        task: {
            "query": banks[task].state_hash("pseudo_query"),
            "alpha": banks[task].state_hash("alpha"),
        }
        for task in FORMAL_TASKS
    }
    partition_hashes_before = {task: partitions[task].sha256 for task in FORMAL_TASKS}
    optimizer, scheduler, consumed, results = train_token_budget_mixture(
        model,
        banks=banks,
        examples_by_task=examples,
        tokenizer=tokenizer,
        training_config=config["training"],
        token_budgets={task: int(config["data"]["token_budget"][task]) for task in FORMAL_TASKS},
        device=device,
        max_steps=max_steps,
        distributed_context=distributed_context,
        seed=int(config["experiment"]["seed"]),
        shuffle=bool(config["data"]["shuffle"]),
    )
    complete = all(consumed[task] >= int(config["data"]["token_budget"][task]) for task in FORMAL_TASKS)
    after_backbone_hash = parameter_hash(
        model,
        backbone_names,
        distributed_context=distributed_context,
    )
    after_bank_hashes = {
        task: {
            "query": banks[task].state_hash("pseudo_query"),
            "alpha": banks[task].state_hash("alpha"),
        }
        for task in FORMAL_TASKS
    }
    partition_hashes_after = {task: partitions[task].sha256 for task in FORMAL_TASKS}
    changed = {
        task: {
            "query": before_bank_hashes[task]["query"] != after_bank_hashes[task]["query"],
            "alpha": before_bank_hashes[task]["alpha"] != after_bank_hashes[task]["alpha"],
        }
        for task in FORMAL_TASKS
    }
    if partition_hashes_after != partition_hashes_before:
        raise RuntimeError("PARTITION_CHANGED_IN_FORMAL_TRAINING_SUMMARY")
    if complete and before_backbone_hash == after_backbone_hash:
        raise RuntimeError("FORMAL_BACKBONE_DID_NOT_UPDATE")
    if complete and not all(changed[task]["query"] for task in FORMAL_TASKS):
        raise RuntimeError("FORMAL_TASK_QUERY_DID_NOT_UPDATE")
    if complete and not all(changed[task]["alpha"] for task in FORMAL_TASKS):
        raise RuntimeError("FORMAL_TASK_ALPHA_DID_NOT_UPDATE")
    summary = {
        "status": "PASS" if complete else "PARTIAL_CONNECTIVITY",
        "identity": identity_record,
        "requested_record_counts": expected,
        "consumed_tokens_per_task": consumed,
        "steps": len(results),
        "last_loss": results[-1].loss if results else None,
        "backbone_grad_norm_last": results[-1].backbone_grad_norm if results else None,
        "query_grad_norm_last": results[-1].query_grad_norm if results else None,
        "alpha_grad_norm_last": results[-1].alpha_grad_norm if results else None,
        "trainability": {
            "backbone_parameter_count": len(backbone_names),
            "query_parameter_count": len(query_names),
            "alpha_parameter_count": len(alpha_names),
            "partition_trainable": train_audit["partition_trainable"],
        },
        "backbone_hash_before": before_backbone_hash,
        "backbone_hash_after": after_backbone_hash,
        "backbone_updated": before_backbone_hash != after_backbone_hash,
        "task_bank_updates": changed,
        "partition_hashes_before": partition_hashes_before,
        "partition_hashes_after": partition_hashes_after,
        "base_checkpoint_sha256": base_hash,
        "data_manifest_sha256": data_hash,
        "partitions": {task: partitions[task].to_dict() for task in FORMAL_TASKS},
    }
    _write_json(paths["joint_checkpoint_output"].parent / "formal_training_summary.json", summary)
    if not complete:
        return summary
    manifest = save_joint_checkpoint(
        paths["joint_checkpoint_output"],
        model=model,
        banks=banks,
        partitions={task: partitions[task].to_dict() for task in FORMAL_TASKS},
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        base_checkpoint_sha256=base_hash,
        data_manifest_sha256=data_hash,
        consumed_tokens=consumed,
        seed=int(config["experiment"]["seed"]),
        identity_test=identity_record,
    )
    summary["checkpoint_manifest_sha256"] = sha256_file(
        paths["joint_checkpoint_output"] / "checkpoint_manifest.json"
    )
    summary["checkpoint_status"] = manifest["checkpoint_kind"]
    _write_json(paths["joint_checkpoint_output"].parent / "formal_training_summary.json", summary)
    return summary


def _probe_data(config: dict[str, Any], tokenizer, data_config, records):
    from src.formal.data import records_by_task_and_stage

    train_records = records_by_task_and_stage(
        records, stage="probe_train", expected_counts=_required_counts(data_config, "probe_train")
    )
    val_records = records_by_task_and_stage(
        records, stage="probe_val", expected_counts=_required_counts(data_config, "probe_val")
    )
    max_length = int(config["pipeline"]["max_sequence_length"])
    train = build_formal_examples(
        train_records,
        data_config=data_config,
        tokenizer=tokenizer,
        target=False,
        max_length=max_length,
    )
    validation = build_formal_examples(
        val_records,
        data_config=data_config,
        tokenizer=tokenizer,
        target=False,
        max_length=max_length,
    )
    train_examples = [example for task in FORMAL_TASKS for example in train[task]]
    train_labels = [index for index, task in enumerate(FORMAL_TASKS) for _ in train[task]]
    val_examples = [example for task in FORMAL_TASKS for example in validation[task]]
    val_labels = [index for index, task in enumerate(FORMAL_TASKS) for _ in validation[task]]
    return train_examples, train_labels, val_examples, val_labels


def _run_probe(config: dict[str, Any], checkpoint: Path) -> dict[str, Any]:
    from transformers import AutoTokenizer
    from src.distributed.fsdp_utils import destroy_distributed, init_distributed

    records, data_config, data_hash, _, _ = _data_context(config)
    partitions, _, base_hash = _load_partitions(config, checkpoint)
    joint_dir = _pipeline_paths(config)["joint_checkpoint_output"]
    manifest_path = joint_dir / "checkpoint_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Formal joint checkpoint is missing: {manifest_path}")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, use_fast=True)
    train_examples, train_labels, val_examples, val_labels = _probe_data(
        config, tokenizer, data_config, records
    )
    distributed_context = None
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        distributed_context = init_distributed(require_cuda=True)
        device = distributed_context.device
    else:
        device = _device()
    original, model, banks, _ = _load_formal_runtime(
        config,
        checkpoint=checkpoint,
        partitions=partitions,
        device=device,
        load_checkpoint_dir=joint_dir,
        expected_base_checkpoint_sha256=base_hash,
        expected_data_manifest_sha256=data_hash,
        expected_config_sha256=sha256_json(config),
        distributed_context=distributed_context,
    )
    del original
    manifest = train_formal_probe(
        model=model,
        banks=banks,
        train_examples=train_examples,
        train_labels=train_labels,
        validation_examples=val_examples,
        validation_labels=val_labels,
        tokenizer=tokenizer,
        device=device,
        output_dir=_pipeline_paths(config)["probe_output"],
        config=config["probe"],
        checkpoint_manifest_sha256=sha256_file(manifest_path),
        data_manifest_sha256=data_hash,
    )
    manifest["partition_sha256_per_task"] = {
        task: partitions[task].sha256 for task in FORMAL_TASKS
    }
    _write_json(_pipeline_paths(config)["probe_output"] / "probe_manifest.json", manifest)
    if distributed_context is not None:
        destroy_distributed(distributed_context)
    return manifest


@torch.no_grad()
def _greedy_generate(model, input_ids, attention_mask, *, maximum_new_tokens: int, eos_token_id: int | None):
    generated = input_ids.clone()
    mask = attention_mask.clone()
    prompt_length = generated.shape[1]
    for _ in range(maximum_new_tokens):
        output = model(
            input_ids=generated,
            attention_mask=mask,
            use_cache=False,
            logits_to_keep=1,
        )
        next_token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
        generated = torch.cat((generated, next_token), dim=1)
        mask = torch.cat((mask, torch.ones_like(next_token)), dim=1)
        if eos_token_id is not None and bool(torch.all(next_token == eos_token_id)):
            break
    return generated[:, prompt_length:]


def _run_evaluation(config: dict[str, Any], checkpoint: Path) -> dict[str, Any]:
    from transformers import AutoTokenizer
    from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

    from src.formal.inference import FormalInferenceEngine
    from src.distributed.fsdp_utils import destroy_distributed, init_distributed, wrap_qwen3_fsdp

    records, data_config, data_hash, _, expected = _validate_data_for_stage(
        config, stage="stage4_final_eval"
    )
    partitions, _, base_hash = _load_partitions(config, checkpoint)
    joint_dir = _pipeline_paths(config)["joint_checkpoint_output"]
    probe_dir = _pipeline_paths(config)["probe_output"]
    checkpoint_manifest_path = joint_dir / "checkpoint_manifest.json"
    if not checkpoint_manifest_path.is_file():
        raise FileNotFoundError("Formal joint checkpoint is required for evaluation")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, use_fast=True)
    eval_records = records_by_task_and_stage(records, stage="stage4_final_eval", expected_counts=expected)
    prompt_records = {task: eval_records[task] for task in FORMAL_TASKS}
    prompt_examples = build_formal_examples(
        prompt_records,
        data_config=data_config,
        tokenizer=tokenizer,
        target=False,
        max_length=int(config["pipeline"]["max_sequence_length"]),
    )
    distributed_context = None
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        distributed_context = init_distributed(require_cuda=True)
        device = distributed_context.device
    else:
        device = _device()
    native, model, banks, _ = _load_formal_runtime(
        config,
        checkpoint=checkpoint,
        partitions=partitions,
        device=device,
        load_checkpoint_dir=joint_dir,
        expected_base_checkpoint_sha256=base_hash,
        expected_data_manifest_sha256=data_hash,
        expected_config_sha256=sha256_json(config),
        distributed_context=distributed_context,
    )
    if distributed_context is not None:
        native = wrap_qwen3_fsdp(
            native,
            distributed_context,
            decoder_layer_classes=(Qwen3DecoderLayer,),
            sync_module_states=False,
            device_id=device,
        )
    else:
        native.to(device)
    native.eval()
    probe_head = load_formal_probe_head(
        probe_dir,
        hidden_size=int(
            model.module.config.hidden_size if hasattr(model, "module") else model.config.hidden_size
        ),
        expected_checkpoint_manifest_sha256=sha256_file(checkpoint_manifest_path),
        expected_data_manifest_sha256=data_hash,
        expected_partition_sha256_per_task={
            task: partitions[task].sha256 for task in FORMAL_TASKS
        },
    )
    engine = FormalInferenceEngine(
        model=model,
        banks=banks,
        probe_head=probe_head,
        device=device,
        probe_task="math",
    )
    max_new_tokens = config["evaluation"]["max_new_tokens"]
    rows: list[dict[str, Any]] = []
    task_scores: dict[str, list[dict[str, float]]] = {task: [] for task in FORMAL_TASKS}
    probe_correct = 0
    total_latency = 0.0
    for task in FORMAL_TASKS:
        for record, prompt in zip(eval_records[task], prompt_examples[task]):
            input_ids = prompt.input_ids.unsqueeze(0).to(device)
            attention_mask = prompt.attention_mask.unsqueeze(0).to(device)
            row, mapping, _ = load_record_row(record, data_config=data_config)
            gold = format_task_target(task, row, mapping)
            started = time.perf_counter()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            formal_result = engine.infer(
                input_ids.cpu(),
                attention_mask.cpu(),
                maximum_new_tokens=int(max_new_tokens[task]),
                eos_token_id=tokenizer.eos_token_id,
            )
            formal_prediction = tokenizer.decode(
                formal_result.generated_ids[0].cpu(), skip_special_tokens=True
            )
            formal_elapsed = time.perf_counter() - started
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                peak_memory = {
                    "allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                    "reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                }
            else:
                peak_memory = None
            baseline_ids = _greedy_generate(
                native,
                input_ids,
                attention_mask,
                maximum_new_tokens=int(max_new_tokens[task]),
                eos_token_id=tokenizer.eos_token_id,
            )
            baseline_prediction = tokenizer.decode(
                baseline_ids[0].cpu(), skip_special_tokens=True
            )
            kwargs = {}
            if task == "code":
                kwargs = {
                    "test_list": list(nested_value(row, str(mapping["test_list"]))),
                    "test_setup_code": str(nested_value(row, str(mapping["test_setup_code"]))),
                }
            formal_score = task_score(task, formal_prediction, gold=gold, **kwargs)
            baseline_score = task_score(task, baseline_prediction, gold=gold, **kwargs)
            task_scores[task].append(formal_score)
            probe_correct += int(formal_result.probe.predicted_task == task)
            total_latency += formal_elapsed
            rows.append(
                {
                    "case_id": record["stable_id"],
                    "dataset": record["dataset"],
                    "true_task": task,
                    "probe_prediction": formal_result.probe.predicted_task,
                    "probe_confidence": max(formal_result.probe.probabilities),
                    "selected_mode": formal_result.selected_task,
                    "partition_hash": formal_result.partition_sha256,
                    "query_hash": formal_result.query_sha256,
                    "alpha_hash": formal_result.alpha_sha256,
                    "alpha_statistics": formal_result.alpha_statistics,
                    "answer": formal_prediction,
                    "gold": gold,
                    "correctness": formal_score,
                    "baseline_answer": baseline_prediction,
                    "baseline_correctness": baseline_score,
                    "latency": formal_elapsed,
                    "peak_memory": peak_memory,
                }
            )
    high_confidence_threshold = float(
        config["probe"].get("high_confidence_threshold", 0.5)
    )
    high_confidence_rows = [
        row for row in rows if row["probe_confidence"] >= high_confidence_threshold
    ]
    high_confidence_wrong_route = sum(
        row["probe_prediction"] != row["true_task"] for row in high_confidence_rows
    )
    mode_specific_accuracy: dict[str, float | None] = {}
    for mode in FORMAL_TASKS:
        selected_rows = [row for row in rows if row["selected_mode"] == mode]
        mode_specific_accuracy[mode] = (
            sum(float(row["correctness"]["accuracy"]) for row in selected_rows)
            / len(selected_rows)
            if selected_rows
            else None
        )
    memory_rows = [row["peak_memory"] for row in rows if row["peak_memory"] is not None]
    result = {
        "status": "PASS",
        "base_checkpoint_sha256": base_hash,
        "data_manifest_sha256": data_hash,
        "counts": expected,
        "routing_accuracy": probe_correct / len(rows),
        "fallback_rate": sum(row["selected_mode"] == "fixed" for row in rows) / len(rows),
        "high_confidence_threshold": high_confidence_threshold,
        "high_confidence_wrong_route_rate": (
            high_confidence_wrong_route / len(high_confidence_rows)
            if high_confidence_rows
            else 0.0
        ),
        "high_confidence_case_count": len(high_confidence_rows),
        "task_specific_accuracy": {
            task: mean_metrics(task_scores[task]) for task in FORMAL_TASKS
        },
        "mode_specific_accuracy": mode_specific_accuracy,
        "end_to_end_accuracy": sum(row["correctness"]["accuracy"] for row in rows) / len(rows),
        "baseline_accuracy": sum(row["baseline_correctness"]["accuracy"] for row in rows) / len(rows),
        "mean_formal_latency_seconds": total_latency / len(rows),
        "latency": {"mean_seconds": total_latency / len(rows)},
        "peak_memory": (
            {
                "mean_allocated_peak_bytes": sum(
                    int(row["allocated_bytes"]) for row in memory_rows
                )
                / len(memory_rows),
                "mean_reserved_peak_bytes": sum(
                    int(row["reserved_bytes"]) for row in memory_rows
                )
                / len(memory_rows),
            }
            if memory_rows
            else None
        ),
        "partitions": {task: partitions[task].to_dict() for task in FORMAL_TASKS},
    }
    output = _pipeline_paths(config)["evaluation_output"]
    output.mkdir(parents=True, exist_ok=True)
    if not dist.is_initialized() or dist.get_rank() == 0:
        with (output / "predictions.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    _write_json(output / "evaluation_results.json", result)
    if distributed_context is not None:
        destroy_distributed(distributed_context)
    return result


def _validate_only(config: dict[str, Any], checkpoint: Path) -> dict[str, Any]:
    records, data_config, data_hash, leakage = _data_context(config)
    native_config = _native_config(checkpoint)
    stages = (
        "stage2_discovery",
        "stage3_adapter_train",
        "stage3_adapter_val",
        "probe_train",
        "probe_val",
        "stage4_final_eval",
    )
    stage_counts: dict[str, dict[str, int]] = {}
    stage_requirements: dict[str, dict[str, int]] = {}
    for stage in stages:
        expected = (
            _formal_discovery_counts(config)
            if stage == "stage2_discovery"
            else _required_counts(data_config, stage)
        )
        actual = {
            task: sum(
                1
                for record in records
                if record.get("task") == task
                and record.get("assigned_split") == stage
            )
            for task in FORMAL_TASKS
        }
        stage_counts[stage] = actual
        stage_requirements[stage] = expected
    data_ready = stage_counts == stage_requirements
    return {
        "status": "PASS" if data_ready else "BLOCKED",
        "model_type": native_config.model_type,
        "num_transformer_blocks": int(native_config.num_hidden_layers),
        "data_manifest_sha256": data_hash,
        "data_leakage": leakage["status"],
        "stage_records_available": stage_counts,
        "stage_records_required": stage_requirements,
        "formal_data_ready": data_ready,
        "status_note": (
            "ready for explicit stage execution"
            if data_ready
            else "formal manifest counts are below qwen3_14b_config.yaml; execution will fail fast"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Explicit formal Qwen3 task-adaptive pipeline")
    parser.add_argument("--config", default="qwen3_14b_config.yaml")
    parser.add_argument(
        "--stage",
        choices=("validate", "connectivity", "discovery", "train", "probe", "evaluate", "all"),
        default="validate",
    )
    parser.add_argument("--connectivity-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--max-steps", type=int, default=None, help="Only for a bounded local training smoke test")
    args = parser.parse_args()
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    config = load_formal_config(args.config)
    stage = "connectivity" if args.connectivity_only else args.stage
    status = formal_discovery_status(config["discovery"])
    print(status)
    checkpoint = resolve_base_model_path(config)
    if stage == "connectivity":
        from src.formal.connectivity import run_tiny_connectivity

        print(json.dumps(run_tiny_connectivity(config["training"]), indent=2, sort_keys=True))
        return
    if stage == "validate":
        print(json.dumps(_validate_only(config, checkpoint), indent=2, sort_keys=True))
        return
    if stage == "discovery":
        print(json.dumps(_run_formal_discovery(config, checkpoint), indent=2, sort_keys=True))
        return
    if stage == "train":
        if int(os.environ.get("WORLD_SIZE", "1")) == 1:
            print(json.dumps(
                _launch_formal_stage(
                    config,
                    checkpoint,
                    config_path=args.config,
                    stage="train",
                    max_steps=args.max_steps,
                ),
                indent=2,
                sort_keys=True,
            ))
            return
        print(json.dumps(_run_train(config, checkpoint, max_steps=args.max_steps), indent=2, sort_keys=True))
        return
    if stage == "probe":
        if int(os.environ.get("WORLD_SIZE", "1")) == 1:
            print(json.dumps(
                _launch_formal_stage(
                    config,
                    checkpoint,
                    config_path=args.config,
                    stage="probe",
                ),
                indent=2,
                sort_keys=True,
            ))
            return
        print(json.dumps(_run_probe(config, checkpoint), indent=2, sort_keys=True))
        return
    if stage == "evaluate":
        if int(os.environ.get("WORLD_SIZE", "1")) == 1:
            print(json.dumps(
                _launch_formal_stage(
                    config,
                    checkpoint,
                    config_path=args.config,
                    stage="evaluate",
                ),
                indent=2,
                sort_keys=True,
            ))
            return
        print(json.dumps(_run_evaluation(config, checkpoint), indent=2, sort_keys=True))
        return
    if args.max_steps is not None:
        raise ValueError("--max-steps is not allowed with --stage all")
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise RuntimeError("Formal all stage must be launched by a single coordinator")
    discovery = _run_formal_discovery(config, checkpoint)
    training = _launch_formal_stage(
        config,
        checkpoint,
        config_path=args.config,
        stage="train",
    )
    if training["status"] != "PASS":
        raise RuntimeError("Formal training did not reach all token budgets")
    probe = _launch_formal_stage(
        config,
        checkpoint,
        config_path=args.config,
        stage="probe",
    )
    evaluation = _launch_formal_stage(
        config,
        checkpoint,
        config_path=args.config,
        stage="evaluate",
    )
    print(json.dumps({"discovery": discovery, "training": training, "probe": probe, "evaluation": evaluation}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
