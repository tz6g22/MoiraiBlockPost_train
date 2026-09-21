from __future__ import annotations

import argparse
import gc
import json
import os
import random
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.distributed as dist
import yaml

from src.common import load_yaml, sha256_file, sha256_json
from src.data.format_tasks import (
    PromptOnlyExample,
    TargetCausalExample,
    collate_target_examples,
    format_task_target,
    nested_value,
)
from src.data.leakage_audit import audit_manifest
from src.discovery.ordinary_residual import (
    formal_discovery_status,
)
from src.formal.checkpoint import converted_config_sha256, load_joint_checkpoint, save_joint_checkpoint
from src.formal.config import (
    enabled_tasks,
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
    validate_training_source_mixture,
    validate_formal_source_policy,
)
from src.formal.probe import load_formal_probe_head, train_formal_probe
from src.formal.runtime import (
    build_joint_optimizer,
    identity_test,
    parameter_hash,
    trainability_audit,
)
from src.formal.task_banks import TaskBank
from src.formal.train_joint import FormalTokenScheduler, train_token_budget_sequential
from src.evaluation.task_metrics import mean_metrics, task_score
from src.evaluation.math_validation import (
    default_math_validation_manifest,
    evaluate_causal_lm,
    load_canonical_math_validation_examples,
)
from src.modeling.partition import MoiraiPartition


def _enabled_tasks(config: dict[str, Any]) -> tuple[str, ...]:
    return enabled_tasks(config)


def _path(raw: str) -> Path:
    return Path(os.path.expandvars(raw))


def _pipeline_paths(config: dict[str, Any]) -> dict[str, Path]:
    values = config["pipeline"]
    paths = {
        key: _path(str(value))
        for key, value in values.items()
        if key != "max_sequence_length" and isinstance(value, str)
    }
    if "run_root" in paths:
        paths["data_root"] = paths["run_root"] / "data"
    return paths


def _run_id(config: dict[str, Any]) -> str:
    value = str(config["pipeline"].get("run_id", "")).strip()
    if not value:
        raise ValueError("Formal run_id is required for run-scoped artifacts")
    return value


def _discovery_partition_paths(config: dict[str, Any]) -> dict[str, Path]:
    root = _pipeline_paths(config)["discovery_output"]
    thresholds = config["discovery"]["similarity_thresholds"]
    return {
        str(task): root / str(task) / f"threshold_{format(float(value), 'g')}" / "partition.json"
        for task, value in thresholds.items()
    }


def _validate_run_binding(config: dict[str, Any], data_hash: str) -> None:
    paths = _pipeline_paths(config)
    if "run_root" not in paths:
        return
    run_id = _run_id(config)
    run_manifest_path = paths["run_root"] / "run_manifest.json"
    if not run_manifest_path.is_file():
        _stale_error("STALE_RUN_MISMATCH", f"run manifest is missing: {run_manifest_path}")
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    if run_manifest.get("run_id") != run_id:
        _stale_error("STALE_RUN_MISMATCH", "run manifest run_id does not match config")
    if run_manifest.get("config_sha256") != sha256_json(config):
        _stale_error("STALE_RUN_MISMATCH", "run manifest config hash does not match config")
    if run_manifest.get("data_manifest_sha256") != data_hash:
        _stale_error("STALE_RUN_MISMATCH", "run manifest data hash does not match manifest")


def _prepare_run_manifest(config: dict[str, Any]) -> dict[str, Any]:
    """Create the isolated data manifest before any formal stage can consume it."""
    paths = _pipeline_paths(config)
    run_root = paths["run_root"]
    if run_root.exists() and any(run_root.iterdir()):
        _stale_error(
            "STALE_RUN_MISMATCH",
            f"formal run directory is non-empty; refusing implicit artifact reuse: {run_root}",
            runtime=True,
        )
    from src.data.prepare_post_data import prepare

    report = prepare(
        paths["data_config"],
        output_dir=paths["data_root"],
        run_id=_run_id(config),
    )
    manifest_path = paths["data_manifest"]
    data_hash = formal_data_manifest_sha256(manifest_path)
    if report.get("status") != "PASS" or report.get("manifest_sha256") != data_hash:
        _stale_error("STALE_RUN_MISMATCH", "generated data manifest report is invalid", runtime=True)
    if report.get("run_id") != _run_id(config):
        _stale_error("STALE_RUN_MISMATCH", "generated data manifest run_id mismatch", runtime=True)
    run_root.mkdir(parents=True, exist_ok=True)
    run_manifest = {
        "run_id": _run_id(config),
        "config_sha256": sha256_json(config),
        "data_config_sha256": sha256_file(paths["data_config"]),
        "data_manifest": str(manifest_path),
        "data_manifest_sha256": data_hash,
        "enabled_tasks": list(_enabled_tasks(config)),
        "stage2_discovery_cases": _formal_discovery_counts(config),
        "token_budget": config["data"]["token_budget"],
        "status": "DATA_READY",
    }
    _write_json(run_root / "run_manifest.json", run_manifest)
    return run_manifest


def _stale_error(status: str, detail: str, *, runtime: bool = False) -> None:
    error = RuntimeError if runtime else ValueError
    raise error(f"{status}: {detail}")


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


@torch.no_grad()
def _validation_loss(
    model,
    banks: Mapping[str, TaskBank],
    examples: Sequence[TargetCausalExample],
    *,
    task: str,
    tokenizer,
    device: torch.device,
    distributed_context,
) -> float:
    banks[task].activate(model)
    value = evaluate_causal_lm(
        model,
        examples,
        pad_token_id=int(tokenizer.pad_token_id),
        device=device,
        distributed_context=distributed_context,
    )["loss"]
    banks[task].activate(model)
    model.train()
    return float(value)


def _summarize_metric_file(path: Path, task: str) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ] if path.is_file() else []
    if not rows:
        raise RuntimeError(f"FORMAL_METRICS_EMPTY: {path}")
    validation = [row for row in rows if row.get("val_loss") is not None]
    total_tokens = sum(int(row["tokens_step"]) for row in rows)
    total_seconds = sum(float(row.get("train_wall_seconds", 0.0)) for row in rows)
    best_val = min((float(row["val_loss"]) for row in validation), default=None)
    final_val = float(validation[-1]["val_loss"]) if validation else None
    return {
        "task": task,
        "steps": len(rows),
        "final_train_loss": float(rows[-1]["train_loss"]),
        "best_val_loss": best_val,
        "final_val_loss": final_val,
        "best_val_ppl": float(np.exp(best_val)) if best_val is not None else None,
        "final_val_ppl": float(np.exp(final_val)) if final_val is not None else None,
        "average_tokens_per_second": total_tokens / total_seconds if total_seconds > 0 else 0.0,
        "peak_memory_gb": max(float(row.get("peak_memory_gb", 0.0)) for row in rows),
    }


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
    resume: bool = False,
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
    if resume:
        command.append("--resume")
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
    tasks = _enabled_tasks(config)
    audit_formal_source_provenance(records, data_config=data_config, enabled_tasks=tasks)
    validate_formal_source_policy(config, data_config, enabled_tasks=tasks)
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
        if "run_root" in paths and report.get("run_id") != _run_id(config):
            raise RuntimeError("STALE_RUN_MISMATCH: data manifest report run_id mismatch")
    _validate_run_binding(config, formal_data_manifest_sha256(paths["data_manifest"]))
    return records, data_config, formal_data_manifest_sha256(paths["data_manifest"]), leakage


def _required_counts(
    data_config: dict[str, Any],
    stage: str,
    tasks: tuple[str, ...],
) -> dict[str, int]:
    from src.data.format_tasks import task_split_count

    return {
        task: task_split_count(data_config, task=task, split_name=stage)
        for task in tasks
    }


def _formal_discovery_counts(config: dict[str, Any]) -> dict[str, int]:
    values = config["discovery"]["cases"]
    return {task: int(values[task]) for task in _enabled_tasks(config)}


def _validate_data_for_stage(
    config: dict[str, Any],
    *,
    stage: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], str, dict[str, tuple[dict[str, Any], ...]], dict[str, int]]:
    records, data_config, data_hash, leakage = _data_context(config)
    expected = (
        _formal_discovery_counts(config)
        if stage == "stage2_discovery"
        else _required_counts(data_config, stage, _enabled_tasks(config))
    )
    grouped = records_by_task_and_stage(
        records,
        stage=stage,
        expected_counts=expected,
        enabled_tasks=_enabled_tasks(config),
    )
    if stage == "stage3_adapter_train" and config["discovery"].get("method") == "linear_cka_min":
        validate_training_source_mixture(
            grouped,
            data_config=data_config,
            enabled_tasks=_enabled_tasks(config),
        )
    return records, data_config, data_hash, grouped, expected


def _load_partitions(
    config: dict[str, Any],
    checkpoint: Path,
    *,
    expected_data_manifest_sha256: str | None = None,
) -> tuple[dict[str, MoiraiPartition], str]:
    discovery = config["discovery"]
    if discovery.get("method") != "linear_cka_min":
        _stale_error(
            "STALE_PARTITION_MISMATCH",
            "formal pipeline is not configured for Linear CKA min Discovery",
        )
    if discovery.get("metric") != "linear_cka":
        _stale_error("STALE_PARTITION_MISMATCH", "formal Discovery metric is not linear_cka")
    if discovery.get("interval_reduction") != "min":
        _stale_error("STALE_PARTITION_MISMATCH", "formal CKA interval reduction is not min")
    thresholds = discovery.get("similarity_thresholds", {})
    if not isinstance(thresholds, dict):
        raise ValueError("Formal CKA similarity_thresholds must be task-specific")
    configured_paths = _discovery_partition_paths(config)
    native_config = _native_config(checkpoint)
    layers = int(native_config.num_hidden_layers)
    partition_config = config["discovery"]["partition"]
    root = _pipeline_paths(config)["discovery_output"]
    manifest_path = root / "linear_cka_manifest.json"
    if not manifest_path.is_file():
        _stale_error("STALE_PARTITION_MISMATCH", f"CKA manifest is missing: {manifest_path}")
    discovery_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if discovery_manifest.get("metric") != "linear_cka_residual_v1":
        _stale_error("STALE_PARTITION_MISMATCH", "CKA manifest metric mismatch")
    if discovery_manifest.get("interval_reduction") != "min":
        _stale_error("STALE_PARTITION_MISMATCH", "CKA manifest reduction mismatch")
    if discovery_manifest.get("mode") != "ordinary_residual_only":
        _stale_error("STALE_PARTITION_MISMATCH", "CKA manifest mode mismatch")
    if discovery_manifest.get("run_id") != _run_id(config):
        _stale_error("STALE_RUN_MISMATCH", "CKA manifest run_id mismatch")
    if discovery_manifest.get("num_transformer_blocks") != layers:
        _stale_error("MODEL_IDENTITY_MISMATCH", "CKA manifest layer count mismatch")
    if set(discovery_manifest.get("tasks", ())) != set(_enabled_tasks(config)):
        _stale_error("STALE_PARTITION_MISMATCH", "CKA manifest task set mismatch")
    manifest_thresholds = discovery_manifest.get("similarity_thresholds_by_task", {})
    if any(
        manifest_thresholds.get(task) != [float(thresholds[task])]
        for task in _enabled_tasks(config)
    ):
        _stale_error("STALE_PARTITION_MISMATCH", "CKA manifest threshold binding mismatch")
    partitions: dict[str, MoiraiPartition] = {}
    base_hash = _native_weight_hash(checkpoint)
    for task in _enabled_tasks(config):
        if task not in configured_paths:
            raise ValueError(f"Formal CKA partition path is missing for task {task}")
        if task not in thresholds:
            raise ValueError(f"Formal CKA similarity threshold is missing for task {task}")
        threshold = float(thresholds[task])
        if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Invalid formal CKA similarity threshold for {task}: {threshold}")
        path = configured_paths[task]
        if not path.is_file():
            _stale_error("STALE_PARTITION_MISMATCH", f"configured partition is missing: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        try:
            partition = MoiraiPartition.from_dict(
                {
                    "task": payload["task"],
                    "num_transformer_blocks": payload["num_transformer_blocks"],
                    "blocks": [
                        {
                            key: block[key]
                            for key in ("block_id", "start", "end", "length")
                        }
                        for block in payload["blocks"]
                    ],
                    "constraints": payload.get("constraints", {}),
                }
            )
        except (KeyError, TypeError, ValueError) as exc:
            _stale_error("STALE_PARTITION_MISMATCH", f"invalid CKA partition for {task}: {exc}")
        if partition.task != task:
            _stale_error("STALE_PARTITION_MISMATCH", f"task mismatch for {task}")
        if partition.num_transformer_blocks != layers:
            _stale_error("MODEL_IDENTITY_MISMATCH", f"partition depth mismatch for {task}")
        if partition.min_block_length < int(partition_config["min_block_length"]):
            _stale_error("STALE_PARTITION_MISMATCH", f"minimum length mismatch for {task}")
        configured_max = partition_config.get("max_block_length")
        if configured_max is not None and partition.max_block_length > int(configured_max):
            _stale_error("STALE_PARTITION_MISMATCH", f"maximum length mismatch for {task}")
        if payload.get("metric") != "linear_cka_residual_v1":
            _stale_error("STALE_PARTITION_MISMATCH", f"partition metric mismatch for {task}")
        if payload.get("interval_reduction") != "min":
            _stale_error("STALE_PARTITION_MISMATCH", f"partition reduction mismatch for {task}")
        if payload.get("task") != task:
            _stale_error("STALE_PARTITION_MISMATCH", f"partition task mismatch for {task}")
        if float(payload.get("similarity_threshold", float("nan"))) != threshold:
            _stale_error("STALE_PARTITION_MISMATCH", f"partition threshold mismatch for {task}")
        if payload.get("block_sizes") != list(partition.lengths):
            _stale_error("STALE_PARTITION_MISMATCH", f"partition block-size payload mismatch for {task}")
        if payload.get("partition_sha256") != partition.sha256:
            _stale_error("STALE_PARTITION_MISMATCH", f"partition hash mismatch for {task}")
        if payload.get("run_id") != _run_id(config):
            _stale_error("STALE_RUN_MISMATCH", f"partition run_id mismatch for {task}")
        if discovery_manifest.get("base_checkpoint_sha256") != base_hash:
            _stale_error("MODEL_IDENTITY_MISMATCH", f"CKA base checkpoint mismatch for {task}")
        if (
            expected_data_manifest_sha256 is not None
            and discovery_manifest.get("data_manifest_sha256") != expected_data_manifest_sha256
        ):
            _stale_error("STALE_PARTITION_MISMATCH", f"data manifest mismatch for {task}")

        expected_cases = int(config["discovery"]["cases"][task])
        statistics_path = path.parent.parent / "statistics.json"
        if not statistics_path.is_file() and discovery_manifest.get("source_output_dir"):
            statistics_path = Path(discovery_manifest["source_output_dir"]) / task / "statistics.json"
        if not statistics_path.is_file():
            _stale_error("STALE_PARTITION_MISMATCH", f"CKA statistics are missing: {statistics_path}")
        statistics = json.loads(statistics_path.read_text(encoding="utf-8"))
        if statistics.get("metric") != "linear_cka_residual_v1":
            _stale_error("STALE_PARTITION_MISMATCH", f"CKA statistics mismatch for {task}")
        if int(statistics.get("case_count", -1)) != expected_cases:
            _stale_error("STALE_PARTITION_MISMATCH", f"Discovery case count mismatch for {task}")

        similarity_path = path.parent.parent / "similarity_matrix.npy"
        interval_path = path.parent.parent / "interval_similarity_matrix.npy"
        if not similarity_path.is_file() or not interval_path.is_file():
            _stale_error("STALE_PARTITION_MISMATCH", f"CKA matrices are missing for {task}")
        matrix_hashes = discovery_manifest.get("similarity_matrix_sha256") or discovery_manifest.get(
            "source_similarity_matrix_sha256", {}
        )
        if matrix_hashes.get(task) != sha256_file(similarity_path):
            _stale_error("STALE_PARTITION_MISMATCH", f"CKA similarity matrix hash mismatch for {task}")
        if payload.get("similarity_matrix_sha256") != sha256_file(similarity_path):
            _stale_error("STALE_PARTITION_MISMATCH", f"partition matrix hash mismatch for {task}")
        if payload.get("base_checkpoint_sha256") != base_hash:
            _stale_error("MODEL_IDENTITY_MISMATCH", f"partition base checkpoint mismatch for {task}")
        if (
            expected_data_manifest_sha256 is not None
            and payload.get("data_manifest_sha256") != expected_data_manifest_sha256
        ):
            _stale_error("STALE_PARTITION_MISMATCH", f"partition data manifest mismatch for {task}")
        similarity = np.load(similarity_path, allow_pickle=False)
        interval_similarity = np.load(interval_path, allow_pickle=False)
        if (
            similarity.shape != (layers, layers)
            or not np.isfinite(similarity).all()
            or not np.allclose(similarity, similarity.T, rtol=0.0, atol=1.0e-6)
            or not np.allclose(np.diag(similarity), 1.0, rtol=0.0, atol=1.0e-6)
        ):
            _stale_error("MODEL_IDENTITY_MISMATCH", f"CKA similarity matrix is invalid for {task}")
        if interval_similarity.shape != (layers, layers):
            _stale_error("MODEL_IDENTITY_MISMATCH", f"CKA interval matrix shape mismatch for {task}")
        for start in range(layers):
            if not np.isfinite(interval_similarity[start, start]):
                _stale_error("MODEL_IDENTITY_MISMATCH", f"CKA singleton score is invalid for {task}")
            for end in range(start + 1, layers):
                if not np.isfinite(interval_similarity[start, end]):
                    _stale_error("MODEL_IDENTITY_MISMATCH", f"CKA interval score is invalid for {task}")
        if int(payload.get("num_transformer_blocks", -1)) != layers:
            _stale_error("MODEL_IDENTITY_MISMATCH", f"partition layer count mismatch for {task}")
        cursor = 0
        for block in partition.blocks:
            if block.start != cursor or block.end >= layers:
                _stale_error("STALE_PARTITION_MISMATCH", f"partition coverage mismatch for {task}")
            if block.length > 1 and interval_similarity[block.start, block.end] + 1.0e-12 < threshold:
                _stale_error(
                    "STALE_PARTITION_MISMATCH",
                    f"block similarity below threshold for {task}: [{block.start},{block.end}]={interval_similarity[block.start, block.end]}",
                )
            cursor = block.end + 1
        if cursor != layers:
            _stale_error("STALE_PARTITION_MISMATCH", f"partition does not cover all layers for {task}")
        partitions[task] = partition
    return partitions, base_hash


def _build_task_banks(
    model,
    partitions: dict[str, MoiraiPartition],
) -> dict[str, TaskBank]:
    banks: dict[str, TaskBank] = {}
    for task in partitions:
        partition = partitions[task]
        model.config.moirai_partition = list(partition.lengths)
        model.config.moirai_task = task
        model.config.attnres_execution = "formal"
        bank = TaskBank.from_model(model, task=task, partition_sha256=partition.sha256)
        bank.partition_lengths = partition.lengths
        banks[task] = bank
    banks[next(iter(banks))].activate(model)
    return banks


def _load_formal_runtime(
    config: dict[str, Any],
    *,
    checkpoint: Path,
    partitions: dict[str, MoiraiPartition],
    device: torch.device,
    identity_examples: Mapping[str, Sequence[TargetCausalExample]] | Sequence[TargetCausalExample] = (),
    identity_pad_token_id: int = 0,
    load_checkpoint_dir: Path | None = None,
    distributed_context=None,
    expected_base_checkpoint_sha256: str | None = None,
    expected_data_manifest_sha256: str | None = None,
    expected_config_sha256: str | None = None,
):
    from src.formal.conversion import convert_qwen3_checkpoint

    initial_task = next(iter(partitions))
    initial_partition = partitions[initial_task]
    original, formal = convert_qwen3_checkpoint(
        checkpoint,
        partition=list(initial_partition.lengths),
        task=initial_task,
        min_block_length=initial_partition.min_block_length,
        max_block_length=initial_partition.max_block_length,
        no_adjacent_singletons=initial_partition.no_adjacent_singletons,
        alpha_init=float(config.get("attnres", {}).get("alpha", {}).get("init", 0.0)),
        use_alpha=bool(config.get("attnres", {}).get("alpha", {}).get("enabled", True)),
        dtype=torch.bfloat16,
        routing_dtype=torch.float32,
    )
    banks = _build_task_banks(formal, partitions)
    identity = None
    run_identity = identity_examples and (
        distributed_context is None or distributed_context.is_rank0
    )
    if run_identity:
        tasks = (
            _enabled_tasks(config)
            if config.get("tasks", {}).get("enabled")
            else tuple(partitions)
        )
        if isinstance(identity_examples, Mapping):
            examples_by_task = {
                task: tuple(identity_examples.get(task, ()))
                for task in tasks
            }
        else:
            examples_by_task = {task: tuple(identity_examples) if index == 0 else () for index, task in enumerate(tasks)}
        if any(not examples_by_task[task] for task in tasks):
            raise RuntimeError(
                "FORMAL_IDENTITY_TEST_MISSING: one identity example is required "
                "for each formal task partition"
            )
        if distributed_context is not None and distributed_context.distributed:
            # A large CPU forward is unnecessarily slow on the coordinator node.
            # Rank 0 owns the validation work and can use its assigned GPU;
            # return both models to CPU before FSDP wraps the formal model.
            original.to(device)
            formal.to(device)
        per_task_identity: dict[str, dict[str, Any]] = {}
        for task in tasks:
            banks[task].activate(formal)
            count = int(config.get("identity_test", {}).get("num_examples", 1))
            selected = examples_by_task[task][:count]
            batch = collate_target_examples(
                selected,
                pad_token_id=identity_pad_token_id,
            )
            ids = batch["input_ids"]
            mask = batch["attention_mask"]
            if distributed_context is not None and distributed_context.distributed:
                ids = ids.to(device)
                mask = mask.to(device)
            task_identity = identity_test(original, formal, ids, mask)
            if task_identity["status"] != "PASS":
                raise RuntimeError(
                    f"IDENTITY_CONVERSION_FAILED for formal task {task}"
                )
            per_task_identity[task] = task_identity
        banks[next(iter(banks))].activate(formal)
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
            active_task=next(iter(banks)),
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
    records, data_config, data_hash, grouped, _ = _validate_data_for_stage(
        config, stage="stage2_discovery"
    )
    del records, data_config, grouped
    discovery = config["discovery"]
    if discovery.get("method") != "linear_cka_min":
        raise RuntimeError("STALE_PARTITION_MISMATCH: formal Discovery must use linear_cka_min")
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
        "run_id": _run_id(config),
        "model_mode": "original_residual_only",
        "formal_discovery": True,
        "ordinary_residual_only": True,
        "discovery_metric": "linear_cka",
        "cka_interval_reduction": "min",
        "base_checkpoint": str(checkpoint),
        "data_manifest": str(paths["data_manifest"]),
        "data_config": str(paths["data_config"]),
        "tasks": list(_enabled_tasks(config)),
        "discovery_cases_per_task": _formal_discovery_counts(config),
        "task_similarity_thresholds": {
            task: float(discovery["similarity_thresholds"][task])
            for task in _enabled_tasks(config)
        },
        "min_block_length": int(config["discovery"]["partition"]["min_block_length"]),
        "max_block_length": config["discovery"]["partition"].get("max_block_length"),
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
    ]
    subprocess.run(command, check=True)
    partitions, base_hash = _load_partitions(
        config, checkpoint, expected_data_manifest_sha256=data_hash
    )
    run_manifest_path = paths["run_root"] / "run_manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    cka_manifest_path = paths["discovery_output"] / "linear_cka_manifest.json"
    cka_manifest = json.loads(cka_manifest_path.read_text(encoding="utf-8"))
    run_manifest["discovery"] = {
        "method": "linear_cka_min",
        "manifest": str(cka_manifest_path),
        "manifest_sha256": sha256_file(cka_manifest_path),
        "similarity_matrix_sha256": cka_manifest["similarity_matrix_sha256"],
        "partition_sha256": {
            task: partitions[task].sha256 for task in _enabled_tasks(config)
        },
        "status": "PASS",
    }
    _write_json(run_manifest_path, run_manifest)
    result = {
        "status": "PASS",
        "base_checkpoint_sha256": base_hash,
        "num_transformer_blocks": int(_native_config(checkpoint).num_hidden_layers),
        "tasks": {
            task: {
                "lengths": list(partitions[task].lengths),
                "num_moirai_blocks": len(partitions[task].blocks),
                "partition_sha256": partitions[task].sha256,
            }
            for task in _enabled_tasks(config)
        },
    }
    _write_json(paths["discovery_output"] / "formal_pipeline_discovery_summary.json", result)
    return result


def _run_train(
    config: dict[str, Any],
    checkpoint: Path,
    *,
    max_steps: int | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    _, data_config, data_hash, grouped, expected = _validate_data_for_stage(
        config, stage="stage3_adapter_train"
    )
    _, _, _, validation_grouped, _ = _validate_data_for_stage(
        config, stage="stage3_adapter_val"
    )
    partitions, base_hash = _load_partitions(
        config, checkpoint, expected_data_manifest_sha256=data_hash
    )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, use_fast=True)
    max_length = int(config["pipeline"]["max_sequence_length"])
    task_order = tuple(config["training"]["task_order"])
    examples = build_formal_examples(
        grouped,
        data_config=data_config,
        tokenizer=tokenizer,
        target=True,
        max_length=max_length,
    )
    canonical_manifest = config.get("pipeline", {}).get(
        "canonical_math_validation_manifest",
        str(default_math_validation_manifest()),
    )
    validation_examples = {
        task: (
            load_canonical_math_validation_examples(
                manifest_path=canonical_manifest,
                data_manifest_path=config["pipeline"]["data_manifest"],
                data_config_path=config["pipeline"]["data_config"],
                tokenizer=tokenizer,
                max_length=max_length,
            )
            if task == "math"
            else build_formal_examples(
                {task: validation_grouped[task]},
                data_config=data_config,
                tokenizer=tokenizer,
                target=True,
                max_length=max_length,
            )[task]
        )
        for task in task_order
    }
    source_registry = {
        key: source
        for section in ("sources", "external_sources")
        for key, source in data_config.get(section, {}).items()
    }
    source_examples = {task: {} for task in task_order}
    for task in task_order:
        for record, example in zip(grouped[task], examples[task]):
            source_examples[task].setdefault(str(record["dataset"]), []).append(example)
    source_weights = {
        task: {
            str(source_registry[key]["dataset_name"]): float(weight)
            for key, weight in data_config["training_source_weights"][task].items()
        }
        for task in task_order
    }
    token_budgets = {
        task: int(config["data"]["token_budget"][task])
        for task in task_order
    }
    paths = _pipeline_paths(config)
    joint_dir = paths["joint_checkpoint_output"]
    manifest_path = joint_dir / "checkpoint_manifest.json"
    if resume and not manifest_path.is_file():
        _stale_error(
            "STALE_CHECKPOINT_MISMATCH",
            f"requested resume checkpoint is missing: {manifest_path}",
            runtime=True,
        )
    if not resume and joint_dir.exists() and any(joint_dir.iterdir()):
        _stale_error(
            "STALE_CHECKPOINT_MISMATCH",
            "existing formal checkpoint output requires explicit --resume; refusing to overwrite",
            runtime=True,
        )
    if not resume and any(
        (joint_dir.parent / f"metrics_{task}.jsonl").is_file()
        for task in task_order
    ):
        _stale_error(
            "STALE_CHECKPOINT_MISMATCH",
            "existing formal metrics require explicit --resume; refusing to overwrite training history",
            runtime=True,
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
        identity_examples=(
            validation_examples
            if not bool(config.get("attnres", {}).get("alpha", {}).get("enabled", True))
            else examples
        ),
        identity_pad_token_id=tokenizer.pad_token_id,
        distributed_context=distributed_context,
    )
    del original
    if identity is None:
        raise RuntimeError("FORMAL_IDENTITY_TEST_MISSING")
    identity_record = {
        **identity,
        "base_checkpoint_sha256": base_hash,
        "converted_config_sha256": converted_config_sha256(model),
    }
    _write_json(paths["joint_checkpoint_output"].parent / "identity_test.json", identity_record)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    optimizer = build_joint_optimizer(
        model,
        backbone_lr=float(config["training"]["optimizer"]["parameter_groups"]["backbone"]["lr"]),
        attnres_lr=float(config["training"]["optimizer"]["parameter_groups"]["attnres"]["lr"]),
        backbone_weight_decay=float(config["training"]["optimizer"]["parameter_groups"]["backbone"]["weight_decay"]),
        attnres_weight_decay=float(config["training"]["optimizer"]["parameter_groups"]["attnres"]["weight_decay"]),
        query_lr=(
            float(config["training"]["optimizer"]["parameter_groups"]["query"]["lr"])
            if "query" in config["training"]["optimizer"]["parameter_groups"]
            else None
        ),
        alpha_lr=(
            float(config["training"]["optimizer"]["parameter_groups"]["alpha"]["lr"])
            if "alpha" in config["training"]["optimizer"]["parameter_groups"]
            else None
        ),
        betas=tuple(config["training"]["optimizer"]["betas"]),
        eps=float(config["training"]["optimizer"]["eps"]),
    )
    scheduler = FormalTokenScheduler(
        optimizer,
        maximum_tokens=sum(token_budgets.values()),
        warmup_ratio=float(config["training"]["scheduler"]["warmup_ratio"]),
        min_lr_ratio=float(config["training"]["scheduler"]["min_lr_ratio"]),
        task_budgets=token_budgets,
    )
    initial_progress: dict[str, Any] = {}
    if resume:
        resume_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        resume_progress = resume_payload.get("training_progress", {})
        resume_task = resume_progress.get("current_task") or task_order[0]
        if resume_task not in task_order:
            raise ValueError("Formal resume checkpoint has an unknown current task")
        resume_manifest = load_joint_checkpoint(
            joint_dir,
            model=model,
            banks=banks,
            optimizer=optimizer,
            scheduler=scheduler,
            active_task=resume_task,
            restore_rng=True,
            expected_base_checkpoint_sha256=base_hash,
            expected_data_manifest_sha256=data_hash,
            expected_config_sha256=sha256_json(config),
        )
        initial_progress = dict(resume_manifest.get("training_progress", {}))
        if tuple(initial_progress.get("task_order", ())) != task_order:
            _stale_error("TASK_ORDER_MISMATCH", "formal resume task order does not match config")
        if initial_progress.get("mode") != "sequential_task_training":
            _stale_error("TRAINING_MODE_MISMATCH", "formal resume checkpoint is not sequential training state")
        expected_metric_files = {
            task: str(paths["joint_checkpoint_output"].parent / f"metrics_{task}.jsonl")
            for task in task_order
        }
        expected_metric_files["combined"] = str(
            paths["joint_checkpoint_output"].parent / "metrics.jsonl"
        )
        if resume_manifest.get("metrics_files") != expected_metric_files:
            raise ValueError("Formal resume metrics files do not match the current pipeline")

    task_bank_updates = {
        task: {
            "query": bool(initial_progress.get("task_bank_updates", {}).get(task, {}).get("query", False)),
            "alpha": bool(initial_progress.get("task_bank_updates", {}).get(task, {}).get("alpha", False)),
        }
        for task in task_order
    }

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
        for task in _enabled_tasks(config)
    }
    partition_hashes_before = {task: partitions[task].sha256 for task in task_order}
    metric_paths = {
        task: paths["joint_checkpoint_output"].parent / f"metrics_{task}.jsonl"
        for task in task_order
    }
    unified_metrics_path = paths["joint_checkpoint_output"].parent / "metrics.jsonl"
    metrics_summary_path = paths["joint_checkpoint_output"].parent / "metrics_summary.json"
    window_size = int(config["training"]["metrics"]["window_size"])
    loss_windows = {task: deque(maxlen=window_size) for task in task_order}
    if resume:
        for task, path in metric_paths.items():
            if path.is_file():
                for line in path.read_text(encoding="utf-8").splitlines()[-window_size:]:
                    if line.strip():
                        loss_windows[task].append(float(json.loads(line)["loss"]))
    last_loss_per_task: dict[str, float | None] = {task: None for task in task_order}
    stage_peak_memory_bytes = {
        task: int(initial_progress.get("stage_peak_memory_bytes", {}).get(task, 0))
        for task in task_order
    }
    current_stage_peak_memory_bytes = int(
        initial_progress.get("current_stage_peak_memory_bytes", 0)
    )

    def write_metric(record: dict[str, Any]) -> None:
        nonlocal current_stage_peak_memory_bytes
        task = str(record["task"])
        task_bank_updates[task]["query"] |= float(record.get("query_parameter_delta", 0.0)) > 0.0
        task_bank_updates[task]["alpha"] |= float(record.get("alpha_parameter_delta", 0.0)) > 0.0
        loss_windows[task].append(float(record["loss"]))
        task_cumulative_tokens = record.get("task_cumulative_tokens") or {}
        tokens_step = int(
            record.get("non_padding_tokens_this_step", record.get("tokens_step", 0))
        )
        task_tokens = int(task_cumulative_tokens.get(task, record.get("task_tokens", 0)))
        global_tokens = int(
            record.get("global_cumulative_tokens", record.get("global_tokens", 0))
        )
        train_wall_seconds = float(record.get("train_wall_seconds", 0.0))
        if torch.cuda.is_available():
            current_stage_peak_memory_bytes = max(
                current_stage_peak_memory_bytes,
                int(torch.cuda.max_memory_allocated(device)),
            )
        validation_interval = int(
            config["training"]["metrics"]["validation_interval_steps"]
        )
        val_loss = None
        if int(record["global_step"]) % validation_interval == 0:
            val_loss = _validation_loss(
                model,
                banks,
                validation_examples[task][: int(config["training"]["metrics"]["validation_examples_per_task"])],
                task=task,
                tokenizer=tokenizer,
                device=device,
                distributed_context=distributed_context,
            )
        peak_bytes = torch.tensor(
            current_stage_peak_memory_bytes,
            dtype=torch.int64,
            device=device,
        )
        if distributed_context is not None and distributed_context.distributed:
            dist.all_reduce(peak_bytes, op=dist.ReduceOp.MAX)
            current_stage_peak_memory_bytes = int(peak_bytes.item())
        peak_memory_gb = current_stage_peak_memory_bytes / (1024 ** 3)
        record = {
            **record,
            "raw_loss": float(record["loss"]),
            "method": "task_adaptive_block_attnres",
            "stage": task,
            "tokens_seen": global_tokens,
            "train_loss": float(record["loss"]),
            "val_loss": val_loss,
            "val_ppl": float(np.exp(val_loss)) if val_loss is not None else None,
            "smoothed_loss": sum(loss_windows[task]) / len(loss_windows[task]),
            "tokens_step": tokens_step,
            "task_tokens": task_tokens,
            "global_tokens": global_tokens,
            "tokens_per_second": tokens_step / max(train_wall_seconds, 1.0e-9),
            "train_wall_seconds": train_wall_seconds,
            "peak_memory_gb": peak_memory_gb,
        }
        last_loss_per_task[task] = float(record["loss"])
        if dist.is_initialized() and dist.get_rank() != 0:
            return
        with metric_paths[task].open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        with unified_metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def save_progress(progress, active_optimizer, active_scheduler) -> None:
        nonlocal current_stage_peak_memory_bytes
        completed_task = (
            progress["completed_tasks"][-1]
            if progress.get("completed_tasks")
            else None
        )
        if completed_task is not None:
            stage_peak_memory_bytes[completed_task] = max(
                stage_peak_memory_bytes.get(completed_task, 0),
                current_stage_peak_memory_bytes,
            )
            current_stage_peak_memory_bytes = 0
        progress = {
            **progress,
            "task_bank_updates": task_bank_updates,
            "stage_peak_memory_bytes": stage_peak_memory_bytes,
            "current_stage_peak_memory_bytes": current_stage_peak_memory_bytes,
        }
        metrics_files = {task: str(metric_paths[task]) for task in task_order}
        metrics_files["combined"] = str(unified_metrics_path)
        save_joint_checkpoint(
            joint_dir,
            model=model,
            banks=banks,
            partitions={task: partitions[task].to_dict() for task in task_order},
            optimizer=active_optimizer,
            scheduler=active_scheduler,
            config=config,
            base_checkpoint_sha256=base_hash,
            data_manifest_sha256=data_hash,
            consumed_tokens=progress["task_cumulative_tokens"],
            seed=int(config["experiment"]["seed"]),
            identity_test=identity_record,
            training_progress=progress,
            metrics_files=metrics_files,
        )
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)

    optimizer, scheduler, consumed, results, progress = train_token_budget_sequential(
        model,
        banks=banks,
        examples_by_task=examples,
        source_examples_by_task=source_examples,
        source_weights_by_task=source_weights,
        tokenizer=tokenizer,
        training_config=config["training"],
        token_budgets=token_budgets,
        task_order=task_order,
        device=device,
        max_steps=max_steps,
        distributed_context=distributed_context,
        seed=int(config["experiment"]["seed"]),
        shuffle=bool(config["data"]["shuffle"]),
        optimizer=optimizer,
        scheduler=scheduler,
        initial_progress=initial_progress,
        on_step=write_metric,
        on_task_boundary=save_progress,
    )
    current_task = progress.get("current_task")
    if current_task in task_order:
        stage_peak_memory_bytes[current_task] = max(
            stage_peak_memory_bytes.get(current_task, 0),
            current_stage_peak_memory_bytes,
        )
    progress = {
        **progress,
        "task_bank_updates": task_bank_updates,
        "stage_peak_memory_bytes": stage_peak_memory_bytes,
        "current_stage_peak_memory_bytes": current_stage_peak_memory_bytes,
    }
    if torch.cuda.is_available():
        from src.distributed.fsdp_utils import DistributedContext, peak_memory_stats

        memory_context = distributed_context or DistributedContext(
            rank=0,
            local_rank=int(device.index or 0),
            world_size=1,
            device=device,
        )
        peak_memory_rows = peak_memory_stats(memory_context)
        peak_memory = {
            "scope": "formal_training_after_identity",
            "per_rank": peak_memory_rows,
            "peak_allocated_bytes": max(
                row["peak_allocated_bytes"] for row in peak_memory_rows or []
            ),
            "peak_reserved_bytes": max(
                row["peak_reserved_bytes"] for row in peak_memory_rows or []
            ),
        } if peak_memory_rows is not None else None
    else:
        peak_memory = None
    complete = all(consumed[task] >= token_budgets[task] for task in task_order)
    metric_summaries = (
        {
            task: _summarize_metric_file(metric_paths[task], task)
            for task in task_order
            if metric_paths[task].is_file()
        }
        if not dist.is_initialized() or dist.get_rank() == 0
        else {}
    )
    if not dist.is_initialized() or dist.get_rank() == 0:
        metrics_summary_path.write_text(
            json.dumps(
                {
                    "method": "task_adaptive_block_attnres",
                    "math": metric_summaries.get("math"),
                    "multihop": metric_summaries.get("multihop"),
                    "overall": {
                        "total_tokens": int(sum(consumed.values())),
                        "total_walltime": float(progress.get("global_training_wall_seconds", 0.0)),
                        "max_peak_memory_gb": max(stage_peak_memory_bytes.values(), default=0) / (1024 ** 3),
                    },
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
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
        for task in _enabled_tasks(config)
    }
    partition_hashes_after = {task: partitions[task].sha256 for task in _enabled_tasks(config)}
    changed = {
        task: {
            "query": task_bank_updates[task]["query"]
            or before_bank_hashes[task]["query"] != after_bank_hashes[task]["query"],
            "alpha": task_bank_updates[task]["alpha"]
            or before_bank_hashes[task]["alpha"] != after_bank_hashes[task]["alpha"],
        }
        for task in task_order
    }
    if partition_hashes_after != partition_hashes_before:
        raise RuntimeError("PARTITION_CHANGED_IN_FORMAL_TRAINING_SUMMARY")
    if complete and before_backbone_hash == after_backbone_hash:
        raise RuntimeError("FORMAL_BACKBONE_DID_NOT_UPDATE")
    if complete and not all(changed[task]["query"] for task in task_order):
        raise RuntimeError("FORMAL_TASK_QUERY_DID_NOT_UPDATE")
    if complete and alpha_names and not all(changed[task]["alpha"] for task in task_order):
        raise RuntimeError("FORMAL_TASK_ALPHA_DID_NOT_UPDATE")
    summary = {
        "status": "PASS" if complete else "PARTIAL_CONNECTIVITY",
        "training_mode": "sequential_task_training",
        "task_order": list(task_order),
        "identity": identity_record,
        "requested_record_counts": expected,
        "consumed_tokens_per_task": consumed,
        "steps": len(results),
        "last_loss": results[-1].loss if results else None,
        "last_loss_per_task": last_loss_per_task,
        "backbone_grad_norm_last": results[-1].backbone_grad_norm if results else None,
        "query_grad_norm_last": results[-1].query_grad_norm if results else None,
        "alpha_grad_norm_last": results[-1].alpha_grad_norm if results else None,
        "query_parameter_delta_last": results[-1].query_parameter_delta if results else None,
        "alpha_parameter_delta_last": results[-1].alpha_parameter_delta if results else None,
        "query_optimizer_membership": all(
            result.query_optimizer_membership for result in results
        ),
        "query_dtype": results[-1].query_dtype if results else None,
        "alpha_dtype": results[-1].alpha_dtype if results else None,
        "query_optimizer_state_dtype": (
            results[-1].query_optimizer_state_dtype if results else None
        ),
        "alpha_optimizer_state_dtype": (
            results[-1].alpha_optimizer_state_dtype if results else None
        ),
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
        "partitions": {task: partitions[task].to_dict() for task in task_order},
        "training_progress": progress,
        "metrics_files": {
            **{task: str(metric_paths[task]) for task in task_order},
            "combined": str(unified_metrics_path),
        },
        "peak_memory": peak_memory,
        "metrics_summary": metric_summaries,
        "metrics_path": str(unified_metrics_path),
        "stage_peak_memory_bytes": stage_peak_memory_bytes,
    }
    _write_json(paths["joint_checkpoint_output"].parent / "formal_training_summary.json", summary)
    manifest = save_joint_checkpoint(
        paths["joint_checkpoint_output"],
        model=model,
        banks=banks,
        partitions={task: partitions[task].to_dict() for task in task_order},
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        base_checkpoint_sha256=base_hash,
        data_manifest_sha256=data_hash,
        consumed_tokens=consumed,
        seed=int(config["experiment"]["seed"]),
        identity_test=identity_record,
        training_progress=progress,
        metrics_files={
            **{task: str(metric_paths[task]) for task in task_order},
            "combined": str(unified_metrics_path),
        },
    )
    summary["checkpoint_manifest_sha256"] = sha256_file(
        paths["joint_checkpoint_output"] / "checkpoint_manifest.json"
    )
    summary["checkpoint_status"] = manifest["checkpoint_kind"]
    _write_json(paths["joint_checkpoint_output"].parent / "formal_training_summary.json", summary)
    return summary


def _probe_data(config: dict[str, Any], tokenizer, data_config, records):
    from src.formal.data import records_by_task_and_stage

    tasks = _enabled_tasks(config)
    train_records = records_by_task_and_stage(
        records,
        stage="probe_train",
        expected_counts=_required_counts(data_config, "probe_train", tasks),
        enabled_tasks=tasks,
    )
    val_records = records_by_task_and_stage(
        records,
        stage="probe_val",
        expected_counts=_required_counts(data_config, "probe_val", tasks),
        enabled_tasks=tasks,
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
    train_examples = [example for task in _enabled_tasks(config) for example in train[task]]
    train_labels = [index for index, task in enumerate(_enabled_tasks(config)) for _ in train[task]]
    val_examples = [example for task in _enabled_tasks(config) for example in validation[task]]
    val_labels = [index for index, task in enumerate(_enabled_tasks(config)) for _ in validation[task]]
    return train_examples, train_labels, val_examples, val_labels


def _run_probe(config: dict[str, Any], checkpoint: Path) -> dict[str, Any]:
    from transformers import AutoTokenizer
    from src.distributed.fsdp_utils import destroy_distributed, init_distributed

    records, data_config, data_hash, _, _ = _data_context(config)
    partitions, base_hash = _load_partitions(
        config, checkpoint, expected_data_manifest_sha256=data_hash
    )
    joint_dir = _pipeline_paths(config)["joint_checkpoint_output"]
    manifest_path = joint_dir / "checkpoint_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"STALE_CHECKPOINT_MISMATCH: formal joint checkpoint is missing: {manifest_path}"
        )
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
        task: partitions[task].sha256 for task in _enabled_tasks(config)
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
    partitions, base_hash = _load_partitions(
        config, checkpoint, expected_data_manifest_sha256=data_hash
    )
    joint_dir = _pipeline_paths(config)["joint_checkpoint_output"]
    probe_dir = _pipeline_paths(config)["probe_output"]
    checkpoint_manifest_path = joint_dir / "checkpoint_manifest.json"
    if not checkpoint_manifest_path.is_file():
        raise FileNotFoundError(
            "STALE_CHECKPOINT_MISMATCH: formal joint checkpoint is required for evaluation"
        )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, use_fast=True)
    eval_records = records_by_task_and_stage(
        records,
        stage="stage4_final_eval",
        expected_counts=expected,
        enabled_tasks=_enabled_tasks(config),
    )
    prompt_records = {task: eval_records[task] for task in _enabled_tasks(config)}
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
        enabled_tasks=_enabled_tasks(config),
        expected_checkpoint_manifest_sha256=sha256_file(checkpoint_manifest_path),
        expected_data_manifest_sha256=data_hash,
        expected_partition_sha256_per_task={
            task: partitions[task].sha256 for task in _enabled_tasks(config)
        },
    )
    engine = FormalInferenceEngine(
        model=model,
        banks=banks,
        probe_head=probe_head,
        device=device,
        probe_task=_enabled_tasks(config)[0],
    )
    max_new_tokens = config["evaluation"]["max_new_tokens"]
    rows: list[dict[str, Any]] = []
    task_scores: dict[str, list[dict[str, float]]] = {task: [] for task in _enabled_tasks(config)}
    probe_correct = 0
    total_latency = 0.0
    for task in _enabled_tasks(config):
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
    for mode in _enabled_tasks(config):
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
            task: mean_metrics(task_scores[task]) for task in _enabled_tasks(config)
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
        "partitions": {task: partitions[task].to_dict() for task in _enabled_tasks(config)},
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
            else _required_counts(data_config, stage, _enabled_tasks(config))
        )
        actual = {
            task: sum(
                1
                for record in records
                if record.get("task") == task
                and record.get("assigned_split") == stage
            )
            for task in _enabled_tasks(config)
        }
        stage_counts[stage] = actual
        stage_requirements[stage] = expected
    data_ready = stage_counts == stage_requirements
    paths = _pipeline_paths(config)
    tasks = _enabled_tasks(config)
    training = config["training"]
    data = config["data"]
    discovery = config["discovery"]
    partition_status = "PASS"
    partition_error = None
    if discovery.get("method") == "linear_cka_min":
        try:
            _load_partitions(config, checkpoint, expected_data_manifest_sha256=data_hash)
        except (RuntimeError, ValueError, FileNotFoundError) as exc:
            partition_status = "FAIL"
            partition_error = str(exc)
    source_mix_status = "PASS"
    source_mix_error = None
    if discovery.get("method") == "linear_cka_min":
        try:
            training_records = records_by_task_and_stage(
                records,
                stage="stage3_adapter_train",
                expected_counts=None,
                enabled_tasks=tasks,
            )
            validate_training_source_mixture(
                training_records,
                data_config=data_config,
                enabled_tasks=tasks,
            )
        except (RuntimeError, ValueError) as exc:
            source_mix_status = "FAIL"
            source_mix_error = str(exc)
    data_ready = data_ready and partition_status == "PASS" and source_mix_status == "PASS"
    mixture_trainer_reachable = False
    return {
        "status": "PASS" if data_ready else "BLOCKED",
        "model_type": native_config.model_type,
        "num_transformer_blocks": int(native_config.num_hidden_layers),
        "data_manifest_sha256": data_hash,
        "data_leakage": leakage["status"],
        "stage_records_available": stage_counts,
        "stage_records_required": stage_requirements,
        "formal_data_ready": data_ready,
        "partition_binding": partition_status,
        "partition_binding_error": partition_error,
        "source_mixture": source_mix_status,
        "source_mixture_error": source_mix_error,
        "resolved": {
            "model": {
                "checkpoint": str(checkpoint),
                "model_type": native_config.model_type,
                "num_hidden_layers": int(native_config.num_hidden_layers),
                "hidden_size": int(native_config.hidden_size),
                "dtype": config["model"]["dtype"],
            },
            "enabled_tasks": list(tasks),
            "task_order": list(training["task_order"]),
            "training_mode": "sequential_task_training",
            "mixture_training": bool(data["mixture_training"]),
            "interleave_across_tasks": bool(data["interleave_across_tasks"]),
            "formal_train_entrypoint": (
                f"{train_token_budget_sequential.__module__}."
                f"{train_token_budget_sequential.__name__}"
            ),
            "mixture_trainer_reachable": mixture_trainer_reachable,
            "discovery": {
                "method": discovery.get("method"),
                "metric": discovery.get("metric"),
                "interval_reduction": discovery.get("interval_reduction"),
                "similarity_thresholds": discovery.get("similarity_thresholds"),
                "partition_root": str(paths["discovery_output"]),
                "partition_paths": {
                    task: str(_discovery_partition_paths(config)[task]) for task in tasks
                } if discovery.get("method") == "linear_cka_min" else {},
            },
            "resume_checkpoint": str(paths["joint_checkpoint_output"]),
            "resume_policy": "explicit --resume only; no latest/glob discovery",
            "output_directories": {
                key: str(paths[key])
                for key in (
                    "discovery_output",
                    "joint_checkpoint_output",
                    "probe_output",
                    "evaluation_output",
                )
            },
        },
        "status_note": (
            "ready for explicit stage execution"
            if data_ready
            else "formal manifest counts do not satisfy the configured stage requirements; execution will fail fast"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Explicit formal Qwen3 task-adaptive pipeline")
    parser.add_argument("--config", required=True, help="Formal YAML configuration; must be explicit")
    parser.add_argument(
        "--stage",
        choices=("validate", "connectivity", "discovery", "train", "probe", "evaluate", "all"),
        default="validate",
    )
    parser.add_argument("--connectivity-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--max-steps", type=int, default=None, help="Only for a bounded local training smoke test")
    parser.add_argument("--resume", action="store_true", help="Resume the sequential formal training checkpoint")
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
        _prepare_run_manifest(config)
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
                    resume=args.resume,
                ),
                indent=2,
                sort_keys=True,
            ))
            return
        print(json.dumps(_run_train(config, checkpoint, max_steps=args.max_steps, resume=args.resume), indent=2, sort_keys=True))
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
    _prepare_run_manifest(config)
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
