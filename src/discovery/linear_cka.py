"""Experimental ordinary-residual Linear CKA Discovery.

This module is deliberately separate from the cosine residual metric.  It
captures the same native Qwen block residuals, aggregates valid-token rows,
and writes an independent CKA matrix and threshold sweep.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoTokenizer, Qwen3ForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

from src.common import load_yaml, sha256_file, sha256_json, tokenizer_sha256
from src.data.format_tasks import encode_prompt_only, load_dataset_pool, load_manifest
from src.data.source_provenance import (
    audit_formal_source_provenance,
    validate_manifest_row_identity,
)
from src.discovery.dynamic_programming import solve_similarity_partition
from src.discovery.ordinary_residual import collect_ordinary_residual_reference
from src.distributed.fsdp_utils import (
    barrier,
    broadcast_object,
    destroy_distributed,
    init_distributed,
    wrap_qwen3_fsdp,
)


METRIC = "linear_cka_residual_v1"


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_json(path: Path, payload: Any) -> None:
    if not dist.is_initialized() or dist.get_rank() == 0:
        _atomic_write_json(path, payload)
    if dist.is_initialized():
        dist.barrier()


def _write_numpy(path: Path, value: np.ndarray) -> None:
    if not dist.is_initialized() or dist.get_rank() == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        with temporary.open("wb") as handle:
            np.save(handle, value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    if dist.is_initialized():
        dist.barrier()


def _write_numpy_rank0(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _resolve_checkpoint_path(raw: str) -> Path:
    resolved = os.path.expandvars(str(raw))
    if "$" in resolved:
        raise RuntimeError("Linear CKA checkpoint path is unresolved")
    return Path(resolved)


def _weight_hash(checkpoint: Path) -> str:
    weights = sorted(checkpoint.glob("model*.safetensors"))
    if not weights:
        raise FileNotFoundError(f"No model*.safetensors files in {checkpoint}")
    return sha256_json({path.name: sha256_file(path) for path in weights})


def _checkpoint_manifest(checkpoint: Path) -> tuple[dict[str, Any], str]:
    config = AutoConfig.from_pretrained(checkpoint, local_files_only=True)
    if getattr(config, "model_type", None) != "qwen3":
        raise ValueError("Linear CKA checkpoint is not a native Qwen3 checkpoint")
    if any(hasattr(config, name) for name in ("attnres_execution", "moirai_partition")):
        raise ValueError("Linear CKA checkpoint contains converted AttnRes state")
    manifest = {
        "architecture": "original_qwen3",
        "model_type": "qwen3",
        "num_hidden_layers": int(config.num_hidden_layers),
        "hidden_size": int(config.hidden_size),
        "model_weights_sha256": _weight_hash(checkpoint),
    }
    return manifest, str(manifest["model_weights_sha256"])


def _task_examples(
    task: str,
    *,
    records: list[dict[str, Any]],
    data_config: dict[str, Any],
    tokenizer,
    expected_count: int,
) -> tuple[list[dict[str, Any]], tuple[Any, ...]]:
    source_counts = data_config["discovery_sources"][task]
    selected_records: list[dict[str, Any]] = []
    examples: list[Any] = []
    for source_name, source_expected in source_counts.items():
        source = data_config["sources"][source_name]
        selected = [
            record
            for record in records
            if record["task"] == task
            and record["dataset"] == source["dataset_name"]
            and record["assigned_split"] == "stage2_discovery"
        ]
        selected.sort(key=lambda record: record["split_key"])
        if len(selected) != int(source_expected):
            raise RuntimeError(
                f"{task}/{source_name} requires exactly {source_expected} "
                f"discovery cases, found {len(selected)}"
            )
        pool = load_dataset_pool(data_config, str(source["dataset_name"]))
        for record in selected:
            dataset, field_mapping = pool[str(record["official_split"])]
            row = dataset[int(record["row_index"])]
            validate_manifest_row_identity(record, row=row, field_mapping=field_mapping)
            selected_records.append(record)
            examples.append(
                encode_prompt_only(
                    tokenizer,
                    task=task,
                    row=row,
                    field_mapping=field_mapping,
                    stable_id=str(record["stable_id"]),
                    max_length=2048,
                )
            )
    if len(selected_records) != expected_count:
        raise RuntimeError(
            f"{task} requires exactly {expected_count} discovery cases, "
            f"found {len(selected_records)}"
        )
    return selected_records, tuple(examples)


def _reference_batch(example, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        example.input_ids.unsqueeze(0).to(device),
        example.attention_mask.unsqueeze(0).to(device),
    )


@torch.no_grad()
def collect_task_residuals(
    model,
    *,
    examples: Iterable[Any],
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    """Collect V_l rows on CPU while each case uses one native forward."""
    examples = tuple(examples)
    token_counts = [int(example.attention_mask.sum().item()) for example in examples]
    total_tokens = sum(token_counts)
    if total_tokens <= 0:
        raise ValueError("Linear CKA requires at least one valid token")

    residuals: torch.Tensor | None = None
    offset = 0
    for example, token_count in zip(examples, token_counts):
        input_ids, attention_mask = _reference_batch(example, device)
        reference = collect_ordinary_residual_reference(
            model,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        if residuals is None:
            layers, _batch, _tokens, hidden = reference.residual_contributions.shape
            residuals = torch.empty(
                (layers, total_tokens, hidden), dtype=torch.float32, device="cpu"
            )
        valid = reference.attention_mask.reshape(-1).to(dtype=torch.bool)
        values = reference.residual_contributions[:, 0, valid, :].detach().float().cpu()
        if values.shape[1] != token_count:
            raise RuntimeError("Linear CKA valid-token count disagrees with the example mask")
        residuals[:, offset : offset + token_count, :].copy_(values)
        offset += token_count
        del reference, values, input_ids, attention_mask
    assert residuals is not None
    if offset != total_tokens:
        raise RuntimeError("Linear CKA residual collection did not cover all tokens")
    return residuals, total_tokens


def _center_residuals(residuals: torch.Tensor) -> torch.Tensor:
    if residuals.ndim != 3:
        raise ValueError("residuals must have shape [layers, tokens, hidden]")
    centered = residuals.float()
    centered -= centered.mean(dim=1, keepdim=True)
    return centered


def linear_cka_similarity(left: torch.Tensor, right: torch.Tensor) -> float:
    """Exact centered linear CKA for two [tokens, hidden] representations."""
    if left.ndim != 2 or right.ndim != 2 or left.shape != right.shape:
        raise ValueError("Linear CKA inputs must have equal [tokens, hidden] shapes")
    left = left.float() - left.float().mean(dim=0, keepdim=True)
    right = right.float() - right.float().mean(dim=0, keepdim=True)
    left_gram = left.transpose(0, 1) @ left
    right_gram = right.transpose(0, 1) @ right
    cross = left.transpose(0, 1) @ right
    denominator = torch.linalg.matrix_norm(left_gram) * torch.linalg.matrix_norm(right_gram)
    if float(denominator) <= 1.0e-12:
        return 0.0
    return float((cross.square().sum() / denominator).item())


@torch.no_grad()
def compute_linear_cka_matrix(
    residuals: torch.Tensor,
    *,
    device: torch.device,
) -> np.ndarray:
    """Compute the layer-pair CKA matrix with FP32 matrix products."""
    centered = _center_residuals(residuals)
    layers = int(centered.shape[0])
    result = np.eye(layers, dtype=np.float64)
    self_norms: list[float] = []
    for index in range(layers):
        value = centered[index].to(device=device)
        gram = value.transpose(0, 1) @ value
        self_norms.append(float(torch.linalg.matrix_norm(gram).item()))
        del value, gram
    for left_index in range(layers):
        left = centered[left_index].to(device=device)
        for right_index in range(left_index + 1, layers):
            right = centered[right_index].to(device=device)
            cross = left.transpose(0, 1) @ right
            denominator = self_norms[left_index] * self_norms[right_index]
            similarity = 0.0 if denominator <= 1.0e-12 else float(
                (cross.square().sum() / denominator).item()
            )
            result[left_index, right_index] = similarity
            result[right_index, left_index] = similarity
            del right, cross
        del left
    return result


def interval_similarity_matrix(
    layer_cka: np.ndarray,
    *,
    reduction: str = "mean",
) -> np.ndarray:
    if reduction not in {"mean", "min"}:
        raise ValueError("cka_interval_reduction must be 'mean' or 'min'")
    if layer_cka.ndim != 2 or layer_cka.shape[0] != layer_cka.shape[1]:
        raise ValueError("layer_cka must be square")
    layers = int(layer_cka.shape[0])
    result = np.full_like(layer_cka, np.nan, dtype=np.float64)
    for start in range(layers):
        result[start, start] = 1.0
        for end in range(start + 1, layers):
            length = end - start + 1
            upper = layer_cka[start : end + 1, start : end + 1][
                np.triu_indices(length, k=1)
            ]
            result[start, end] = float(upper.mean() if reduction == "mean" else upper.min())
    return result


def _threshold_label(value: float) -> str:
    return format(float(value), "g")


def _partition_payload(
    *,
    task: str,
    partition,
    interval_similarity: np.ndarray,
    similarity_threshold: float,
    reduction: str,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    blocks = []
    cursor = 0
    for block_id, length in enumerate(partition.lengths):
        end = cursor + length - 1
        blocks.append(
            {
                "block_id": block_id,
                "start": cursor,
                "end": end,
                "length": length,
                "similarity": 1.0
                if length == 1
                else float(interval_similarity[cursor, end]),
            }
        )
        cursor = end + 1
    return {
        **(provenance or {}),
        "metric": METRIC,
        "interval_reduction": reduction,
        "task": task,
        "similarity_threshold": float(similarity_threshold),
        "block_sizes": list(partition.lengths),
        "num_moirai_blocks": len(partition.blocks),
        "num_transformer_blocks": partition.num_transformer_blocks,
        "blocks": blocks,
        "partition_sha256": partition.sha256,
        "constraints": partition.to_dict()["constraints"],
    }


def _save_threshold_sweep(
    *,
    task: str,
    task_output: Path,
    interval_similarity: np.ndarray,
    thresholds: Iterable[float],
    reduction: str,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for raw_threshold in thresholds:
        threshold = float(raw_threshold)
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("Linear CKA similarity thresholds must be in [0, 1]")
        selected = solve_similarity_partition(
            interval_similarity,
            similarity_threshold=threshold,
            task=task,
            candidate_lengths=range(1, interval_similarity.shape[0] + 1),
        )
        payload = _partition_payload(
            task=task,
            partition=selected.partition,
            interval_similarity=interval_similarity,
            similarity_threshold=threshold,
            reduction=reduction,
            provenance=provenance,
        )
        payload["total_similarity"] = float(
            sum(block["similarity"] for block in payload["blocks"])
        )
        threshold_dir = task_output / f"threshold_{_threshold_label(threshold)}"
        _atomic_write_json(threshold_dir / "partition.json", payload)
        results[_threshold_label(threshold)] = payload
    return results


def sweep_saved_similarity(
    output_dir: str | Path,
    *,
    thresholds: Iterable[float],
    tasks: Iterable[str] = ("math", "multihop"),
    source_dir: str | Path | None = None,
    reduction: str = "mean",
) -> dict[str, dict[str, Any]]:
    """Run additional threshold sweeps without loading a model or forwarding."""
    if reduction not in {"mean", "min"}:
        raise ValueError("cka_interval_reduction must be 'mean' or 'min'")
    root = Path(output_dir)
    source_root = Path(source_dir) if source_dir is not None else root
    threshold_list = [float(value) for value in thresholds]
    source_manifest_path = source_root / "linear_cka_manifest.json"
    source_manifest = (
        json.loads(source_manifest_path.read_text(encoding="utf-8"))
        if source_manifest_path.is_file()
        else {}
    )
    task_list = list(tasks)
    results: dict[str, dict[str, Any]] = {}
    for task in task_list:
        source_task = source_root / task
        task_output = root / task
        similarity_path = source_task / "similarity_matrix.npy"
        layer_cka = np.load(similarity_path)
        if (
            layer_cka.ndim != 2
            or layer_cka.shape[0] != layer_cka.shape[1]
            or not np.isfinite(layer_cka).all()
            or not np.allclose(layer_cka, layer_cka.T, rtol=0.0, atol=1.0e-6)
            or not np.allclose(np.diag(layer_cka), 1.0, rtol=0.0, atol=1.0e-6)
        ):
            raise ValueError(f"Invalid saved Linear CKA matrix: {similarity_path}")
        interval_similarity = interval_similarity_matrix(layer_cka, reduction=reduction)
        _write_numpy(task_output / "similarity_matrix.npy", layer_cka)
        _write_numpy(task_output / "interval_similarity_matrix.npy", interval_similarity)
        results[task] = _save_threshold_sweep(
            task=task,
            task_output=task_output,
            interval_similarity=interval_similarity,
            thresholds=threshold_list,
            reduction=reduction,
            provenance={
                "run_id": source_manifest.get("run_id"),
                "base_checkpoint_sha256": source_manifest.get("base_checkpoint_sha256"),
                "data_manifest_sha256": source_manifest.get("data_manifest_sha256"),
                "similarity_matrix_sha256": sha256_file(similarity_path),
            },
        )
    _write_json(
        root / "linear_cka_manifest.json",
        {
            **source_manifest,
            "interval_reduction": reduction,
            "source_output_dir": str(source_root),
            "source_similarity_matrix_sha256": {
                task: sha256_file(source_root / task / "similarity_matrix.npy")
                for task in task_list
            },
            "tasks": task_list,
            "similarity_thresholds": threshold_list,
            "forward_reused": True,
            "forward_rerun": False,
        },
    )
    return results


def run_linear_cka_discovery(config: dict[str, Any], args) -> None:
    run_id = str(config.get("run_id", "legacy-unbound")).strip()
    task_thresholds = config.get("task_similarity_thresholds")
    if task_thresholds is not None:
        if not isinstance(task_thresholds, dict):
            raise ValueError("task_similarity_thresholds must be a mapping")
        thresholds_by_task = {
            str(task): (float(value),)
            for task, value in task_thresholds.items()
        }
    else:
        thresholds = tuple(float(value) for value in config.get("similarity_thresholds", ()))
        if not thresholds:
            raise ValueError("Linear CKA config must declare similarity_thresholds")
        thresholds_by_task = {}
    reduction = str(config.get("cka_interval_reduction", "mean"))
    if reduction not in {"mean", "min"}:
        raise ValueError("cka_interval_reduction must be 'mean' or 'min'")
    if args.resume:
        raise ValueError("Linear CKA does not resume partial forwards; use saved-matrix sweep")

    context = init_distributed()
    checkpoint = _resolve_checkpoint_path(args.checkpoint or config["base_checkpoint"])
    checkpoint_manifest, checkpoint_hash = broadcast_object(
        _checkpoint_manifest(checkpoint) if context.is_rank0 else None,
        context,
    )
    data_manifest_path = Path(args.data_manifest or config["data_manifest"])
    data_manifest_sha256 = sha256_file(data_manifest_path)
    records = load_manifest(data_manifest_path)
    data_config = load_yaml(config["data_config"])
    audit_formal_source_provenance(
        records,
        data_config=data_config,
        enabled_tasks=tuple(config["tasks"]),
    )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, use_fast=True)
    checkpoint_manifest["tokenizer_sha256"] = tokenizer_sha256(tokenizer)
    model = Qwen3ForCausalLM.from_pretrained(
        checkpoint,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model = wrap_qwen3_fsdp(
        model,
        context,
        decoder_layer_classes=(Qwen3DecoderLayer,),
        sync_module_states=False,
        device_id=context.device,
    )

    output_root = Path(args.output_dir or config["output_dir"])
    task_results: dict[str, Any] = {}
    tasks = [args.task] if args.task else list(config["tasks"])
    if task_thresholds is not None and set(thresholds_by_task) != set(tasks):
        raise ValueError("task_similarity_thresholds must cover every enabled task")
    for task in tasks:
        records_for_task, examples = _task_examples(
            task,
            records=records,
            data_config=data_config,
            tokenizer=tokenizer,
            expected_count=int(config["discovery_cases_per_task"][task]),
        )
        task_output = output_root / task
        if context.is_rank0 and (task_output / "similarity_matrix.npy").exists():
            raise FileExistsError(f"Linear CKA output already exists: {task_output}")
        residuals, total_tokens = collect_task_residuals(
            model,
            examples=examples,
            device=context.device,
        )
        if context.is_rank0:
            layer_cka = compute_linear_cka_matrix(residuals, device=context.device)
            interval_similarity = interval_similarity_matrix(layer_cka, reduction=reduction)
            _write_numpy_rank0(task_output / "similarity_matrix.npy", layer_cka)
            _write_numpy_rank0(task_output / "interval_similarity_matrix.npy", interval_similarity)
            _atomic_write_json(
                task_output / "statistics.json",
                {
                    "run_id": run_id,
                    "metric": METRIC,
                    "interval_reduction": reduction,
                    "task": task,
                    "case_count": len(records_for_task),
                    "valid_token_count": total_tokens,
                    "num_transformer_blocks": int(layer_cka.shape[0]),
                    "hidden_size": int(residuals.shape[-1]),
                    "centered": True,
                    "fp32_accumulation": True,
                    "padding_mask": True,
                    "ordinary_residual_definition": "v_l = h_(l+1) - h_l",
                    "source_counts": dict(
                        sorted(Counter(str(record["dataset"]) for record in records_for_task).items())
                    ),
                    "stable_ids_sha256": hashlib.sha256(
                        "\n".join(sorted(str(record["stable_id"]) for record in records_for_task)).encode()
                    ).hexdigest(),
                },
            )
            task_results[task] = _save_threshold_sweep(
                task=task,
                task_output=task_output,
                interval_similarity=interval_similarity,
                thresholds=(thresholds_by_task[task] if task_thresholds is not None else thresholds),
                reduction=reduction,
                provenance={
                    "run_id": run_id,
                    "base_checkpoint_sha256": checkpoint_hash,
                    "data_manifest_sha256": data_manifest_sha256,
                    "similarity_matrix_sha256": sha256_file(task_output / "similarity_matrix.npy"),
                },
            )
        del residuals
        barrier(context)

    _write_json(
        output_root / "linear_cka_manifest.json",
        {
            "run_id": run_id,
            "metric": METRIC,
            "interval_reduction": reduction,
            "mode": "ordinary_residual_only",
            "base_checkpoint_sha256": checkpoint_hash,
            "data_manifest_sha256": data_manifest_sha256,
            "num_transformer_blocks": int(checkpoint_manifest["num_hidden_layers"]),
            "hidden_size": int(checkpoint_manifest["hidden_size"]),
            "tasks": tasks,
            "discovery_cases_per_task": config["discovery_cases_per_task"],
            "similarity_thresholds": (
                list(thresholds)
                if task_thresholds is None
                else sorted({value for values in thresholds_by_task.values() for value in values})
            ),
            "similarity_thresholds_by_task": {
                task: list(values) for task, values in thresholds_by_task.items()
            },
            "similarity_matrix_sha256": {
                task: sha256_file(output_root / task / "similarity_matrix.npy")
                for task in tasks
            },
            "interval_similarity_matrix_sha256": {
                task: sha256_file(output_root / task / "interval_similarity_matrix.npy")
                for task in tasks
            },
            "padding_mask": True,
            "centered_representations": True,
            "fp32_accumulation": True,
            "attnres_accessed": False,
            "query_accessed": False,
            "alpha_accessed": False,
            "backward_used": False,
            "replay_used": False,
        },
    )
    barrier(context)
    destroy_distributed(context)
