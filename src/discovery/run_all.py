from __future__ import annotations

import argparse
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
from src.data.format_tasks import (
    SUPERVISED_TASKS,
    encode_prompt_only,
    load_dataset_pool,
    load_manifest,
)
from src.data.source_provenance import (
    audit_formal_source_provenance,
    validate_manifest_row_identity,
)
from src.discovery.dynamic_programming import (
    solve_partition,
    valid_interval_mask,
)
from src.discovery.ordinary_residual import (
    OrdinaryResidualReference,
    collect_ordinary_residual_reference,
    pairwise_directional_interval_costs,
    require_defined_residual_cost,
)
from src.distributed.fsdp_utils import (
    barrier,
    broadcast_object,
    destroy_distributed,
    init_distributed,
    wrap_qwen3_fsdp,
)
from src.modeling.partition import MoiraiPartition


EXPECTED_TASKS = list(SUPERVISED_TASKS)
COST_METHOD = "ordinary_residual_pairwise_directional_v1"


def validate_discovery_config(config: dict[str, Any]) -> None:
    discovery_metric = str(config.get("discovery_metric", "cosine_residual"))
    if discovery_metric not in {"cosine_residual", "linear_cka"}:
        raise ValueError(f"Unsupported discovery_metric: {discovery_metric}")
    tasks = config.get("tasks")
    if (
        not isinstance(tasks, list)
        or not tasks
        or len(set(tasks)) != len(tasks)
        or any(task not in EXPECTED_TASKS for task in tasks)
    ):
        raise ValueError(f"Discovery tasks must be a non-empty subset of {EXPECTED_TASKS}")
    if config.get("model_mode") != "original_residual_only":
        raise ValueError("Discovery must run in original_residual_only mode")
    if config.get("formal_discovery") is not True:
        raise ValueError("Formal Discovery must be explicitly enabled")
    if discovery_metric == "linear_cka":
        if config.get("ordinary_residual_only") is not True:
            raise RuntimeError("Linear CKA Discovery must use ordinary residuals only")
        if config.get("cka_interval_reduction") != "min":
            raise ValueError("Formal Linear CKA must use min interval reduction")
        task_thresholds = config.get("task_similarity_thresholds")
        if not isinstance(task_thresholds, dict) or set(task_thresholds) != set(tasks):
            raise ValueError("Linear CKA thresholds must cover every task")
        for task, value in task_thresholds.items():
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"Invalid Linear CKA threshold for {task}")
    else:
        if config.get("ordinary_residual_cost_defined") is not True:
            raise RuntimeError("RESIDUAL_DISCOVERY_COST_UNDEFINED")
        if "merge_cost_threshold" not in config:
            raise ValueError("Discovery config must declare merge_cost_threshold")
        threshold = config.get("merge_cost_threshold")
        if threshold is not None and (
            not isinstance(threshold, (int, float))
            or not math.isfinite(float(threshold))
            or float(threshold) < 0
        ):
            raise ValueError("merge_cost_threshold must be null or finite and non-negative")

    case_counts = config.get("discovery_cases_per_task")
    if (
        not isinstance(case_counts, dict)
        or set(case_counts) != set(tasks)
        or any(int(value) <= 0 for value in case_counts.values())
    ):
        raise ValueError(
            "discovery_cases_per_task must give a positive count for "
            f"{', '.join(tasks)}"
        )
    min_length = config.get("min_block_length")
    max_length = config.get("max_block_length")
    if not isinstance(min_length, int) or min_length != 1:
        raise ValueError("Discovery must allow singleton blocks with min_block_length=1")
    if max_length is not None and (
        not isinstance(max_length, int) or max_length < min_length
    ):
        raise ValueError("max_block_length must be null or an inclusive range maximum")
    if config.get("no_adjacent_singletons", False) is True:
        raise ValueError("Discovery must allow adjacent singleton blocks")
    for key in ("base_checkpoint", "data_manifest", "data_config", "output_dir"):
        if key not in config:
            raise ValueError(f"Discovery config is missing {key}")


def require_defined_merge_threshold(config: dict[str, Any]) -> float:
    threshold = config.get("merge_cost_threshold")
    if threshold is None:
        raise RuntimeError("MERGE_THRESHOLD_UNDEFINED")
    if not isinstance(threshold, (int, float)) or not math.isfinite(float(threshold)):
        raise ValueError("merge_cost_threshold must be finite")
    if float(threshold) < 0:
        raise ValueError("merge_cost_threshold must be non-negative")
    return float(threshold)


def _weight_hash(checkpoint: Path) -> str:
    weights = sorted(checkpoint.glob("model*.safetensors"))
    if not weights:
        raise FileNotFoundError(f"No model*.safetensors files in {checkpoint}")
    return sha256_json({path.name: sha256_file(path) for path in weights})


def _resolve_checkpoint_path(raw: str) -> Path:
    resolved = os.path.expandvars(str(raw))
    if "$" in resolved:
        raise RuntimeError(
            "Original Qwen3 checkpoint path is unresolved; set the configured Qwen3 path"
        )
    return Path(resolved)


def _checkpoint_manifest(checkpoint: Path) -> tuple[dict[str, Any], str]:
    config = AutoConfig.from_pretrained(checkpoint, local_files_only=True)
    if getattr(config, "model_type", None) != "qwen3":
        raise ValueError("Discovery checkpoint is not a native Qwen3 checkpoint")
    if any(hasattr(config, name) for name in ("attnres_execution", "moirai_partition")):
        raise ValueError("Discovery checkpoint contains converted AttnRes config state")
    manifest_path = checkpoint / "checkpoint_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else {}
    )
    if any("fixed" in str(key).lower() for key in manifest):
        raise ValueError("Discovery checkpoint manifest contains Fixed state")
    manifest = {
        **manifest,
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
    connectivity_cases: int | None = None,
):
    source_counts = data_config["discovery_sources"][task]
    target_count = expected_count if connectivity_cases is None else connectivity_cases
    if target_count <= 0 or target_count > expected_count:
        raise ValueError("connectivity_cases must be between 1 and the configured count")
    all_records = []
    all_examples = []
    remaining = target_count
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
        take = min(len(selected), remaining)
        selected = selected[:take]
        remaining -= take
        pool = load_dataset_pool(data_config, str(source["dataset_name"]))
        all_records.extend(selected)
        for record in selected:
            dataset, field_mapping = pool[str(record["official_split"])]
            row_index = int(record["row_index"])
            if row_index < 0 or row_index >= len(dataset):
                raise RuntimeError(
                    f"Discovery manifest row_index {row_index} is outside "
                    f"{source['dataset_name']}/{record['official_split']}"
                )
            row = dataset[row_index]
            validate_manifest_row_identity(
                record,
                row=row,
                field_mapping=field_mapping,
            )
            all_examples.append(
                encode_prompt_only(
                    tokenizer,
                    task=task,
                    row=row,
                    field_mapping=field_mapping,
                    stable_id=str(record["stable_id"]),
                    max_length=2048,
                )
            )
    if len(all_records) != target_count:
        raise RuntimeError(
            f"{task} requires exactly {target_count} discovery cases, "
            f"found {len(all_records)}"
        )
    return all_records, tuple(all_examples)


def _reference_batch(example, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        example.input_ids.unsqueeze(0).to(device),
        example.attention_mask.unsqueeze(0).to(device),
    )


def _valid_intervals(
    num_transformer_blocks: int,
    candidate_lengths: Iterable[int],
) -> tuple[tuple[int, int], ...]:
    lengths = tuple(sorted(set(int(value) for value in candidate_lengths)))
    return tuple(
        (start, start + length - 1)
        for start in range(num_transformer_blocks)
        for length in lengths
        if start + length <= num_transformer_blocks
    )


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_json(path: Path, payload: Any) -> None:
    if not dist.is_initialized() or dist.get_rank() == 0:
        _atomic_write_text(
            path,
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
    if dist.is_initialized():
        dist.barrier()


def _write_numpy(path: Path, value: np.ndarray) -> None:
    if not dist.is_initialized() or dist.get_rank() == 0:
        temporary = path.with_name(f".{path.name}.tmp")
        with temporary.open("wb") as handle:
            np.save(handle, value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    if dist.is_initialized():
        dist.barrier()


def _validate_cost_record(
    record: dict[str, Any],
    *,
    expected_index: int,
    expected_stable_id: str,
    intervals: tuple[tuple[int, int], ...],
) -> None:
    if int(record.get("case_index", -1)) != expected_index:
        raise ValueError(f"Cost record case index mismatch at {expected_index}")
    if record.get("stable_id") != expected_stable_id:
        raise ValueError(f"Cost record stable ID mismatch at {expected_index}")
    if record.get("cost_method") != COST_METHOD:
        raise ValueError("Cost record method mismatch")
    costs = record.get("interval_costs")
    if not isinstance(costs, list) or len(costs) != len(intervals):
        raise ValueError(f"Cost record interval count mismatch at {expected_index}")
    for saved, expected in zip(costs, intervals):
        if (
            not isinstance(saved, list)
            or len(saved) != 3
            or (int(saved[0]), int(saved[1])) != expected
            or not np.isfinite(float(saved[2]))
            or float(saved[2]) < -1.0e-7
        ):
            raise ValueError(f"Invalid cost record interval at case {expected_index}: {saved!r}")


def _read_cost_records(
    path: Path,
    *,
    case_records,
    intervals: tuple[tuple[int, int], ...],
) -> tuple[list[dict[str, Any]], int]:
    if not path.is_file():
        return [], 0
    raw = path.read_bytes()
    records: list[dict[str, Any]] = []
    valid_bytes = 0
    lines = raw.splitlines(keepends=True)
    for line_index, line in enumerate(lines):
        if not line.endswith(b"\n"):
            break
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            if line_index != len(lines) - 1:
                raise
            break
        if not isinstance(record, dict) or len(records) >= len(case_records):
            raise ValueError("Cost record file contains an invalid or extra record")
        _validate_cost_record(
            record,
            expected_index=len(records),
            expected_stable_id=str(case_records[len(records)]["stable_id"]),
            intervals=intervals,
        )
        records.append(record)
        valid_bytes += len(line)
    return records, valid_bytes


def _prepare_cost_resume(
    *,
    task: str,
    case_records,
    output_dir: Path,
    source_counts: dict[str, int],
    checkpoint_hash: str,
    data_manifest_sha256: str,
    intervals: tuple[tuple[int, int], ...],
    resume: bool,
) -> tuple[Path, list[dict[str, Any]]]:
    cases_path = output_dir / "cases_manifest.json"
    costs_path = output_dir / "cost_records.jsonl"
    cases_manifest = {
        "task": task,
        "case_count": len(case_records),
        "source_counts": source_counts,
        "base_checkpoint_sha256": checkpoint_hash,
        "data_manifest_sha256": data_manifest_sha256,
        "cost_method": COST_METHOD,
        "records": [
            {
                "case_index": index,
                "stable_id": str(record["stable_id"]),
                "dataset": str(record["dataset"]),
                "dataset_revision": str(record["dataset_revision"]),
                "official_split": str(record["official_split"]),
                "row_index": int(record["row_index"]),
                "content_sha256": str(record["content_sha256"]),
            }
            for index, record in enumerate(case_records)
        ],
    }
    if resume and cases_path.is_file():
        if json.loads(cases_path.read_text(encoding="utf-8")) != cases_manifest:
            raise ValueError("Discovery cost resume identity mismatch")
    elif resume and costs_path.is_file() and costs_path.stat().st_size:
        raise ValueError("Cost records exist without a matching cases manifest")
    else:
        _write_json(cases_path, cases_manifest)

    if not resume:
        if not dist.is_initialized() or dist.get_rank() == 0:
            _atomic_write_text(costs_path, "")
        if dist.is_initialized():
            dist.barrier()
        return costs_path, []

    if not dist.is_initialized() or dist.get_rank() == 0:
        _, valid_bytes = _read_cost_records(
            costs_path,
            case_records=case_records,
            intervals=intervals,
        )
        if costs_path.is_file() and valid_bytes != costs_path.stat().st_size:
            with costs_path.open("r+b") as handle:
                handle.truncate(valid_bytes)
                handle.flush()
                os.fsync(handle.fileno())
    if dist.is_initialized():
        dist.barrier()
    completed, _ = _read_cost_records(
        costs_path,
        case_records=case_records,
        intervals=intervals,
    )
    return costs_path, completed


def _append_cost_record(path: Path, record: dict[str, Any]) -> None:
    if not dist.is_initialized() or dist.get_rank() == 0:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    if dist.is_initialized():
        dist.barrier()


def _compute_case_cost_record(
    *,
    task: str,
    case_index: int,
    case_record: dict[str, Any],
    example,
    model,
    device: torch.device,
    intervals: tuple[tuple[int, int], ...],
) -> dict[str, Any]:
    input_ids, attention_mask = _reference_batch(example, device)
    reference: OrdinaryResidualReference = collect_ordinary_residual_reference(
        model,
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    interval_costs = pairwise_directional_interval_costs(reference, intervals)
    values = [value for value in interval_costs.values()]
    return {
        "case_index": case_index,
        "stable_id": str(case_record["stable_id"]),
        "sequence_length": int(attention_mask.shape[-1]),
        "valid_tokens": int(attention_mask.sum().item()),
        "cost_method": COST_METHOD,
        "interval_costs": [
            [start, end, interval_costs[(start, end)]]
            for start, end in intervals
        ],
        "cost_min": min(values),
        "cost_max": max(values),
    }


def _finish_task_discovery(
    *,
    task: str,
    case_records,
    output_dir: Path,
    checkpoint_hash: str,
    data_manifest_sha256: str,
    cost_mean: np.ndarray,
    merge_cost_threshold: float,
    candidate_block_lengths: list[int],
    source_counts: dict[str, int],
) -> dict[str, Any]:
    stable_ids_hash = hashlib.sha256(
        "\n".join(sorted(str(record["stable_id"]) for record in case_records)).encode()
    ).hexdigest()
    selected = solve_partition(
        cost_mean,
        merge_cost_threshold=merge_cost_threshold,
        task=task,
        candidate_lengths=candidate_block_lengths,
    )
    partition_payload = selected.partition.to_dict()
    partition_payload.update(
        {
            "discovery_checkpoint_sha256": checkpoint_hash,
            "data_manifest_sha256": data_manifest_sha256,
            "cost_method": COST_METHOD,
            "merge_cost_threshold": merge_cost_threshold,
        }
    )
    _write_json(output_dir / "partition.json", partition_payload)
    result = {
        "task": task,
        "base_checkpoint_sha256": checkpoint_hash,
        "data_manifest_sha256": data_manifest_sha256,
        "cost_method": COST_METHOD,
        "discovery_case_count": len(case_records),
        "discovery_source_counts": source_counts,
        "discovery_stable_ids_sha256": stable_ids_hash,
        "candidate_block_lengths": candidate_block_lengths,
        "merge_cost_threshold": merge_cost_threshold,
        "num_moirai_blocks": len(selected.partition.blocks),
        "final_partition": selected.partition.to_dict(),
        "final_dp_cost": float(selected.cost),
    }
    _write_json(output_dir / "discovery_result.json", result)
    _write_json(
        output_dir / "stage2_manifest.json",
        {
            "task": task,
            "case_count": len(case_records),
            "base_checkpoint_sha256": checkpoint_hash,
            "data_manifest_sha256": data_manifest_sha256,
            "cost_method": COST_METHOD,
            "cost_records_sha256": sha256_file(output_dir / "cost_records.jsonl"),
            "cost_mean_sha256": sha256_file(output_dir / "cost_mean.npy"),
            "partition_sha256": selected.partition.sha256,
        },
    )
    return result


def run_task_discovery(
    *,
    task: str,
    model,
    examples,
    case_records,
    output_dir: Path,
    device: torch.device,
    checkpoint_hash: str,
    data_manifest_sha256: str,
    num_transformer_blocks: int,
    merge_cost_threshold: float,
    candidate_block_lengths: list[int],
    resume: bool = False,
) -> dict[str, Any]:
    if not dist.is_initialized() or dist.get_rank() == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()
    mask = valid_interval_mask(num_transformer_blocks, candidate_block_lengths)
    intervals = _valid_intervals(num_transformer_blocks, candidate_block_lengths)
    source_counts = dict(sorted(Counter(str(record["dataset"]) for record in case_records).items()))
    costs_path, cost_records = _prepare_cost_resume(
        task=task,
        case_records=case_records,
        output_dir=output_dir,
        source_counts=source_counts,
        checkpoint_hash=checkpoint_hash,
        data_manifest_sha256=data_manifest_sha256,
        intervals=intervals,
        resume=resume,
    )
    costs_by_interval: dict[tuple[int, int], list[float]] = {interval: [] for interval in intervals}
    for record in cost_records:
        for start, end, value in record["interval_costs"]:
            costs_by_interval[(int(start), int(end))].append(float(value))

    for case_index in range(len(cost_records), len(case_records)):
        record = _compute_case_cost_record(
            task=task,
            case_index=case_index,
            case_record=case_records[case_index],
            example=examples[case_index],
            model=model,
            device=device,
            intervals=intervals,
        )
        if dist.is_initialized():
            payload = [record if dist.get_rank() == 0 else None]
            dist.broadcast_object_list(payload, src=0)
            record = payload[0]
        _validate_cost_record(
            record,
            expected_index=case_index,
            expected_stable_id=str(case_records[case_index]["stable_id"]),
            intervals=intervals,
        )
        _append_cost_record(costs_path, record)
        cost_records.append(record)
        for start, end, value in record["interval_costs"]:
            costs_by_interval[(int(start), int(end))].append(float(value))

    cost_mean = np.full((num_transformer_blocks, num_transformer_blocks), np.inf, dtype=np.float64)
    cost_std = np.full_like(cost_mean, np.inf)
    for interval, values in costs_by_interval.items():
        if len(values) != len(examples):
            raise RuntimeError(f"Interval {interval} has {len(values)} costs, expected {len(examples)}")
        cost_mean[interval] = np.mean(values, dtype=np.float64)
        cost_std[interval] = np.std(values, dtype=np.float64)
    if not np.isfinite(cost_mean[mask]).all() or not np.isfinite(cost_std[mask]).all():
        raise FloatingPointError("Residual coalescence cost matrix contains non-finite valid entries")
    if (cost_mean[mask] < -1.0e-7).any():
        raise FloatingPointError("Residual coalescence cost matrix contains negative entries")

    _write_numpy(output_dir / "cost_mean.npy", cost_mean)
    _write_numpy(output_dir / "cost_std.npy", cost_std)
    _write_numpy(output_dir / "valid_interval_mask.npy", mask)
    _write_json(
        output_dir / "numerical_audit.json",
        {
            "task": task,
            "cost_method": COST_METHOD,
            "case_count": len(cost_records),
            "valid_interval_count": int(mask.sum()),
            "all_costs_finite": bool(np.isfinite(cost_mean[mask]).all()),
            "cost_mean_min": float(cost_mean[mask].min()),
            "cost_mean_max": float(cost_mean[mask].max()),
            "cost_std_min": float(cost_std[mask].min()),
            "cost_std_max": float(cost_std[mask].max()),
        },
    )
    return _finish_task_discovery(
        task=task,
        case_records=case_records,
        output_dir=output_dir,
        checkpoint_hash=checkpoint_hash,
        data_manifest_sha256=data_manifest_sha256,
        cost_mean=cost_mean,
        merge_cost_threshold=merge_cost_threshold,
        candidate_block_lengths=candidate_block_lengths,
        source_counts=source_counts,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage2_discovery.yaml")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--resume", default="")
    parser.add_argument("--data-manifest", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--task", choices=EXPECTED_TASKS, default="")
    parser.add_argument(
        "--connectivity-cases",
        type=int,
        default=0,
        help="Run at most this many local cases per task for connectivity only",
    )
    return parser.parse_args()


def _validate_resume_outputs(
    *,
    task: str,
    partition_path: Path,
    result_path: Path,
    checkpoint_hash: str,
    data_manifest_sha256: str,
    merge_cost_threshold: float,
) -> None:
    partition = MoiraiPartition.from_json(partition_path)
    partition_payload = json.loads(partition_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if partition.task != task or result.get("task") != task:
        raise ValueError(f"STALE_PARTITION_MISMATCH: Discovery resume task mismatch for {task}")
    if result.get("cost_method") != COST_METHOD or partition_payload.get("cost_method") != COST_METHOD:
        raise ValueError("STALE_PARTITION_MISMATCH: Discovery resume cost method mismatch")
    if partition_payload.get("discovery_checkpoint_sha256") != checkpoint_hash:
        raise ValueError(f"MODEL_IDENTITY_MISMATCH: Discovery resume checkpoint hash mismatch for {task}")
    if partition_payload.get("data_manifest_sha256") != data_manifest_sha256:
        raise ValueError(f"STALE_PARTITION_MISMATCH: Discovery resume data manifest hash mismatch for {task}")
    if result.get("data_manifest_sha256") != data_manifest_sha256:
        raise ValueError(f"STALE_PARTITION_MISMATCH: Discovery resume data manifest hash mismatch for {task}")
    if float(result.get("merge_cost_threshold", float("nan"))) != merge_cost_threshold:
        raise ValueError(f"STALE_PARTITION_MISMATCH: Discovery resume merge threshold mismatch for {task}")
    if float(partition_payload.get("merge_cost_threshold", float("nan"))) != merge_cost_threshold:
        raise ValueError(f"STALE_PARTITION_MISMATCH: Discovery partition merge threshold mismatch for {task}")
    final_payload = result.get("final_partition")
    if not isinstance(final_payload, dict):
        raise ValueError("STALE_PARTITION_MISMATCH: Discovery resume final partition is missing")
    if MoiraiPartition.from_dict(final_payload).sha256 != partition.sha256:
        raise ValueError("STALE_PARTITION_MISMATCH: Discovery resume final partition disagrees")


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    validate_discovery_config(config)
    if str(config.get("discovery_metric", "cosine_residual")) == "linear_cka":
        from src.discovery.linear_cka import run_linear_cka_discovery

        run_linear_cka_discovery(config, args)
        return
    require_defined_residual_cost(config)
    merge_cost_threshold = require_defined_merge_threshold(config)
    context = init_distributed()
    checkpoint = _resolve_checkpoint_path(args.checkpoint or config["base_checkpoint"])
    checkpoint_manifest, checkpoint_hash = broadcast_object(
        _checkpoint_manifest(checkpoint) if context.is_rank0 else None,
        context,
    )
    model_num_layers = int(checkpoint_manifest["num_hidden_layers"])
    max_length = config.get("max_block_length")
    candidate_block_lengths = list(
        range(
            int(config["min_block_length"]),
            (int(max_length) if max_length is not None else model_num_layers) + 1,
        )
    )
    if args.connectivity_cases < 0:
        raise ValueError("--connectivity-cases must be non-negative")
    connectivity_cases = args.connectivity_cases or None
    data_manifest_path = Path(args.data_manifest or config["data_manifest"])
    if not data_manifest_path.is_file():
        raise FileNotFoundError(f"Missing split manifest: {data_manifest_path}")
    data_manifest_sha256 = sha256_file(data_manifest_path)
    records = load_manifest(data_manifest_path)
    data_config = load_yaml(config["data_config"])
    audit_formal_source_provenance(
        records,
        data_config=data_config,
        enabled_tasks=tuple(config["tasks"]),
    )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, use_fast=True)
    if tokenizer_sha256(tokenizer) != checkpoint_manifest.get("tokenizer_sha256"):
        if checkpoint_manifest.get("tokenizer_sha256") is not None:
            raise ValueError("Discovery checkpoint tokenizer hash mismatch")
        checkpoint_manifest["tokenizer_sha256"] = tokenizer_sha256(tokenizer)

    tasks = [args.task] if args.task else list(config["tasks"])
    task_payloads = {
        task: _task_examples(
            task,
            records=records,
            data_config=data_config,
            tokenizer=tokenizer,
            expected_count=int(config["discovery_cases_per_task"][task]),
            connectivity_cases=connectivity_cases,
        )
        for task in tasks
    }
    if args.resume and Path(args.resume).resolve() != Path(config["output_dir"]).resolve():
        raise ValueError("Discovery resume path must equal configured output_dir")

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
    task_results: dict[str, dict[str, Any]] = {}
    for task in tasks:
        task_output = output_root / task
        existing_partition = task_output / "partition.json"
        existing_result = task_output / "discovery_result.json"
        if args.resume and existing_partition.is_file() and existing_result.is_file():
            _validate_resume_outputs(
                task=task,
                partition_path=existing_partition,
                result_path=existing_result,
                checkpoint_hash=checkpoint_hash,
                data_manifest_sha256=data_manifest_sha256,
                merge_cost_threshold=merge_cost_threshold,
            )
            task_results[task] = json.loads(existing_result.read_text(encoding="utf-8"))
            continue
        case_records, examples = task_payloads[task]
        task_results[task] = run_task_discovery(
            task=task,
            model=model,
            examples=examples,
            case_records=case_records,
            output_dir=task_output,
            device=context.device,
            checkpoint_hash=checkpoint_hash,
            data_manifest_sha256=data_manifest_sha256,
            num_transformer_blocks=int(checkpoint_manifest["num_hidden_layers"]),
            merge_cost_threshold=merge_cost_threshold,
            candidate_block_lengths=candidate_block_lengths,
            resume=bool(args.resume),
        )
    peak_memory_bytes = 0
    if torch.cuda.is_available():
        peak_memory = torch.tensor(
            torch.cuda.max_memory_allocated(context.device),
            dtype=torch.int64,
            device=context.device,
        )
        if dist.is_initialized():
            dist.all_reduce(peak_memory, op=dist.ReduceOp.MAX)
        peak_memory_bytes = int(peak_memory.item())
    _write_json(
        output_root / (
            "connectivity_manifest.json" if connectivity_cases else "discovery_manifest.json"
        ),
        {
            "mode": "connectivity" if connectivity_cases else "formal_discovery",
            "base_checkpoint_sha256": checkpoint_hash,
            "data_manifest_sha256": data_manifest_sha256,
            "model_type": checkpoint_manifest["model_type"],
            "num_transformer_blocks": model_num_layers,
            "merge_cost_threshold": merge_cost_threshold,
            "min_block_length": int(config["min_block_length"]),
            "max_block_length": max_length,
            "connectivity_cases_per_task": connectivity_cases,
            "world_size": context.world_size,
            "dtype": "bfloat16",
            "use_cache": False,
            "ordinary_residual_only": True,
            "attnres_accessed": False,
            "query_accessed": False,
            "alpha_accessed": False,
            "backward_used": False,
            "replay_used": False,
            "peak_cuda_memory_allocated_bytes": peak_memory_bytes,
            "tasks": {
                task: {
                    "case_count": int(result["discovery_case_count"]),
                    "partition_sha256": result["final_partition"]["partition_sha256"],
                    "num_moirai_blocks": int(result["num_moirai_blocks"]),
                }
                for task, result in sorted(task_results.items())
            },
        },
    )
    barrier(context)
    destroy_distributed(context)


if __name__ == "__main__":
    main()
