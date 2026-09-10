from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
from collections import Counter
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from src.common import load_yaml, sha256_file, tokenizer_sha256
from src.data.format_tasks import (
    SUPERVISED_TASKS,
    TASK_TO_SOURCE,
    encode_prompt_target,
    load_dataset_pool,
    load_manifest,
)
from src.discovery.collect_reference import collect_full_reference
from src.discovery.collect_reference import FullReference
from src.discovery.dynamic_programming import solve_partition, valid_interval_mask
from src.discovery.local_cost import local_surrogate_interval_cost
from src.discovery.refine import refine_partition
from src.discovery.replay import replay_partition
from src.discovery.ordinary_residual import require_defined_residual_cost
from src.distributed.fsdp_utils import (
    barrier,
    broadcast_object,
    destroy_distributed,
    init_distributed,
    wrap_qwen3_fsdp,
)
from src.modeling.full_attnres import MoiraiQwen3DecoderLayer, MoiraiQwen3ForCausalLM
from src.modeling.partition import MoiraiPartition
from src.training.checkpointing import (
    pseudo_query_sha256,
    validate_post_training_base_manifest,
)


EXPECTED_TASKS = list(SUPERVISED_TASKS)


def _weight_file(checkpoint: Path) -> Path:
    candidates = sorted(checkpoint.glob("model*.safetensors"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one safetensors weight file in {checkpoint}, got {candidates}"
        )
    return candidates[0]


def validate_discovery_config(config: dict[str, Any]) -> None:
    transformer_blocks = int(config.get("num_transformer_blocks", 0))
    tasks = config.get("tasks")
    if (
        not isinstance(tasks, list)
        or not tasks
        or len(set(tasks)) != len(tasks)
        or any(task not in EXPECTED_TASKS for task in tasks)
    ):
        raise ValueError(f"Stage 2 tasks must be a non-empty subset of {EXPECTED_TASKS}")
    case_counts = config.get("discovery_cases_per_task")
    if (
        not isinstance(case_counts, dict)
        or set(case_counts) != set(EXPECTED_TASKS)
        or any(int(value) <= 0 for value in case_counts.values())
    ):
        raise ValueError(
            "Stage 2 discovery_cases_per_task must give a positive count "
            f"for {', '.join(EXPECTED_TASKS)}"
        )
    expected = {
        "seed": 42,
        "num_transformer_blocks": 40,
        "observation_site_count": 2 * transformer_blocks + 1,
        "candidate_block_lengths": [1, 2, 3, 4],
        "num_moirai_blocks": list(range(10, 17)),
        "no_adjacent_singletons": True,
        "near_optimal_ratio": 0.02,
        "boundary_refinement_sweeps": 5,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(
                f"Stage 2 config mismatch for {key}: expected {value!r}, "
                f"got {config.get(key)!r}"
            )
    for key in {"base_checkpoint", "data_manifest", "data_config", "output_dir"}:
        if key not in config:
            raise ValueError(f"Stage 2 config is missing {key}")


def _checkpoint_manifest(checkpoint: Path) -> tuple[dict[str, Any], str]:
    manifest_path = checkpoint / "checkpoint_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing HF bootstrap manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_post_training_base_manifest(manifest)
    if manifest.get("architecture") != "Full AttnRes":
        raise ValueError("Discovery bootstrap is not the converted AttnRes model")
    weight_hash = sha256_file(_weight_file(checkpoint))
    if weight_hash != manifest.get("model_weights_sha256"):
        raise ValueError("HF bootstrap weight hash does not match its manifest")
    return manifest, weight_hash


def _task_examples(
    task: str,
    *,
    records: list[dict[str, Any]],
    data_config: dict[str, Any],
    tokenizer,
    expected_count: int = 100,
):
    source_counts = data_config["discovery_sources"][task]
    all_records = []
    all_examples = []
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
        all_records.extend(selected)
        for record in selected:
            dataset, field_mapping = pool[str(record["official_split"])]
            all_examples.append(
                encode_prompt_target(
                    tokenizer,
                    task=task,
                    row=dataset[int(record["row_index"])],
                    field_mapping=field_mapping,
                    stable_id=str(record["stable_id"]),
                    max_length=2048,
                )
            )
    if len(all_records) != expected_count:
        raise RuntimeError(
            f"{task} requires exactly {expected_count} discovery cases, "
            f"found {len(all_records)}"
        )
    return all_records, tuple(all_examples)


def _reference_batch(example, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    full_sequence = torch.cat((example.input_ids, example.labels[-1:]))
    input_ids = full_sequence.unsqueeze(0).to(device)
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


def _score_partition(
    model,
    partition: MoiraiPartition,
    examples,
    device: torch.device,
) -> tuple[float, list[float]]:
    scores: list[float] = []
    for example in examples:
        input_ids, attention_mask = _reference_batch(example, device)
        full_reference = collect_full_reference(
            model,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        replay_reference = FullReference(
            observations=full_reference.observations,
            residual_sources=(),
            attention_outputs=(),
            mlp_outputs=(),
            attention_mask=full_reference.attention_mask,
        )
        del full_reference
        replay = replay_partition(
            model,
            partition,
            input_ids=input_ids,
            attention_mask=attention_mask,
            reference=replay_reference,
        )
        scores.append(replay.mean_distortion)
        del replay_reference, replay, input_ids, attention_mask
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return sum(scores) / len(scores), scores


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


def _valid_intervals(num_transformer_blocks: int) -> tuple[tuple[int, int], ...]:
    return tuple(
        (start, end)
        for start in range(num_transformer_blocks)
        for end in range(start, min(num_transformer_blocks, start + 4))
    )


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
    costs = record.get("interval_costs")
    if not isinstance(costs, list) or len(costs) != len(intervals):
        raise ValueError(f"Cost record interval count mismatch at {expected_index}")
    for saved, expected in zip(costs, intervals):
        if (
            not isinstance(saved, list)
            or len(saved) != 3
            or (int(saved[0]), int(saved[1])) != expected
            or not np.isfinite(float(saved[2]))
        ):
            raise ValueError(
                f"Invalid cost record interval at case {expected_index}: {saved!r}"
            )


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
    q_full_hash: str,
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
        "q_full_sha256": q_full_hash,
        "records": [
            {
                "case_index": index,
                "stable_id": str(record["stable_id"]),
                "dataset": str(record["dataset"]),
            }
            for index, record in enumerate(case_records)
        ],
    }
    if resume and cases_path.is_file():
        saved_manifest = json.loads(cases_path.read_text(encoding="utf-8"))
        if saved_manifest != cases_manifest:
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
    # The old implementation below used Full AttnRes observations.  Keep the
    # legacy code available for historical artifact readers, but never allow it
    # to execute as the formal Discovery objective.
    raise RuntimeError(
        "RESIDUAL_DISCOVERY_COST_UNDEFINED: legacy AttnRes local cost is not "
        "a valid ordinary-residual Discovery cost"
    )

    # pragma: no cover - retained only as a historical reference path.
    input_ids, attention_mask = _reference_batch(example, device)
    reference = collect_full_reference(
        model,
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    interval_costs: list[list[int | float]] = []
    try:
        for start, end in intervals:
            cost, _ = local_surrogate_interval_cost(
                model,
                reference,
                start=start,
                end=end,
            )
            value = float(cost.cpu())
            del cost
            if not np.isfinite(value):
                raise FloatingPointError(
                    f"Non-finite local cost for {task} {start}:{end}"
                )
            interval_costs.append([start, end, value])
        return {
            "case_index": case_index,
            "stable_id": str(case_record["stable_id"]),
            "sequence_length": int(reference.attention_mask.shape[-1]),
            "interval_costs": interval_costs,
        }
    finally:
        del reference, input_ids, attention_mask
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _finish_task_discovery(
    *,
    task: str,
    model,
    examples,
    case_records,
    output_dir: Path,
    device: torch.device,
    checkpoint_hash: str,
    q_full_hash: str,
    cost_mean: np.ndarray,
    candidate_block_counts: list[int],
    maximum_refinement_sweeps: int,
    source_counts: dict[str, int],
) -> dict[str, Any]:
    stable_ids_hash = hashlib.sha256(
        "\n".join(
            sorted(str(record["stable_id"]) for record in case_records)
        ).encode()
    ).hexdigest()
    progress_path = output_dir / "replay_progress.json"
    progress_identity = {
        "task": task,
        "base_checkpoint_sha256": checkpoint_hash,
        "q_full_sha256": q_full_hash,
        "discovery_stable_ids_sha256": stable_ids_hash,
        "cost_mean_sha256": hashlib.sha256(cost_mean.tobytes()).hexdigest(),
        "source_counts": source_counts,
    }
    cached_candidates: dict[str, Any] = {}
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("identity") != progress_identity:
            raise ValueError("Replay progress identity mismatch")
        if not isinstance(progress.get("candidates"), dict):
            raise ValueError("Replay progress candidates are invalid")
        cached_candidates = progress["candidates"]

    candidate_payload: dict[str, Any] = {}
    replay_payload: dict[str, Any] = {}
    candidates: dict[int, MoiraiPartition] = {}
    replay_means: dict[int, float] = {}
    for num_blocks in candidate_block_counts:
        result = solve_partition(
            cost_mean,
            num_blocks=num_blocks,
            task=task,
        )
        candidates[num_blocks] = result.partition
        key = str(num_blocks)
        candidate_payload[key] = {
            "surrogate_cost": result.cost,
            "partition": result.partition.to_dict(),
        }
        if key in cached_candidates:
            cached = cached_candidates[key]
            cached_partition = MoiraiPartition.from_dict(cached["partition"])
            if cached_partition.sha256 != result.partition.sha256:
                raise ValueError(f"Cached replay partition mismatch for N={num_blocks}")
            case_scores = cached.get("case_distortions")
            if not isinstance(case_scores, list) or len(case_scores) != len(examples):
                raise ValueError(f"Cached replay case count mismatch for N={num_blocks}")
            replay_mean = float(cached["mean_distortion"])
        else:
            replay_mean, case_scores = _score_partition(
                model,
                result.partition,
                examples,
                device,
            )
            cached_candidates[key] = {
                **candidate_payload[key],
                "mean_distortion": replay_mean,
                "case_distortions": case_scores,
            }
            _write_json(
                progress_path,
                {"identity": progress_identity, "candidates": cached_candidates},
            )
        replay_means[num_blocks] = replay_mean
        replay_payload[key] = {
            "mean_distortion": replay_mean,
            "case_distortions": case_scores,
        }

    minimum = min(replay_means.values())
    tolerance = max(0.02 * minimum, 1.0e-8)
    acceptable = [
        num_blocks
        for num_blocks, score in replay_means.items()
        if score <= minimum + tolerance
    ]
    selected_n = min(acceptable)
    initial_partition = candidates[selected_n]

    def score(candidate: MoiraiPartition) -> float:
        mean, _ = _score_partition(model, candidate, examples, device)
        return mean

    refined = refine_partition(
        initial_partition,
        score,
        initial_score=replay_means[selected_n],
        maximum_sweeps=maximum_refinement_sweeps,
    )
    partition_payload = refined.partition.to_dict()
    partition_payload.update(
        {
            "discovery_checkpoint_sha256": checkpoint_hash,
            "q_full_sha256": q_full_hash,
        }
    )
    _write_json(output_dir / "partition.json", partition_payload)
    result = {
        "task": task,
        "base_checkpoint_sha256": checkpoint_hash,
        "q_full_sha256": q_full_hash,
        "discovery_case_count": len(examples),
        "discovery_source_counts": source_counts,
        "discovery_stable_ids_sha256": stable_ids_hash,
        "candidates": {
            key: {**candidate_payload[key], **replay_payload[key]}
            for key in candidate_payload
        },
        "minimum_replay_distortion": minimum,
        "near_optimal_tolerance": tolerance,
        "acceptable_num_blocks": acceptable,
        "selected_num_blocks": selected_n,
        "final_partition": refined.partition.to_dict(),
        "refinement_accepted": [
            record.to_dict() for record in refined.records if record.accepted
        ],
        "final_replay_distortion": refined.score,
    }
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("Stage 2 unexpectedly produced parameter gradients")
    _write_json(output_dir / "discovery_result.json", result)
    _write_json(
        output_dir / "stage2_manifest.json",
        {
            "task": task,
            "case_count": len(examples),
            "base_checkpoint_sha256": checkpoint_hash,
            "q_full_sha256": q_full_hash,
            "cost_records_sha256": sha256_file(output_dir / "cost_records.jsonl"),
            "cost_mean_sha256": sha256_file(output_dir / "cost_mean.npy"),
            "partition_sha256": refined.partition.sha256,
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
    q_full_hash: str,
    num_transformer_blocks: int,
    candidate_block_counts: list[int],
    maximum_refinement_sweeps: int,
    resume: bool = False,
) -> dict[str, Any]:
    if not dist.is_initialized() or dist.get_rank() == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()
    mask = valid_interval_mask(num_transformer_blocks)
    intervals = _valid_intervals(num_transformer_blocks)
    source_counts = dict(sorted(Counter(
        str(record["dataset"]) for record in case_records
    ).items()))
    costs_path, cost_records = _prepare_cost_resume(
        task=task,
        case_records=case_records,
        output_dir=output_dir,
        source_counts=source_counts,
        checkpoint_hash=checkpoint_hash,
        q_full_hash=q_full_hash,
        intervals=intervals,
        resume=resume,
    )
    costs_by_interval: dict[tuple[int, int], list[float]] = {
        interval: [] for interval in intervals
    }
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

    cost_mean = np.full(
        (num_transformer_blocks, num_transformer_blocks),
        np.inf,
        dtype=np.float64,
    )
    cost_std = np.full_like(cost_mean, np.inf)
    for interval, values in costs_by_interval.items():
        if len(values) != len(examples):
            raise RuntimeError(
                f"Interval {interval} has {len(values)} costs, "
                f"expected {len(examples)}"
            )
        cost_mean[interval] = np.mean(values, dtype=np.float64)
        cost_std[interval] = np.std(values, dtype=np.float64)
    if not np.isfinite(cost_mean[mask]).all():
        raise FloatingPointError("Stage 2 cost matrix contains non-finite valid entries")
    if not np.isfinite(cost_std[mask]).all():
        raise FloatingPointError("Stage 2 cost std matrix contains non-finite valid entries")

    _write_numpy(output_dir / "cost_mean.npy", cost_mean)
    _write_numpy(output_dir / "cost_std.npy", cost_std)
    _write_numpy(output_dir / "valid_interval_mask.npy", mask)
    _write_json(
        output_dir / "numerical_audit.json",
        {
            "task": task,
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
        model=model,
        examples=examples,
        case_records=case_records,
        output_dir=output_dir,
        device=device,
        checkpoint_hash=checkpoint_hash,
        q_full_hash=q_full_hash,
        cost_mean=cost_mean,
        candidate_block_counts=candidate_block_counts,
        maximum_refinement_sweeps=maximum_refinement_sweeps,
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
    return parser.parse_args()


def _validate_resume_outputs(
    *,
    task: str,
    partition_path: Path,
    result_path: Path,
    checkpoint_hash: str,
    q_full_hash: str,
    candidate_block_counts: list[int],
) -> None:
    partition = MoiraiPartition.from_json(partition_path)
    partition_payload = json.loads(partition_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if partition.task != task or result.get("task") != task:
        raise ValueError(f"Stage 2 resume task mismatch for {task}")
    if partition_payload.get("discovery_checkpoint_sha256") != checkpoint_hash:
        raise ValueError(f"Stage 2 resume checkpoint hash mismatch for {task}")
    if result.get("base_checkpoint_sha256") != checkpoint_hash:
        raise ValueError(f"Stage 2 resume result checkpoint hash mismatch for {task}")
    if partition_payload.get("q_full_sha256") != q_full_hash:
        raise ValueError(f"Stage 2 resume Q_full hash mismatch for {task}")
    if result.get("q_full_sha256") != q_full_hash:
        raise ValueError(f"Stage 2 resume result Q_full hash mismatch for {task}")
    if partition_payload.get("partition_sha256") != partition.sha256:
        raise ValueError(f"Stage 2 resume partition hash mismatch for {task}")

    candidates = result.get("candidates")
    if not isinstance(candidates, dict) or set(candidates) != {
        str(value) for value in candidate_block_counts
    }:
        raise ValueError(f"Stage 2 resume candidates differ from configuration for {task}")
    for value in candidate_block_counts:
        candidate = candidates[str(value)]
        if not isinstance(candidate, dict) or "partition" not in candidate:
            raise ValueError(f"Stage 2 resume candidate N={value} lacks partition")
        candidate_partition = MoiraiPartition.from_dict(candidate["partition"])
        if len(candidate_partition.blocks) != value or candidate_partition.task != task:
            raise ValueError(f"Stage 2 resume candidate N={value} is inconsistent")
        distortion = candidate.get("mean_distortion")
        if not isinstance(distortion, (int, float)) or not np.isfinite(distortion):
            raise ValueError(f"Stage 2 resume candidate N={value} lacks replay distortion")

    selected_n = result.get("selected_num_blocks")
    if not isinstance(selected_n, int) or selected_n not in candidate_block_counts:
        raise ValueError(f"Stage 2 resume final N selection is invalid for {task}")
    final_payload = result.get("final_partition")
    if not isinstance(final_payload, dict):
        raise ValueError(f"Stage 2 resume final partition is missing for {task}")
    final_partition = MoiraiPartition.from_dict(final_payload)
    if final_partition.sha256 != partition.sha256:
        raise ValueError(f"Stage 2 resume final partition disagrees for {task}")


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    if config.get("model_mode") == "original_residual_only" or config.get(
        "formal_discovery", False
    ):
        require_defined_residual_cost(config)
    validate_discovery_config(config)
    context = init_distributed()
    candidate_block_counts = [int(value) for value in config["num_moirai_blocks"]]
    checkpoint = Path(args.checkpoint or config["base_checkpoint"])
    checkpoint_identity = broadcast_object(
        _checkpoint_manifest(checkpoint) if context.is_rank0 else None,
        context,
    )
    checkpoint_manifest, checkpoint_hash = checkpoint_identity
    if int(checkpoint_manifest.get("num_hidden_layers", 0)) != int(
        config["num_transformer_blocks"]
    ):
        raise ValueError("Discovery depth does not match the HF checkpoint manifest")
    data_manifest_path = Path(args.data_manifest or config["data_manifest"])
    if not data_manifest_path.is_file():
        raise FileNotFoundError(f"Missing split manifest: {data_manifest_path}")
    records = load_manifest(data_manifest_path)
    data_config = load_yaml(config["data_config"])
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        local_files_only=True,
        use_fast=True,
    )
    if tokenizer_sha256(tokenizer) != checkpoint_manifest.get("tokenizer_sha256"):
        raise ValueError("Stage 2 checkpoint tokenizer hash mismatch")
    tasks = [args.task] if args.task else list(config["tasks"])
    task_payloads = {
        task: _task_examples(
            task,
            records=records,
            data_config=data_config,
            tokenizer=tokenizer,
            expected_count=int(config["discovery_cases_per_task"][task]),
        )
        for task in tasks
    }
    if args.resume:
        resume_root = Path(args.resume)
        if resume_root.resolve() != Path(config["output_dir"]).resolve():
            raise ValueError("Stage 2 resume path must equal the configured output_dir")
    model = MoiraiQwen3ForCausalLM.from_pretrained(
        checkpoint,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.config.attnres_execution = "full"
    model.config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    query_identity = broadcast_object(
        pseudo_query_sha256(model) if context.is_rank0 else None,
        context,
    )
    query_names, actual_q_full_hash = query_identity
    if query_names != checkpoint_manifest.get("q_full_parameter_names"):
        raise ValueError("Stage 2 Q_full parameter names mismatch")
    if actual_q_full_hash != checkpoint_manifest.get("q_full_sha256"):
        raise ValueError("Stage 2 Q_full hash mismatch")
    model = wrap_qwen3_fsdp(
        model,
        context,
        decoder_layer_classes=(MoiraiQwen3DecoderLayer,),
    )

    output_root = Path(args.output_dir or config["output_dir"])
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
                q_full_hash=checkpoint_manifest["q_full_sha256"],
                candidate_block_counts=candidate_block_counts,
            )
            continue
        case_records, examples = task_payloads[task]
        run_task_discovery(
            task=task,
            model=model,
            examples=examples,
            case_records=case_records,
            output_dir=task_output,
            device=context.device,
            checkpoint_hash=checkpoint_hash,
            q_full_hash=checkpoint_manifest["q_full_sha256"],
            num_transformer_blocks=int(config["num_transformer_blocks"]),
            candidate_block_counts=candidate_block_counts,
            maximum_refinement_sweeps=int(config["boundary_refinement_sweeps"]),
            resume=bool(args.resume),
        )
    barrier(context)
    destroy_distributed(context)


if __name__ == "__main__":
    main()
