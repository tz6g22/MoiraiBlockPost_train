from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
from collections import Counter
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

from src.common import load_yaml, sha256_file, tokenizer_sha256
from src.data.format_tasks import (
    TASK_TO_SOURCE,
    load_local_split,
    load_manifest,
)
from src.data.streams import ManifestTaskRows
from src.discovery.collect_reference import collect_full_reference
from src.discovery.collect_reference import FullReference
from src.discovery.dynamic_programming import solve_partition, valid_interval_mask
from src.discovery.local_cost import local_surrogate_interval_cost
from src.discovery.refine import refine_partition
from src.discovery.replay import replay_partition
from src.modeling.full_attnres import MoiraiQwen3ForCausalLM
from src.modeling.partition import MoiraiPartition
from src.training.checkpointing import (
    pseudo_query_sha256,
    validate_post_training_base_manifest,
)


EXPECTED_TASKS = ["math", "multihop"]


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
            "for math and multihop"
        )
    expected = {
        "seed": 42,
        "num_transformer_blocks": 28,
        "observation_site_count": 2 * transformer_blocks + 1,
        "candidate_block_lengths": [1, 2, 3, 4],
        "num_moirai_blocks": list(range(9, 17)),
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
        dataset = load_local_split(source["local_path"], str(source["official_split"]))
        rows = ManifestTaskRows(
            task=task,
            records=tuple(selected),
            dataset=dataset,
            field_mapping=source["field_mapping"],
        )
        all_records.extend(selected)
        all_examples.extend(rows.target_examples(tokenizer, max_length=2048))
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


def _reference_storage_bytes(reference: FullReference) -> int:
    tensors = (*reference.observations, reference.attention_mask)
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


def _cache_replay_references(
    references: list[FullReference],
    device: torch.device,
) -> tuple[list[FullReference], bool]:
    """Keep replay-only reference tensors on CUDA when they safely fit."""
    if device.type != "cuda" or not references:
        return references, False

    required_bytes = sum(_reference_storage_bytes(value) for value in references)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    reserve_bytes = max(2 * 1024**3, total_bytes // 4)
    if required_bytes > max(0, free_bytes - reserve_bytes):
        return references, False

    cached: list[FullReference] = []
    try:
        for reference in references:
            cached.append(
                FullReference(
                    observations=tuple(
                        value.to(device=device) for value in reference.observations
                    ),
                    residual_sources=(),
                    attention_outputs=(),
                    mlp_outputs=(),
                    attention_mask=reference.attention_mask.to(device=device),
                )
            )
    except torch.OutOfMemoryError:
        del cached
        gc.collect()
        torch.cuda.empty_cache()
        return references, False
    return cached, True


def _score_partition(
    model,
    partition: MoiraiPartition,
    examples,
    references,
    device: torch.device,
) -> tuple[float, list[float]]:
    scores: list[float] = []
    for example, cpu_reference in zip(examples, references):
        input_ids, attention_mask = _reference_batch(example, device)
        reference = FullReference(
            observations=tuple(value.to(device) for value in cpu_reference.observations),
            residual_sources=(),
            attention_outputs=(),
            mlp_outputs=(),
            attention_mask=cpu_reference.attention_mask.to(device),
        )
        replay = replay_partition(
            model,
            partition,
            input_ids=input_ids,
            attention_mask=attention_mask,
            reference=reference,
        )
        scores.append(replay.mean_distortion)
        del reference, replay, input_ids, attention_mask
    return sum(scores) / len(scores), scores


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_saved_activation_state(
    *,
    task: str,
    case_records: list[dict[str, Any]],
    examples,
    output_dir: Path,
    num_transformer_blocks: int,
    source_counts: dict[str, int],
    model,
    device: torch.device,
    recompute_costs: bool = False,
) -> tuple[np.ndarray, list[FullReference]]:
    manifest_path = output_dir / "activation_manifest.json"
    cost_path = output_dir / "cost_mean.npy"
    if not manifest_path.is_file() or (
        not recompute_costs and not cost_path.is_file()
    ):
        raise FileNotFoundError(
            "Activation resume requires activation_manifest.json and, unless "
            "costs are being recomputed, cost_mean.npy"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("task") != task:
        raise ValueError("Activation manifest task mismatch")
    if int(manifest.get("case_count", -1)) != len(case_records):
        raise ValueError("Activation manifest case count mismatch")
    if manifest.get("source_counts") != source_counts:
        raise ValueError("Activation manifest source counts mismatch")
    saved_records = manifest.get("records")
    if not isinstance(saved_records, list) or len(saved_records) != len(case_records):
        raise ValueError("Activation manifest records are incomplete")

    references: list[FullReference] = []
    costs_by_interval: dict[tuple[int, int], list[float]] = {}
    expected_observations = 2 * num_transformer_blocks + 1
    for case_index, (saved, current, example) in enumerate(
        zip(saved_records, case_records, examples)
    ):
        if int(saved.get("case_index", -1)) != case_index:
            raise ValueError(f"Activation case index mismatch at {case_index}")
        if saved.get("stable_id") != str(current["stable_id"]):
            raise ValueError(f"Activation stable ID mismatch at {case_index}")
        activation_path = Path(saved["file"])
        if not activation_path.is_file():
            raise FileNotFoundError(f"Missing saved activation: {activation_path}")
        if sha256_file(activation_path) != saved.get("sha256"):
            raise ValueError(f"Saved activation hash mismatch: {activation_path}")
        tensors = load_file(activation_path, device="cpu")
        observation_keys = sorted(
            key for key in tensors if key.startswith("observation_")
        )
        if len(observation_keys) != expected_observations:
            raise ValueError(
                f"Saved activation {case_index} has {len(observation_keys)} "
                f"observations; expected {expected_observations}"
            )
        cpu_observations = tuple(tensors[key] for key in observation_keys)
        cpu_reference = FullReference(
            observations=cpu_observations,
            residual_sources=(),
            attention_outputs=(),
            mlp_outputs=(),
            attention_mask=tensors["attention_mask"],
        )
        references.append(cpu_reference)

        if recompute_costs:
            attention_keys = sorted(
                key
                for key in tensors
                if key.startswith("attention_") and key != "attention_mask"
            )
            mlp_keys = sorted(key for key in tensors if key.startswith("mlp_"))
            if len(attention_keys) != num_transformer_blocks:
                raise ValueError(
                    f"Saved activation {case_index} has {len(attention_keys)} "
                    f"Attention outputs; expected {num_transformer_blocks}"
                )
            if len(mlp_keys) != num_transformer_blocks:
                raise ValueError(
                    f"Saved activation {case_index} has {len(mlp_keys)} MLP "
                    f"outputs; expected {num_transformer_blocks}"
                )
            input_ids, attention_mask = _reference_batch(example, device)
            if not torch.equal(tensors["attention_mask"], attention_mask.cpu()):
                raise ValueError(
                    f"Saved activation attention mask mismatch at {case_index}"
                )
            gpu_observations = tuple(value.to(device) for value in cpu_observations)
            gpu_attention = tuple(tensors[key].to(device) for key in attention_keys)
            gpu_mlp = tuple(tensors[key].to(device) for key in mlp_keys)
            embedding = model.model.embed_tokens(input_ids)
            residual_sources = [embedding]
            for attention_output, mlp_output in zip(gpu_attention, gpu_mlp):
                residual_sources.extend((attention_output, mlp_output))
            full_reference = FullReference(
                observations=gpu_observations,
                residual_sources=tuple(residual_sources),
                attention_outputs=gpu_attention,
                mlp_outputs=gpu_mlp,
                attention_mask=attention_mask,
            )
            for start in range(num_transformer_blocks):
                for end in range(
                    start, min(num_transformer_blocks, start + 4)
                ):
                    cost, _ = local_surrogate_interval_cost(
                        model,
                        full_reference,
                        start=start,
                        end=end,
                    )
                    value = float(cost.cpu())
                    if not np.isfinite(value):
                        raise FloatingPointError(
                            f"Non-finite restored local cost for {task} {start}:{end}"
                        )
                    costs_by_interval.setdefault((start, end), []).append(value)
            del full_reference, residual_sources, embedding
            del gpu_observations, gpu_attention, gpu_mlp
            del input_ids, attention_mask
            gc.collect()
            torch.cuda.empty_cache()
        del tensors

    expected_shape = (num_transformer_blocks, num_transformer_blocks)
    if recompute_costs:
        cost_mean = np.full(expected_shape, np.inf, dtype=np.float64)
        for interval, values in costs_by_interval.items():
            if len(values) != len(examples):
                raise RuntimeError(
                    f"Restored interval {interval} has {len(values)} costs, "
                    f"expected {len(examples)}"
                )
            cost_mean[interval] = np.mean(values, dtype=np.float64)
        np.save(cost_path, cost_mean)
    else:
        cost_mean = np.load(cost_path)
    if cost_mean.shape != expected_shape:
        raise ValueError(
            f"Saved cost matrix has shape {cost_mean.shape}, expected {expected_shape}"
        )
    mask = valid_interval_mask(num_transformer_blocks)
    if not np.isfinite(cost_mean[mask]).all():
        raise FloatingPointError("Saved cost matrix has non-finite valid entries")
    return cost_mean, references


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
    replay_references: list[FullReference],
    candidate_block_counts: list[int],
    maximum_refinement_sweeps: int,
    source_counts: dict[str, int],
) -> dict[str, Any]:
    replay_references, references_cached = _cache_replay_references(
        replay_references,
        device,
    )
    reference_bytes = sum(
        _reference_storage_bytes(reference) for reference in replay_references
    )
    print(
        "Replay reference cache: "
        f"task={task} device_cached={references_cached} "
        f"bytes={reference_bytes}"
    )
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
                replay_references,
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
        mean, _ = _score_partition(
            model, candidate, examples, replay_references, device
        )
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
    resume_activations: bool = False,
    recompute_activation_costs: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    costs_by_interval: dict[tuple[int, int], list[float]] = {}
    replay_references: list[FullReference] = []
    activation_records: list[dict[str, Any]] = []
    activation_dir = output_dir / "activations"
    activation_dir.mkdir(parents=True, exist_ok=True)
    mask = valid_interval_mask(num_transformer_blocks)
    source_counts = dict(sorted(Counter(
        str(record["dataset"]) for record in case_records
    ).items()))

    if resume_activations:
        cost_mean, replay_references = _load_saved_activation_state(
            task=task,
            case_records=case_records,
            examples=examples,
            output_dir=output_dir,
            num_transformer_blocks=num_transformer_blocks,
            source_counts=source_counts,
            model=model,
            device=device,
            recompute_costs=recompute_activation_costs,
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
            replay_references=replay_references,
            candidate_block_counts=candidate_block_counts,
            maximum_refinement_sweeps=maximum_refinement_sweeps,
            source_counts=source_counts,
        )

    for case_index, (case_record, example) in enumerate(zip(case_records, examples)):
        input_ids, attention_mask = _reference_batch(example, device)
        reference = collect_full_reference(
            model,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        replay_references.append(
            FullReference(
                observations=tuple(
                    value.detach().to(device="cpu", dtype=torch.bfloat16)
                    for value in reference.observations
                ),
                residual_sources=(),
                attention_outputs=(),
                mlp_outputs=(),
                attention_mask=reference.attention_mask.detach().cpu(),
            )
        )
        for start in range(num_transformer_blocks):
            for end in range(start, min(num_transformer_blocks, start + 4)):
                cost, _ = local_surrogate_interval_cost(
                    model,
                    reference,
                    start=start,
                    end=end,
                )
                value = float(cost.cpu())
                if not np.isfinite(value):
                    raise FloatingPointError(
                        f"Non-finite local cost for {task} {start}:{end}"
                    )
                costs_by_interval.setdefault((start, end), []).append(value)
        activation_path = activation_dir / f"case_{case_index:02d}.safetensors"
        activation_payload = {
            **{
                f"observation_{index:02d}": value.detach()
                .to(device="cpu", dtype=torch.bfloat16)
                .contiguous()
                for index, value in enumerate(reference.observations)
            },
            **{
                f"attention_{index:02d}": value.detach()
                .to(device="cpu", dtype=torch.bfloat16)
                .contiguous()
                for index, value in enumerate(reference.attention_outputs)
            },
            **{
                f"mlp_{index:02d}": value.detach()
                .to(device="cpu", dtype=torch.bfloat16)
                .contiguous()
                for index, value in enumerate(reference.mlp_outputs)
            },
            "attention_mask": reference.attention_mask.detach().cpu().contiguous(),
        }
        save_file(activation_payload, activation_path)
        activation_records.append(
            {
                "case_index": case_index,
                "stable_id": str(case_record["stable_id"]),
                "file": str(activation_path),
                "sha256": sha256_file(activation_path),
                "sequence_length": int(reference.attention_mask.shape[-1]),
                "observation_site_count": len(reference.observations),
                "attention_activation_count": len(reference.attention_outputs),
                "mlp_activation_count": len(reference.mlp_outputs),
            }
        )
        del activation_payload
        del reference, input_ids, attention_mask
        gc.collect()
        torch.cuda.empty_cache()

    cost_mean = np.full(
        (num_transformer_blocks, num_transformer_blocks),
        np.inf,
        dtype=np.float64,
    )
    for interval, values in costs_by_interval.items():
        if len(values) != len(examples):
            raise RuntimeError(
                f"Interval {interval} has {len(values)} costs, "
                f"expected {len(examples)}"
            )
        cost_mean[interval] = np.mean(values, dtype=np.float64)
    if not np.isfinite(cost_mean[mask]).all():
        raise FloatingPointError("Stage 2 cost matrix contains non-finite valid entries")

    np.save(output_dir / "cost_mean.npy", cost_mean)
    _write_json(
        output_dir / "activation_manifest.json",
        {
            "task": task,
            "case_count": len(activation_records),
            "source_counts": source_counts,
            "dtype": "bfloat16",
            "records": activation_records,
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
        replay_references=replay_references,
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
    parser.add_argument("--resume-activations", action="store_true")
    parser.add_argument(
        "--recompute-costs-from-activations",
        action="store_true",
        help="Rebuild cost_mean.npy from a validated saved activation manifest",
    )
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
        raise ValueError(f"Stage 2 resume candidates must cover N=9-16 for {task}")
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
    if args.recompute_costs_from_activations and not args.resume_activations:
        raise ValueError(
            "--recompute-costs-from-activations requires --resume-activations"
        )
    config = load_yaml(args.config)
    validate_discovery_config(config)
    candidate_block_counts = [int(value) for value in config["num_moirai_blocks"]]
    checkpoint = Path(args.checkpoint or config["base_checkpoint"])
    checkpoint_manifest, checkpoint_hash = _checkpoint_manifest(checkpoint)
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
    if not torch.cuda.is_available():
        raise RuntimeError("Formal Stage 2 discovery requires CUDA")
    device = torch.device("cuda:0")
    model = MoiraiQwen3ForCausalLM.from_pretrained(
        checkpoint,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
    ).to(device)
    model.eval()
    model.config.attnres_execution = "full"
    model.config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    query_names, actual_q_full_hash = pseudo_query_sha256(model)
    if query_names != checkpoint_manifest.get("q_full_parameter_names"):
        raise ValueError("Stage 2 Q_full parameter names mismatch")
    if actual_q_full_hash != checkpoint_manifest.get("q_full_sha256"):
        raise ValueError("Stage 2 Q_full hash mismatch")

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
            device=device,
            checkpoint_hash=checkpoint_hash,
            q_full_hash=checkpoint_manifest["q_full_sha256"],
            num_transformer_blocks=int(config["num_transformer_blocks"]),
            candidate_block_counts=candidate_block_counts,
            maximum_refinement_sweeps=int(config["boundary_refinement_sweeps"]),
            resume_activations=bool(args.resume_activations),
            recompute_activation_costs=bool(
                args.recompute_costs_from_activations
            ),
        )


if __name__ == "__main__":
    main()
