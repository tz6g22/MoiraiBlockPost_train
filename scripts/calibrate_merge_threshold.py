from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoTokenizer, Qwen3ForCausalLM
from torch.nn.utils.rnn import pad_sequence

from src.common import load_yaml, sha256_file, sha256_json
from src.data.format_tasks import load_manifest
from src.data.source_provenance import audit_formal_source_provenance
from src.discovery.ordinary_residual import collect_ordinary_residual_reference
from src.discovery.run_all import _task_examples, _valid_intervals


def _summary(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    quantiles = {
        str(q): ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1)))]
        for q in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
    }
    return {
        "count": len(values),
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "std": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
        "quantiles": quantiles,
    }


def _interval_costs(reference, intervals: tuple[tuple[int, int], ...]) -> dict[tuple[int, int], torch.Tensor]:
    values = reference.residual_contributions.float()
    directions = values / (torch.linalg.vector_norm(values, dim=-1, keepdim=True) + 1.0e-8)
    prefix = torch.cat(
        (torch.zeros_like(directions[:1]), directions.cumsum(dim=0)),
        dim=0,
    )
    squared_norms = directions.square().sum(dim=-1)
    squared_prefix = torch.cat(
        (torch.zeros_like(squared_norms[:1]), squared_norms.cumsum(dim=0)),
        dim=0,
    )
    valid = reference.attention_mask.to(dtype=torch.float32)
    denominator = valid.sum() + 1.0e-8
    costs: dict[tuple[int, int], torch.Tensor] = {}
    for start, end in intervals:
        length = end - start + 1
        if length == 1:
            costs[(start, end)] = torch.zeros(values.shape[1], dtype=torch.float32)
            continue
        summed = prefix[end + 1] - prefix[start]
        sum_squared_norms = squared_prefix[end + 1] - squared_prefix[start]
        pair_dot_sum = 0.5 * (summed.square().sum(dim=-1) - sum_squared_norms)
        pair_count = length * (length - 1) // 2
        per_token = 1.0 - pair_dot_sum / float(pair_count)
        value = (per_token * valid).sum(dim=1) / denominator
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"Pairwise directional cost is not finite for {start}:{end}")
        costs[(start, end)] = value.cpu()
    return costs


@torch.no_grad()
def calibrate(
    *,
    checkpoint: Path,
    data_config_path: Path,
    data_manifest_path: Path,
    output_path: Path,
    tasks: tuple[str, ...],
    cases_per_task: int | None,
    batch_size: int,
) -> dict[str, object]:
    data_config = load_yaml(data_config_path)
    records = load_manifest(data_manifest_path)
    audit_formal_source_provenance(
        records,
        data_config=data_config,
        enabled_tasks=tasks,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        local_files_only=True,
        use_fast=True,
    )
    model = Qwen3ForCausalLM.from_pretrained(
        checkpoint,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    layers = int(model.config.num_hidden_layers)
    intervals = _valid_intervals(layers, range(1, layers + 1))
    values_by_task_length: dict[str, dict[int, list[float]]] = {
        task: defaultdict(list) for task in tasks
    }
    for task in tasks:
        expected = sum(int(value) for value in data_config["discovery_sources"][task].values())
        count = expected if cases_per_task is None else cases_per_task
        case_records, examples = _task_examples(
            task,
            records=records,
            data_config=data_config,
            tokenizer=tokenizer,
            expected_count=count,
        )
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id
        for start in range(0, len(examples), batch_size):
            batch_examples = examples[start : start + batch_size]
            input_ids = pad_sequence(
                [example.input_ids for example in batch_examples],
                batch_first=True,
                padding_value=int(pad_token_id),
            )
            attention_mask = pad_sequence(
                [example.attention_mask for example in batch_examples],
                batch_first=True,
                padding_value=0,
            )
            reference = collect_ordinary_residual_reference(
                model,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            costs = _interval_costs(reference, intervals)
            for (interval_start, interval_end), values in costs.items():
                length = interval_end - interval_start + 1
                values_by_task_length[task][length].extend(values.tolist())
            del reference
    result: dict[str, object] = {
        "status": "PASS",
        "model": {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_json({
                path.name: sha256_file(path)
                for path in sorted(checkpoint.glob("model*.safetensors"))
            }),
            "num_hidden_layers": layers,
            "hidden_size": int(model.config.hidden_size),
            "torch_dtype": str(model.dtype),
        },
        "data_manifest_sha256": sha256_file(data_manifest_path),
        "tasks": list(tasks),
        "cases_per_task": {
            task: sum(len(values) for values in values_by_task_length[task].values()) // len(intervals)
            for task in tasks
        },
        "interval_count": len(intervals),
        "batch_size": batch_size,
        "cost_method": "ordinary_residual_pairwise_directional_v1",
        "forward_only": True,
        "attnres_accessed": False,
        "query_accessed": False,
        "alpha_accessed": False,
        "backward_used": False,
        "by_task_and_length": {
            task: {
                str(length): _summary(values)
                for length, values in sorted(values_by_task_length[task].items())
            }
            for task in tasks
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases-per-task", type=int)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    config = load_yaml("qwen3_1.7b_config.yaml")
    tasks = tuple(str(task) for task in config["tasks"]["enabled"])
    result = calibrate(
        checkpoint=args.checkpoint,
        data_config_path=args.data_config,
        data_manifest_path=args.data_manifest,
        output_path=args.output,
        tasks=tasks,
        cases_per_task=args.cases_per_task,
        batch_size=args.batch_size,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
