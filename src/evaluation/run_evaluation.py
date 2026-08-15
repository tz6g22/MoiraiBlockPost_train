from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from src.common import load_yaml, sha256_file, tokenizer_sha256
from src.data.format_tasks import (
    SUPERVISED_TASKS,
    encode_prompt_only,
    encode_prompt_target,
    format_task_target,
    load_dataset_pool,
    load_manifest,
    nested_value,
)
from src.discovery.collect_reference import collect_full_reference
from src.discovery.replay import replay_partition
from src.distributed.fsdp_utils import (
    barrier,
    broadcast_object,
    destroy_distributed,
    init_distributed,
    load_selected_parameter_state,
    wrap_qwen3_fsdp,
)
from src.evaluation.distortion import distortion_summary
from src.evaluation.efficiency import (
    benchmark_cuda,
    measure_peak_memory,
    runtime_environment,
    source_storage_counts,
)
from src.evaluation.task_metrics import mean_metrics, task_score
from src.modeling.config_bundle import MoiraiConfigBundle
from src.modeling.full_attnres import MoiraiQwen3DecoderLayer, MoiraiQwen3ForCausalLM
from src.modeling.partition import fixed_kimi_partition
from src.probe.inference import (
    ALLOWED_PROBE_CLASSES,
    ALLOWED_SELECTED_CONFIGS,
    MoiraiInferenceEngine,
)
from src.training.checkpointing import validate_post_training_base_manifest


METHODS = ("full_attnres", "kimi_fixed_raw", "kimi_fixed_adapted", "moiraiblock")
TASKS = SUPERVISED_TASKS
GENERATION_LIMITS = {"math": 256, "multihop": 64, "code": 512}


def validate_evaluation_config(config: dict[str, Any]) -> None:
    expected = {
        "seed": 42,
        "use_cache": False,
        "do_sample": False,
        "num_beams": 1,
        "diagnostic_examples_per_task": 10,
        "evaluation_examples_per_task": 10,
        "primary_metric": "accuracy",
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(
                f"Evaluation config mismatch for {key}: expected {value!r}, "
                f"got {config.get(key)!r}"
            )
    efficiency = config.get("efficiency", {})
    if efficiency.get("lengths") != [128, 512, 2048]:
        raise ValueError("Evaluation lengths must remain [128, 512, 2048]")
    for key in ("warmup_runs", "timed_runs"):
        if int(efficiency.get(key, 0)) <= 0:
            raise ValueError(f"Evaluation {key} must be positive")
    for key in {
        "base_checkpoint",
        "data_manifest",
        "data_config",
        "partition_root",
        "query_root",
        "probe_config",
        "output_dir",
    }:
        if key not in config:
            raise ValueError(f"Evaluation config is missing {key}")


def _base_checkpoint_hash(checkpoint: Path) -> str:
    weights = sorted(checkpoint.glob("model*.safetensors"))
    if len(weights) != 1:
        raise RuntimeError("Evaluation requires exactly one base model weight file")
    manifest = json.loads(
        (checkpoint / "checkpoint_manifest.json").read_text(encoding="utf-8")
    )
    actual = sha256_file(weights[0])
    if actual != manifest.get("model_weights_sha256"):
        raise ValueError("Evaluation base checkpoint hash mismatch")
    validate_post_training_base_manifest(manifest)
    return actual


def _load_bundle(
    *,
    partition_path: Path,
    query_manifest_path: Path,
    base_hash: str,
) -> MoiraiConfigBundle:
    return MoiraiConfigBundle.load(
        partition_path=partition_path,
        query_manifest_path=query_manifest_path,
        expected_base_checkpoint_sha256=base_hash,
    )


def _query_state(model) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if "pseudo_query" in name
    }


def _restore_query_state(model, state: dict[str, torch.Tensor]) -> None:
    load_selected_parameter_state(
        model,
        state,
        lambda name, _parameter: "pseudo_query" in name,
    )


@torch.no_grad()
def _greedy_generate(
    model,
    input_ids: torch.LongTensor,
    attention_mask: torch.LongTensor,
    *,
    maximum_new_tokens: int,
    eos_token_id: int,
) -> torch.LongTensor:
    generated = input_ids.clone()
    mask = attention_mask.clone()
    prompt_length = generated.shape[1]
    for _ in range(maximum_new_tokens):
        outputs = model(
            input_ids=generated,
            attention_mask=mask,
            use_cache=False,
            logits_to_keep=1,
        )
        next_token = outputs.logits[:, -1].argmax(dim=-1, keepdim=True)
        generated = torch.cat((generated, next_token), dim=1)
        mask = torch.cat((mask, torch.ones_like(next_token)), dim=1)
        if bool(torch.all(next_token == eos_token_id)):
            break
    return generated[:, prompt_length:]


def _evaluation_rows(
    task: str,
    *,
    manifest: list[dict[str, Any]],
    data_config: dict[str, Any],
    expected_count: int,
):
    source = data_config["evaluation_sources"][task]
    selected = [
        record
        for record in manifest
        if record.get("assigned_split") == "stage4_final_eval"
        and record.get("task") == task
        and record.get("dataset") == source["dataset_name"]
    ]
    selected.sort(key=lambda record: record["split_key"])
    if len(selected) != expected_count:
        raise RuntimeError(
            f"{task} final evaluation requires exactly {expected_count} cases, "
            f"found {len(selected)}"
        )
    return selected, load_dataset_pool(data_config, str(source["dataset_name"]))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _commit_hash() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/evaluation.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--data-manifest")
    parser.add_argument("--partition-root")
    parser.add_argument("--query-root")
    parser.add_argument("--probe-config")
    parser.add_argument("--output-dir")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    overrides = {
        "base_checkpoint": args.checkpoint,
        "data_manifest": args.data_manifest,
        "partition_root": args.partition_root,
        "query_root": args.query_root,
        "probe_config": args.probe_config,
        "output_dir": args.output_dir,
    }
    for key, value in overrides.items():
        if value:
            config[key] = value
    validate_evaluation_config(config)
    context = init_distributed()
    rank = context.rank
    device = context.device
    checkpoint = Path(config["base_checkpoint"])
    base_hash = broadcast_object(
        _base_checkpoint_hash(checkpoint) if context.is_rank0 else None,
        context,
    )
    manifest_path = Path(config["data_manifest"])
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Evaluation manifest is missing: {manifest_path}")
    manifest = load_manifest(manifest_path)
    data_config = load_yaml(config["data_config"])
    main_bundles = {
        task: _load_bundle(
            partition_path=Path(config["partition_root"]) / task / "partition.json",
            query_manifest_path=Path(config["query_root"])
            / task
            / "query_manifest.json",
            base_hash=base_hash,
        )
        for task in TASKS
    }
    fixed_root = Path(config["query_root"]) / "fixed"
    fixed_bundle = _load_bundle(
        partition_path=fixed_root / "partition.json",
        query_manifest_path=fixed_root / "query_manifest.json",
        base_hash=base_hash,
    )
    evaluation_data = {
        task: _evaluation_rows(
            task,
            manifest=manifest,
            data_config=data_config,
            expected_count=int(config["evaluation_examples_per_task"]),
        )
        for task in TASKS
    }
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        local_files_only=True,
        use_fast=True,
    )
    checkpoint_manifest = json.loads(
        (checkpoint / "checkpoint_manifest.json").read_text(encoding="utf-8")
    )
    if tokenizer_sha256(tokenizer) != checkpoint_manifest.get("tokenizer_sha256"):
        raise ValueError("Evaluation checkpoint tokenizer hash mismatch")
    baseline_model = MoiraiQwen3ForCausalLM.from_pretrained(
        checkpoint,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    baseline_model.eval()
    for parameter in baseline_model.parameters():
        parameter.requires_grad_(False)
    q_full = _query_state(baseline_model)
    baseline_parameter_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in baseline_model.parameters()
    )
    baseline_model = wrap_qwen3_fsdp(
        baseline_model,
        context,
        decoder_layer_classes=(MoiraiQwen3DecoderLayer,),
    )
    engine = MoiraiInferenceEngine.from_config(
        config["probe_config"],
        device=device,
        model=baseline_model,
        tokenizer=tokenizer,
    )

    predictions: list[dict[str, Any]] = []
    scores: dict[str, dict[str, list[dict[str, float]]]] = {
        method: {task: [] for task in TASKS}
        for method in METHODS
    }
    confusion = {
        task: {predicted: 0 for predicted in TASKS}
        for task in TASKS
    }
    probe_correct = 0
    probe_total = 0
    selected_config_counts = {
        selected: 0 for selected in sorted(ALLOWED_SELECTED_CONFIGS)
    }

    try:
        for task in TASKS:
            records, pool = evaluation_data[task]
            for record in records:
                dataset, mapping = pool[str(record["official_split"])]
                row = dataset[int(record["row_index"])]
                prompt = encode_prompt_only(
                    tokenizer,
                    task=task,
                    row=row,
                    field_mapping=mapping,
                    stable_id=record["stable_id"],
                    max_length=2048 - GENERATION_LIMITS[task],
                )
                input_ids = prompt.input_ids.unsqueeze(0).to(device)
                attention_mask = prompt.attention_mask.unsqueeze(0).to(device)
                gold = format_task_target(task, row, mapping)
                method_predictions: dict[str, str] = {}
                _restore_query_state(baseline_model, q_full)
                baseline_model.config.attnres_execution = "full"
                full_ids = _greedy_generate(
                    baseline_model,
                    input_ids,
                    attention_mask,
                    maximum_new_tokens=GENERATION_LIMITS[task],
                    eos_token_id=tokenizer.eos_token_id,
                )
                method_predictions["full_attnres"] = tokenizer.decode(
                    full_ids[0].cpu(),
                    skip_special_tokens=True,
                )

                _restore_query_state(baseline_model, q_full)
                fixed = fixed_kimi_partition(
                    task="fixed",
                    num_transformer_blocks=baseline_model.config.num_hidden_layers,
                )
                baseline_model.config.attnres_execution = "moirai"
                baseline_model.config.moirai_partition = list(fixed.lengths)
                baseline_model.config.moirai_task = "fixed"
                fixed_raw_ids = _greedy_generate(
                    baseline_model,
                    input_ids,
                    attention_mask,
                    maximum_new_tokens=GENERATION_LIMITS[task],
                    eos_token_id=tokenizer.eos_token_id,
                )
                method_predictions["kimi_fixed_raw"] = tokenizer.decode(
                    fixed_raw_ids[0].cpu(),
                    skip_special_tokens=True,
                )

                fixed_bundle.apply_to_model(baseline_model)
                fixed_adapted_ids = _greedy_generate(
                    baseline_model,
                    input_ids,
                    attention_mask,
                    maximum_new_tokens=GENERATION_LIMITS[task],
                    eos_token_id=tokenizer.eos_token_id,
                )
                method_predictions["kimi_fixed_adapted"] = tokenizer.decode(
                    fixed_adapted_ids[0].cpu(),
                    skip_special_tokens=True,
                )

                moirai = engine.infer(
                    input_ids,
                    attention_mask,
                    maximum_new_tokens=GENERATION_LIMITS[task],
                )
                probe_predicted_task = moirai.probe.predicted_task
                selected_config = moirai.selected_config
                assert probe_predicted_task in ALLOWED_PROBE_CLASSES
                assert selected_config in ALLOWED_SELECTED_CONFIGS
                method_predictions["moiraiblock"] = tokenizer.decode(
                    moirai.generated_ids[0].cpu(),
                    skip_special_tokens=True,
                )
                probe_total += 1
                probe_correct += int(probe_predicted_task == task)
                confusion[task][probe_predicted_task] += 1
                selected_config_counts[selected_config] += 1

                row_scores: dict[str, dict[str, float]] = {}
                for method, prediction in method_predictions.items():
                    metric = task_score(
                        task,
                        prediction,
                        gold=gold,
                        test_list=(
                            list(
                                nested_value(
                                    row,
                                    str(mapping["test_list"]),
                                )
                            )
                            if task == "code"
                            else None
                        ),
                        test_setup_code=(
                            str(
                                nested_value(
                                    row,
                                    str(mapping["test_setup_code"]),
                                )
                            )
                            if task == "code"
                            else ""
                        ),
                    )
                    scores[method][task].append(metric)
                    row_scores[method] = metric
                main_metric = row_scores["moiraiblock"]
                predictions.append(
                    {
                        "stable_id": record["stable_id"],
                        "true_task": task,
                        "probe_predicted_task": probe_predicted_task,
                        "selected_config": selected_config,
                        "probe_logits": list(moirai.probe.logits),
                        "probe_probabilities": list(moirai.probe.probabilities),
                        "probe_confidence": moirai.probe.confidence,
                        "config_partition_sha256": moirai.partition_sha256,
                        "config_query_sha256": moirai.query_sha256,
                        "prediction": method_predictions["moiraiblock"],
                        "gold": gold,
                        "correct": bool(
                            main_metric["accuracy"]
                        ),
                        "baseline_predictions": method_predictions,
                        "baseline_metrics": row_scores,
                    }
                )
    except Exception:
        raise

    task_metrics = {
        method: {
            task: mean_metrics(scores[method][task])
            for task in TASKS
        }
        for method in METHODS
    }
    primary_scores = {
        method: {
            **{
                task: task_metrics[method][task]["accuracy"]
                for task in TASKS
            },
            "macro_average": sum(
                task_metrics[method][task]["accuracy"] for task in TASKS
            )
            / len(TASKS),
        }
        for method in METHODS
    }
    final_eval_probe_accuracy = probe_correct / probe_total
    final_eval_probe_metrics = {
        "final_eval_probe_accuracy": final_eval_probe_accuracy,
        "correct": probe_correct,
        "total": probe_total,
        "confusion_by_true_task": confusion,
        "selected_config_counts": selected_config_counts,
        "fixed_selection_rate": selected_config_counts["fixed"] / probe_total,
    }

    distortion_values = {
        method: {task: [] for task in TASKS}
        for method in METHODS
    }
    for task in TASKS:
        records, pool = evaluation_data[task]
        diagnostic_records = records[: int(config["diagnostic_examples_per_task"])]
        for record in diagnostic_records:
            dataset, mapping = pool[str(record["official_split"])]
            row = dataset[int(record["row_index"])]
            example = encode_prompt_target(
                tokenizer,
                task=task,
                row=row,
                field_mapping=mapping,
                stable_id=record["stable_id"],
                max_length=2048,
            )
            sequence = torch.cat((example.input_ids, example.labels[-1:]))
            input_ids = sequence.unsqueeze(0).to(device)
            attention_mask = torch.ones_like(input_ids)
            _restore_query_state(baseline_model, q_full)
            baseline_model.config.attnres_execution = "full"
            reference = collect_full_reference(
                baseline_model,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            distortion_values["full_attnres"][task].append(0.0)
            fixed = fixed_kimi_partition(
                task="fixed",
                num_transformer_blocks=baseline_model.config.num_hidden_layers,
            )
            raw = replay_partition(
                baseline_model,
                fixed,
                input_ids=input_ids,
                attention_mask=attention_mask,
                reference=reference,
            )
            distortion_values["kimi_fixed_raw"][task].append(raw.mean_distortion)
            fixed_bundle.apply_to_model(baseline_model)
            baseline_model.config.attnres_execution = "full"
            adapted = replay_partition(
                baseline_model,
                fixed,
                input_ids=input_ids,
                attention_mask=attention_mask,
                reference=reference,
            )
            distortion_values["kimi_fixed_adapted"][task].append(
                adapted.mean_distortion
            )
            main_bundles[task].apply_to_model(baseline_model)
            baseline_model.config.attnres_execution = "full"
            moirai_distortion = replay_partition(
                baseline_model,
                main_bundles[task].partition,
                input_ids=input_ids,
                attention_mask=attention_mask,
                reference=reference,
            )
            distortion_values["moiraiblock"][task].append(
                moirai_distortion.mean_distortion
            )
            del reference, raw, adapted, moirai_distortion
            torch.cuda.empty_cache()
    distortion = {
        method: {
            task: distortion_summary(distortion_values[method][task])
            for task in TASKS
        }
        for method in METHODS
    }

    benchmark_tokens: list[int] = []
    for task in TASKS:
        records, pool = evaluation_data[task]
        for record in records:
            dataset, mapping = pool[str(record["official_split"])]
            prompt = encode_prompt_only(
                tokenizer,
                task=task,
                row=dataset[int(record["row_index"])],
                field_mapping=mapping,
                stable_id=record["stable_id"],
                max_length=2048,
            )
            benchmark_tokens.extend(prompt.input_ids.tolist())
            if len(benchmark_tokens) >= 2048:
                break
        if len(benchmark_tokens) >= 2048:
            break
    if not benchmark_tokens:
        raise RuntimeError("Final evaluation prompts provide no benchmark tokens")

    latency_rows: list[dict[str, Any]] = []
    memory_rows: list[dict[str, Any]] = []
    efficiency = config["efficiency"]
    engine_parameter_bytes = baseline_parameter_bytes + sum(
        parameter.numel() * parameter.element_size()
        for parameter in engine.probe_head.parameters()
    )
    for length in efficiency["lengths"]:
        actual_length = min(length, len(benchmark_tokens))
        ids = torch.tensor(
            benchmark_tokens[:actual_length],
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)
        mask = torch.ones_like(ids)
        for method in METHODS:
            if method == "full_attnres":
                _restore_query_state(baseline_model, q_full)
                baseline_model.config.attnres_execution = "full"
                operation = lambda: baseline_model(
                    input_ids=ids,
                    attention_mask=mask,
                    use_cache=False,
                )
            elif method == "kimi_fixed_raw":
                _restore_query_state(baseline_model, q_full)
                fixed = fixed_kimi_partition(
                    task="math",
                    num_transformer_blocks=baseline_model.config.num_hidden_layers,
                )
                baseline_model.config.attnres_execution = "moirai"
                baseline_model.config.moirai_partition = list(fixed.lengths)
                baseline_model.config.moirai_task = "math"
                operation = lambda: baseline_model(
                    input_ids=ids,
                    attention_mask=mask,
                    use_cache=False,
                )
            elif method == "kimi_fixed_adapted":
                fixed_bundle.apply_to_model(baseline_model)
                operation = lambda: baseline_model(
                    input_ids=ids,
                    attention_mask=mask,
                    use_cache=False,
                )
            else:
                probe_operation = lambda: engine.classify(ids, mask)
                main_bundles["math"].apply_to_model(engine.model)
                main_operation = lambda: engine.model(
                    input_ids=ids,
                    attention_mask=mask,
                    use_cache=False,
                )
                operation = lambda: engine.infer(
                    ids,
                    mask,
                    maximum_new_tokens=1,
                )
                probe_latency = benchmark_cuda(
                    probe_operation,
                    warmup_runs=int(efficiency["warmup_runs"]),
                    timed_runs=int(efficiency["timed_runs"]),
                    processed_nonpadding_tokens=int(mask.sum().item()),
                )
                main_latency = benchmark_cuda(
                    main_operation,
                    warmup_runs=int(efficiency["warmup_runs"]),
                    timed_runs=int(efficiency["timed_runs"]),
                    processed_nonpadding_tokens=int(mask.sum().item()),
                )
                latency_rows.append(
                    {
                        "method": method,
                        "component": "probe",
                        "length": actual_length,
                        "requested_length": length,
                        **probe_latency,
                    }
                )
                latency_rows.append(
                    {
                        "method": method,
                        "component": "main",
                        "length": actual_length,
                        "requested_length": length,
                        **main_latency,
                    }
                )
            latency = benchmark_cuda(
                operation,
                warmup_runs=int(efficiency["warmup_runs"]),
                timed_runs=int(efficiency["timed_runs"]),
                processed_nonpadding_tokens=int(mask.sum().item()),
            )
            memory = measure_peak_memory(operation)
            latency_rows.append(
                {
                    "method": method,
                    "component": "total" if method == "moiraiblock" else "main",
                    "length": actual_length,
                    "requested_length": length,
                    **latency,
                }
            )
            parameter_bytes = (
                engine_parameter_bytes
                if method == "moiraiblock"
                else baseline_parameter_bytes
            )
            if method == "full_attnres":
                counts = source_storage_counts(
                    num_transformer_blocks=baseline_model.config.num_hidden_layers,
                    partition_lengths=None,
                )
            elif method in {"kimi_fixed_raw", "kimi_fixed_adapted"}:
                counts = source_storage_counts(
                    num_transformer_blocks=baseline_model.config.num_hidden_layers,
                    partition_lengths=fixed_kimi_partition(
                        num_transformer_blocks=baseline_model.config.num_hidden_layers
                    ).lengths,
                )
            else:
                counts = source_storage_counts(
                    num_transformer_blocks=baseline_model.config.num_hidden_layers,
                    partition_lengths=main_bundles["math"].partition.lengths,
                )
            source_peak_bytes = (
                counts["maximum_live_source_tensors"]
                * actual_length
                * baseline_model.config.hidden_size
                * 2
            )
            memory_rows.append(
                {
                    "method": method,
                    "length": actual_length,
                    "requested_length": length,
                    **memory,
                    "parameter_bytes": parameter_bytes,
                    "non_parameter_peak_bytes": max(
                        0,
                        memory["allocated_peak_bytes"] - parameter_bytes,
                    ),
                    "source_tensor_peak_bytes": source_peak_bytes,
                }
            )

    source_counts = {
        "full_attnres": source_storage_counts(
            num_transformer_blocks=baseline_model.config.num_hidden_layers,
            partition_lengths=None,
        ),
        "kimi_fixed_raw": source_storage_counts(
            num_transformer_blocks=baseline_model.config.num_hidden_layers,
            partition_lengths=fixed_kimi_partition(
                num_transformer_blocks=baseline_model.config.num_hidden_layers
            ).lengths,
        ),
        "kimi_fixed_adapted": source_storage_counts(
            num_transformer_blocks=baseline_model.config.num_hidden_layers,
            partition_lengths=fixed_kimi_partition(
                num_transformer_blocks=baseline_model.config.num_hidden_layers
            ).lengths,
        ),
        "moiraiblock": {
            task: source_storage_counts(
                num_transformer_blocks=baseline_model.config.num_hidden_layers,
                partition_lengths=main_bundles[task].partition.lengths,
            )
            for task in TASKS
        },
    }

    output_dir = Path(config["output_dir"])
    if context.is_rank0:
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
            for prediction in predictions:
                handle.write(json.dumps(prediction, ensure_ascii=False, sort_keys=True) + "\n")
        _write_json(
            output_dir / "evaluation_results.json",
            {
                "seed": int(config["seed"]),
                "base_checkpoint_sha256": base_hash,
                "primary_metric": config["primary_metric"],
                "primary_scores": primary_scores,
                "task_metrics": task_metrics,
                "final_eval_probe_accuracy": final_eval_probe_accuracy,
                "final_eval_probe": final_eval_probe_metrics,
                "distortion": distortion,
                "source_counts": source_counts,
            },
        )
    efficiency_rows = [
        {"measurement": "latency", **row} for row in latency_rows
    ] + [{"measurement": "memory", **row} for row in memory_rows]
    fieldnames = sorted({key for row in efficiency_rows for key in row})
    if context.is_rank0:
        with (output_dir / "efficiency.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(efficiency_rows)
        environment = runtime_environment(_commit_hash())
        _write_json(output_dir / "environment.json", environment)
    barrier(context)
    destroy_distributed(context)


if __name__ == "__main__":
    main()
