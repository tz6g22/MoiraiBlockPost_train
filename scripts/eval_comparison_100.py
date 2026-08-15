from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from src.baselines.evaluate_svamp import (
    _greedy_generate as baseline_greedy_generate,
    _load_trained_baseline,
    _numeric_accuracy,
)
from src.baselines.training import validate_baseline_config
from src.common import load_yaml, sha256_json
from src.data.format_tasks import (
    canonical_content_sha256,
    encode_prompt_only,
    format_task_target,
    load_local_split,
    load_manifest,
)
from src.data.prepare_post_data import _entry, _ordered_unique
from src.evaluation.task_metrics import task_score
from src.probe.inference import ALLOWED_PROBE_CLASSES, MoiraiInferenceEngine


METHODS = ("moiraiblock", "full_attnres", "fixed_block_attnres")
TASKS = ("math", "multihop")
EVALUATION_CASES = 100
GENERATION_LIMITS = {"math": 256, "multihop": 64}
OUTPUT_ROOT = Path("outputs/comparison_100")


def select_evaluation_records(
    *,
    task: str,
    data_config: dict[str, Any],
    manifest: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], Any, dict[str, Any]]:
    if task == "math":
        source = data_config["sources"]["svamp"]
    elif task == "multihop":
        # The official clean-2-hop test split has only 38 rows. Select a
        # deterministic held-out set from the 4,054-row train source while
        # excluding every ID and semantic-content hash already in the project
        # manifest (Discovery, training, validation, Probe, and final eval).
        source = data_config["sources"]["clutrr"]
    else:
        raise ValueError(f"Unsupported comparison task: {task}")
    split = str(source["official_split"])
    dataset = load_local_split(source["local_path"], split)
    ordered = _ordered_unique(
        dataset,
        task=task,
        source=source,
        split=split,
        seed=int(data_config["split_seed"]),
    )
    used_ids = {str(record["stable_id"]) for record in manifest}
    used_content = {str(record["content_sha256"]) for record in manifest}
    selected: list[dict[str, Any]] = []
    for _key, row_index, row in ordered:
        record = _entry(
            task=task,
            source=source,
            split=split,
            row_index=row_index,
            row=row,
            assigned_split=f"comparison_100_{task}",
            seed=int(data_config["split_seed"]),
        )
        if (
            record["stable_id"] in used_ids
            or record["content_sha256"] in used_content
        ):
            continue
        selected.append(record)
        used_ids.add(record["stable_id"])
        used_content.add(record["content_sha256"])
        if len(selected) == EVALUATION_CASES:
            break
    if len(selected) != EVALUATION_CASES:
        raise RuntimeError(
            f"{task} comparison requires {EVALUATION_CASES} unused cases; "
            f"found {len(selected)}"
        )
    manifest_ids = {str(record["stable_id"]) for record in manifest}
    manifest_content = {str(record["content_sha256"]) for record in manifest}
    if any(record["stable_id"] in manifest_ids for record in selected):
        raise RuntimeError("Comparison selection overlaps a manifest stable ID")
    if any(record["content_sha256"] in manifest_content for record in selected):
        raise RuntimeError("Comparison selection overlaps manifest semantic content")
    if len({record["stable_id"] for record in selected}) != EVALUATION_CASES:
        raise RuntimeError("Comparison stable IDs are not unique")
    if len({record["content_sha256"] for record in selected}) != EVALUATION_CASES:
        raise RuntimeError("Comparison content hashes are not unique")
    return selected, dataset, source["field_mapping"]


def _baseline_spec(method: str) -> tuple[str, Path]:
    if method == "full_attnres":
        return "full_attnres", Path("configs/baselines/full.yaml")
    if method == "fixed_block_attnres":
        return "fixed_block_attnres", Path("configs/baselines/fixed.yaml")
    raise ValueError(f"Not a baseline method: {method}")


def _write_evaluation_set(
    *,
    task: str,
    records: list[dict[str, Any]],
) -> str:
    stable_ids = [record["stable_id"] for record in records]
    selection_hash = sha256_json(stable_ids)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_ROOT / f"{task}_evaluation_set.json"
    payload = {
        "task": task,
        "dataset": "svamp" if task == "math" else "clutrr",
        "source_split": "all" if task == "math" else "train",
        "selection": (
            "seed42 ordered, unique, excluding every existing manifest "
            "stable ID and semantic-content hash"
        ),
        "evaluation_cases": EVALUATION_CASES,
        "selection_sha256": selection_hash,
        "records": records,
    }
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError(f"Existing comparison set differs: {path}")
    else:
        path.write_text(serialized, encoding="utf-8")
    return selection_hash


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--task", required=True, choices=TASKS)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Comparison evaluation requires CUDA")

    data_config = load_yaml("configs/data.yaml")
    manifest = load_manifest(data_config["output_dir"] + "/splits.json")
    records, dataset, mapping = select_evaluation_records(
        task=args.task,
        data_config=data_config,
        manifest=manifest,
    )
    selection_hash = _write_evaluation_set(task=args.task, records=records)
    device = torch.device("cuda:0")

    engine = None
    model = None
    tokenizer = None
    training_manifest = None
    if args.method == "moiraiblock":
        engine = MoiraiInferenceEngine.from_config(
            "configs/probe.yaml",
            device=device,
        )
        tokenizer = engine.tokenizer
    else:
        baseline_type, config_path = _baseline_spec(args.method)
        baseline_config = load_yaml(config_path)
        validate_baseline_config(baseline_config, expected_type=baseline_type)
        model, tokenizer, training_manifest = _load_trained_baseline(
            baseline_type=baseline_type,
            task=args.task,
            config=baseline_config,
            device=device,
        )

    output_dir = OUTPUT_ROOT / args.method / args.task
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite comparison output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    metric_rows: list[dict[str, float]] = []
    route_counts = {task: 0 for task in TASKS}
    route_correct = 0

    with predictions_path.open("x", encoding="utf-8") as handle:
        for index, record in enumerate(records, start=1):
            row = dataset[int(record["row_index"])]
            prompt = encode_prompt_only(
                tokenizer,
                task=args.task,
                row=row,
                field_mapping=mapping,
                stable_id=record["stable_id"],
                max_length=2048 - GENERATION_LIMITS[args.task],
            )
            input_ids = prompt.input_ids.unsqueeze(0).to(device)
            attention_mask = prompt.attention_mask.unsqueeze(0).to(device)
            probe_predicted_task = None
            selected_config = args.task
            if engine is not None:
                inference = engine.infer(
                    input_ids,
                    attention_mask,
                    maximum_new_tokens=GENERATION_LIMITS[args.task],
                )
                generated_ids = inference.generated_ids
                probe_predicted_task = inference.probe.predicted_task
                selected_config = inference.selected_config
                if probe_predicted_task not in ALLOWED_PROBE_CLASSES:
                    raise RuntimeError("Main method produced an illegal Probe class")
                if selected_config != probe_predicted_task:
                    raise RuntimeError("Main method route differs from Probe prediction")
                route_counts[selected_config] += 1
                route_correct += int(selected_config == args.task)
            else:
                generated_ids = baseline_greedy_generate(
                    model,
                    input_ids,
                    attention_mask,
                    eos_token_id=tokenizer.eos_token_id,
                    maximum_new_tokens=GENERATION_LIMITS[args.task],
                )
            prediction = tokenizer.decode(
                generated_ids[0].cpu(),
                skip_special_tokens=True,
            )
            gold = format_task_target(args.task, row, mapping)
            metrics = (
                {"accuracy": _numeric_accuracy(prediction, gold)}
                if args.task == "math"
                else task_score(args.task, prediction, gold=gold)
            )
            metric_rows.append(metrics)
            result = {
                "case": index,
                "stable_id": record["stable_id"],
                "row_index": int(record["row_index"]),
                "true_task": args.task,
                "probe_predicted_task": probe_predicted_task,
                "selected_config": selected_config,
                "gold": gold,
                "prediction": prediction,
                "metrics": metrics,
            }
            handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            if index % 10 == 0:
                current_correct = sum(int(row["accuracy"]) for row in metric_rows)
                print(
                    f"{args.method} {args.task}: {index}/{EVALUATION_CASES}, "
                    f"correct={current_correct}",
                    flush=True,
                )

    mean_metrics = {
        key: sum(float(row[key]) for row in metric_rows) / len(metric_rows)
        for key in sorted(metric_rows[0])
    }
    result_payload = {
        "method": args.method,
        "task": args.task,
        "dataset": "svamp" if args.task == "math" else "clutrr",
        "evaluation_cases": EVALUATION_CASES,
        "correct": sum(int(row["accuracy"]) for row in metric_rows),
        "accuracy": mean_metrics["accuracy"],
        "metrics": mean_metrics,
        "generation_limit": GENERATION_LIMITS[args.task],
        "evaluation_selection_sha256": selection_hash,
        "stable_ids": [record["stable_id"] for record in records],
        "route_counts": route_counts if engine is not None else None,
        "route_correct": route_correct if engine is not None else None,
        "route_accuracy": (
            route_correct / EVALUATION_CASES if engine is not None else None
        ),
        "training_query_after_hash": (
            training_manifest["query_after_hash"]
            if training_manifest is not None
            else None
        ),
    }
    (output_dir / "evaluation_results.json").write_text(
        json.dumps(result_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result_payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
