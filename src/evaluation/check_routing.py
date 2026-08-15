from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.common import load_yaml
from src.data.format_tasks import encode_prompt_only, load_manifest
from src.evaluation.run_evaluation import (
    GENERATION_LIMITS,
    TASKS,
    _evaluation_rows,
    validate_evaluation_config,
)
from src.probe.inference import (
    ALLOWED_PROBE_CLASSES,
    ALLOWED_SELECTED_CONFIGS,
    MoiraiInferenceEngine,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/evaluation.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    validate_evaluation_config(config)
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 4 routing check requires CUDA")

    manifest = load_manifest(Path(config["data_manifest"]))
    data_config = load_yaml(config["data_config"])
    evaluation_data = {
        task: _evaluation_rows(
            task,
            manifest=manifest,
            data_config=data_config,
            expected_count=int(config["evaluation_examples_per_task"]),
        )
        for task in TASKS
    }
    engine = MoiraiInferenceEngine.from_config(
        config["probe_config"],
        device=torch.device("cuda:0"),
    )
    tokenizer = engine.tokenizer

    rows: list[dict[str, str]] = []
    true_counts = {task: 0 for task in TASKS}
    predicted_counts = {task: 0 for task in TASKS}
    confusion = {
        task: {predicted: 0 for predicted in TASKS}
        for task in TASKS
    }
    correct = 0

    for true_task in TASKS:
        records, dataset, mapping = evaluation_data[true_task]
        for record in records:
            prompt = encode_prompt_only(
                tokenizer,
                task=true_task,
                row=dataset[int(record["row_index"])],
                field_mapping=mapping,
                stable_id=record["stable_id"],
                max_length=2048 - GENERATION_LIMITS[true_task],
            )
            prediction, selected_config, _bundle = engine.select_config(
                prompt.input_ids.unsqueeze(0),
                prompt.attention_mask.unsqueeze(0),
            )
            probe_predicted_task = prediction.predicted_task
            assert probe_predicted_task in ALLOWED_PROBE_CLASSES
            assert selected_config in ALLOWED_SELECTED_CONFIGS
            assert selected_config == probe_predicted_task

            true_counts[true_task] += 1
            predicted_counts[probe_predicted_task] += 1
            confusion[true_task][probe_predicted_task] += 1
            correct += int(true_task == probe_predicted_task)
            rows.append(
                {
                    "stable_id": record["stable_id"],
                    "true_task": true_task,
                    "probe_predicted_task": probe_predicted_task,
                    "selected_config": selected_config,
                }
            )

    total = len(rows)
    if total != sum(true_counts.values()) or total == 0:
        raise RuntimeError("Invalid Stage 4 routing-check sample count")
    result = {
        "true_counts": true_counts,
        "predicted_counts": predicted_counts,
        "confusion_matrix": confusion,
        "final_eval_probe_accuracy": correct / total,
        "correct": correct,
        "total": total,
        "rows": rows,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
