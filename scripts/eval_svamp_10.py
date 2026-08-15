from __future__ import annotations

import json
from decimal import Decimal, InvalidOperation
from pathlib import Path

import torch

from src.common import load_yaml
from src.data.format_tasks import (
    encode_prompt_only,
    format_task_target,
    load_local_split,
    load_manifest,
)
from src.data.prepare_post_data import _entry, _ordered_unique
from src.evaluation.task_metrics import extract_math_answer
from src.probe.inference import (
    ALLOWED_PROBE_CLASSES,
    ALLOWED_SELECTED_CONFIGS,
    MoiraiInferenceEngine,
)


EVALUATION_CASES = 10
MAXIMUM_NEW_TOKENS = 256
OUTPUT_DIR = Path("outputs/formal/evaluation_svamp_10")


def _svamp_numeric_accuracy(prediction: str, gold: str) -> float:
    predicted_answer = extract_math_answer(prediction)
    gold_answer = extract_math_answer(gold)
    if not predicted_answer or not gold_answer:
        return 0.0
    try:
        return float(Decimal(predicted_answer) == Decimal(gold_answer))
    except InvalidOperation:
        return 0.0


def _select_unused_svamp_cases(
    *,
    data_config: dict,
    manifest: list[dict],
) -> tuple[list[dict], object, dict]:
    source = data_config["sources"]["svamp"]
    split = str(source["official_split"])
    dataset = load_local_split(source["local_path"], split)
    ordered = _ordered_unique(
        dataset,
        task="math",
        source=source,
        split=split,
        seed=int(data_config["split_seed"]),
    )
    used_ids = {str(record["stable_id"]) for record in manifest}
    used_content = {str(record["content_sha256"]) for record in manifest}
    selected: list[dict] = []
    for _key, row_index, row in ordered:
        entry = _entry(
            task="math",
            source=source,
            split=split,
            row_index=row_index,
            row=row,
            assigned_split="svamp_eval_10",
            seed=int(data_config["split_seed"]),
        )
        if entry["stable_id"] in used_ids or entry["content_sha256"] in used_content:
            continue
        selected.append(entry)
        used_ids.add(entry["stable_id"])
        used_content.add(entry["content_sha256"])
        if len(selected) == EVALUATION_CASES:
            break
    if len(selected) != EVALUATION_CASES:
        raise RuntimeError(
            f"SVAMP evaluation requires {EVALUATION_CASES} unused cases, "
            f"found {len(selected)}"
        )
    return selected, dataset, source["field_mapping"]


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("SVAMP evaluation requires CUDA")
    data_config = load_yaml("configs/data.yaml")
    manifest = load_manifest(data_config["output_dir"] + "/splits.json")
    records, dataset, mapping = _select_unused_svamp_cases(
        data_config=data_config,
        manifest=manifest,
    )
    engine = MoiraiInferenceEngine.from_config(
        "configs/probe.yaml",
        device=torch.device("cuda:0"),
    )
    predictions: list[dict] = []
    correct = 0
    route_counts = {task: 0 for task in sorted(ALLOWED_PROBE_CLASSES)}
    for record in records:
        row = dataset[int(record["row_index"])]
        prompt = encode_prompt_only(
            engine.tokenizer,
            task="math",
            row=row,
            field_mapping=mapping,
            stable_id=record["stable_id"],
            max_length=2048 - MAXIMUM_NEW_TOKENS,
        )
        result = engine.infer(
            prompt.input_ids.unsqueeze(0),
            prompt.attention_mask.unsqueeze(0),
            maximum_new_tokens=MAXIMUM_NEW_TOKENS,
        )
        probe_predicted_task = result.probe.predicted_task
        selected_config = result.selected_config
        assert probe_predicted_task in ALLOWED_PROBE_CLASSES
        assert selected_config in ALLOWED_SELECTED_CONFIGS
        assert selected_config == probe_predicted_task
        prediction = engine.tokenizer.decode(
            result.generated_ids[0].cpu(),
            skip_special_tokens=True,
        )
        gold = format_task_target("math", row, mapping)
        accuracy = _svamp_numeric_accuracy(prediction, gold)
        correct += int(accuracy)
        route_counts[selected_config] += 1
        predictions.append(
            {
                "stable_id": record["stable_id"],
                "row_index": int(record["row_index"]),
                "dataset": "svamp",
                "true_task": "math",
                "probe_predicted_task": probe_predicted_task,
                "selected_config": selected_config,
                "question": row[mapping["question"]],
                "gold": gold,
                "prediction": prediction,
                "accuracy": accuracy,
            }
        )

    result_payload = {
        "dataset": "svamp",
        "evaluation_cases": EVALUATION_CASES,
        "correct": correct,
        "accuracy": correct / EVALUATION_CASES,
        "selected_config_counts": route_counts,
        "stable_ids": [record["stable_id"] for record in records],
        "selection": "seed42 ordered, excluding every existing manifest ID/content hash",
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "predictions.jsonl").write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in predictions
        ),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "evaluation_results.json").write_text(
        json.dumps(result_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result_payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
