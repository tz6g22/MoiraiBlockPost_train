from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import load_yaml
from src.data.format_tasks import format_task_prompt, format_task_target, load_manifest, load_local_split
from src.data.prepare_post_data import _entry, _ordered_unique


KIMI_ROOT = ROOT.parent / "KimiBlockAttnRes"
OUTPUT = ROOT / "outputs/formal/evaluation_math_svamp_50_taskmatched"
CASE_COUNT = 50


def _stable_ids(value: object) -> set[str]:
    if isinstance(value, dict):
        return set().union(*(_stable_ids(item) for item in value.values())) if value else set()
    if isinstance(value, list):
        return set().union(*(_stable_ids(item) for item in value)) if value else set()
    if isinstance(value, str) and value.startswith("svamp::"):
        return {value}
    return set()


def _prediction_ids(root: Path) -> set[str]:
    ids: set[str] = set()
    for path in root.glob("outputs/formal/evaluation*/**/predictions.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                ids.update(_stable_ids(json.loads(line)))
    return ids


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {OUTPUT}")

    config = load_yaml(ROOT / "configs/data.yaml")
    source = config["sources"]["svamp"]
    split = str(source["official_split"])
    dataset = load_local_split(source["local_path"], split)
    split_manifest = load_manifest(ROOT / "outputs/data/splits.json")

    discovery_ids = {
        row["stable_id"]
        for row in split_manifest
        if row["task"] == "math" and row["assigned_split"] == "stage2_discovery"
    }
    moirai_train = json.loads(
        (ROOT / "outputs/formal/adapter_lr3e-5/math/query_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    moirai_train_ids = _stable_ids(moirai_train)
    kimi_train = json.loads(
        (KIMI_ROOT / "outputs/formal/task_specific/math/training_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    kimi_train_ids = _stable_ids(kimi_train)
    probe_ids = _stable_ids(
        json.loads((ROOT / "outputs/formal/probe/probe_manifest.json").read_text(encoding="utf-8"))
    )
    prior_evaluation_ids = _prediction_ids(ROOT) | _prediction_ids(KIMI_ROOT)
    used_ids = (
        discovery_ids
        | moirai_train_ids
        | kimi_train_ids
        | probe_ids
        | prior_evaluation_ids
    )

    candidates: list[tuple[dict, dict]] = []
    exclusions = {
        "discovery": 0,
        "moirai_query_training": 0,
        "kimi_query_training": 0,
        "probe": 0,
        "prior_evaluation": 0,
        "any": 0,
    }
    for _key, row_index, row in _ordered_unique(
        dataset,
        task="math",
        source=source,
        split=split,
        seed=int(config["split_seed"]),
    ):
        record = _entry(
            task="math",
            source=source,
            split=split,
            row_index=row_index,
            row=row,
            assigned_split="evaluation_svamp_50",
            seed=int(config["split_seed"]),
        )
        stable_id = record["stable_id"]
        flags = {
            "discovery": stable_id in discovery_ids,
            "moirai_query_training": stable_id in moirai_train_ids,
            "kimi_query_training": stable_id in kimi_train_ids,
            "probe": stable_id in probe_ids,
            "prior_evaluation": stable_id in prior_evaluation_ids,
        }
        if any(flags.values()):
            exclusions["any"] += 1
            for key, present in flags.items():
                exclusions[key] += int(present)
            continue
        candidates.append((record, row))

    if len(candidates) < CASE_COUNT:
        raise RuntimeError(f"SVAMP has only {len(candidates)} held-out candidates, need {CASE_COUNT}")

    tokenizer = AutoTokenizer.from_pretrained(
        ROOT / "outputs/base/qwen3_14b_full_attnres", local_files_only=True, use_fast=True
    )
    cases = []
    for record, row in candidates[:CASE_COUNT]:
        prompt = format_task_prompt("math", row, source["field_mapping"])
        token_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        cases.append(
            {
                "stable_id": record["stable_id"],
                "task": "math",
                "dataset": "svamp",
                "official_split": split,
                "row_index": int(record["row_index"]),
                "raw_problem": str(row[source["field_mapping"]["question"]]),
                "gold": format_task_target("math", row, source["field_mapping"]),
                "prompt": prompt,
                "prompt_token_ids": token_ids,
            }
        )

    selected_ids = [case["stable_id"] for case in cases]
    if len(selected_ids) != CASE_COUNT or len(set(selected_ids)) != CASE_COUNT:
        raise RuntimeError("SVAMP selection must contain 50 unique stable IDs")
    if set(selected_ids) & used_ids:
        raise RuntimeError("SVAMP selection leaked a prior-use stable ID")

    OUTPUT.mkdir(parents=True)
    _write_jsonl(OUTPUT / "evaluation_manifest.jsonl", cases)
    audit = {
        "dataset": "svamp",
        "candidate_cases": len(dataset),
        "excluded_due_to_leakage": exclusions["any"],
        "final_evaluation_cases": len(cases),
        "discovery_overlap": len(set(selected_ids) & discovery_ids),
        "moirai_query_training_overlap": len(set(selected_ids) & moirai_train_ids),
        "kimi_query_training_overlap": len(set(selected_ids) & kimi_train_ids),
        "probe_overlap": len(set(selected_ids) & probe_ids),
        "prior_evaluation_overlap": len(set(selected_ids) & prior_evaluation_ids),
        "exclusions_by_use": exclusions,
        "stable_id_sha256": hashlib.sha256(
            json.dumps(selected_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }
    (OUTPUT / "leakage_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
