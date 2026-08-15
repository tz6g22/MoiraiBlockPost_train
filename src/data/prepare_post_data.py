from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from src.common import canonical_json, load_yaml
from src.data.format_tasks import (
    TASK_TO_SOURCE,
    canonical_content_sha256,
    load_local_split,
    nested_value,
)
from src.data.leakage_audit import audit_manifest


def _stable_id(
    *,
    dataset_name: str,
    split: str,
    row: dict[str, Any],
    mapping: dict[str, Any],
) -> str:
    field = mapping.get("id")
    try:
        value = nested_value(row, str(field)) if field else None
    except KeyError:
        value = None
    if value is None or not str(value):
        identity_fields = [
            mapping.get("question"),
            mapping.get("target"),
            mapping.get("answer"),
        ]
        text_fields = mapping.get("text_fields", [])
        identity_fields.extend(text_fields if isinstance(text_fields, list) else [])
        values = []
        for identity_field in identity_fields:
            if identity_field:
                values.append(str(nested_value(row, str(identity_field))))
        if not values:
            raise ValueError(f"{dataset_name}/{split} row has no stable identifier")
        value = "\n".join(values)
    digest = hashlib.sha256(str(value).encode("utf-8")).hexdigest()
    return f"{dataset_name}::{split}::{digest}"


def _order_key(seed: int, stable_id: str) -> str:
    return hashlib.sha256(f"{seed}::{stable_id}".encode("utf-8")).hexdigest()


def _load_source(source: dict[str, Any]):
    path = Path(source["local_path"])
    if not path.exists():
        raise FileNotFoundError(
            f"Required original-protocol dataset is missing: {path}. "
            "Dataset substitution is disabled."
        )
    return load_local_split(path, str(source["official_split"]))


def _entry(
    *,
    task: str,
    source: dict[str, Any],
    split: str,
    row_index: int,
    row: dict[str, Any],
    assigned_split: str,
    seed: int,
) -> dict[str, Any]:
    dataset_name = str(source["dataset_name"])
    mapping = source["field_mapping"]
    stable_id = _stable_id(
        dataset_name=dataset_name,
        split=split,
        row=row,
        mapping=mapping,
    )
    return {
        "stable_id": stable_id,
        "dataset": dataset_name,
        "dataset_revision": source["revision"],
        "official_split": split,
        "row_index": row_index,
        "content_sha256": canonical_content_sha256(
            str(source["dataset_name"]),
            row,
            mapping,
        ),
        "split_key": _order_key(seed, stable_id),
        "assigned_split": assigned_split,
        "task": task,
    }


def _ordered_unique(
    dataset,
    *,
    task: str,
    source: dict[str, Any],
    split: str,
    seed: int,
) -> list[tuple[str, int, dict[str, Any]]]:
    rows: list[tuple[str, int, dict[str, Any]]] = []
    seen: set[str] = set()
    for row_index in range(len(dataset)):
        row = dataset[row_index]
        stable_id = _stable_id(
            dataset_name=str(source["dataset_name"]),
            split=split,
            row=row,
            mapping=source["field_mapping"],
        )
        if stable_id in seen:
            continue
        seen.add(stable_id)
        rows.append((_order_key(seed, stable_id), row_index, row))
    rows.sort(key=lambda value: value[0])
    return rows


def _append_assignments(
    *,
    entries: list[dict[str, Any]],
    used_content_hashes: set[str],
    rows: list[tuple[str, int, dict[str, Any]]],
    assignments: tuple[tuple[str, int], ...],
    task: str,
    source: dict[str, Any],
    split: str,
    seed: int,
) -> None:
    cursor = 0
    for assigned_split, count in assignments:
        selected = 0
        while selected < count and cursor < len(rows):
            _, row_index, row = rows[cursor]
            cursor += 1
            entry = _entry(
                task=task,
                source=source,
                split=split,
                row_index=row_index,
                row=row,
                assigned_split=assigned_split,
                seed=seed,
            )
            if entry["content_sha256"] in used_content_hashes:
                continue
            entries.append(entry)
            used_content_hashes.add(entry["content_sha256"])
            selected += 1
        if selected != count:
            raise RuntimeError(
                f"{task}/{assigned_split} needs {count} non-overlapping rows, "
                f"found {selected}"
            )


def prepare(config_path: str | Path) -> dict[str, Any]:
    config = load_yaml(config_path)
    counts = config["counts"]
    seed = int(config["split_seed"])
    output_dir = Path(config["output_dir"])
    manifest_path = output_dir / "splits.json"
    entries: list[dict[str, Any]] = []
    used_content_hashes: set[str] = set()
    report: dict[str, Any] = {"tasks": {}, "counts": counts}
    discovery_sources = config["discovery_sources"]

    for task, source_key in TASK_TO_SOURCE.items():
        source = config["sources"][source_key]
        dataset = _load_source(source)
        train_split = str(source["official_split"])
        train_rows = _ordered_unique(
            dataset,
            task=task,
            source=source,
            split=train_split,
            seed=seed,
        )
        validation_source = config.get("validation_sources", {}).get(task)
        if validation_source is None:
            train_assignments = (
                (
                    "stage2_discovery",
                    int(discovery_sources[task].get(source_key, 0)),
                ),
                ("stage3_adapter_train", int(counts["stage3_adapter_train"])),
                ("stage3_adapter_val", int(counts["stage3_adapter_val"])),
                ("probe_train", int(counts["probe_train"])),
                ("probe_val", int(counts["probe_val"])),
            )
        else:
            train_assignments = (
                (
                    "stage2_discovery",
                    int(discovery_sources[task].get(source_key, 0)),
                ),
                ("stage3_adapter_train", int(counts["stage3_adapter_train"])),
                ("probe_train", int(counts["probe_train"])),
            )
        required = sum(count for _, count in train_assignments)
        if len(train_rows) < required:
            raise RuntimeError(
                f"{task} train needs {required} unique rows, got {len(train_rows)}"
            )
        _append_assignments(
            entries=entries,
            used_content_hashes=used_content_hashes,
            rows=train_rows,
            assignments=train_assignments,
            task=task,
            source=source,
            split=train_split,
            seed=seed,
        )

        validation_available = len(train_rows)
        validation_assignments: tuple[tuple[str, int], ...] = ()
        if validation_source is not None:
            validation_dataset = _load_source(validation_source)
            validation_split = str(validation_source["official_split"])
            validation_rows = _ordered_unique(
                validation_dataset,
                task=task,
                source=validation_source,
                split=validation_split,
                seed=seed,
            )
            validation_assignments = (
                ("stage3_adapter_val", int(counts["stage3_adapter_val"])),
                ("probe_val", int(counts["probe_val"])),
            )
            required_validation = sum(
                count for _, count in validation_assignments
            )
            if len(validation_rows) < required_validation:
                raise RuntimeError(
                    f"{task} validation needs {required_validation} unique rows, "
                    f"got {len(validation_rows)}"
                )
            _append_assignments(
                entries=entries,
                used_content_hashes=used_content_hashes,
                rows=validation_rows,
                assignments=validation_assignments,
                task=task,
                source=validation_source,
                split=validation_split,
                seed=seed,
            )
            validation_available = len(validation_rows)

        evaluation = config["evaluation_sources"][task]
        evaluation_dataset = _load_source(evaluation)
        evaluation_split = str(evaluation["official_split"])
        evaluation_rows = _ordered_unique(
            evaluation_dataset,
            task=task,
            source=evaluation,
            split=evaluation_split,
            seed=seed,
        )
        evaluation_count = int(counts["stage4_final_eval"])
        if len(evaluation_rows) < evaluation_count:
            raise RuntimeError(
                f"{task} evaluation needs {evaluation_count} rows, "
                f"got {len(evaluation_rows)}"
            )
        _append_assignments(
            entries=entries,
            used_content_hashes=used_content_hashes,
            rows=evaluation_rows,
            assignments=(("stage4_final_eval", evaluation_count),),
            task=task,
            source=evaluation,
            split=evaluation_split,
            seed=seed,
        )
        report["tasks"][task] = {
            "train_available": len(train_rows),
            "validation_available": validation_available,
            "evaluation_available": len(evaluation_rows),
            "selected": {
                **dict(train_assignments),
                **dict(validation_assignments),
                "stage2_discovery": int(counts["stage2_discovery"][task]),
                "stage2_discovery_by_source": discovery_sources[task],
                "stage4_final_eval": evaluation_count,
            },
        }

    for task, source_counts in discovery_sources.items():
        primary_source = TASK_TO_SOURCE[task]
        for source_key, requested_count in source_counts.items():
            if source_key == primary_source:
                continue
            source = config["sources"][source_key]
            split = str(source["official_split"])
            dataset = _load_source(source)
            rows = _ordered_unique(
                dataset,
                task=task,
                source=source,
                split=split,
                seed=seed,
            )
            count = int(requested_count)
            if len(rows) < count:
                raise RuntimeError(
                    f"{task} discovery source {source_key} needs {count} unique rows, "
                    f"got {len(rows)}"
                )
            _append_assignments(
                entries=entries,
                used_content_hashes=used_content_hashes,
                rows=rows,
                assignments=(("stage2_discovery", count),),
                task=task,
                source=source,
                split=split,
                seed=seed,
            )

    entries.sort(key=lambda row: (row["task"], row["assigned_split"], row["split_key"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(canonical_json(entry) + "\n")
    temporary.replace(manifest_path)
    leakage = audit_manifest(manifest_path)
    if leakage["status"] != "PASS":
        raise RuntimeError("Generated post-training manifest contains data leakage")
    report.update(
        {
            "status": "PASS",
            "manifest_path": str(manifest_path),
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "leakage_audit": leakage,
        }
    )
    (output_dir / "data_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(prepare(parse_args().config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
