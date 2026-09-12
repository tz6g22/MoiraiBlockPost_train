from __future__ import annotations

from collections import defaultdict
from typing import Any

from src.data.format_tasks import (
    TASK_TO_SOURCE,
    canonical_content_sha256,
    canonical_stable_id,
)

FORMAL_TASKS = ("math", "multihop", "code")
FORMAL_STAGES = (
    "stage2_discovery",
    "stage3_adapter_train",
    "stage3_adapter_val",
    "probe_train",
    "probe_val",
    "stage4_final_eval",
)


def stage_source_sections(
    data_config: dict[str, Any],
    *,
    dataset_name: str,
    stage: str,
) -> tuple[str, ...]:
    """Return the configured source sections allowed to produce one stage."""
    if stage == "stage4_final_eval":
        return ("evaluation_sources",)
    if stage == "stage3_adapter_val":
        has_validation_source = any(
            source.get("dataset_name") == dataset_name
            for source in data_config.get("validation_sources", {}).values()
        )
        return ("validation_sources",) if has_validation_source else ("sources",)
    if stage in {"probe_train", "probe_val"}:
        return ("probe_sources", "sources")
    return ("sources",)


def task_source_datasets(
    data_config: dict[str, Any],
    *,
    task: str,
    stage: str,
) -> frozenset[str]:
    """Return dataset names legal for one task/stage in the formal manifest."""
    source_keys: tuple[str, ...]
    if stage == "stage2_discovery":
        source_keys = tuple(
            str(value)
            for value in data_config.get("discovery_sources", {}).get(task, {})
        )
    elif stage == "stage4_final_eval":
        source_keys = (task,)
        if task not in data_config.get("evaluation_sources", {}):
            source_keys = ()
    elif stage == "stage3_adapter_train":
        source_keys = tuple(
            str(value)
            for value in data_config.get("training_sources", {}).get(
                task,
                (TASK_TO_SOURCE[task],),
            )
        )
    elif stage == "stage3_adapter_val" and task in data_config.get(
        "validation_sources", {}
    ):
        source_keys = (task,)
    else:
        source_keys = (TASK_TO_SOURCE[task],)

    datasets: set[str] = set()
    for source_key in source_keys:
        source = data_config.get("sources", {}).get(source_key)
        if source is None:
            source = data_config.get("validation_sources", {}).get(source_key)
        if source is None:
            source = data_config.get("evaluation_sources", {}).get(source_key)
        if source is not None and source.get("dataset_name"):
            datasets.add(str(source["dataset_name"]))
    return frozenset(datasets)


def audit_formal_source_provenance(
    records: list[dict[str, Any]],
    *,
    data_config: dict[str, Any],
) -> dict[str, Any]:
    """Reject records whose assigned stage uses the wrong configured source."""
    configured: dict[str, set[tuple[str, str, str]]] = {}
    for section_name in (
        "sources",
        "validation_sources",
        "probe_sources",
        "evaluation_sources",
    ):
        configured[section_name] = {
            (
                str(source.get("dataset_name", "")),
                str(source.get("revision", "")),
                str(source.get("official_split", "")),
            )
            for source in data_config.get(section_name, {}).values()
        }

    checked = 0
    by_stage: dict[str, int] = defaultdict(int)
    for record in records:
        task = str(record.get("task", ""))
        stage = str(record.get("assigned_split", ""))
        if task not in FORMAL_TASKS:
            raise RuntimeError("Formal data manifest contains an unknown task")
        if stage not in FORMAL_STAGES:
            raise RuntimeError(
                f"FORMAL_DATA_SOURCE_PROVENANCE_MISMATCH: unknown stage {stage!r}"
            )
        source_key = (
            str(record.get("dataset", "")),
            str(record.get("dataset_revision", "")),
            str(record.get("official_split", "")),
        )
        allowed_task_datasets = task_source_datasets(
            data_config,
            task=task,
            stage=stage,
        )
        if allowed_task_datasets and source_key[0] not in allowed_task_datasets:
            raise RuntimeError(
                "FORMAL_DATA_TASK_SOURCE_MISMATCH: "
                f"{task}/{stage} record {record.get('stable_id')} uses dataset "
                f"{source_key[0]!r}, allowed datasets={sorted(allowed_task_datasets)}"
            )
        allowed_sections = stage_source_sections(
            data_config,
            dataset_name=source_key[0],
            stage=stage,
        )
        if not any(source_key in configured[section] for section in allowed_sections):
            raise RuntimeError(
                "FORMAL_DATA_SOURCE_PROVENANCE_MISMATCH: "
                f"{task}/{stage} record {record.get('stable_id')} uses "
                f"{source_key}, allowed sections={allowed_sections}"
            )
        checked += 1
        by_stage[stage] += 1
    return {
        "status": "PASS",
        "records_checked": checked,
        "records_by_stage": dict(sorted(by_stage.items())),
    }


def validate_manifest_row_identity(
    record: dict[str, Any],
    *,
    row: dict[str, Any],
    field_mapping: dict[str, Any],
) -> None:
    """Bind a manifest record to the exact row selected by its row index."""
    dataset = str(record["dataset"])
    split = str(record["official_split"])
    expected_stable_id = canonical_stable_id(dataset, split, row, field_mapping)
    if str(record.get("stable_id", "")) != expected_stable_id:
        raise RuntimeError(
            "FORMAL_DATA_ROW_IDENTITY_MISMATCH: stable_id does not match "
            f"{dataset}/{split}/row_index={record.get('row_index')}"
        )
    expected_content_hash = canonical_content_sha256(dataset, row, field_mapping)
    if str(record.get("content_sha256", "")) != expected_content_hash:
        raise RuntimeError(
            "FORMAL_DATA_ROW_IDENTITY_MISMATCH: content_sha256 does not match "
            f"{dataset}/{split}/row_index={record.get('row_index')}"
        )
