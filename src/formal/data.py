from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from src.common import sha256_file
from src.data.format_tasks import (
    PromptOnlyExample,
    TargetCausalExample,
    TASK_TO_SOURCE,
    encode_prompt_only,
    encode_prompt_target,
    load_dataset_pool,
    load_manifest,
)
from src.data.leakage_audit import normalize_stage_overlap_pairs
from src.data.source_provenance import (
    audit_formal_source_provenance,
    validate_manifest_row_identity,
)


def validate_formal_source_policy(
    formal_config: dict[str, Any],
    data_config: dict[str, Any],
    enabled_tasks: tuple[str, ...] | None = None,
) -> None:
    """Ensure the formal config's declared source policy is executable."""
    data = formal_config.get("data", {})
    tasks = tuple(
        enabled_tasks
        or formal_config.get("tasks", {}).get("enabled", ())
        or TASK_TO_SOURCE
    )
    policy = data.get("source_policy")
    if policy != "local_first_then_external_topup":
        raise ValueError(
            "Formal data source policy must be local_first_then_external_topup"
        )

    configured_keys = {
        str(key)
        for section_name in (
            "sources",
            "validation_sources",
            "probe_sources",
            "evaluation_sources",
        )
        for key in data_config.get(section_name, {})
    }
    missing_local = {
        task: tuple(
            str(source)
            for source in data.get(task, {}).get("local_sources", ())
            if str(source) not in configured_keys
        )
        for task in tasks
    }
    missing_local = {
        task: values for task, values in missing_local.items() if values
    }
    if missing_local:
        raise RuntimeError(
            "CONFIG_DECLARED_BUT_NOT_ENFORCED: formal local sources are not "
            f"registered in the data manifest config: {missing_local}"
        )

    external_sources = data_config.get("external_sources", {})
    missing_external = {
        task: tuple(
            str(source)
            for source in data.get(task, {}).get("external_topup_priority", ())
            if str(source) not in external_sources
        )
        for task in tasks
    }
    missing_external = {
        task: values for task, values in missing_external.items() if values
    }
    if missing_external:
        raise RuntimeError(
            "CONFIG_DECLARED_BUT_NOT_ENFORCED: external top-up sources are "
            f"not registered or executable: {missing_external}"
        )
    allowed_overlap_pairs = normalize_stage_overlap_pairs(
        data_config.get("allowed_cross_stage_reuse", ())
    )
    if allowed_overlap_pairs != frozenset(
        {frozenset(("stage2_discovery", "probe_train"))}
    ):
        raise ValueError(
            "Formal data may allow only stage2_discovery/probe_train reuse"
        )


def validate_training_source_mixture(
    records_by_task: dict[str, tuple[dict[str, Any], ...]],
    *,
    data_config: dict[str, Any],
    enabled_tasks: tuple[str, ...],
) -> None:
    """Require every configured task-local source to be available to the sampler."""
    weights_by_task = data_config.get("training_source_weights", {})
    source_registry = {
        key: source
        for section in ("sources", "external_sources")
        for key, source in data_config.get(section, {}).items()
    }
    for task in enabled_tasks:
        weights = weights_by_task.get(task)
        if not isinstance(weights, dict) or not weights:
            raise RuntimeError(f"DATA_SOURCE_MIX_MISMATCH: missing source weights for {task}")
        total = sum(float(value) for value in weights.values())
        if abs(total - 1.0) > 1.0e-8 or any(float(value) <= 0.0 for value in weights.values()):
            raise RuntimeError(f"DATA_SOURCE_MIX_MISMATCH: invalid source weights for {task}")
        expected_datasets = set()
        for source_key in weights:
            if source_key not in source_registry:
                raise RuntimeError(
                    f"DATA_SOURCE_MIX_MISMATCH: unregistered source {source_key} for {task}"
                )
            expected_datasets.add(str(source_registry[source_key]["dataset_name"]))
        actual_datasets = {
            str(record.get("dataset")) for record in records_by_task.get(task, ())
        }
        if actual_datasets != expected_datasets:
            raise RuntimeError(
                f"DATA_SOURCE_MIX_MISMATCH: {task} manifest sources "
                f"{sorted(actual_datasets)} != configured {sorted(expected_datasets)}"
            )


def load_formal_records(path: str | Path) -> list[dict[str, Any]]:
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Formal data manifest is missing: {manifest_path}")
    records = load_manifest(manifest_path)
    if not records:
        raise ValueError("Formal data manifest is empty")
    return records


def formal_data_manifest_sha256(path: str | Path) -> str:
    return sha256_file(path)


def records_by_task_and_stage(
    records: list[dict[str, Any]],
    *,
    stage: str,
    expected_counts: dict[str, int] | None = None,
    enabled_tasks: tuple[str, ...] | None = None,
) -> dict[str, tuple[dict[str, Any], ...]]:
    tasks = tuple(enabled_tasks or expected_counts or ())
    if not tasks:
        raise ValueError("records_by_task_and_stage requires enabled tasks")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        task = str(record.get("task", ""))
        if task in tasks and record.get("assigned_split") == stage:
            grouped[task].append(record)
    result: dict[str, tuple[dict[str, Any], ...]] = {}
    for task in tasks:
        values = sorted(grouped.get(task, []), key=lambda item: str(item["split_key"]))
        if expected_counts is not None and len(values) != int(expected_counts[task]):
            raise RuntimeError(
                f"{stage}/{task} requires {expected_counts[task]} records, "
                f"found {len(values)} in the local manifest"
            )
        result[task] = tuple(values)
    return result


def load_record_row(
    record: dict[str, Any],
    *,
    data_config: dict[str, Any],
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    dataset_name = str(record["dataset"])
    pool = load_dataset_pool(data_config, dataset_name)
    split = str(record["official_split"])
    if split not in pool:
        raise RuntimeError(
            f"Formal manifest references {dataset_name}/{split}, but no local pool exists"
        )
    dataset, field_mapping = pool[split]
    row_index = int(record["row_index"])
    if row_index < 0 or row_index >= len(dataset):
        raise RuntimeError(
            f"Formal manifest row_index {row_index} is outside "
            f"{dataset_name}/{split} with {len(dataset)} rows"
        )
    row = dataset[row_index]
    validate_manifest_row_identity(record, row=row, field_mapping=field_mapping)
    return row, field_mapping, record


def load_record_example(
    record: dict[str, Any],
    *,
    data_config: dict[str, Any],
    tokenizer,
    target: bool,
    max_length: int,
) -> PromptOnlyExample | TargetCausalExample:
    row, field_mapping, _ = load_record_row(record, data_config=data_config)
    common = {
        "tokenizer": tokenizer,
        "task": str(record["task"]),
        "row": row,
        "field_mapping": field_mapping,
        "stable_id": str(record["stable_id"]),
        "max_length": max_length,
    }
    return encode_prompt_target(**common) if target else encode_prompt_only(**common)


def build_formal_examples(
    records_by_task: dict[str, tuple[dict[str, Any], ...]],
    *,
    data_config: dict[str, Any],
    tokenizer,
    target: bool,
    max_length: int,
) -> dict[str, tuple[PromptOnlyExample | TargetCausalExample, ...]]:
    return {
        task: tuple(
            load_record_example(
                record,
                data_config=data_config,
                tokenizer=tokenizer,
                target=target,
                max_length=max_length,
            )
            for record in task_records
        )
        for task, task_records in records_by_task.items()
    }
