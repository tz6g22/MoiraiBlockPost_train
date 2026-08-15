from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from datasets import DatasetDict, load_from_disk
import torch


def nested_value(row: dict[str, Any], field: str) -> Any:
    value: Any = row
    for component in field.split("."):
        if not isinstance(value, dict) or component not in value:
            raise KeyError(f"Missing mapped field {field!r}")
        value = value[component]
    return value


def load_local_split(path: str | Path, official_split: str):
    dataset = load_from_disk(str(path))
    if isinstance(dataset, DatasetDict):
        if official_split not in dataset:
            raise KeyError(
                f"Local DatasetDict {path} does not contain split {official_split!r}"
            )
        return dataset[official_split]
    return dataset


def format_clutrr_prompt(
    row: dict,
    field_mapping: dict[str, Any] | None = None,
) -> str:
    mapping = field_mapping or {}
    story = str(nested_value(row, str(mapping.get("story", "story")))).strip()
    query = str(nested_value(row, str(mapping.get("query", "query")))).strip()
    if not story or not query:
        raise ValueError("CLUTRR row requires non-empty story and query fields")
    return (
        f"Story: {story}\n"
        f"Query: {query}\n"
        "Relationship:"
    )


TASK_TO_SOURCE = {
    "math": "gsm8k",
    "multihop": "clutrr",
}


def format_task_prompt(
    task: str,
    row: dict[str, Any],
    field_mapping: dict[str, Any],
) -> str:
    if task == "math":
        question = str(nested_value(row, str(field_mapping["question"]))).strip()
        return f"Question: {question}\nAnswer:"
    if task == "multihop":
        return format_clutrr_prompt(row, field_mapping)
    raise ValueError(f"Unknown task: {task}")


def format_task_target(
    task: str,
    row: dict[str, Any],
    field_mapping: dict[str, Any],
) -> str:
    if task == "math":
        target = str(nested_value(row, str(field_mapping["target"]))).strip()
    elif task == "multihop":
        target = str(nested_value(row, str(field_mapping["target"]))).strip()
    else:
        raise ValueError(f"Unknown task: {task}")
    if not target:
        raise ValueError(f"{task} target must be non-empty")
    return target


def canonical_content_sha256(
    dataset_name: str,
    row: dict[str, Any],
    field_mapping: dict[str, Any],
) -> str:
    """Hash fixed, dataset-native semantic fields, independent of stage formatting."""
    if dataset_name == "clutrr":
        payload = {
            "story": nested_value(row, str(field_mapping["story"])),
            "query": nested_value(row, str(field_mapping["query"])),
            "target": nested_value(row, str(field_mapping["target"])),
        }
    elif dataset_name in {"gsm8k", "svamp"}:
        payload = {
            "question": nested_value(row, str(field_mapping["question"])),
            "target": nested_value(row, str(field_mapping["target"])),
        }
    else:
        raise ValueError(f"Unsupported dataset for canonical content: {dataset_name}")
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TargetCausalExample:
    input_ids: torch.LongTensor
    labels: torch.LongTensor
    attention_mask: torch.LongTensor
    target_mask: torch.BoolTensor
    stable_id: str


@dataclass(frozen=True)
class PromptOnlyExample:
    input_ids: torch.LongTensor
    attention_mask: torch.LongTensor
    stable_id: str
    truncated: bool
    truncated_tokens: int


def _encode_multihop_prompt(
    tokenizer,
    row: dict[str, Any],
    field_mapping: dict[str, Any],
    *,
    token_budget: int,
) -> tuple[list[int], bool, int]:
    story = str(nested_value(row, str(field_mapping["story"]))).strip()
    query = str(nested_value(row, str(field_mapping["query"]))).strip()
    prefix = "Story: "
    context = story
    suffix = f"\nQuery: {query}\nRelationship:"
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    context_ids = tokenizer.encode(context, add_special_tokens=False)
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
    fixed = len(prefix_ids) + len(suffix_ids)
    if fixed > token_budget:
        raise ValueError("CLUTRR query plus relationship marker exceeds token budget")
    keep = token_budget - fixed
    truncated_tokens = max(0, len(context_ids) - keep)
    context_ids = context_ids[:keep]
    return prefix_ids + context_ids + suffix_ids, truncated_tokens > 0, truncated_tokens


def encode_prompt_only(
    tokenizer,
    *,
    task: str,
    row: dict[str, Any],
    field_mapping: dict[str, Any],
    stable_id: str,
    max_length: int = 2048,
) -> PromptOnlyExample:
    unstructured_math = (
        task == "math"
        and "question" not in field_mapping
        and bool(field_mapping.get("text_fields"))
    )
    if unstructured_math:
        raise ValueError(
            "Math Probe input is unavailable: the dataset schema does not "
            "separate the input from the answer or solution"
        )
    if task == "multihop":
        token_ids, truncated, truncated_tokens = _encode_multihop_prompt(
            tokenizer,
            row,
            field_mapping,
            token_budget=max_length,
        )
    else:
        token_ids = tokenizer.encode(
            format_task_prompt(task, row, field_mapping),
            add_special_tokens=False,
        )
        if len(token_ids) > max_length:
            raise ValueError(f"{task} prompt exceeds {max_length} without context to trim")
        truncated = False
        truncated_tokens = 0
    input_ids = torch.tensor(token_ids, dtype=torch.long)
    return PromptOnlyExample(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        stable_id=stable_id,
        truncated=truncated,
        truncated_tokens=truncated_tokens,
    )


def encode_prompt_target(
    tokenizer,
    *,
    task: str,
    row: dict[str, Any],
    field_mapping: dict[str, Any],
    stable_id: str,
    max_length: int = 2048,
) -> TargetCausalExample:
    target_ids = tokenizer.encode(
        format_task_target(task, row, field_mapping),
        add_special_tokens=False,
    ) + [tokenizer.eos_token_id]
    if len(target_ids) > max_length:
        raise ValueError(f"{task} target exceeds the fixed sequence length")
    prompt_budget = max_length - len(target_ids)
    if task == "multihop":
        prompt_ids, _, _ = _encode_multihop_prompt(
            tokenizer,
            row,
            field_mapping,
            token_budget=prompt_budget,
        )
    else:
        prompt_ids = tokenizer.encode(
            format_task_prompt(task, row, field_mapping),
            add_special_tokens=False,
        )
        if len(prompt_ids) > prompt_budget:
            raise ValueError(f"{task} question plus target exceeds the fixed sequence length")
    tokens = prompt_ids + target_ids
    input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
    labels = torch.tensor(tokens[1:], dtype=torch.long)
    target_mask = torch.zeros_like(labels, dtype=torch.bool)
    target_mask[max(0, len(prompt_ids) - 1) :] = True
    return TargetCausalExample(
        input_ids=input_ids,
        labels=labels,
        attention_mask=torch.ones_like(input_ids),
        target_mask=target_mask,
        stable_id=stable_id,
    )


def collate_target_examples(
    samples: Sequence[TargetCausalExample],
    *,
    pad_token_id: int,
) -> dict[str, torch.Tensor]:
    if not samples:
        raise ValueError("Cannot collate an empty target batch")
    width = max(sample.input_ids.numel() for sample in samples)
    batch = len(samples)
    input_ids = torch.full((batch, width), pad_token_id, dtype=torch.long)
    labels = torch.full((batch, width), -100, dtype=torch.long)
    attention_mask = torch.zeros((batch, width), dtype=torch.long)
    target_mask = torch.zeros((batch, width), dtype=torch.bool)
    for index, sample in enumerate(samples):
        length = sample.input_ids.numel()
        input_ids[index, :length] = sample.input_ids
        labels[index, :length] = sample.labels
        attention_mask[index, :length] = sample.attention_mask
        target_mask[index, :length] = sample.target_mask
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "target_mask": target_mask,
    }


def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Manifest line {line_number} is not an object")
            records.append(value)
    return records
