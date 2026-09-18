from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import ssl
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

import certifi

# The Hub's Xet transport can stall on large parquet files in restricted runners.
# Plain HTTP preserves the exact repository revision and supports the same cache.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")

from datasets import (
    Dataset,
    DatasetDict,
    Features,
    Value,
    concatenate_datasets,
    load_dataset,
)
from huggingface_hub import hf_hub_download

from src.common import load_yaml
from src.data.format_tasks import load_local_split, nested_value


SOURCE_GROUPS = (
    "sources",
    "validation_sources",
    "probe_sources",
    "evaluation_sources",
    "external_sources",
)

MATH_SUBSETS = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)

CLUTRR_FIELDS = (
    "id",
    "story",
    "query",
    "target",
    "target_text",
    "clean_story",
    "proof_state",
    "f_comb",
    "task_name",
    "story_edges",
    "edge_types",
    "query_edge",
    "genders",
    "task_split",
)
CLUTRR_FEATURES = Features(
    {
        field: Value("int32" if field == "target" else "string")
        for field in CLUTRR_FIELDS
    }
)
MBPP_FIELDS = {
    "stable_id",
    "task_id",
    "official_split",
    "prompt",
    "target",
    "test_list",
    "test_setup_code",
    "challenge_test_list",
}


def _urlopen(request: urllib.request.Request):
    return urllib.request.urlopen(
        request,
        timeout=600,
        context=ssl.create_default_context(cafile=certifi.where()),
    )


def _required_fields(source: dict[str, Any]) -> list[str]:
    mapping = source["field_mapping"]
    return [
        str(mapping[key])
        for key in (
            "id",
            "question",
            "story",
            "query",
            "target",
            "context",
            "answer",
            "prompt",
            "messages",
            "test_list",
            "test_setup_code",
            "challenge_test_list",
        )
        if mapping.get(key)
    ]


def _validate_local_source(source: dict[str, Any]) -> int:
    dataset = load_local_split(source["local_path"], source["official_split"])
    if not len(dataset):
        raise RuntimeError(f"Downloaded dataset is empty: {source['local_path']}")
    row = dataset[0]
    for field in _required_fields(source):
        nested_value(row, field)
    if source["dataset_name"] == "clutrr":
        task_name = str(source.get("task_name", ""))
        if set(dataset.unique("task_name")) != {task_name}:
            raise RuntimeError(
                f"Local CLUTRR split contains rows outside {task_name}"
            )
        if dataset.features != CLUTRR_FEATURES:
            raise RuntimeError("Local CLUTRR schema does not match official HF schema")
    if source["dataset_name"] == "mbpp":
        if set(dataset.column_names) != MBPP_FIELDS:
            raise RuntimeError(
                "Local MBPP schema does not match the validated preprocessing schema"
            )
        if set(dataset.unique("official_split")) != {source["official_split"]}:
            raise RuntimeError("Local MBPP rows do not match their official split")
    expected_columns = source.get("expected_columns")
    if expected_columns is not None and set(dataset.column_names) != set(
        expected_columns
    ):
        raise RuntimeError(
            f"{source['dataset_name']} expected columns {sorted(expected_columns)}, "
            f"got {dataset.column_names}"
        )
    expected_rows = source.get("expected_rows")
    if expected_rows is not None and len(dataset) != int(expected_rows):
        raise RuntimeError(
            f"{source['dataset_name']} expected {expected_rows} rows, got {len(dataset)}"
        )
    return len(dataset)


def _download_svamp(source: dict[str, Any]) -> Dataset:
    request = urllib.request.Request(
        source["source_url"],
        headers={"User-Agent": "MoiraiBlock-post-training"},
    )
    with _urlopen(request) as response:
        payload = json.load(response)
    if not isinstance(payload, list):
        raise TypeError("Official SVAMP JSON must contain a list")
    rows = []
    for row in payload:
        body = str(row["Body"]).strip()
        question = str(row["Question"]).strip()
        rows.append(
            {
                "id": str(row["ID"]),
                "question": f"{body} {question}".strip(),
                "answer": str(row["Answer"]),
                "equation": str(row["Equation"]),
                "type": str(row["Type"]),
            }
        )
    return Dataset.from_list(rows)


def _download_clutrr(source: dict[str, Any]) -> DatasetDict:
    split_urls = source.get("split_urls")
    if not isinstance(split_urls, dict) or set(split_urls) != {
        "train",
        "validation",
        "test",
    }:
        raise ValueError("CLUTRR requires official train/validation/test URLs")
    task_name = str(source.get("task_name", ""))
    if task_name != "task_1.2":
        raise ValueError("CLUTRR Multihop source must be clean task_1.2")

    splits: dict[str, Dataset] = {}
    for split, url in split_urls.items():
        request = urllib.request.Request(
            str(url),
            headers={"User-Agent": "MoiraiBlock-post-training"},
        )
        with _urlopen(request) as response:
            reader = csv.DictReader(
                line.decode("utf-8-sig") for line in response
            )
            rows = []
            for source_row in reader:
                if source_row.get("task_name") != task_name:
                    continue
                missing = [field for field in CLUTRR_FIELDS if field not in source_row]
                if missing:
                    raise ValueError(
                        f"Official CLUTRR {split} row is missing fields: {missing}"
                    )
                row = {field: str(source_row[field]) for field in CLUTRR_FIELDS}
                row["target"] = int(row["target"])
                rows.append(row)
        if not rows:
            raise RuntimeError(f"Official CLUTRR {split} has no {task_name} rows")
        splits[split] = Dataset.from_list(rows, features=CLUTRR_FEATURES)
    return DatasetDict(splits)


def _download_math(source: dict[str, Any]) -> Dataset:
    subsets = tuple(source.get("subsets", MATH_SUBSETS))
    if subsets != MATH_SUBSETS:
        raise ValueError(
            "MATH source must include all seven Hendrycks MATH configurations"
        )
    datasets = [
        load_dataset(
            source["repo_id"],
            subset,
            split=source["official_split"],
            revision=source["revision"],
        )
        for subset in subsets
    ]
    return concatenate_datasets(datasets)


def _download_source(source: dict[str, Any]) -> int:
    destination = Path(source["local_path"])
    if destination.exists():
        return _validate_local_source(source)

    destination.parent.mkdir(parents=True, exist_ok=True)
    data_file = source.get("data_file")
    if source.get("source_url"):
        if source["dataset_name"] != "svamp":
            raise ValueError("Only the official SVAMP JSON source is supported")
        dataset = _download_svamp(source)
    elif source["dataset_name"] == "clutrr":
        dataset = _download_clutrr(source)
    elif source["dataset_name"] == "math":
        dataset = _download_math(source)
    elif data_file:
        filenames = [data_file] if isinstance(data_file, str) else list(data_file)
        if not filenames or not all(isinstance(filename, str) for filename in filenames):
            raise ValueError("data_file must be a non-empty filename or filename list")
        local_files = [
            hf_hub_download(
                repo_id=source["repo_id"],
                filename=filename,
                repo_type="dataset",
                revision=source["revision"],
            )
            for filename in filenames
        ]
        file_format = source.get("file_format")
        if file_format is None:
            file_format = "json" if str(data_file).endswith((".json", ".jsonl")) else "parquet"
        dataset = load_dataset(
            file_format,
            data_files={source["official_split"]: local_files},
            split=source["official_split"],
        )
    else:
        dataset = load_dataset(
            source["repo_id"],
            source["subset"],
            split=source["official_split"],
            revision=source["revision"],
        )
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.incomplete-",
            dir=destination.parent,
        )
    )
    try:
        dataset.save_to_disk(temporary)
        temporary.replace(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return _validate_local_source(source)


def prepare(config_path: str | Path) -> dict[str, Any]:
    config = load_yaml(config_path)
    unique_sources: dict[tuple[str, str, str, str, str, str, str], dict[str, Any]] = {}
    for group in SOURCE_GROUPS:
        for source in config.get(group, {}).values():
            required = {
                "repo_id",
                "subset",
                "revision",
                "official_split",
                "local_path",
                "field_mapping",
            }
            missing = sorted(required - source.keys())
            if missing:
                raise ValueError(f"Dataset source is missing keys: {missing}")
            key = (
                str(source["repo_id"]),
                str(source["subset"]),
                str(source["revision"]),
                str(source["official_split"]),
                str(source["local_path"]),
                json.dumps(source.get("data_file", ""), ensure_ascii=False, sort_keys=True),
                str(source.get("source_url", "")),
            )
            unique_sources[key] = source

    prepared = []
    for key, source in sorted(unique_sources.items()):
        row_count = _download_source(source)
        prepared.append(
            {
                "dataset_name": source.get("dataset_name"),
                "repo_id": key[0],
                "subset": key[1],
                "revision": key[2],
                "split": key[3],
                "local_path": key[4],
                "data_file": source.get("data_file"),
                "source_url": key[6] or None,
                "subsets": source.get("subsets"),
                "source_reference": source.get("source_reference"),
                "field_mapping": source.get("field_mapping"),
                "rows": row_count,
            }
        )

    report = {"status": "PASS", "datasets": prepared}
    output = Path("artifacts/data/source_manifest.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
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
