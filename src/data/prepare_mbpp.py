from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, Features, Sequence, Value, load_dataset
from huggingface_hub import hf_hub_download

from src.common import sha256_file, sha256_json


REPO_ID = "google-research-datasets/mbpp"
REVISION = "4bb6404fdc6cacfda99d4ac4205087b89d32030c"
CONFIG_NAME = "full"
SPLIT_FILES = {
    "train": "full/train-00000-of-00001.parquet",
    "validation": "full/validation-00000-of-00001.parquet",
    "test": "full/test-00000-of-00001.parquet",
    "prompt": "full/prompt-00000-of-00001.parquet",
}
EXPECTED_TASK_IDS = {
    "prompt": range(1, 11),
    "test": range(11, 511),
    "validation": range(511, 601),
    "train": range(601, 975),
}
PROCESSED_FEATURES = Features(
    {
        "stable_id": Value("string"),
        "task_id": Value("int32"),
        "official_split": Value("string"),
        "prompt": Value("string"),
        "target": Value("string"),
        "test_list": Sequence(Value("string")),
        "test_setup_code": Value("string"),
        "challenge_test_list": Sequence(Value("string")),
    }
)


def _download_source(source_dir: Path) -> dict[str, Path]:
    source_dir.mkdir(parents=True, exist_ok=True)
    return {
        split: Path(
            hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                repo_type="dataset",
                revision=REVISION,
                local_dir=source_dir,
            )
        )
        for split, filename in SPLIT_FILES.items()
    }


def _validate_source(dataset: DatasetDict) -> None:
    if set(dataset) != set(EXPECTED_TASK_IDS):
        raise ValueError("MBPP source split names differ from the official split set")
    required_fields = {
        "task_id",
        "text",
        "code",
        "test_list",
        "test_setup_code",
        "challenge_test_list",
    }
    all_task_ids: list[int] = []
    for split, expected_range in EXPECTED_TASK_IDS.items():
        rows = dataset[split]
        if not required_fields.issubset(rows.column_names):
            missing = sorted(required_fields - set(rows.column_names))
            raise ValueError(f"Official MBPP {split} is missing fields: {missing}")
        task_ids = [int(value) for value in rows["task_id"]]
        if set(task_ids) != set(expected_range) or len(task_ids) != len(expected_range):
            raise ValueError(f"Official MBPP {split} task IDs differ from the published split")
        for row in rows:
            if not str(row["text"]).strip() or not str(row["code"]).strip():
                raise ValueError(f"Official MBPP {split} contains an empty prompt or solution")
            if not row["test_list"]:
                raise ValueError(f"Official MBPP {split} contains a task without tests")
        all_task_ids.extend(task_ids)
    if len(all_task_ids) != 974 or len(set(all_task_ids)) != 974:
        raise RuntimeError("Official MBPP task IDs are not 974 globally unique values")


def _preprocess_split(rows: Dataset, split: str) -> Dataset:
    processed = [
        {
            "stable_id": f"mbpp-full-{int(row['task_id']):04d}",
            "task_id": int(row["task_id"]),
            "official_split": split,
            "prompt": str(row["text"]),
            "target": str(row["code"]),
            "test_list": [str(value) for value in row["test_list"]],
            "test_setup_code": str(row["test_setup_code"]),
            "challenge_test_list": [
                str(value) for value in row["challenge_test_list"]
            ],
        }
        for row in rows
    ]
    return Dataset.from_list(processed, features=PROCESSED_FEATURES)


def _file_record(path: Path, root: Path) -> dict[str, Any]:
    path = path.resolve()
    root = root.resolve()
    return {
        "path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def prepare(source_dir: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing MBPP output: {output_dir}")
    source_files = _download_source(source_dir)
    raw = load_dataset(
        "parquet",
        data_files={split: str(path) for split, path in source_files.items()},
    )
    if not isinstance(raw, DatasetDict):
        raise TypeError("Official MBPP download did not produce a DatasetDict")
    _validate_source(raw)
    processed = DatasetDict(
        {
            split: _preprocess_split(raw[split], split)
            for split in SPLIT_FILES
        }
    )
    stable_ids = [
        stable_id
        for split in SPLIT_FILES
        for stable_id in processed[split]["stable_id"]
    ]
    if len(stable_ids) != 974 or len(set(stable_ids)) != 974:
        raise RuntimeError("Preprocessed MBPP stable IDs are not globally unique")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.incomplete-",
            dir=output_dir.parent,
        )
    )
    try:
        processed.save_to_disk(temporary)
        processed_files = sorted(path for path in temporary.rglob("*") if path.is_file())
        manifest = {
            "status": "PASS",
            "source_repo_id": REPO_ID,
            "source_revision": REVISION,
            "source_config": CONFIG_NAME,
            "source_dir": str(source_dir),
            "output_dir": str(output_dir),
            "rows": {split: len(processed[split]) for split in SPLIT_FILES},
            "total_rows": sum(len(processed[split]) for split in SPLIT_FILES),
            "stable_id_sha256": sha256_json(stable_ids),
            "field_mapping": {"prompt": "text", "target": "code"},
            "preserved_fields": [
                "test_list",
                "test_setup_code",
                "challenge_test_list",
            ],
            "source_files": {
                split: _file_record(path, source_dir)
                for split, path in source_files.items()
            },
            "processed_files": [
                _file_record(path, temporary) for path in processed_files
            ],
        }
        (temporary / "preprocessing_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("artifacts/data/mbpp_full_source"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/data/mbpp_full_preprocessed"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = prepare(args.source_dir, args.output_dir)
    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
