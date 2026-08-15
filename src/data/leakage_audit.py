from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


def audit_manifest(path: str | Path) -> dict:
    split_ids: dict[str, set[str]] = defaultdict(set)
    split_content: dict[str, set[str]] = defaultdict(set)
    duplicate_ids: list[dict[str, str]] = []
    duplicate_content: list[dict[str, str]] = []

    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            for key in {
                "stable_id",
                "dataset",
                "dataset_revision",
                "official_split",
                "assigned_split",
                "content_sha256",
            }:
                if key not in row:
                    raise ValueError(f"Manifest line {line_number} is missing {key}")
            split_name = row["assigned_split"]
            stable_id = row["stable_id"]
            content_hash = row["content_sha256"]
            for other_split, ids in split_ids.items():
                if other_split != split_name and stable_id in ids:
                    duplicate_ids.append(
                        {"stable_id": stable_id, "a": other_split, "b": split_name}
                    )
            for other_split, hashes in split_content.items():
                if other_split != split_name and content_hash in hashes:
                    duplicate_content.append(
                        {
                            "content_sha256": content_hash,
                            "a": other_split,
                            "b": split_name,
                        }
                    )
            split_ids[split_name].add(stable_id)
            split_content[split_name].add(content_hash)

    return {
        "status": "PASS" if not duplicate_ids and not duplicate_content else "FAILED",
        "id_intersections": duplicate_ids,
        "content_hash_intersections": duplicate_content,
        "split_unique_counts": {
            split_name: len(ids) for split_name, ids in sorted(split_ids.items())
        },
    }

