from __future__ import annotations

from src.common import load_yaml, sha256_json
from src.data.format_tasks import load_manifest
from scripts.eval_comparison_100 import EVALUATION_CASES, select_evaluation_records


def test_comparison_sets_are_unique_and_outside_manifest() -> None:
    data_config = load_yaml("configs/data.yaml")
    manifest = load_manifest(data_config["output_dir"] + "/splits.json")
    manifest_ids = {record["stable_id"] for record in manifest}
    manifest_content = {record["content_sha256"] for record in manifest}
    hashes: dict[str, str] = {}
    for task in ("math", "multihop"):
        records, _dataset, _mapping = select_evaluation_records(
            task=task,
            data_config=data_config,
            manifest=manifest,
        )
        ids = [record["stable_id"] for record in records]
        content = [record["content_sha256"] for record in records]
        assert len(ids) == len(set(ids)) == EVALUATION_CASES
        assert len(content) == len(set(content)) == EVALUATION_CASES
        assert not set(ids) & manifest_ids
        assert not set(content) & manifest_content
        hashes[task] = sha256_json(ids)
    assert hashes["math"] != hashes["multihop"]
