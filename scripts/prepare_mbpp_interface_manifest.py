from __future__ import annotations

import argparse
from pathlib import Path

from transformers import AutoTokenizer

from src.common import load_yaml
from src.data.format_tasks import load_dataset_pool
from src.data.format_tasks import load_manifest
from scripts.mbpp_interface import write_interface_manifest


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_yaml(ROOT / "configs/data.yaml")
    source = config["validation_sources"]["code"]
    pool = load_dataset_pool(config, "mbpp")
    dataset, mapping = pool[str(source["official_split"])]
    tokenizer = AutoTokenizer.from_pretrained(
        ROOT / "outputs/base/qwen3_14b_full_attnres", local_files_only=True, use_fast=True
    )
    rows = write_interface_manifest(
        source_manifest=args.source_manifest,
        dataset=dataset,
        mapping=mapping,
        tokenizer=tokenizer,
        output=args.output,
    )
    print(f"interface manifest: {args.output}")
    print(f"cases: {len(rows)}")
    print(f"interface_parse_failed: {sum(int(row['interface_parse_failed']) for row in rows)}")


if __name__ == "__main__":
    main()
