from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.math_validation import build_canonical_math_validation_manifest


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-manifest", default="outputs/formal_retrain/shared/data/splits.json")
    parser.add_argument("--data-config", default="configs/data_qwen3_1.7b.yaml")
    parser.add_argument("--output", default="outputs/formal_retrain/shared/math_validation_manifest.json")
    parser.add_argument("--checkpoint", default=os.environ.get("QWEN3_1_7B_PATH", "artifacts/models/Qwen3-1.7B"))
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        (root / args.checkpoint).resolve() if not Path(args.checkpoint).is_absolute() else args.checkpoint,
        local_files_only=True,
        use_fast=True,
    )
    path = build_canonical_math_validation_manifest(
        shared_manifest_path=args.data_manifest,
        data_config_path=args.data_config,
        tokenizer=tokenizer,
        output_path=args.output,
        repo_root=root,
    )
    print(path)


if __name__ == "__main__":
    main()
