from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from statistics import mean, median

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.math_validation import (
    _hash_lines,
    default_math_validation_manifest,
    evaluate_causal_lm,
    load_canonical_math_validation_examples,
)
from src.formal.conversion import convert_qwen3_checkpoint
from kimiattnres.modeling_qwen3_kimiattnres import convert_pretrained_qwen3


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/models/Qwen3-1.7B"
DATA_MANIFEST = ROOT / "outputs/formal_retrain/shared/data/splits.json"
DATA_CONFIG = ROOT / "configs/data_qwen3_1.7b.yaml"
CANONICAL = default_math_validation_manifest(ROOT)


def _load_main(path: Path):
    metadata = json.loads((path / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    blocks = metadata["partition_per_task"]["math"]["blocks"]
    partition = [int(block["length"]) for block in blocks]
    _original, model = convert_qwen3_checkpoint(
        BASE,
        partition=partition,
        task="math",
        min_block_length=1,
        max_block_length=max(partition),
        no_adjacent_singletons=False,
        alpha_init=0.0,
        use_alpha=False,
        dtype=torch.bfloat16,
        routing_dtype=torch.float32,
    )
    state = load_file(str(path / "shared_backbone.safetensors"), device="cpu")
    state.update(load_file(str(path / "math/query.safetensors"), device="cpu"))
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"MAIN_CHECKPOINT_LOAD_MISMATCH: {incompatible}")
    return model


def _load_kimi(path: Path, mode: str):
    block_sizes = [4, 4, 4, 4, 4, 4, 4] if mode == "block" else None
    _original, model, _conversion = convert_pretrained_qwen3(
        BASE,
        mode=mode,
        block_sizes=block_sizes,
        dtype=torch.bfloat16,
    )
    state = torch.load(path / "model_state.pt", map_location="cpu", weights_only=False)
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"KIMI_CHECKPOINT_LOAD_MISMATCH: {incompatible}")
    return model


def _summary(rows: list[dict[str, object]]) -> dict[str, float]:
    values = [float(row["loss"]) for row in rows]
    return {
        "mean": mean(values),
        "median": median(values),
        "min": min(values),
        "max": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="outputs/formal_retrain/shared/unified_math_validation_results.json",
    )
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True, use_fast=True)
    examples = load_canonical_math_validation_examples(
        manifest_path=CANONICAL,
        data_manifest_path=DATA_MANIFEST,
        data_config_path=DATA_CONFIG,
        tokenizer=tokenizer,
        max_length=4096,
        repo_root=ROOT,
    )
    manifest = json.loads(CANONICAL.read_text(encoding="utf-8"))
    stable_ids = [str(record["stable_id"]) for record in manifest["records"]]
    result: dict[str, object] = {
        "canonical_manifest": str(CANONICAL),
        "canonical_count": len(examples),
        "stable_id_sha256": _hash_lines(stable_ids),
        "tokenizer": str(BASE),
        "evaluator": "src.evaluation.math_validation.evaluate_causal_lm",
        "checkpoints": {},
    }
    checkpoints = {
        "main_no_alpha": ROOT / "outputs/formal_retrain/main_math_no_alpha_tau045_seed42/checkpoint",
        "kimi_block": ROOT / "outputs/formal_retrain/kimi_block_math_independent/final",
        "kimi_full": ROOT / "outputs/formal_retrain/kimi_full_independent/math/final",
    }
    for name, path in checkpoints.items():
        if not path.is_dir():
            raise FileNotFoundError(f"Missing checkpoint directory: {path}")
        if name == "main_no_alpha":
            model = _load_main(path)
        else:
            model = _load_kimi(path, "block" if name == "kimi_block" else "full")
        model.config.use_cache = False
        evaluation = evaluate_causal_lm(
            model,
            examples,
            pad_token_id=int(tokenizer.pad_token_id),
            device=torch.device("cpu"),
            return_per_example=True,
        )
        rows = evaluation["per_example"]
        result["checkpoints"][name] = {
            "path": str(path),
            "loss": float(evaluation["loss"]),
            "target_tokens": int(evaluation["tokens"]),
            "per_example": _summary(rows),
            "rows": rows,
        }
        del model
        gc.collect()

    main_rows = {
        str(row["stable_id"]): float(row["loss"])
        for row in result["checkpoints"]["main_no_alpha"]["rows"]
    }
    block_rows = {
        str(row["stable_id"]): float(row["loss"])
        for row in result["checkpoints"]["kimi_block"]["rows"]
    }
    differences = [main_rows[key] - block_rows[key] for key in stable_ids]
    result["paired_main_vs_kimi_block"] = {
        "mean_loss_difference": mean(differences),
        "median_loss_difference": median(differences),
        "main_lower_count": sum(main_rows[key] < block_rows[key] for key in stable_ids),
        "kimi_lower_count": sum(block_rows[key] < main_rows[key] for key in stable_ids),
        "tie_count": sum(main_rows[key] == block_rows[key] for key in stable_ids),
        "differences": [
            {
                "stable_id": key,
                "main_loss": main_rows[key],
                "kimi_block_loss": block_rows[key],
                "difference": main_rows[key] - block_rows[key],
            }
            for key in stable_ids
        ],
    }
    output = ROOT / args.output if not Path(args.output).is_absolute() else Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
