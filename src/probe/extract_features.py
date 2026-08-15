from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer

from src.common import load_yaml, sha256_file, sha256_json, tokenizer_sha256
from src.data.format_tasks import (
    PromptOnlyExample,
    TASK_TO_SOURCE,
    load_local_split,
    load_manifest,
)
from src.data.streams import ManifestTaskRows
from src.modeling.config_bundle import MoiraiConfigBundle
from src.modeling.full_attnres import MoiraiQwen3ForCausalLM
from src.training.checkpointing import validate_post_training_base_manifest


CLASS_TO_TASK = {0: "math", 1: "multihop"}
TASK_TO_CLASS = {task: index for index, task in CLASS_TO_TASK.items()}


def _weight_hash(checkpoint: Path) -> str:
    files = sorted(checkpoint.glob("model*.safetensors"))
    if len(files) != 1:
        raise RuntimeError(f"Expected one base model weight file, found {files}")
    manifest = json.loads(
        (checkpoint / "checkpoint_manifest.json").read_text(encoding="utf-8")
    )
    actual = sha256_file(files[0])
    if actual != manifest.get("model_weights_sha256"):
        raise ValueError("Probe base checkpoint hash mismatch")
    validate_post_training_base_manifest(manifest)
    return actual


def validate_probe_config(config: dict[str, Any]) -> None:
    expected = {
        "seed": 42,
        "probe_entry_task": "math",
        "feature_site": "transformer_block_0_output",
        "pooling": "mask_aware_mean",
        "classifier": "Linear(1024,2)",
        "max_epochs": 20,
        "learning_rate": 1.0e-3,
        "weight_decay": 1.0e-4,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(
                f"Probe config mismatch for {key}: expected {value!r}, "
                f"got {config.get(key)!r}"
            )
    for key in ("feature_batch_size", "batch_size"):
        if int(config.get(key, 0)) <= 0:
            raise ValueError(f"Probe {key} must be positive")
    if config.get("classes") != CLASS_TO_TASK:
        raise ValueError("Probe classes must be exactly math/multihop")
    for key in {
        "base_checkpoint",
        "data_manifest",
        "data_config",
        "partition_root",
        "query_root",
        "output_dir",
    }:
        if key not in config:
            raise ValueError(f"Probe config is missing {key}")


def load_task_bundle(
    config: dict[str, Any],
    *,
    task: str,
    base_checkpoint_hash: str,
) -> MoiraiConfigBundle:
    return MoiraiConfigBundle.load(
        partition_path=Path(config["partition_root"]) / task / "partition.json",
        query_manifest_path=Path(config["query_root"]) / task / "query_manifest.json",
        expected_base_checkpoint_sha256=base_checkpoint_hash,
    )


def _prompt_examples(
    *,
    split_name: str,
    records: list[dict[str, Any]],
    data_config: dict[str, Any],
    tokenizer,
    expected_per_class: int | None = None,
) -> tuple[list[PromptOnlyExample], list[int]]:
    examples: list[PromptOnlyExample] = []
    labels: list[int] = []
    for task in CLASS_TO_TASK.values():
        if task == "math":
            source_key = "gsm8k_main_train"
            source = data_config["probe_sources"][source_key]
        else:
            source_key = TASK_TO_SOURCE[task]
            source = data_config["sources"][source_key]
            if split_name == "probe_val":
                source = data_config.get("validation_sources", {}).get(
                    task,
                    source,
                )
        dataset_name = str(source.get("dataset_name", source_key))
        selected = [
            record
            for record in records
            if record["dataset"] == dataset_name
            and record["task"] == task
            and record["assigned_split"] == split_name
        ]
        selected.sort(key=lambda record: record["split_key"])
        expected = (
            int(expected_per_class)
            if expected_per_class is not None
            else int(data_config["counts"][split_name])
        )
        if len(selected) != expected:
            raise RuntimeError(
                f"{task} {split_name} requires {expected} examples, found {len(selected)}"
            )
        dataset = load_local_split(source["local_path"], str(source["official_split"]))
        rows = ManifestTaskRows(
            task=task,
            records=tuple(selected),
            dataset=dataset,
            field_mapping=source["field_mapping"],
        )
        task_examples = rows.prompt_examples(tokenizer, max_length=2048)
        examples.extend(task_examples)
        labels.extend([TASK_TO_CLASS[task]] * len(task_examples))
    return examples, labels


def collate_prompt_examples(
    examples: list[PromptOnlyExample],
    *,
    pad_token_id: int,
) -> tuple[torch.LongTensor, torch.LongTensor]:
    if not examples:
        raise ValueError("Cannot collate an empty prompt batch")
    width = max(example.input_ids.numel() for example in examples)
    input_ids = torch.full(
        (len(examples), width),
        pad_token_id,
        dtype=torch.long,
    )
    attention_mask = torch.zeros((len(examples), width), dtype=torch.long)
    for index, example in enumerate(examples):
        length = example.input_ids.numel()
        input_ids[index, :length] = example.input_ids
        attention_mask[index, :length] = example.attention_mask
    return input_ids, attention_mask


@torch.no_grad()
def extract_split_features(
    model,
    examples: list[PromptOnlyExample],
    labels: list[int],
    *,
    tokenizer,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.LongTensor, list[dict[str, Any]]]:
    if len(examples) != len(labels):
        raise ValueError("Prompt examples and labels differ in length")
    features: list[torch.Tensor] = []
    metadata: list[dict[str, Any]] = []
    model.eval()
    for start in range(0, len(examples), batch_size):
        batch_examples = examples[start : start + batch_size]
        input_ids, attention_mask = collate_prompt_examples(
            batch_examples,
            pad_token_id=tokenizer.pad_token_id,
        )
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            shallow = model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                probe_layer0_only=True,
            ).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(shallow.dtype)
        pooled = (shallow * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        if not torch.isfinite(pooled).all():
            raise FloatingPointError("Probe feature contains NaN or Inf")
        features.append(pooled.float().cpu())
        metadata.extend(
            {
                "stable_id": example.stable_id,
                "truncated": example.truncated,
                "truncated_tokens": example.truncated_tokens,
            }
            for example in batch_examples
        )
        del shallow, pooled, input_ids, attention_mask
    return (
        torch.cat(features, dim=0),
        torch.tensor(labels, dtype=torch.long),
        metadata,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/probe.yaml")
    parser.add_argument("--base-checkpoint")
    parser.add_argument("--data-manifest")
    parser.add_argument("--partition-root")
    parser.add_argument("--query-root")
    parser.add_argument("--output-dir")
    return parser.parse_args()


def apply_probe_overrides(
    config: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    overrides = {
        "base_checkpoint": args.base_checkpoint,
        "data_manifest": args.data_manifest,
        "partition_root": args.partition_root,
        "query_root": args.query_root,
        "output_dir": args.output_dir,
    }
    for key, value in overrides.items():
        if value:
            config[key] = value
    return config


def main() -> None:
    args = parse_args()
    config = apply_probe_overrides(load_yaml(args.config), args)
    validate_probe_config(config)
    run_config_hash = sha256_json(config)
    checkpoint = Path(config["base_checkpoint"])
    base_hash = _weight_hash(checkpoint)
    probe_entry_bundle = load_task_bundle(
        config,
        task="math",
        base_checkpoint_hash=base_hash,
    )
    manifest_path = Path(config["data_manifest"])
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Probe data manifest is missing: {manifest_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        local_files_only=True,
        use_fast=True,
    )
    checkpoint_manifest = json.loads(
        (checkpoint / "checkpoint_manifest.json").read_text(encoding="utf-8")
    )
    if tokenizer_sha256(tokenizer) != checkpoint_manifest.get("tokenizer_sha256"):
        raise ValueError("Probe checkpoint tokenizer hash mismatch")
    records = load_manifest(manifest_path)
    data_config = load_yaml(config["data_config"])
    train_examples, train_labels = _prompt_examples(
        split_name="probe_train",
        records=records,
        data_config=data_config,
        tokenizer=tokenizer,
        expected_per_class=None,
    )
    validation_examples, validation_labels = _prompt_examples(
        split_name="probe_val",
        records=records,
        data_config=data_config,
        tokenizer=tokenizer,
        expected_per_class=None,
    )
    output_dir = Path(config["output_dir"])
    feature_manifest_path = output_dir / "feature_manifest.json"
    if not torch.cuda.is_available():
        raise RuntimeError("Formal Probe feature extraction requires CUDA")
    device = torch.device("cuda:0")
    model = MoiraiQwen3ForCausalLM.from_pretrained(
        checkpoint,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
    ).to(device)
    probe_entry_bundle.apply_to_model(model)
    if model.config.attnres_execution != "moirai" or model.config.moirai_task != "math":
        raise RuntimeError("Probe entry did not load Config_math")
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    train_features, train_label_tensor, _ = extract_split_features(
        model,
        train_examples,
        train_labels,
        tokenizer=tokenizer,
        batch_size=int(config["feature_batch_size"]),
        device=device,
    )
    val_features, val_label_tensor, _ = extract_split_features(
        model,
        validation_examples,
        validation_labels,
        tokenizer=tokenizer,
        batch_size=int(config["feature_batch_size"]),
        device=device,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    train_file = output_dir / "train_features.safetensors"
    val_file = output_dir / "validation_features.safetensors"
    save_file({"features": train_features, "labels": train_label_tensor}, train_file)
    save_file({"features": val_features, "labels": val_label_tensor}, val_file)
    feature_manifest_path.write_text(
        json.dumps(
            {
                "status": "PASS",
                "run_config_sha256": run_config_hash,
                "base_checkpoint_sha256": base_hash,
                "probe_entry_task": "math",
                "probe_entry_partition_sha256": probe_entry_bundle.partition.sha256,
                "probe_entry_query_sha256": probe_entry_bundle.query_sha256,
                "prompt_only": True,
                "train_feature_sha256": sha256_file(train_file),
                "validation_feature_sha256": sha256_file(val_file),
                "train_examples": len(train_examples),
                "validation_examples": len(validation_examples),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
