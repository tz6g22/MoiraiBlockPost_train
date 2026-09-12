from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from src.common import load_yaml, sha256_file, sha256_json
from src.probe.extract_features import (
    CLASS_TO_TASK,
    apply_probe_overrides,
    validate_probe_config,
)


def save_final_probe_head(head: torch.nn.Linear, output_dir: Path) -> Path:
    path = output_dir / "probe_head.safetensors"
    save_file(
        {
            name: parameter.detach().cpu().contiguous()
            for name, parameter in head.state_dict().items()
        },
        path,
    )
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/probe.yaml")
    parser.add_argument("--base-checkpoint")
    parser.add_argument("--data-manifest")
    parser.add_argument("--partition-root")
    parser.add_argument("--query-root")
    parser.add_argument("--output-dir")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = apply_probe_overrides(load_yaml(args.config), args)
    validate_probe_config(config)
    run_config_hash = sha256_json(config)
    output_dir = Path(config["output_dir"])
    feature_manifest = json.loads(
        (output_dir / "feature_manifest.json").read_text(encoding="utf-8")
    )
    if feature_manifest.get("run_config_sha256") != run_config_hash:
        raise ValueError("Probe feature config hash mismatch")
    train_path = output_dir / "train_features.safetensors"
    val_path = output_dir / "validation_features.safetensors"
    if sha256_file(train_path) != feature_manifest["train_feature_sha256"]:
        raise ValueError("Probe train feature hash mismatch")
    if sha256_file(val_path) != feature_manifest["validation_feature_sha256"]:
        raise ValueError("Probe validation feature hash mismatch")
    train = load_file(train_path)
    validation = load_file(val_path)
    random.seed(int(config["seed"]))
    torch.manual_seed(int(config["seed"]))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    train_features = train["features"].to(device)
    train_labels = train["labels"].long().to(device)
    val_features = validation["features"].to(device)
    val_labels = validation["labels"].long().to(device)
    hidden_size = int(train_features.shape[1])
    if int(val_features.shape[1]) != hidden_size:
        raise ValueError("Probe train and validation feature widths differ")
    validate_probe_config(config, hidden_size=hidden_size)
    class_count = len(CLASS_TO_TASK)
    expected_labels = set(range(class_count))
    if set(train_labels.cpu().tolist()) != expected_labels:
        raise ValueError("Probe training features do not cover every configured class")
    if set(val_labels.cpu().tolist()) != expected_labels:
        raise ValueError("Probe validation features do not cover every configured class")
    head = torch.nn.Linear(hidden_size, class_count).to(device)
    initial_head = {
        name: parameter.detach().clone()
        for name, parameter in head.named_parameters()
    }
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    optimizer_steps = 0
    last_train_loss = None
    last_gradient_norm = None
    batch_size = int(config["batch_size"])
    for epoch in range(int(config["max_epochs"])):
        generator = torch.Generator(device="cpu").manual_seed(
            int(config["seed"]) + epoch
        )
        permutation = torch.randperm(
            train_features.shape[0],
            generator=generator,
        ).to(device)
        head.train()
        for start in range(0, len(permutation), batch_size):
            indices = permutation[start : start + batch_size]
            logits = head(train_features[indices])
            loss = F.cross_entropy(logits.float(), train_labels[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradients = [
                parameter.grad
                for parameter in head.parameters()
                if parameter.grad is not None
            ]
            if not gradients or not all(torch.isfinite(grad).all() for grad in gradients):
                raise FloatingPointError("Probe gradients are missing or contain NaN/Inf")
            gradient_norm = torch.linalg.vector_norm(
                torch.stack([grad.detach().float().norm() for grad in gradients])
            )
            if not torch.isfinite(gradient_norm) or float(gradient_norm) <= 0.0:
                raise FloatingPointError("Probe gradient norm must be positive and finite")
            optimizer.step()
            optimizer_steps += 1
            last_train_loss = float(loss.detach().cpu())
            last_gradient_norm = float(gradient_norm.detach().cpu())

    head.eval()
    with torch.no_grad():
        val_logits = head(val_features)
        accuracy = float((val_logits.argmax(dim=-1) == val_labels).float().mean())
        val_loss = float(F.cross_entropy(val_logits.float(), val_labels).cpu())
    if not torch.isfinite(val_logits).all():
        raise FloatingPointError("Probe validation logits contain NaN/Inf")
    save_final_probe_head(head, output_dir)
    changed_parameters = sorted(
        name
        for name, parameter in head.named_parameters()
        if not torch.equal(initial_head[name], parameter.detach())
    )
    if optimizer_steps <= 0 or not changed_parameters:
        raise RuntimeError("Probe training completed without a real parameter update")
    head_path = output_dir / "probe_head.safetensors"
    (output_dir / "probe_manifest.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "run_config_sha256": run_config_hash,
                "probe_head_sha256": sha256_file(head_path),
                "class_mapping": {str(index): task for index, task in CLASS_TO_TASK.items()},
                "confidence_threshold": float(config["confidence_threshold"]),
                "feature_config_sha256": feature_manifest["run_config_sha256"],
                "train_feature_sha256": feature_manifest["train_feature_sha256"],
                "validation_feature_sha256": feature_manifest["validation_feature_sha256"],
                "base_checkpoint_sha256": feature_manifest["base_checkpoint_sha256"],
                "partition_sha256": feature_manifest["probe_entry_partition_sha256"],
                "query_sha256": feature_manifest["probe_entry_query_sha256"],
                "validation_loss": val_loss,
                "validation_accuracy": accuracy,
                "trainable_parameters": ["weight", "bias"],
                "changed_parameters": changed_parameters,
                "optimizer_steps": optimizer_steps,
                "last_train_loss": last_train_loss,
                "last_gradient_norm": last_gradient_norm,
                "backbone_and_pseudo_query_frozen": True,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "feature_manifest.json").unlink()


if __name__ == "__main__":
    main()
