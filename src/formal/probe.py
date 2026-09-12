from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.distributed as dist
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from src.data.format_tasks import PromptOnlyExample
from src.formal.task_banks import TaskBank


FORMAL_TASKS = ("math", "multihop", "code")


def _collate_prompt_examples(
    examples: Sequence[PromptOnlyExample],
    *,
    pad_token_id: int,
) -> tuple[torch.LongTensor, torch.LongTensor]:
    if not examples:
        raise ValueError("Cannot collate an empty Probe batch")
    width = max(int(example.input_ids.numel()) for example in examples)
    input_ids = torch.full(
        (len(examples), width), pad_token_id, dtype=torch.long
    )
    attention_mask = torch.zeros((len(examples), width), dtype=torch.long)
    for index, example in enumerate(examples):
        length = int(example.input_ids.numel())
        input_ids[index, :length] = example.input_ids
        attention_mask[index, :length] = example.attention_mask
    return input_ids, attention_mask


@torch.no_grad()
def extract_formal_probe_features(
    model,
    *,
    bank: TaskBank,
    examples: Sequence[PromptOnlyExample],
    labels: Sequence[int],
    tokenizer,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, torch.LongTensor]:
    if len(examples) != len(labels) or not examples:
        raise ValueError("Probe examples and labels must be non-empty and aligned")
    model.eval()
    features: list[torch.Tensor] = []
    for start in range(0, len(examples), batch_size):
        batch_examples = examples[start : start + batch_size]
        input_ids, attention_mask = _collate_prompt_examples(
            batch_examples,
            pad_token_id=int(tokenizer.pad_token_id),
        )
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                probe_layer0_only=True,
            )
            hidden = output.probe_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        if not torch.isfinite(pooled).all():
            raise FloatingPointError("Formal Probe feature contains NaN or Inf")
        features.append(pooled.float().cpu())
        del hidden, pooled, output, input_ids, attention_mask
    return torch.cat(features, dim=0), torch.tensor(labels, dtype=torch.long)


def train_formal_probe(
    *,
    model,
    banks: dict[str, TaskBank],
    train_examples: Sequence[PromptOnlyExample],
    train_labels: Sequence[int],
    validation_examples: Sequence[PromptOnlyExample],
    validation_labels: Sequence[int],
    tokenizer,
    device: torch.device,
    output_dir: str | Path,
    config: dict[str, Any],
    checkpoint_manifest_sha256: str,
    data_manifest_sha256: str,
) -> dict[str, Any]:
    if set(banks) != set(FORMAL_TASKS):
        raise ValueError("Formal Probe requires all three task banks")
    for key in (
        "discard_hidden_after_routing",
        "discard_kv_cache_after_routing",
        "discard_all_probe_state",
    ):
        if config.get(key) is not True:
            raise ValueError(f"Formal Probe must enforce {key}")
    seed = int(config.get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)
    batch_size = int(config.get("batch_size", 32))
    feature_batch_size = int(config.get("feature_batch_size", 1))
    train_features, train_targets = extract_formal_probe_features(
        model,
        bank=banks["math"],
        examples=train_examples,
        labels=train_labels,
        tokenizer=tokenizer,
        device=device,
        batch_size=feature_batch_size,
    )
    validation_features, validation_targets = extract_formal_probe_features(
        model,
        bank=banks["math"],
        examples=validation_examples,
        labels=validation_labels,
        tokenizer=tokenizer,
        device=device,
        batch_size=feature_batch_size,
    )
    hidden_size = int(train_features.shape[1])
    if validation_features.shape[1] != hidden_size:
        raise ValueError("Formal Probe feature widths differ")
    if set(train_targets.tolist()) != set(range(3)):
        raise ValueError("Formal Probe training data must cover all three labels")
    if set(validation_targets.tolist()) != set(range(3)):
        raise ValueError("Formal Probe validation data must cover all three labels")

    head = torch.nn.Linear(hidden_size, 3).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=float(config.get("learning_rate", 1.0e-3)),
        weight_decay=float(config.get("weight_decay", 1.0e-4)),
    )
    max_epochs = int(config.get("max_epochs", 20))
    if max_epochs <= 0:
        raise ValueError("Formal Probe max_epochs must be positive")
    train_features = train_features.to(device)
    train_targets = train_targets.to(device)
    validation_features = validation_features.to(device)
    validation_targets = validation_targets.to(device)
    steps = 0
    for epoch in range(max_epochs):
        generator = torch.Generator(device="cpu").manual_seed(seed + epoch)
        permutation = torch.randperm(train_features.shape[0], generator=generator).to(device)
        head.train()
        for start in range(0, len(permutation), batch_size):
            indices = permutation[start : start + batch_size]
            logits = head(train_features[indices])
            loss = F.cross_entropy(logits.float(), train_targets[indices])
            if not torch.isfinite(loss):
                raise FloatingPointError("Formal Probe loss is NaN or Inf")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            steps += 1
    head.eval()
    with torch.no_grad():
        validation_logits = head(validation_features).float()
        validation_loss = F.cross_entropy(validation_logits, validation_targets)
        validation_accuracy = float(
            (validation_logits.argmax(dim=-1) == validation_targets).float().mean().item()
        )
    if not torch.isfinite(validation_logits).all() or not torch.isfinite(validation_loss):
        raise FloatingPointError("Formal Probe validation contains NaN or Inf")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    head_path = root / "probe_head.safetensors"
    is_rank0 = not dist.is_initialized() or dist.get_rank() == 0
    if is_rank0:
        save_file(
            {name: value.detach().cpu().contiguous() for name, value in head.state_dict().items()},
            head_path,
        )
    if dist.is_initialized():
        dist.barrier()
    manifest = {
        "status": "PASS",
        "checkpoint_manifest_sha256": checkpoint_manifest_sha256,
        "data_manifest_sha256": data_manifest_sha256,
        "class_mapping": {str(index): task for index, task in enumerate(FORMAL_TASKS)},
        "routing_rule": "argmax_three_way",
        "head_file": head_path.name,
        "head_sha256": _file_sha256(head_path),
        "hidden_size": hidden_size,
        "train_examples": len(train_examples),
        "validation_examples": len(validation_examples),
        "optimizer_steps": steps,
        "validation_loss": float(validation_loss.cpu()),
        "validation_accuracy": validation_accuracy,
        "discard_hidden_after_routing": True,
        "discard_kv_cache_after_routing": True,
        "discard_all_probe_state": True,
    }
    if is_rank0:
        (root / "probe_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if dist.is_initialized():
        dist.barrier()
        payload = [manifest if dist.get_rank() == 0 else None]
        dist.broadcast_object_list(payload, src=0)
        manifest = payload[0]
    return manifest


def _file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_formal_probe_head(
    output_dir: str | Path,
    *,
    hidden_size: int,
    expected_checkpoint_manifest_sha256: str,
    expected_data_manifest_sha256: str | None = None,
    expected_partition_sha256_per_task: dict[str, str] | None = None,
) -> torch.nn.Linear:
    manifest = load_formal_probe_manifest(
        output_dir,
        expected_checkpoint_manifest_sha256=expected_checkpoint_manifest_sha256,
        expected_data_manifest_sha256=expected_data_manifest_sha256,
        expected_partition_sha256_per_task=expected_partition_sha256_per_task,
    )
    root = Path(output_dir)
    head_path = root / manifest["head_file"]
    if _file_sha256(head_path) != manifest["head_sha256"]:
        raise ValueError("Formal Probe head hash mismatch")
    if int(manifest.get("hidden_size", -1)) != hidden_size:
        raise ValueError("Formal Probe hidden size mismatch")
    state = load_file(head_path, device="cpu")
    head = torch.nn.Linear(hidden_size, 3)
    head.load_state_dict(state, strict=True)
    return head


def load_formal_probe_manifest(
    output_dir: str | Path,
    *,
    expected_checkpoint_manifest_sha256: str,
    expected_data_manifest_sha256: str | None = None,
    expected_partition_sha256_per_task: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Load and validate the complete formal Probe provenance record."""
    root = Path(output_dir)
    manifest = json.loads((root / "probe_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "PASS":
        raise ValueError("Formal Probe manifest is missing a passing status")
    if manifest.get("checkpoint_manifest_sha256") != expected_checkpoint_manifest_sha256:
        raise ValueError("Formal Probe checkpoint identity mismatch")
    if expected_data_manifest_sha256 is not None and manifest.get("data_manifest_sha256") != expected_data_manifest_sha256:
        raise ValueError("Formal Probe data manifest identity mismatch")
    if manifest.get("class_mapping") != {"0": "math", "1": "multihop", "2": "code"}:
        raise ValueError("Formal Probe class mapping is not the formal three-way mapping")
    if manifest.get("routing_rule") != "argmax_three_way":
        raise ValueError("Formal Probe routing rule mismatch")
    if any("fixed" in str(key).lower() for key in manifest):
        raise ValueError("Formal Probe manifest contains a Fixed entry")
    for key in (
        "discard_hidden_after_routing",
        "discard_kv_cache_after_routing",
        "discard_all_probe_state",
    ):
        if manifest.get(key) is not True:
            raise ValueError(f"Formal Probe manifest does not enforce {key}")
    if expected_partition_sha256_per_task is not None:
        actual = manifest.get("partition_sha256_per_task")
        if actual != expected_partition_sha256_per_task:
            raise ValueError("Formal Probe partition provenance mismatch")
    return manifest
