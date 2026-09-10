from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

from src.common import sha256_file, sha256_json
from src.formal.task_banks import TaskBank


def save_task_bank(
    output_dir: str | Path,
    *,
    bank: TaskBank,
    base_checkpoint_sha256: str,
    partition_sha256: str,
    data_manifest_sha256: str,
    optimizer_config: dict[str, Any],
    scheduler_config: dict[str, Any],
    consumed_tokens: int,
) -> dict[str, Any]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    query_state = {name: value for name, value in bank.state.items() if "pseudo_query" in name}
    alpha_state = {name: value for name, value in bank.state.items() if "alpha" in name}
    query_path = root / "query.safetensors"
    alpha_path = root / "alpha.safetensors"
    save_file(query_state, query_path)
    save_file(alpha_state, alpha_path)
    manifest = {
        "task": bank.task,
        "base_checkpoint_sha256": base_checkpoint_sha256,
        "partition_sha256": partition_sha256,
        "query_file": query_path.name,
        "query_sha256": sha256_file(query_path),
        "alpha_file": alpha_path.name,
        "alpha_sha256": sha256_file(alpha_path),
        "data_manifest_sha256": data_manifest_sha256,
        "optimizer_config": optimizer_config,
        "scheduler_config": scheduler_config,
        "consumed_tokens": int(consumed_tokens),
        "trainable_parameters": sorted(bank.state),
        "checkpoint_kind": "task_specific_q_alpha",
    }
    manifest["manifest_sha256"] = sha256_json(manifest)
    (root / "training_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
