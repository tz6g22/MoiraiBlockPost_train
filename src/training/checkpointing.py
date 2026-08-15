from __future__ import annotations

from typing import Any

import torch

from src.common import sha256_json


def validate_post_training_base_manifest(manifest: dict[str, Any]) -> None:
    """Accept only the tutorial-defined converted Hugging Face Qwen3 base."""
    required = {
        "architecture": "Full AttnRes",
        "checkpoint_origin": "huggingface_post_training_bootstrap",
        "stage1_skipped": True,
        "source_repo_id": "Qwen/Qwen3-0.6B",
        "conversion": "copy_qwen3_backbone_and_zero_initialize_attnres_parameters",
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise ValueError(
                f"Post-training base manifest mismatch for {key}: "
                f"expected {expected!r}, got {manifest.get(key)!r}"
            )
    if not manifest.get("q_full_parameter_names") or not manifest.get(
        "q_full_sha256"
    ):
        raise ValueError("Post-training base manifest is missing query identity")


def pseudo_query_sha256(
    model,
    *,
    state_dict: dict[str, torch.Tensor] | None = None,
) -> tuple[list[str], str]:
    state = (
        state_dict
        if state_dict is not None
        else {
            name: parameter.detach()
            for name, parameter in model.named_parameters()
        }
    )
    query_names = sorted(name for name in state if "pseudo_query" in name)
    if not query_names:
        raise ValueError("Model state contains no pseudo-query parameters")
    query_payload = {
        name: state[name].detach().float().cpu().tolist()
        for name in query_names
    }
    return query_names, sha256_json(query_payload)
