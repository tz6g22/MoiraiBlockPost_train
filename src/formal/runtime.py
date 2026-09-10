from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable

import torch


def alpha_parameter_names(model) -> tuple[str, ...]:
    names = tuple(sorted(name for name, _ in model.named_parameters() if "alpha" in name))
    if not names:
        raise ValueError("Formal model has no alpha parameters")
    return names


def trainability_audit(model) -> dict[str, object]:
    """Return the formal trainability split without silently freezing backbone."""
    alpha = set(alpha_parameter_names(model))
    query = {name for name, _ in model.named_parameters() if "pseudo_query" in name}
    if not query:
        raise ValueError("Formal model has no pseudo-query parameters")
    named = dict(model.named_parameters())
    backbone = set(named) - query - alpha
    for name in backbone | query | alpha:
        named[name].requires_grad_(True)
    if any(not named[name].requires_grad for name in backbone | query | alpha):
        raise RuntimeError("Formal backbone/query/alpha trainability audit failed")
    return {
        "backbone": tuple(sorted(backbone)),
        "query": tuple(sorted(query)),
        "alpha": tuple(sorted(alpha)),
        "partition_trainable": False,
    }


def assert_partition_unchanged(model, partition_sha256: str, before: str) -> None:
    """Fail fast if a caller tried to mutate the frozen task partition."""
    if partition_sha256 != before:
        raise RuntimeError("PARTITION_CHANGED_DURING_TRAINING")


def build_joint_optimizer(
    model,
    *,
    backbone_lr: float,
    attnres_lr: float,
    backbone_weight_decay: float,
    attnres_weight_decay: float = 0.0,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1.0e-8,
) -> torch.optim.Optimizer:
    audit = trainability_audit(model)
    named = dict(model.named_parameters())
    backbone = [named[name] for name in audit["backbone"]]
    attnres = [named[name] for name in (*audit["query"], *audit["alpha"])]
    if not backbone or not attnres:
        raise ValueError("Formal optimizer groups cannot be empty")
    return torch.optim.AdamW(
        [
            {"params": backbone, "lr": backbone_lr, "weight_decay": backbone_weight_decay},
            {"params": attnres, "lr": attnres_lr, "weight_decay": attnres_weight_decay},
        ],
        betas=betas,
        eps=eps,
    )


def _logit_diff(reference: torch.Tensor, converted: torch.Tensor) -> tuple[float, float]:
    delta = (reference.float() - converted.float()).abs()
    return float(delta.max().item()), float(delta.mean().item())


@torch.no_grad()
def identity_test(
    original_model,
    converted_model,
    input_ids: torch.LongTensor,
    attention_mask: torch.LongTensor,
) -> dict[str, float | str]:
    """Compare a native Qwen forward with the zero-alpha formal conversion."""
    original_model.eval()
    converted_model.eval()
    reference = original_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        logits_to_keep=0,
    ).logits
    converted = converted_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        logits_to_keep=0,
    ).logits
    max_abs, mean_abs = _logit_diff(reference, converted)
    if not torch.isfinite(converted).all():
        raise FloatingPointError("Converted alpha-zero logits contain NaN or Inf")
    return {
        "max_abs_logit_diff": max_abs,
        "mean_abs_logit_diff": mean_abs,
        "status": "PASS" if max_abs <= 5.0e-2 else "IDENTITY_CONVERSION_FAILED",
    }


def parameter_hash(model, names: Iterable[str] | None = None) -> str:
    selected = set(names) if names is not None else None
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        if selected is not None and name not in selected:
            continue
        tensor = parameter.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()
