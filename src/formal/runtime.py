from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from src.distributed.fsdp_utils import (
    DistributedContext,
    named_parameters as distributed_named_parameters,
    root_module,
    selected_parameter_sha256,
)
from src.modeling.partition import MoiraiPartition


def alpha_parameter_names(model) -> tuple[str, ...]:
    return tuple(
        sorted(name for name, _ in distributed_named_parameters(model) if "alpha" in name)
    )


def task_routing_parameter_names(model) -> tuple[str, ...]:
    """Return the task-specific Q/Alpha parameters for one task bank."""
    names = tuple(
        sorted(
            name
            for name, _ in distributed_named_parameters(model)
            if "pseudo_query" in name or "alpha" in name
        )
    )
    if not names:
        raise ValueError("Formal model has no routed Block AttnRes parameters")
    return names


def attnres_parameter_names(model) -> tuple[str, ...]:
    """Return all shared and task-specific parameters introduced by AttnRes."""
    names = tuple(
        sorted(
            name
            for name, _ in distributed_named_parameters(model)
            if (
                "pseudo_query" in name
                or "alpha" in name
                or "key_norm" in name
            )
        )
    )
    if not names:
        raise ValueError("Formal model has no AttnRes parameters")
    return names


def trainability_audit(model) -> dict[str, object]:
    """Return the formal trainability split without silently freezing backbone."""
    alpha = set(alpha_parameter_names(model))
    query = {
        name for name, _ in distributed_named_parameters(model) if "pseudo_query" in name
    }
    if not query:
        raise ValueError("Formal model has no pseudo-query parameters")
    named = dict(distributed_named_parameters(model))
    attnres = set(attnres_parameter_names(model))
    backbone = set(named) - attnres
    for name in backbone | attnres:
        named[name].requires_grad_(True)
    if any(not named[name].requires_grad for name in backbone | attnres):
        raise RuntimeError("Formal backbone/query/alpha trainability audit failed")
    return {
        "backbone": tuple(sorted(backbone)),
        "query": tuple(sorted(query)),
        "alpha": tuple(sorted(alpha)),
        "attnres": tuple(sorted(attnres)),
        "partition_trainable": False,
    }


def assert_partition_unchanged(model, partition_sha256: str, before: str) -> None:
    """Fail fast if a caller tried to mutate the frozen task partition."""
    config = root_module(model).config
    current = MoiraiPartition.from_lengths(
        config.moirai_partition,
        task=config.moirai_task,
        num_transformer_blocks=int(config.num_hidden_layers),
        min_length=int(getattr(config, "moirai_min_block_length", 1)),
        max_length=int(getattr(config, "moirai_max_block_length", 4)),
        no_adjacent_singletons=bool(
            getattr(config, "moirai_no_adjacent_singletons", True)
        ),
    ).sha256
    if partition_sha256 != before or current != before:
        raise RuntimeError("PARTITION_CHANGED_DURING_TRAINING")


def build_joint_optimizer(
    model,
    *,
    backbone_lr: float,
    attnres_lr: float,
    backbone_weight_decay: float,
    attnres_weight_decay: float = 0.0,
    query_lr: float | None = None,
    alpha_lr: float | None = None,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1.0e-8,
) -> torch.optim.Optimizer:
    audit = trainability_audit(model)
    named = dict(distributed_named_parameters(model))
    backbone = [named[name] for name in audit["backbone"]]
    attnres = [named[name] for name in audit["attnres"]]
    if not backbone or not attnres:
        raise ValueError("Formal optimizer groups cannot be empty")
    if query_lr is None and alpha_lr is not None:
        raise ValueError("alpha_lr cannot be provided without query_lr")
    if query_lr is None:
        groups = [
            {"name": "backbone", "params": backbone, "lr": backbone_lr, "weight_decay": backbone_weight_decay},
            {"name": "attnres", "params": attnres, "lr": attnres_lr, "weight_decay": attnres_weight_decay},
        ]
    else:
        query_names = set(audit["query"])
        alpha_names = set(audit["alpha"])
        query = [named[name] for name in sorted(query_names)]
        alpha = [named[name] for name in sorted(alpha_names)]
        other_attnres = [
            named[name]
            for name in audit["attnres"]
            if name not in query_names and name not in alpha_names
        ]
        if not query:
            raise ValueError("Formal query optimizer group cannot be empty")
        groups = [
            {"name": "backbone", "params": backbone, "lr": backbone_lr, "weight_decay": backbone_weight_decay},
            {"name": "query", "params": query, "lr": query_lr, "weight_decay": attnres_weight_decay},
        ]
        if alpha:
            if alpha_lr is None:
                raise ValueError("alpha_lr is required when Alpha parameters exist")
            groups.append(
                {"name": "alpha", "params": alpha, "lr": alpha_lr, "weight_decay": attnres_weight_decay}
            )
        if other_attnres:
            groups.append(
                {
                    "name": "attnres",
                    "params": other_attnres,
                    "lr": attnres_lr,
                    "weight_decay": attnres_weight_decay,
                }
            )
    return torch.optim.AdamW(
        groups,
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
    if getattr(converted_model.config, "attnres_execution", None) != "formal":
        raise RuntimeError("Identity test must execute the formal converted runtime")
    if not getattr(converted_model.config, "moirai_partition", None):
        raise RuntimeError("Identity test requires a converted task partition")
    query_names = tuple(
        name for name, _ in converted_model.named_parameters() if "pseudo_query" in name
    )
    alpha_names = tuple(
        name for name, _ in converted_model.named_parameters() if "alpha" in name
    )
    if not query_names:
        raise RuntimeError("Identity test requires converted pseudo-query parameters")
    if not alpha_names:
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
            raise FloatingPointError("Converted no-alpha logits contain NaN or Inf")
    return {
            "max_abs_logit_diff": max_abs,
            "mean_abs_logit_diff": mean_abs,
            "num_examples": int(input_ids.shape[0]),
            "identity_preserving": False,
            "mode": "no_alpha_perturbation",
            "status": "PASS",
        }
    saved_alpha = {
        name: converted_model.get_parameter(name).detach().clone()
        for name in alpha_names
    }
    with torch.no_grad():
        for name in alpha_names:
            converted_model.get_parameter(name).zero_()
    original_model.eval()
    converted_model.eval()
    try:
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
    finally:
        with torch.no_grad():
            for name, value in saved_alpha.items():
                converted_model.get_parameter(name).copy_(value)
    return {
        "max_abs_logit_diff": max_abs,
        "mean_abs_logit_diff": mean_abs,
        "num_examples": int(input_ids.shape[0]),
        "status": "PASS" if max_abs <= 5.0e-2 else "IDENTITY_CONVERSION_FAILED",
    }


def parameter_hash(
    model,
    names: Iterable[str] | None = None,
    *,
    distributed_context: DistributedContext | None = None,
) -> str:
    if isinstance(model, FSDP):
        if distributed_context is None:
            if not dist.is_initialized():
                raise RuntimeError("FSDP parameter hashing requires an initialized process group")
            distributed_context = DistributedContext(
                rank=dist.get_rank(),
                local_rank=int(torch.cuda.current_device()),
                world_size=dist.get_world_size(),
                device=torch.device("cuda", torch.cuda.current_device()),
            )
        selected = set(names) if names is not None else None
        return selected_parameter_sha256(
            model,
            distributed_context,
            lambda name, _parameter: selected is None or name in selected,
        )
    selected = set(names) if names is not None else None
    digest = hashlib.sha256()
    for name, parameter in sorted(distributed_named_parameters(model)):
        if selected is not None and name not in selected:
            continue
        tensor = parameter.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()
