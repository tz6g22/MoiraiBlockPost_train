from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch

from src.data.format_tasks import collate_target_examples
from src.formal.runtime import build_joint_optimizer, trainability_audit
from src.formal.task_banks import TaskBank


@dataclass(frozen=True)
class JointStepResult:
    loss: float
    backbone_grad_norm: float
    query_grad_norm: float
    alpha_grad_norm: float


def _group_norm(model, names: Iterable[str]) -> float:
    selected = set(names)
    values = [
        parameter.grad.detach().float().norm() ** 2
        for name, parameter in model.named_parameters()
        if name in selected and parameter.grad is not None
    ]
    return float(torch.stack(values).sum().sqrt().item()) if values else 0.0


def train_joint_step(
    model,
    *,
    bank: TaskBank,
    example,
    tokenizer,
    training_config: dict[str, Any],
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device,
) -> tuple[torch.optim.Optimizer, JointStepResult]:
    """Run one real full-parameter step for the active task bank.

    Other task banks remain CPU-owned and therefore cannot receive gradients or
    optimizer updates.  The partition is metadata on the model and has no
    trainable parameter.
    """
    bank.activate(model)
    audit = trainability_audit(model)
    optimizer = optimizer or build_joint_optimizer(
        model,
        backbone_lr=float(training_config["optimizer"]["parameter_groups"]["backbone"]["lr"]),
        attnres_lr=float(training_config["optimizer"]["parameter_groups"]["attnres"]["lr"]),
        backbone_weight_decay=float(training_config["optimizer"]["parameter_groups"]["backbone"]["weight_decay"]),
        attnres_weight_decay=float(training_config["optimizer"]["parameter_groups"]["attnres"]["weight_decay"]),
        betas=tuple(training_config["optimizer"]["betas"]),
        eps=float(training_config["optimizer"]["eps"]),
    )
    model.train()
    batch = collate_target_examples([example], pad_token_id=tokenizer.pad_token_id)
    target_mask = batch.pop("target_mask").to(device)
    labels = batch.pop("labels").to(device)
    inputs = {key: value.to(device) for key, value in batch.items()}
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        logits = model(**inputs, labels=labels, use_cache=False, logits_to_keep=0).logits
    if logits.shape[:2] != labels.shape:
        raise ValueError("Formal joint training requires pre-shifted labels")
    loss = torch.nn.functional.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=-100,
    )
    loss.backward()
    backbone_grad = _group_norm(model, audit["backbone"])
    query_grad = _group_norm(model, audit["query"])
    alpha_grad = _group_norm(model, audit["alpha"])
    torch.nn.utils.clip_grad_norm_(model.parameters(), float(training_config["clipping"]["max_grad_norm"]))
    optimizer.step()
    bank.capture(model)
    return optimizer, JointStepResult(
        loss=float(loss.detach().cpu()),
        backbone_grad_norm=backbone_grad,
        query_grad_norm=query_grad,
        alpha_grad_norm=alpha_grad,
    )
