from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any, Iterable, Mapping, Sequence

import torch

from src.data.format_tasks import collate_target_examples
from src.distributed.fsdp_utils import (
    clip_grad_norm,
    named_parameters as distributed_named_parameters,
)
from src.formal.runtime import (
    assert_partition_unchanged,
    build_joint_optimizer,
    task_routing_parameter_names,
    trainability_audit,
)
from src.formal.task_banks import TaskBank


@dataclass(frozen=True)
class JointStepResult:
    loss: float
    backbone_grad_norm: float
    query_grad_norm: float
    alpha_grad_norm: float
    nonpadding_input_tokens: int
    target_tokens: int


def _group_norm(model, names: Iterable[str]) -> float:
    selected = set(names)
    values = [
        parameter.grad.detach().float().norm() ** 2
        for name, parameter in distributed_named_parameters(model)
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
    distributed_context=None,
) -> tuple[torch.optim.Optimizer, JointStepResult]:
    """Run one real full-parameter step for the active task bank.

    Other task banks remain CPU-owned and therefore cannot receive gradients or
    optimizer updates.  The partition is metadata on the model and has no
    trainable parameter.
    """
    bank.activate(model)
    partition_before = bank._model_partition(model).sha256
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
    bank_parameter_names = task_routing_parameter_names(model)
    bank.restore_optimizer_state(optimizer, model, bank_parameter_names)
    model.train()
    batch = collate_target_examples([example], pad_token_id=tokenizer.pad_token_id)
    target_mask = batch.pop("target_mask").to(device)
    labels = batch.pop("labels").to(device)
    labels = labels.masked_fill(~target_mask, -100)
    target_tokens = int((labels != -100).sum().item())
    if target_tokens <= 0:
        raise RuntimeError("Formal joint step has no target/EOS tokens")
    nonpadding_input_tokens = int(batch["attention_mask"].sum().item())
    if nonpadding_input_tokens <= 0:
        raise RuntimeError("Formal joint step has no non-padding input tokens")
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
    clip_grad_norm(
        model,
        model.parameters(),
        float(training_config["clipping"]["max_grad_norm"]),
    )
    optimizer.step()
    assert_partition_unchanged(model, bank.partition_sha256, partition_before)
    bank.capture(model, distributed_context=distributed_context)
    bank.capture_optimizer_state(optimizer, model, bank_parameter_names)
    return optimizer, JointStepResult(
        loss=float(loss.detach().cpu()),
        backbone_grad_norm=backbone_grad,
        query_grad_norm=query_grad,
        alpha_grad_norm=alpha_grad,
        nonpadding_input_tokens=nonpadding_input_tokens,
        target_tokens=target_tokens,
    )


class FormalTokenScheduler:
    """Cosine schedule whose progress is measured in actual input tokens."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        maximum_tokens: int,
        warmup_ratio: float,
        min_lr_ratio: float,
    ) -> None:
        if maximum_tokens <= 0:
            raise ValueError("maximum_tokens must be positive")
        if not 0.0 <= warmup_ratio < 1.0:
            raise ValueError("warmup_ratio must be in [0, 1)")
        if not 0.0 <= min_lr_ratio <= 1.0:
            raise ValueError("min_lr_ratio must be in [0, 1]")
        self.optimizer = optimizer
        self.maximum_tokens = int(maximum_tokens)
        self.warmup_tokens = int(self.maximum_tokens * warmup_ratio)
        self.min_lr_ratio = float(min_lr_ratio)
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        self.trained_tokens = 0
        self.step(0)

    def step(self, trained_tokens: int) -> None:
        self.trained_tokens = int(trained_tokens)
        if self.trained_tokens < self.warmup_tokens:
            ratio = self.trained_tokens / max(1, self.warmup_tokens)
        else:
            progress = min(
                1.0,
                (self.trained_tokens - self.warmup_tokens)
                / max(1, self.maximum_tokens - self.warmup_tokens),
            )
            ratio = self.min_lr_ratio + (1.0 - self.min_lr_ratio) * 0.5 * (
                1.0 + math.cos(math.pi * progress)
            )
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base_lr * ratio

    def state_dict(self) -> dict[str, Any]:
        return {
            "maximum_tokens": self.maximum_tokens,
            "warmup_tokens": self.warmup_tokens,
            "min_lr_ratio": self.min_lr_ratio,
            "base_lrs": self.base_lrs,
            "trained_tokens": self.trained_tokens,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        expected = {
            "maximum_tokens": self.maximum_tokens,
            "warmup_tokens": self.warmup_tokens,
            "min_lr_ratio": self.min_lr_ratio,
        }
        for key, value in expected.items():
            if state.get(key) != value:
                raise ValueError(f"Scheduler state mismatch for {key}")
        if list(state.get("base_lrs", [])) != self.base_lrs:
            raise ValueError("Scheduler base learning rates differ")
        self.step(int(state["trained_tokens"]))


def choose_next_task(
    consumed_tokens: Mapping[str, int],
    budgets: Mapping[str, int],
    *,
    task_order: Sequence[str],
) -> str | None:
    """Choose the least-complete task with deterministic config-order ties."""
    candidates = [
        task for task in task_order
        if consumed_tokens.get(task, 0) < budgets[task]
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda task: (
            consumed_tokens.get(task, 0) / max(1, budgets[task]),
            task_order.index(task),
        ),
    )


def train_token_budget_mixture(
    model,
    *,
    banks: Mapping[str, TaskBank],
    examples_by_task: Mapping[str, Sequence[Any]],
    tokenizer,
    training_config: dict[str, Any],
    token_budgets: Mapping[str, int],
    device: torch.device,
    max_steps: int | None = None,
    distributed_context=None,
    seed: int = 42,
    shuffle: bool = True,
) -> tuple[torch.optim.Optimizer, FormalTokenScheduler, dict[str, int], list[JointStepResult]]:
    """Run shared-backbone interleaved training until per-task token budgets."""
    batching = training_config.get("batching", {})
    if int(batching.get("micro_batch_size", 0)) != 1 or int(
        batching.get("gradient_accumulation_steps", 0)
    ) != 1:
        raise ValueError(
            "Formal token-budget mixture currently supports only 1 case per optimizer step"
        )
    task_order = tuple(banks)
    if set(task_order) != set(token_budgets) or set(task_order) != set(examples_by_task):
        raise ValueError("Mixture tasks, banks, budgets, and examples must match")
    if any(not examples_by_task[task] for task in task_order):
        raise ValueError("Every formal mixture task requires at least one example")
    maximum_tokens = sum(int(token_budgets[task]) for task in task_order)
    optimizer = build_joint_optimizer(
        model,
        backbone_lr=float(training_config["optimizer"]["parameter_groups"]["backbone"]["lr"]),
        attnres_lr=float(training_config["optimizer"]["parameter_groups"]["attnres"]["lr"]),
        backbone_weight_decay=float(training_config["optimizer"]["parameter_groups"]["backbone"]["weight_decay"]),
        attnres_weight_decay=float(training_config["optimizer"]["parameter_groups"]["attnres"]["weight_decay"]),
        betas=tuple(training_config["optimizer"]["betas"]),
        eps=float(training_config["optimizer"]["eps"]),
    )
    scheduler_config = training_config["scheduler"]
    scheduler = FormalTokenScheduler(
        optimizer,
        maximum_tokens=maximum_tokens,
        warmup_ratio=float(scheduler_config["warmup_ratio"]),
        min_lr_ratio=float(scheduler_config["min_lr_ratio"]),
    )
    consumed = {task: 0 for task in task_order}
    orders: dict[str, list[int]] = {}
    positions = {task: 0 for task in task_order}
    epochs = {task: 0 for task in task_order}
    for task_index, task in enumerate(task_order):
        orders[task] = list(range(len(examples_by_task[task])))
        if shuffle:
            random.Random(int(seed) + task_index).shuffle(orders[task])
    results: list[JointStepResult] = []
    while (task := choose_next_task(consumed, token_budgets, task_order=task_order)) is not None:
        inactive_before = {
            other: banks[other].state_hash()
            for other in task_order
            if other != task
        }
        if positions[task] == len(orders[task]):
            epochs[task] += 1
            positions[task] = 0
            orders[task] = list(range(len(examples_by_task[task])))
            if shuffle:
                task_index = task_order.index(task)
                random.Random(int(seed) + epochs[task] * len(task_order) + task_index).shuffle(
                    orders[task]
                )
        example = examples_by_task[task][orders[task][positions[task]]]
        positions[task] += 1
        optimizer, result = train_joint_step(
            model,
            bank=banks[task],
            example=example,
            tokenizer=tokenizer,
            training_config=training_config,
            optimizer=optimizer,
            device=device,
            distributed_context=distributed_context,
        )
        for other, before_hash in inactive_before.items():
            if banks[other].state_hash() != before_hash:
                raise RuntimeError(
                    f"INACTIVE_TASK_BANK_UPDATED: active={task}, inactive={other}"
                )
        consumed[task] += result.nonpadding_input_tokens
        scheduler.step(sum(consumed.values()))
        results.append(result)
        if max_steps is not None and len(results) >= max_steps:
            break
    return optimizer, scheduler, consumed, results
