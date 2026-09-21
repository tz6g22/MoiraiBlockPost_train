from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
import math
import random
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch
import torch.distributed as dist

from src.data.format_tasks import TargetCausalExample, collate_target_examples
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
    learning_rate_backbone: float
    learning_rate_query: float
    learning_rate_alpha: float
    alpha_parameter_norm: float
    alpha_parameter_delta: float
    query_parameter_norm: float
    query_parameter_delta: float
    query_optimizer_membership: bool
    query_dtype: str
    alpha_dtype: str
    query_optimizer_state_dtype: str
    alpha_optimizer_state_dtype: str


def _group_norm(
    model,
    names: Iterable[str],
    *,
    distributed_context=None,
) -> float:
    selected = set(names)
    values = [
        parameter.grad.detach().float().norm() ** 2
        for name, parameter in distributed_named_parameters(model)
        if name in selected and parameter.grad is not None
    ]
    if values:
        squared = torch.stack(values).sum()
    else:
        parameter = next(iter(model.parameters()))
        squared = torch.zeros((), device=parameter.device, dtype=torch.float32)
    if distributed_context is not None and distributed_context.distributed:
        dist.all_reduce(squared, op=dist.ReduceOp.SUM)
    return float(squared.sqrt().item())


def _state_norm(state: Mapping[str, torch.Tensor], contains: str) -> float:
    values = [value.detach().float().pow(2).sum() for name, value in state.items() if contains in name]
    return float(torch.stack(values).sum().sqrt().item()) if values else 0.0


def _state_delta(
    before: Mapping[str, torch.Tensor],
    after: Mapping[str, torch.Tensor],
    contains: str,
) -> float:
    values = [
        (after[name].detach().float() - before[name].detach().float()).pow(2).sum()
        for name in before
        if contains in name
    ]
    return float(torch.stack(values).sum().sqrt().item()) if values else 0.0


def _optimizer_contains(
    optimizer: torch.optim.Optimizer,
    model,
    names: Iterable[str],
) -> bool:
    selected_ids = {
        id(parameter)
        for name, parameter in distributed_named_parameters(model)
        if name in set(names)
    }
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    return bool(selected_ids) and selected_ids <= optimizer_ids


def _optimizer_group_lr(
    optimizer: torch.optim.Optimizer,
    name: str,
    fallback_index: int,
) -> float:
    for group in optimizer.param_groups:
        if group.get("name") == name:
            return float(group["lr"])
    if fallback_index < len(optimizer.param_groups):
        return float(optimizer.param_groups[fallback_index]["lr"])
    return 0.0


def _optimizer_state_dtype(
    optimizer: torch.optim.Optimizer,
    model,
    names: Iterable[str],
) -> str:
    selected = set(names)
    dtypes = {
        str(value.dtype)
        for name, parameter in distributed_named_parameters(model)
        if name in selected
        for key, value in optimizer.state.get(parameter, {}).items()
        if key in {"exp_avg", "exp_avg_sq"} and isinstance(value, torch.Tensor)
    }
    return ",".join(sorted(dtypes)) if dtypes else "unavailable"


def _source_cycle_spec(
    source_weights: Mapping[str, float],
) -> tuple[tuple[str, ...], dict[str, int]]:
    fractions = {
        source: Fraction(str(weight)).limit_denominator(1000)
        for source, weight in source_weights.items()
    }
    denominator = math.lcm(*(value.denominator for value in fractions.values()))
    counts = {
        source: value.numerator * (denominator // value.denominator)
        for source, value in fractions.items()
    }
    schedule = tuple(source for source, count in counts.items() for _ in range(count))
    if not schedule:
        raise ValueError("Formal source sampler has an empty cycle")
    return schedule, counts


def _source_example_for_step(
    *,
    task: str,
    task_step: int,
    task_index: int,
    source_examples: Mapping[str, Sequence[Any]],
    source_weights: Mapping[str, float],
    seed: int,
) -> Any:
    schedule_template, source_counts = _source_cycle_spec(source_weights)
    cycle_length = len(schedule_template)
    cycle_index, slot = divmod(int(task_step), cycle_length)
    cycle = list(schedule_template)
    random.Random(int(seed) + 1009 * task_index + cycle_index).shuffle(cycle)
    source = cycle[slot]
    occurrence = cycle_index * source_counts[source] + cycle[:slot].count(source)
    values = source_examples[source]
    epoch, position = divmod(occurrence, len(values))
    order = list(range(len(values)))
    random.Random(int(seed) + 1000003 * task_index + epoch).shuffle(order)
    return values[order[position]]


def _fit_example_to_token_budget(
    example: TargetCausalExample,
    remaining_tokens: int,
) -> TargetCausalExample:
    """Keep the final supervised tokens when the last step would overshoot."""
    if remaining_tokens <= 0:
        raise ValueError("Remaining token budget must be positive")
    input_tokens = int(example.attention_mask.sum().item())
    if input_tokens <= remaining_tokens:
        return example
    start = example.input_ids.numel() - remaining_tokens
    fitted = replace(
        example,
        input_ids=example.input_ids[start:],
        labels=example.labels[start:],
        attention_mask=example.attention_mask[start:],
        target_mask=example.target_mask[start:],
    )
    if not bool(fitted.target_mask.any()):
        raise RuntimeError("Final budget slice contains no supervised target tokens")
    return fitted


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
    query_before = {
        name: value.detach().cpu().clone()
        for name, value in bank.state.items()
        if "pseudo_query" in name
    }
    alpha_before = {
        name: value.detach().cpu().clone()
        for name, value in bank.state.items()
        if "alpha" in name
    }
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
    loss_sum = torch.nn.functional.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )
    loss = loss_sum / target_tokens
    if not torch.isfinite(loss):
        raise FloatingPointError("FORMAL_LOSS_NAN_OR_INF")
    reported_loss_sum = loss_sum.detach().float()
    reported_tokens = torch.tensor(
        float(target_tokens), device=reported_loss_sum.device, dtype=torch.float32
    )
    if distributed_context is not None and distributed_context.distributed:
        dist.all_reduce(reported_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(reported_tokens, op=dist.ReduceOp.SUM)
    reported_loss = reported_loss_sum / reported_tokens.clamp_min(1.0)
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
        raise FloatingPointError("FORMAL_GRADIENT_NAN_OR_INF")
    backbone_grad = _group_norm(
        model, audit["backbone"], distributed_context=distributed_context
    )
    query_grad = _group_norm(
        model, audit["query"], distributed_context=distributed_context
    )
    alpha_grad = _group_norm(
        model, audit["alpha"], distributed_context=distributed_context
    )
    clipped_norm = clip_grad_norm(
        model,
        model.parameters(),
        float(training_config["clipping"]["max_grad_norm"]),
    )
    if not torch.isfinite(clipped_norm):
        raise FloatingPointError("FORMAL_CLIPPED_GRADIENT_NAN_OR_INF")
    optimizer.step()
    assert_partition_unchanged(model, bank.partition_sha256, partition_before)
    bank.capture(model, distributed_context=distributed_context)
    bank.capture_optimizer_state(optimizer, model, bank_parameter_names)
    query_after = {
        name: value for name, value in bank.state.items() if "pseudo_query" in name
    }
    alpha_after = {name: value for name, value in bank.state.items() if "alpha" in name}
    query_parameters = [
        parameter
        for name, parameter in distributed_named_parameters(model)
        if name in audit["query"]
    ]
    alpha_parameters = [
        parameter
        for name, parameter in distributed_named_parameters(model)
        if name in audit["alpha"]
    ]
    return optimizer, JointStepResult(
        loss=float(reported_loss.cpu()),
        backbone_grad_norm=backbone_grad,
        query_grad_norm=query_grad,
        alpha_grad_norm=alpha_grad,
        nonpadding_input_tokens=nonpadding_input_tokens,
        target_tokens=target_tokens,
        learning_rate_backbone=_optimizer_group_lr(optimizer, "backbone", 0),
        learning_rate_query=_optimizer_group_lr(optimizer, "query", 1),
        learning_rate_alpha=(
            _optimizer_group_lr(optimizer, "alpha", 2)
            if audit["alpha"]
            else 0.0
        ),
        alpha_parameter_norm=_state_norm(alpha_after, "alpha"),
        alpha_parameter_delta=_state_delta(alpha_before, alpha_after, "alpha"),
        query_parameter_norm=_state_norm(query_after, "pseudo_query"),
        query_parameter_delta=_state_delta(query_before, query_after, "pseudo_query"),
        query_optimizer_membership=_optimizer_contains(optimizer, model, audit["query"]),
        query_dtype=str(query_parameters[0].dtype) if query_parameters else "unavailable",
        alpha_dtype=str(alpha_parameters[0].dtype) if alpha_parameters else "absent",
        query_optimizer_state_dtype=_optimizer_state_dtype(optimizer, model, audit["query"]),
        alpha_optimizer_state_dtype=(
            _optimizer_state_dtype(optimizer, model, audit["alpha"])
            if audit["alpha"]
            else "absent"
        ),
    )


class FormalTokenScheduler:
    """Global shared-parameter schedule plus task-local Q/Alpha schedules."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        maximum_tokens: int,
        warmup_ratio: float,
        min_lr_ratio: float,
        task_budgets: Mapping[str, int] | None = None,
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
        self.group_names = [group.get("name") for group in optimizer.param_groups]
        self.task_budgets = {
            str(task): int(budget)
            for task, budget in (task_budgets or {}).items()
        }
        if any(budget <= 0 for budget in self.task_budgets.values()):
            raise ValueError("Task-local scheduler budgets must be positive")
        self.task_local_enabled = bool(
            self.task_budgets
            and {"query", "alpha"}.issubset(set(self.group_names))
        )
        self.active_task: str | None = None
        self.task_trained_tokens = 0
        self.trained_tokens = 0
        self.step(0)
        if self.task_local_enabled:
            self._apply_task_ratio(0.0)

    @staticmethod
    def _ratio(
        trained_tokens: int,
        maximum_tokens: int,
        warmup_ratio: float,
        min_lr_ratio: float,
    ) -> float:
        warmup_tokens = int(maximum_tokens * warmup_ratio)
        if trained_tokens < warmup_tokens:
            return trained_tokens / max(1, warmup_tokens)
        progress = min(
            1.0,
            (trained_tokens - warmup_tokens)
            / max(1, maximum_tokens - warmup_tokens),
        )
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

    def _apply_task_ratio(self, ratio: float) -> None:
        for group, base_lr, name in zip(
            self.optimizer.param_groups, self.base_lrs, self.group_names
        ):
            if name in {"query", "alpha"}:
                group["lr"] = base_lr * ratio

    def activate_task(self, task: str, trained_tokens: int = 0) -> None:
        if not self.task_local_enabled:
            return
        if task not in self.task_budgets:
            raise ValueError(f"Unknown task-local scheduler task: {task}")
        if not 0 <= int(trained_tokens) <= self.task_budgets[task]:
            raise ValueError(f"Invalid task-local token position for {task}")
        self.active_task = task
        self.task_trained_tokens = int(trained_tokens)
        self._apply_task_ratio(
            self._ratio(
                self.task_trained_tokens,
                self.task_budgets[task],
                self.warmup_tokens / self.maximum_tokens,
                self.min_lr_ratio,
            )
        )

    def step_task(self, trained_tokens: int) -> None:
        if not self.task_local_enabled:
            return
        if self.active_task is None:
            raise RuntimeError("Task-local scheduler has no active task")
        budget = self.task_budgets[self.active_task]
        if not 0 <= int(trained_tokens) <= budget:
            raise ValueError(f"Invalid task-local token position for {self.active_task}")
        self.task_trained_tokens = int(trained_tokens)
        self._apply_task_ratio(
            self._ratio(
                self.task_trained_tokens,
                budget,
                self.warmup_tokens / self.maximum_tokens,
                self.min_lr_ratio,
            )
        )

    def step(self, trained_tokens: int) -> None:
        self.trained_tokens = int(trained_tokens)
        ratio = self._ratio(
            self.trained_tokens,
            self.maximum_tokens,
            self.warmup_tokens / self.maximum_tokens,
            self.min_lr_ratio,
        )
        for group, base_lr, name in zip(
            self.optimizer.param_groups, self.base_lrs, self.group_names
        ):
            if not self.task_local_enabled or name in {"backbone", "attnres"}:
                group["lr"] = base_lr * ratio

    def state_dict(self) -> dict[str, Any]:
        return {
            "maximum_tokens": self.maximum_tokens,
            "warmup_tokens": self.warmup_tokens,
            "min_lr_ratio": self.min_lr_ratio,
            "base_lrs": self.base_lrs,
            "trained_tokens": self.trained_tokens,
            "task_local": {
                "enabled": self.task_local_enabled,
                "task_budgets": self.task_budgets,
                "active_task": self.active_task,
                "trained_tokens": self.task_trained_tokens,
            },
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
        task_local = state.get("task_local", {})
        if self.task_local_enabled:
            if not task_local.get("enabled"):
                raise ValueError("Task-local scheduler state is missing")
            if task_local.get("task_budgets") != self.task_budgets:
                raise ValueError("Task-local scheduler budgets differ")
            active_task = task_local.get("active_task")
            if active_task is not None:
                self.activate_task(active_task, int(task_local.get("trained_tokens", 0)))


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


def _sequential_progress(
    *,
    task_order: Sequence[str],
    consumed: Mapping[str, int],
    task_steps: Mapping[str, int],
    current_task_index: int,
    global_step: int,
    task_positions: Mapping[str, int],
    task_epochs: Mapping[str, int],
    task_training_wall_seconds: Mapping[str, float] | None = None,
    global_training_wall_seconds: float = 0.0,
) -> dict[str, Any]:
    completed = list(task_order[:current_task_index])
    current_task = (
        task_order[current_task_index]
        if current_task_index < len(task_order)
        else None
    )
    consumed_copy = {task: int(consumed[task]) for task in task_order}
    return {
        "mode": "sequential_task_training",
        "task_order": list(task_order),
        "current_task_index": int(current_task_index),
        "current_task": current_task,
        "completed_tasks": completed,
        "global_step": int(global_step),
        "task_steps": {task: int(task_steps[task]) for task in task_order},
        "task_cumulative_tokens": consumed_copy,
        "global_cumulative_tokens": sum(consumed_copy.values()),
        "task_positions": {task: int(task_positions[task]) for task in task_order},
        "task_epochs": {task: int(task_epochs[task]) for task in task_order},
        "task_training_wall_seconds": {
            task: float((task_training_wall_seconds or {}).get(task, 0.0))
            for task in task_order
        },
        "global_training_wall_seconds": float(global_training_wall_seconds),
    }


def train_token_budget_sequential(
    model,
    *,
    banks: Mapping[str, TaskBank],
    examples_by_task: Mapping[str, Sequence[Any]],
    source_examples_by_task: Mapping[str, Mapping[str, Sequence[Any]]] | None = None,
    source_weights_by_task: Mapping[str, Mapping[str, float]] | None = None,
    tokenizer,
    training_config: dict[str, Any],
    token_budgets: Mapping[str, int],
    task_order: Sequence[str],
    device: torch.device,
    max_steps: int | None = None,
    distributed_context=None,
    seed: int = 42,
    shuffle: bool = True,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: FormalTokenScheduler | None = None,
    initial_progress: Mapping[str, Any] | None = None,
    on_step: Callable[[dict[str, Any]], None] | None = None,
    on_task_boundary: Callable[[dict[str, Any], torch.optim.Optimizer, FormalTokenScheduler], None] | None = None,
) -> tuple[
    torch.optim.Optimizer,
    FormalTokenScheduler,
    dict[str, int],
    list[JointStepResult],
    dict[str, Any],
]:
    """Train one shared backbone through complete task stages in config order."""
    batching = training_config.get("batching", {})
    if int(batching.get("micro_batch_size", 0)) != 1 or int(
        batching.get("gradient_accumulation_steps", 0)
    ) != 1:
        raise ValueError("Sequential formal training requires one case per optimizer step")
    order = tuple(str(task) for task in task_order)
    if not order or set(order) != set(banks) or set(order) != set(token_budgets):
        raise ValueError("Sequential tasks, banks, and token budgets must match")
    if set(order) != set(examples_by_task) or any(not examples_by_task[task] for task in order):
        raise ValueError("Every sequential task requires at least one example")
    if (source_examples_by_task is None) != (source_weights_by_task is None):
        raise ValueError("Source examples and source weights must be provided together")
    if source_examples_by_task is not None and source_weights_by_task is not None:
        if set(source_examples_by_task) != set(order) or set(source_weights_by_task) != set(order):
            raise ValueError("Source sampler tasks must match the sequential task order")
        for task in order:
            groups = source_examples_by_task[task]
            weights = source_weights_by_task[task]
            if set(groups) != set(weights) or any(not groups[source] for source in groups):
                raise ValueError(f"Source sampler groups do not match data for {task}")
            if abs(sum(float(value) for value in weights.values()) - 1.0) > 1.0e-8:
                raise ValueError(f"Source sampler weights must sum to one for {task}")

    if optimizer is None:
        optimizer = build_joint_optimizer(
            model,
            backbone_lr=float(training_config["optimizer"]["parameter_groups"]["backbone"]["lr"]),
            attnres_lr=float(training_config["optimizer"]["parameter_groups"]["attnres"]["lr"]),
            backbone_weight_decay=float(training_config["optimizer"]["parameter_groups"]["backbone"]["weight_decay"]),
            attnres_weight_decay=float(training_config["optimizer"]["parameter_groups"]["attnres"]["weight_decay"]),
            betas=tuple(training_config["optimizer"]["betas"]),
            eps=float(training_config["optimizer"]["eps"]),
        )
    if scheduler is None:
        scheduler_config = training_config["scheduler"]
        scheduler = FormalTokenScheduler(
            optimizer,
            maximum_tokens=sum(int(token_budgets[task]) for task in order),
            warmup_ratio=float(scheduler_config["warmup_ratio"]),
            min_lr_ratio=float(scheduler_config["min_lr_ratio"]),
            task_budgets=token_budgets,
        )

    progress = dict(initial_progress or {})
    consumed = {
        task: int(progress.get("task_cumulative_tokens", {}).get(task, 0))
        for task in order
    }
    task_steps = {
        task: int(progress.get("task_steps", {}).get(task, 0))
        for task in order
    }
    task_positions = {
        task: int(progress.get("task_positions", {}).get(task, 0))
        for task in order
    }
    task_epochs = {
        task: int(progress.get("task_epochs", {}).get(task, 0))
        for task in order
    }
    task_training_wall_seconds = {
        task: float(progress.get("task_training_wall_seconds", {}).get(task, 0.0))
        for task in order
    }
    global_training_wall_seconds = float(progress.get("global_training_wall_seconds", 0.0))
    global_step = int(progress.get("global_step", 0))
    start_index = int(progress.get("current_task_index", 0))
    if not 0 <= start_index <= len(order):
        raise ValueError("Invalid sequential current_task_index")

    orders: dict[str, list[int]] = {}
    for task_index, task in enumerate(order):
        orders[task] = list(range(len(examples_by_task[task])))
        if shuffle:
            random.Random(
                int(seed) + task_epochs[task] * len(order) + task_index
            ).shuffle(orders[task])

    results: list[JointStepResult] = []
    for task_index in range(start_index, len(order)):
        task = order[task_index]
        scheduler.activate_task(task, consumed[task])
        while consumed[task] < int(token_budgets[task]):
            inactive_before = {
                other: banks[other].state_hash()
                for other in order
                if other != task
            }
            if source_examples_by_task is not None and source_weights_by_task is not None:
                example = _source_example_for_step(
                    task=task,
                    task_step=task_steps[task],
                    task_index=task_index,
                    source_examples=source_examples_by_task[task],
                    source_weights=source_weights_by_task[task],
                    seed=seed,
                )
            else:
                if task_positions[task] == len(orders[task]):
                    task_epochs[task] += 1
                    task_positions[task] = 0
                    orders[task] = list(range(len(examples_by_task[task])))
                    if shuffle:
                        random.Random(
                            int(seed) + task_epochs[task] * len(order) + task_index
                        ).shuffle(orders[task])
                example = examples_by_task[task][orders[task][task_positions[task]]]
            example = _fit_example_to_token_budget(
                example,
                int(token_budgets[task]) - consumed[task],
            )
            task_positions[task] += 1
            step_started = time.perf_counter()
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
            step_wall_seconds = max(0.0, time.perf_counter() - step_started)
            for other, before_hash in inactive_before.items():
                if banks[other].state_hash() != before_hash:
                    raise RuntimeError(
                        f"INACTIVE_TASK_BANK_UPDATED: active={task}, inactive={other}"
                    )
            consumed[task] += result.nonpadding_input_tokens
            task_steps[task] += 1
            global_step += 1
            task_training_wall_seconds[task] += step_wall_seconds
            global_training_wall_seconds += step_wall_seconds
            scheduler.step(sum(consumed.values()))
            scheduler.step_task(consumed[task])
            results.append(result)
            current_progress = _sequential_progress(
                task_order=order,
                consumed=consumed,
                task_steps=task_steps,
                current_task_index=task_index,
                global_step=global_step,
                task_positions=task_positions,
                task_epochs=task_epochs,
                task_training_wall_seconds=task_training_wall_seconds,
                global_training_wall_seconds=global_training_wall_seconds,
            )
            current_progress["task_step"] = task_steps[task]
            if on_step is not None:
                on_step({
                    **current_progress,
                    "stage": "stage3_adapter_train",
                    "task": task,
                    "loss": result.loss,
                    "learning_rate_backbone": result.learning_rate_backbone,
                    "learning_rate_query": result.learning_rate_query,
                    "learning_rate_alpha": result.learning_rate_alpha,
                    "non_padding_tokens_this_step": result.nonpadding_input_tokens,
                    "target_tokens_this_step": result.target_tokens,
                    "alpha_parameter_norm": result.alpha_parameter_norm,
                    "alpha_parameter_delta": result.alpha_parameter_delta,
                    "alpha_grad_norm": result.alpha_grad_norm,
                    "query_parameter_norm": result.query_parameter_norm,
                    "query_parameter_delta": result.query_parameter_delta,
                    "query_grad_norm": result.query_grad_norm,
                    "query_optimizer_membership": result.query_optimizer_membership,
                    "query_dtype": result.query_dtype,
                    "alpha_dtype": result.alpha_dtype,
                    "query_optimizer_state_dtype": result.query_optimizer_state_dtype,
                    "alpha_optimizer_state_dtype": result.alpha_optimizer_state_dtype,
                    "train_wall_seconds": step_wall_seconds,
                    "tokens_per_second": result.nonpadding_input_tokens / max(step_wall_seconds, 1.0e-9),
                })
            if max_steps is not None and global_step >= max_steps:
                if consumed[task] >= int(token_budgets[task]):
                    completed_progress = _sequential_progress(
                        task_order=order,
                        consumed=consumed,
                        task_steps=task_steps,
                        current_task_index=task_index + 1,
                        global_step=global_step,
                        task_positions=task_positions,
                        task_epochs=task_epochs,
                        task_training_wall_seconds=task_training_wall_seconds,
                        global_training_wall_seconds=global_training_wall_seconds,
                    )
                    if on_task_boundary is not None:
                        on_task_boundary(completed_progress, optimizer, scheduler)
                    return (
                        optimizer,
                        scheduler,
                        consumed,
                        results,
                        completed_progress,
                    )
                return (
                    optimizer,
                    scheduler,
                    consumed,
                    results,
                    current_progress,
                )
        next_progress = _sequential_progress(
            task_order=order,
            consumed=consumed,
            task_steps=task_steps,
            current_task_index=task_index + 1,
            global_step=global_step,
            task_positions=task_positions,
            task_epochs=task_epochs,
            task_training_wall_seconds=task_training_wall_seconds,
            global_training_wall_seconds=global_training_wall_seconds,
        )
        if on_task_boundary is not None:
            if task_index + 1 < len(order):
                scheduler.activate_task(order[task_index + 1], consumed[order[task_index + 1]])
            on_task_boundary(next_progress, optimizer, scheduler)

    final_progress = _sequential_progress(
        task_order=order,
        consumed=consumed,
        task_steps=task_steps,
        current_task_index=len(order),
        global_step=global_step,
        task_positions=task_positions,
        task_epochs=task_epochs,
        task_training_wall_seconds=task_training_wall_seconds,
        global_training_wall_seconds=global_training_wall_seconds,
    )
    return optimizer, scheduler, consumed, results, final_progress
