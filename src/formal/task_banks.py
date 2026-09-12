from __future__ import annotations

import hashlib
from copy import deepcopy
from dataclasses import dataclass, field

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from src.distributed.fsdp_utils import (
    broadcast_object,
    load_selected_parameter_state,
    named_parameters as distributed_named_parameters,
    selected_parameter_state,
)
from src.modeling.partition import MoiraiPartition


def _is_task_routing_parameter(name: str) -> bool:
    # The formal task-specific bundle is exactly (P_t, Q_t, Alpha_t).  Key
    # normalization is a shared AttnRes parameter and must not be copied into
    # or restored from an individual task bank.
    return "pseudo_query" in name or "alpha" in name


@dataclass
class TaskBank:
    """CPU-owned task routing state and frozen partition metadata."""

    task: str
    state: dict[str, torch.Tensor]
    partition_sha256: str
    partition_lengths: tuple[int, ...] = field(default_factory=tuple)
    optimizer_state: dict[str, dict[str, object]] = field(default_factory=dict)

    def state_hash(self, contains: str | None = None) -> str:
        digest = hashlib.sha256()
        for name, value in sorted(self.state.items()):
            if contains is not None and contains not in name:
                continue
            tensor = value.detach().cpu().contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    @staticmethod
    def _model_partition(model) -> MoiraiPartition:
        config = model.module.config if isinstance(model, FSDP) else model.config
        lengths = getattr(config, "moirai_partition", None)
        task = getattr(config, "moirai_task", None)
        if not lengths or not task:
            raise ValueError("Task bank activation requires an assigned model partition")
        return MoiraiPartition.from_lengths(
            lengths,
            task=task,
            num_transformer_blocks=int(config.num_hidden_layers),
            min_length=int(getattr(config, "moirai_min_block_length", 1)),
            max_length=int(getattr(config, "moirai_max_block_length", 4)),
            no_adjacent_singletons=bool(
                getattr(config, "moirai_no_adjacent_singletons", True)
            ),
        )

    @classmethod
    def from_model(cls, model, *, task: str, partition_sha256: str) -> "TaskBank":
        state = {
            name: parameter.detach().cpu().clone()
            for name, parameter in distributed_named_parameters(model)
            if _is_task_routing_parameter(name)
        }
        if not state:
            raise ValueError("Task bank cannot be created without Q/Alpha parameters")
        lengths = tuple(int(value) for value in model.config.moirai_partition or ())
        if not lengths:
            raise ValueError("Task bank requires frozen partition lengths")
        return cls(
            task=task,
            state=state,
            partition_sha256=partition_sha256,
            partition_lengths=lengths,
        )

    def activate(self, model) -> None:
        config = model.module.config if isinstance(model, FSDP) else model.config
        if getattr(config, "attnres_execution", None) != "formal":
            raise ValueError("Task banks can only activate on the formal runtime")
        if self.partition_lengths:
            config.moirai_partition = list(self.partition_lengths)
            config.moirai_task = self.task
        partition = self._model_partition(model)
        if partition.task != self.task or partition.sha256 != self.partition_sha256:
            raise ValueError(
                f"Task bank {self.task} does not match the active partition"
            )
        named = dict(distributed_named_parameters(model))
        if set(self.state) != {
            name for name in named if _is_task_routing_parameter(name)
        }:
            raise ValueError(f"Task bank {self.task} does not match the model Q/Alpha sites")
        if isinstance(model, FSDP):
            load_selected_parameter_state(
                model,
                self.state,
                lambda name, _parameter: _is_task_routing_parameter(name),
            )
            return
        with torch.no_grad():
            for name, value in self.state.items():
                named[name].copy_(value.to(device=named[name].device, dtype=named[name].dtype))

    def capture(self, model, *, distributed_context=None) -> None:
        if isinstance(model, FSDP):
            if distributed_context is None:
                raise ValueError("FSDP task-bank capture requires distributed context")
            state = selected_parameter_state(
                model,
                distributed_context,
                lambda name, _parameter: _is_task_routing_parameter(name),
            )
            state = broadcast_object(state, distributed_context)
            if state is None:
                raise RuntimeError("FSDP task-bank capture returned no state")
            self.state = {name: value.clone() for name, value in state.items()}
            return
        named = dict(distributed_named_parameters(model))
        with torch.no_grad():
            self.state = {
                name: named[name].detach().cpu().clone()
                for name in self.state
            }

    def restore_optimizer_state(self, optimizer, model, names: tuple[str, ...]) -> None:
        """Restore only this bank's Adam state for the shared model parameters."""
        named = dict(distributed_named_parameters(model))
        for name in names:
            parameter = named[name]
            saved = self.optimizer_state.get(name)
            if saved is None:
                optimizer.state.pop(parameter, None)
                continue
            optimizer.state[parameter] = {
                key: value.to(parameter.device)
                if isinstance(value, torch.Tensor)
                else deepcopy(value)
                for key, value in saved.items()
            }

    def capture_optimizer_state(self, optimizer, model, names: tuple[str, ...]) -> None:
        named = dict(distributed_named_parameters(model))
        self.optimizer_state = {
            name: {
                key: value.detach().cpu().clone()
                if isinstance(value, torch.Tensor)
                else deepcopy(value)
                for key, value in optimizer.state.get(named[name], {}).items()
            }
            for name in names
            if named[name] in optimizer.state
        }
