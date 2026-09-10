from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class TaskBank:
    """CPU-owned task Q/Alpha state for one shared backbone."""

    task: str
    state: dict[str, torch.Tensor]
    partition_sha256: str

    @classmethod
    def from_model(cls, model, *, task: str, partition_sha256: str) -> "TaskBank":
        state = {
            name: parameter.detach().cpu().clone()
            for name, parameter in model.named_parameters()
            if "pseudo_query" in name or "alpha" in name
        }
        if not state:
            raise ValueError("Task bank cannot be created without Q/Alpha parameters")
        return cls(task=task, state=state, partition_sha256=partition_sha256)

    def activate(self, model) -> None:
        named = dict(model.named_parameters())
        if set(self.state) != {
            name for name in named if "pseudo_query" in name or "alpha" in name
        }:
            raise ValueError(f"Task bank {self.task} does not match the model Q/Alpha sites")
        with torch.no_grad():
            for name, value in self.state.items():
                named[name].copy_(value.to(device=named[name].device, dtype=named[name].dtype))

    def capture(self, model) -> None:
        named = dict(model.named_parameters())
        with torch.no_grad():
            self.state = {
                name: named[name].detach().cpu().clone()
                for name in self.state
            }
