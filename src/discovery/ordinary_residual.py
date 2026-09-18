"""Strict ordinary-Qwen residual Discovery primitives.

This module deliberately has no dependency on the AttnRes implementation.  It
captures the input/output of each native Qwen3 decoder block and computes the
residual-coalescence statistic over that one reference trajectory.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any, Iterable

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from src.distributed.fsdp_utils import root_module


RESIDUAL_DISCOVERY_COST_UNDEFINED = "RESIDUAL_DISCOVERY_COST_UNDEFINED"


@dataclass(frozen=True)
class OrdinaryResidualReference:
    """Native complete-block states from one ordinary Qwen3 forward."""

    block_inputs: tuple[torch.Tensor, ...]
    block_outputs: tuple[torch.Tensor, ...]
    residual_contributions: torch.Tensor
    final_hidden_state: torch.Tensor
    attention_mask: torch.Tensor

    @property
    def num_transformer_blocks(self) -> int:
        return len(self.block_inputs)


def require_defined_residual_cost(config: dict[str, Any]) -> None:
    if not bool(config.get("ordinary_residual_cost_defined", False)):
        raise RuntimeError(
            f"{RESIDUAL_DISCOVERY_COST_UNDEFINED}: ordinary residual coalescence "
            "cost is not enabled in the Discovery configuration"
        )


def formal_discovery_status(config: dict[str, Any]) -> dict[str, object]:
    cka = config.get("method") == "linear_cka_min"
    return {
        "model_mode": "original_residual_only",
        "attnres_accessed": False,
        "query_accessed": False,
        "alpha_accessed": False,
        "backward_used": False,
        "true_moirai_replay": False,
        "cost_status": (
            "DEFINED" if cka or bool(config.get("ordinary_residual_cost_defined", False))
            else RESIDUAL_DISCOVERY_COST_UNDEFINED
        ),
        "method": config.get("method", "legacy_cosine"),
        "metric": config.get("metric") if cka else "legacy_cosine",
        "interval_reduction": config.get("interval_reduction") if cka else None,
    }


def _ordinary_backbone(model):
    candidate = root_module(model)
    if "Moirai" in type(candidate).__name__ or "full_attnres" in type(candidate).__module__:
        raise ValueError("Discovery requires the original Qwen3 model, not an AttnRes wrapper")
    config = getattr(candidate, "config", None)
    if getattr(config, "model_type", None) != "qwen3":
        raise ValueError("Discovery model must expose a native Qwen3 config")
    if any(
        hasattr(config, name)
        for name in ("attnres_execution", "moirai_partition", "moirai_task")
    ):
        raise ValueError("Discovery model config contains formal AttnRes state")
    backbone = getattr(candidate, "model", candidate)
    layers = getattr(backbone, "layers", None)
    if layers is None or not len(layers):
        raise ValueError("Native Qwen3 decoder layers are unavailable")
    return candidate, backbone, tuple(layers)


def _layer_input_hook(captured: list[torch.Tensor | None], index: int):
    def hook(_module, args):
        if not args:
            raise RuntimeError("Qwen3 decoder block did not receive hidden_states")
        captured[index] = args[0].detach()

    return hook


def _layer_output_hook(captured: list[torch.Tensor | None], index: int):
    def hook(_module, _args, output):
        value = output[0] if isinstance(output, tuple) else output
        if not isinstance(value, torch.Tensor):
            raise RuntimeError("Qwen3 decoder block output is not a tensor")
        captured[index] = value.detach()

    return hook


@torch.no_grad()
def collect_ordinary_residual_reference(
    model,
    *,
    input_ids: torch.LongTensor,
    attention_mask: torch.Tensor,
) -> OrdinaryResidualReference:
    """Capture native Qwen3 complete-block residual increments in one forward."""
    if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
        raise ValueError("input_ids and attention_mask must both have shape [batch, tokens]")
    candidate, backbone, layers = _ordinary_backbone(model)
    del backbone
    forward_model = model if isinstance(model, FSDP) else candidate
    model.eval()
    block_inputs: list[torch.Tensor | None] = [None] * len(layers)
    block_outputs: list[torch.Tensor | None] = [None] * len(layers)
    with ExitStack() as hooks:
        for index, layer in enumerate(layers):
            hooks.enter_context(
                _removable_hook(layer.register_forward_pre_hook(_layer_input_hook(block_inputs, index)))
            )
            hooks.enter_context(
                _removable_hook(layer.register_forward_hook(_layer_output_hook(block_outputs, index)))
            )
        if hasattr(candidate, "model"):
            forward_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                logits_to_keep=1,
            )
        else:
            forward_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )

    if any(value is None for value in block_inputs + block_outputs):
        raise RuntimeError("Native Qwen3 forward did not expose every decoder block boundary")
    inputs = tuple(value for value in block_inputs if value is not None)
    outputs = tuple(value for value in block_outputs if value is not None)
    if len(inputs) != len(layers) or len(outputs) != len(layers):
        raise RuntimeError("Native Qwen3 block capture count mismatch")
    residuals = torch.stack(
        tuple(output.float() - input_.float() for input_, output in zip(inputs, outputs)),
        dim=0,
    )
    return OrdinaryResidualReference(
        block_inputs=inputs,
        block_outputs=outputs,
        residual_contributions=residuals,
        final_hidden_state=outputs[-1],
        attention_mask=attention_mask.detach(),
    )


class _removable_hook:
    def __init__(self, handle) -> None:
        self.handle = handle

    def __enter__(self):
        return self.handle

    def __exit__(self, exc_type, exc_value, traceback):
        self.handle.remove()
        return False


def pairwise_directional_interval_cost(
    residual_contributions: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    start: int,
    end: int,
    epsilon: float = 1.0e-8,
) -> torch.Tensor:
    """Compute mean pairwise directional disagreement for one interval."""
    if residual_contributions.ndim != 4:
        raise ValueError("residual_contributions must have shape [layers, batch, tokens, hidden]")
    layers, batch, tokens, _hidden = residual_contributions.shape
    if attention_mask.shape != (batch, tokens):
        raise ValueError("attention_mask shape does not match residual contributions")
    if not (0 <= start <= end < layers):
        raise ValueError("interval is outside the captured Transformer depth")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    values = residual_contributions[start : end + 1].float()
    interval_length = end - start + 1
    if interval_length == 1:
        return torch.zeros((), dtype=torch.float32, device=values.device)

    norms = torch.linalg.vector_norm(values, dim=-1, keepdim=True)
    directions = values / (norms + epsilon)
    pairwise_dots = torch.einsum("ibth,jbth->ijbt", directions, directions)
    upper_triangle = torch.triu(
        torch.ones(
            (interval_length, interval_length),
            dtype=torch.bool,
            device=values.device,
        ),
        diagonal=1,
    )
    pairwise_disagreement = (1.0 - pairwise_dots[upper_triangle]).sum(dim=0)
    pair_count = interval_length * (interval_length - 1) // 2
    per_token_cost = pairwise_disagreement / float(pair_count)
    valid = attention_mask.to(dtype=torch.float32)
    cost = (per_token_cost * valid).sum() / (valid.sum() + epsilon)
    if not torch.isfinite(cost):
        raise FloatingPointError(
            f"Pairwise directional cost is not finite for {start}:{end}"
        )
    return cost


def pairwise_directional_interval_costs(
    reference: OrdinaryResidualReference,
    intervals: Iterable[tuple[int, int]],
    *,
    epsilon: float = 1.0e-8,
) -> dict[tuple[int, int], float]:
    costs: dict[tuple[int, int], float] = {}
    for start, end in intervals:
        costs[(start, end)] = float(
            pairwise_directional_interval_cost(
                reference.residual_contributions,
                reference.attention_mask,
                start=start,
                end=end,
                epsilon=epsilon,
            ).cpu()
        )
    return costs
