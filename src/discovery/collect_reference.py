from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FullReference:
    observations: tuple[torch.Tensor, ...]
    residual_sources: tuple[torch.Tensor, ...]
    attention_outputs: tuple[torch.Tensor, ...]
    mlp_outputs: tuple[torch.Tensor, ...]
    attention_mask: torch.Tensor


@torch.no_grad()
def collect_full_reference(
    model,
    *,
    input_ids: torch.LongTensor,
    attention_mask: torch.Tensor,
) -> FullReference:
    if getattr(model.config, "attnres_execution", None) == "formal":
        raise RuntimeError(
            "Formal Discovery cannot collect AttnRes observations; use the "
            "ordinary residual reference path"
        )
    if model.config.attnres_execution != "full":
        raise ValueError("Reference collection requires Full AttnRes execution")
    model.eval()
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        return_attnres_observations=True,
    )
    observations = tuple(outputs.attnres_observations)
    residual_sources = tuple(outputs.attnres_sources)
    attention_outputs = tuple(outputs.attnres_attention_outputs)
    mlp_outputs = tuple(outputs.attnres_mlp_outputs)
    expected_sites = 2 * model.config.num_hidden_layers + 1
    if len(observations) != expected_sites:
        raise RuntimeError(
            f"Expected {expected_sites} Full observation sites, got {len(observations)}"
        )
    expected_sources = 2 * model.config.num_hidden_layers + 1
    if len(residual_sources) != expected_sources:
        raise RuntimeError(
            f"Expected {expected_sources} Full AttnRes sources, got {len(residual_sources)}"
        )
    if len(attention_outputs) != model.config.num_hidden_layers:
        raise RuntimeError("Reference is missing Transformer Attention intermediates")
    if len(mlp_outputs) != model.config.num_hidden_layers:
        raise RuntimeError("Reference is missing Transformer MLP intermediates")
    return FullReference(
        observations=observations,
        residual_sources=residual_sources,
        attention_outputs=attention_outputs,
        mlp_outputs=mlp_outputs,
        attention_mask=attention_mask,
    )
