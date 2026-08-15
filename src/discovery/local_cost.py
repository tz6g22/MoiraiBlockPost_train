from __future__ import annotations

import torch

from src.discovery.collect_reference import FullReference
from src.modeling.block_attnres import sum_block_sources
from src.modeling.full_attnres import attnres_aggregate


def masked_normalized_frobenius(
    full: torch.Tensor,
    compared: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    epsilon: float = 1.0e-8,
) -> torch.Tensor:
    """Masked FP32 normalized Frobenius error from specification section 14."""
    if full.shape != compared.shape:
        raise ValueError("Full and compared tensors must have identical shapes")
    if full.ndim != 3:
        raise ValueError("Expected tensors with shape [batch, tokens, hidden]")
    if attention_mask.shape != full.shape[:2]:
        raise ValueError("attention_mask must have shape [batch, tokens]")

    expanded_mask = attention_mask.to(dtype=torch.float32).unsqueeze(-1)
    full_fp32 = full.float()
    compared_fp32 = compared.float()
    denominator = torch.linalg.vector_norm(expanded_mask * full_fp32)
    if denominator.item() < 1.0e-12:
        raise FloatingPointError("Full-state Frobenius denominator is below 1e-12")
    numerator = torch.linalg.vector_norm(expanded_mask * (full_fp32 - compared_fp32))
    error = numerator / (denominator + epsilon)
    if not torch.isfinite(error):
        raise FloatingPointError("Distortion is NaN or Inf")
    return error


def mean_site_distortion(
    full_sites: tuple[torch.Tensor, ...],
    compared_sites: tuple[torch.Tensor, ...],
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    if len(full_sites) != len(compared_sites):
        raise ValueError("Full and compared observation counts differ")
    if not full_sites:
        raise ValueError("At least one observation site is required")
    errors = [
        masked_normalized_frobenius(full, compared, attention_mask)
        for full, compared in zip(full_sites, compared_sites)
    ]
    return torch.stack(errors).mean()


@torch.no_grad()
def local_surrogate_interval_cost(
    model,
    reference: FullReference,
    *,
    start: int,
    end: int,
) -> tuple[torch.Tensor, int]:
    """Section 15 local intervention without re-running Attention or MLP."""
    transformer_blocks = model.config.num_hidden_layers
    if not (0 <= start <= end < transformer_blocks):
        raise ValueError("Candidate interval is outside the Transformer depth")
    if not 1 <= end - start + 1 <= 4:
        raise ValueError("Candidate interval length must be in [1, 4]")
    if model.config.attnres_execution != "full":
        raise ValueError("Local surrogate requires the frozen Full model")

    sources = reference.residual_sources
    interval_start = 1 + 2 * start
    interval_stop = 1 + 2 * (end + 1)
    block_summary = sum_block_sources(sources[interval_start:interval_stop])

    def compressed_sources(available_count: int) -> tuple[torch.Tensor, ...]:
        if available_count < interval_stop:
            raise ValueError("Local comparison site is not downstream of interval")
        return (
            sources[:interval_start]
            + (block_summary,)
            + sources[interval_stop:available_count]
        )

    errors: list[torch.Tensor] = []
    for layer_index in range(end + 1, transformer_blocks):
        layer = model.model.layers[layer_index]
        completed = compressed_sources(1 + 2 * layer_index)
        z_attn = attnres_aggregate(
            completed,
            layer.attn_pseudo_query,
            layer.attn_key_norm,
        )
        errors.append(
            masked_normalized_frobenius(
                reference.observations[2 * layer_index],
                z_attn,
                reference.attention_mask,
            )
        )
        z_mlp = attnres_aggregate(
            completed + (reference.attention_outputs[layer_index],),
            layer.mlp_pseudo_query,
            layer.mlp_key_norm,
        )
        errors.append(
            masked_normalized_frobenius(
                reference.observations[2 * layer_index + 1],
                z_mlp,
                reference.attention_mask,
            )
        )

    z_final = attnres_aggregate(
        compressed_sources(len(sources)),
        model.model.final_pseudo_query,
        model.model.final_key_norm,
    )
    errors.append(
        masked_normalized_frobenius(
            reference.observations[-1],
            z_final,
            reference.attention_mask,
        )
    )
    return torch.stack(errors).mean(), len(errors)

