from __future__ import annotations

import torch

from src.discovery.collect_reference import FullReference


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

    result = model(
        local_surrogate_request={
            "residual_sources": reference.residual_sources,
            "attention_outputs": reference.attention_outputs,
            "start": start,
            "end": end,
        },
        use_cache=False,
    )
    full_sites = reference.observations[2 * (end + 1) :]
    compared_sites = tuple(result.local_surrogate_sites)
    if len(full_sites) != len(compared_sites):
        raise RuntimeError("Local surrogate observation count changed")
    errors = [
        masked_normalized_frobenius(full, compared, reference.attention_mask)
        for full, compared in zip(full_sites, compared_sites)
    ]
    return torch.stack(errors).mean(), len(errors)
