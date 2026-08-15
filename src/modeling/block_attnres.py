from __future__ import annotations

from collections.abc import Sequence

import torch

from src.modeling.partition import MoiraiPartition


def sum_block_sources(sources: Sequence[torch.Tensor]) -> torch.Tensor:
    """Return the parameter-free residual sum for one complete MoiraiBlock."""
    if not sources:
        raise ValueError("A MoiraiBlock summary requires at least one source")
    reference_shape = sources[0].shape
    if any(source.shape != reference_shape for source in sources):
        raise ValueError("All block residual sources must have identical shapes")
    summary = sources[0]
    for source in sources[1:]:
        summary = summary + source
    return summary


def compress_full_sources(
    full_sources: Sequence[torch.Tensor],
    partition: MoiraiPartition,
) -> tuple[torch.Tensor, ...]:
    """Compress embedding + ordered Attention/MLP sources by a partition."""
    partition.validate()
    expected = 1 + 2 * partition.num_transformer_blocks
    if len(full_sources) != expected:
        raise ValueError(
            "Full source inventory must contain embedding plus separate "
            "Attention/MLP sources for every Transformer block: "
            f"expected {expected}, got {len(full_sources)}"
        )
    embedding = full_sources[0]
    sublayer_sources = full_sources[1:]
    compressed = [embedding]
    for block in partition.blocks:
        compressed.append(
            sum_block_sources(
                sublayer_sources[2 * block.start : 2 * (block.end + 1)]
            )
        )
    return tuple(compressed)

