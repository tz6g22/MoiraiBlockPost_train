from __future__ import annotations

import math


FIXED_BLOCK_COUNT_POLICY = "ceil_num_layers_over_fixed_block_size"


def resolve_fixed_num_blocks(
    model_num_layers: int,
    *,
    fixed_block_size: int,
    policy: str,
) -> int:
    """Resolve one shared N from the native model depth and fixed block size."""
    if model_num_layers <= 0:
        raise ValueError("model_num_layers must be positive")
    if fixed_block_size <= 0:
        raise ValueError("fixed_block_size must be positive")
    if policy != FIXED_BLOCK_COUNT_POLICY:
        raise ValueError(f"Unsupported fixed block count policy: {policy!r}")
    return math.ceil(model_num_layers / fixed_block_size)
