from src.modeling.block_attnres import compress_full_sources, sum_block_sources
from src.modeling.partition import MoiraiBlock, MoiraiPartition, fixed_kimi_partition


def __getattr__(name):
    # Keep the public modeling imports available without importing Transformers
    # when callers only need partition or source-compression utilities.
    if name in {
        "MoiraiQwen3Config",
        "MoiraiQwen3ForCausalLM",
        "MoiraiQwen3Model",
        "attnres_aggregate",
    }:
        from src.modeling.full_attnres import (
            MoiraiQwen3Config,
            MoiraiQwen3ForCausalLM,
            MoiraiQwen3Model,
            attnres_aggregate,
        )

        return {
            "MoiraiQwen3Config": MoiraiQwen3Config,
            "MoiraiQwen3ForCausalLM": MoiraiQwen3ForCausalLM,
            "MoiraiQwen3Model": MoiraiQwen3Model,
            "attnres_aggregate": attnres_aggregate,
        }[name]
    raise AttributeError(name)

__all__ = [
    "MoiraiBlock",
    "MoiraiPartition",
    "MoiraiQwen3Config",
    "MoiraiQwen3ForCausalLM",
    "MoiraiQwen3Model",
    "attnres_aggregate",
    "compress_full_sources",
    "fixed_kimi_partition",
    "sum_block_sources",
]
