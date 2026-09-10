from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import Qwen3ForCausalLM

from src.modeling.full_attnres import MoiraiQwen3Config, MoiraiQwen3ForCausalLM


def formal_config_from_qwen(config) -> MoiraiQwen3Config:
    payload: dict[str, Any] = config.to_dict()
    for key in ("architectures", "model_type", "transformers_version", "_name_or_path"):
        payload.pop(key, None)
    payload.update(
        {
            "attnres_execution": "formal",
            "moirai_partition": None,
            "moirai_task": "unassigned",
            "use_cache": False,
        }
    )
    return MoiraiQwen3Config(**payload)


def convert_qwen3_checkpoint(
    checkpoint: str | Path,
    *,
    partition: list[int],
    task: str,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[Qwen3ForCausalLM, MoiraiQwen3ForCausalLM]:
    """Load native Qwen3 and create the post-Discovery formal runtime.

    The native model is retained for the identity comparison.  The converted
    model receives only the task partition and zero-initialized Q/Alpha banks;
    no Discovery state or Fixed bundle is consulted.
    """
    original = Qwen3ForCausalLM.from_pretrained(
        checkpoint,
        local_files_only=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    formal = MoiraiQwen3ForCausalLM(formal_config_from_qwen(original.config)).to(dtype=dtype)
    incompatible = formal.load_state_dict(original.state_dict(), strict=False, assign=True)
    allowed_missing = {
        name for name, _ in formal.named_parameters()
        if "pseudo_query" in name or "alpha" in name or "key_norm" in name
    }
    unexpected_missing = [name for name in incompatible.missing_keys if name not in allowed_missing]
    if unexpected_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "Native Qwen3 to formal runtime conversion mismatch: "
            f"missing={unexpected_missing}, unexpected={incompatible.unexpected_keys}"
        )
    formal.config.moirai_partition = list(partition)
    formal.config.moirai_task = task
    formal.config.attnres_execution = "formal"
    formal.config.use_cache = False
    return original, formal
