from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from transformers import Qwen3ForCausalLM

from src.modeling.full_attnres import MoiraiQwen3Config, MoiraiQwen3ForCausalLM


def _set_named_parameter(module: nn.Module, name: str, value: torch.Tensor) -> None:
    parent_name, _, leaf_name = name.rpartition(".")
    parent = module.get_submodule(parent_name) if parent_name else module
    setattr(parent, leaf_name, nn.Parameter(value))


def _materialize_formal_extras(
    model: MoiraiQwen3ForCausalLM,
    names: set[str],
    *,
    dtype: torch.dtype,
    routing_dtype: torch.dtype,
) -> None:
    """Materialize only parameters absent from the native Qwen3 state dict."""
    formal_parameters = dict(model.named_parameters())
    for name in names:
        parameter = formal_parameters[name]
        if parameter.device.type != "meta":
            continue
        if "alpha" in name:
            fill = float(getattr(model.config, "formal_alpha_init", 0.0))
        else:
            fill = 1.0 if "key_norm" in name else 0.0
        parameter_dtype = (
            routing_dtype
            if "pseudo_query" in name or "alpha" in name
            else dtype
        )
        _set_named_parameter(
            model,
            name,
            torch.full(parameter.shape, fill, dtype=parameter_dtype, device="cpu"),
        )

    # Qwen3's rotary frequency is a non-persistent buffer and is therefore not
    # present in the native state dict.  Recreate that buffer on CPU; all other
    # meta buffers are safe to materialize as zeros because they are derived
    # or unused during the no-cache formal forward.
    for name, buffer in tuple(model.named_buffers()):
        if buffer.device.type != "meta":
            continue
        parent_name, _, leaf_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        if name.endswith("rotary_emb.inv_freq"):
            from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

            replacement = Qwen3RotaryEmbedding(config=model.config).inv_freq
        else:
            replacement = torch.zeros(buffer.shape, dtype=buffer.dtype, device="cpu")
        parent.register_buffer(leaf_name, replacement, persistent=False)


def formal_config_from_qwen(
    config,
    *,
    min_block_length: int,
    max_block_length: int,
    no_adjacent_singletons: bool,
    alpha_init: float = 0.0,
    use_alpha: bool = True,
) -> MoiraiQwen3Config:
    payload: dict[str, Any] = config.to_dict()
    for key in ("architectures", "model_type", "transformers_version", "_name_or_path"):
        payload.pop(key, None)
    payload.update(
        {
            "attnres_execution": "formal",
            "moirai_partition": None,
            "moirai_task": "unassigned",
            "moirai_min_block_length": int(min_block_length),
            "moirai_max_block_length": int(max_block_length),
            "moirai_no_adjacent_singletons": bool(no_adjacent_singletons),
            "formal_alpha_init": float(alpha_init),
            "formal_use_alpha": bool(use_alpha),
            "use_cache": False,
        }
    )
    return MoiraiQwen3Config(**payload)


def convert_qwen3_checkpoint(
    checkpoint: str | Path,
    *,
    partition: list[int],
    task: str,
    min_block_length: int,
    max_block_length: int,
    no_adjacent_singletons: bool,
    alpha_init: float = 0.0,
    use_alpha: bool = True,
    dtype: torch.dtype = torch.bfloat16,
    routing_dtype: torch.dtype = torch.float32,
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
    formal_config = formal_config_from_qwen(
        original.config,
        min_block_length=min_block_length,
        max_block_length=max_block_length,
        no_adjacent_singletons=no_adjacent_singletons,
        alpha_init=alpha_init,
        use_alpha=use_alpha,
    )
    # Avoid a second fully initialized large CPU model during conversion.  The
    # native tensors are assigned directly below; only new formal parameters
    # are then allocated explicitly.
    with torch.device("meta"):
        formal = MoiraiQwen3ForCausalLM(formal_config)
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
    _materialize_formal_extras(
        formal,
        allowed_missing,
        dtype=dtype,
        routing_dtype=routing_dtype,
    )
    formal.config.moirai_partition = list(partition)
    formal.config.moirai_task = task
    formal.config.moirai_min_block_length = int(min_block_length)
    formal.config.moirai_max_block_length = int(max_block_length)
    formal.config.moirai_no_adjacent_singletons = bool(no_adjacent_singletons)
    formal.config.attnres_execution = "formal"
    formal.config.use_cache = False
    return original, formal
