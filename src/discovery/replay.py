from __future__ import annotations

from dataclasses import dataclass

import torch

from src.discovery.collect_reference import FullReference, collect_full_reference
from src.discovery.local_cost import masked_normalized_frobenius
from src.modeling.partition import MoiraiPartition


@dataclass(frozen=True)
class ReplayResult:
    mean_distortion: float
    site_distortions: tuple[float, ...]
    site_count: int


@torch.no_grad()
def replay_partition(
    model,
    partition: MoiraiPartition,
    *,
    input_ids: torch.LongTensor,
    attention_mask: torch.Tensor,
    reference: FullReference | None = None,
) -> ReplayResult:
    """True compressed forward; this is intentionally separate from local cost."""
    if getattr(model.config, "attnres_execution", None) == "formal":
        raise RuntimeError("Formal Discovery cannot run compressed Block AttnRes replay")
    if partition.num_transformer_blocks != model.config.num_hidden_layers:
        raise ValueError("Partition depth does not match model depth")
    partition.validate()
    if model.config.attnres_execution != "full":
        raise ValueError("Replay must start from the frozen Full reference model")
    if reference is None:
        reference = collect_full_reference(
            model,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    old_execution = model.config.attnres_execution
    old_partition = model.config.moirai_partition
    old_task = model.config.moirai_task
    try:
        model.config.attnres_execution = "moirai"
        model.config.moirai_partition = list(partition.lengths)
        model.config.moirai_task = partition.task
        compared = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_attnres_observations=True,
        )
    finally:
        model.config.attnres_execution = old_execution
        model.config.moirai_partition = old_partition
        model.config.moirai_task = old_task

    compared_sites = tuple(compared.attnres_observations)
    if len(compared_sites) != len(reference.observations):
        raise RuntimeError("Replay and Full observation site counts differ")
    errors = tuple(
        float(
            masked_normalized_frobenius(
                full,
                compressed,
                attention_mask,
            ).cpu()
        )
        for full, compressed in zip(reference.observations, compared_sites)
    )
    return ReplayResult(
        mean_distortion=sum(errors) / len(errors),
        site_distortions=errors,
        site_count=len(errors),
    )

