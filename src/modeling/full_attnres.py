from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3MLP,
    Qwen3PreTrainedModel,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
)

from src.modeling.block_attnres import sum_block_sources
from src.modeling.partition import MoiraiPartition


ExecutionMode = Literal["full", "moirai", "formal"]


class MoiraiQwen3Config(Qwen3Config):
    """Qwen3 configuration for MoiraiBlock discovery and query training."""

    model_type = "qwen3_attnres"

    def __init__(
        self,
        *,
        attnres_execution: ExecutionMode = "full",
        moirai_partition: list[int] | None = None,
        moirai_task: str = "unassigned",
        moirai_min_block_length: int = 1,
        moirai_max_block_length: int = 4,
        moirai_no_adjacent_singletons: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if attnres_execution not in {"full", "moirai", "formal"}:
            raise ValueError("attnres_execution must be 'full', 'moirai', or 'formal'")
        self.attnres_execution = attnres_execution
        self.moirai_partition = moirai_partition
        self.moirai_task = moirai_task
        self.moirai_min_block_length = moirai_min_block_length
        self.moirai_max_block_length = moirai_max_block_length
        self.moirai_no_adjacent_singletons = moirai_no_adjacent_singletons


@dataclass(frozen=True)
class AttnResAggregation:
    hidden_states: torch.Tensor
    weights: torch.Tensor


def attnres_aggregate(
    sources: tuple[torch.Tensor, ...] | list[torch.Tensor],
    pseudo_query: torch.Tensor,
    key_norm: nn.Module,
    *,
    return_weights: bool = False,
) -> torch.Tensor | AttnResAggregation:
    """Kimi AttnRes softmax aggregation over depth sources.

    The pseudo-query is one learned d-dimensional vector for one observation
    site. Keys are RMS-normalized source values. Softmax is evaluated in FP32
    for numerical stability and values retain their original dtype.
    """
    if not sources:
        raise ValueError("AttnRes requires at least one source")
    reference_shape = sources[0].shape
    if any(source.shape != reference_shape for source in sources):
        raise ValueError("All AttnRes sources must have the same shape")
    if pseudo_query.ndim != 1 or pseudo_query.shape[0] != reference_shape[-1]:
        raise ValueError("pseudo_query must have shape [hidden_size]")

    values = torch.stack(tuple(sources), dim=0)
    keys = key_norm(values)
    logits = torch.einsum(
        "d,nbtd->nbt",
        pseudo_query.float(),
        keys.float(),
    )
    weights = logits.softmax(dim=0)
    hidden_states = torch.einsum(
        "nbt,nbtd->btd",
        weights.to(dtype=values.dtype),
        values,
    )
    if return_weights:
        return AttnResAggregation(hidden_states=hidden_states, weights=weights)
    return hidden_states


def _create_mask_compat(factory, **kwargs):
    signature = inspect.signature(factory)
    filtered = {name: value for name, value in kwargs.items() if name in signature.parameters}
    return factory(**filtered)


class MoiraiQwen3DecoderLayer(nn.Module):
    """One complete Transformer block with two Kimi-compatible query sites."""

    def __init__(self, config: MoiraiQwen3Config, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = Qwen3Attention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.attention_type = config.layer_types[layer_idx]

        # No gate, router, projection summary, task token, or recency bias.
        self.attn_pseudo_query = nn.Parameter(torch.zeros(config.hidden_size))
        self.attn_key_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp_pseudo_query = nn.Parameter(torch.zeros(config.hidden_size))
        self.mlp_key_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # Formal task-specific gates are zero at conversion time.  They blend
        # the Kimi-routed branch with the native residual stream without
        # changing the Kimi cumulative block source semantics.
        # FSDP requires managed parameters to have at least one dimension.
        # A length-one tensor remains a per-site scalar under broadcasting.
        self.attn_alpha = nn.Parameter(torch.zeros(1))
        self.mlp_alpha = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        completed_sources: tuple[torch.Tensor, ...],
        partial_block: torch.Tensor | None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        surrogate_attention_output: torch.Tensor | None = None,
        native_hidden: torch.Tensor | None = None,
        native_only: bool = False,
    ) -> torch.Tensor | tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if native_only:
            if native_hidden is None:
                raise ValueError("Native-only decoder execution requires hidden states")
            if position_ids is None or cache_position is None or position_embeddings is None:
                raise ValueError("Native-only decoder execution requires position information")
            attn_output, _ = self.self_attn(
                hidden_states=self.input_layernorm(native_hidden),
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=False,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            native_after_attention = native_hidden + attn_output
            mlp_output = self.mlp(
                self.post_attention_layernorm(native_after_attention)
            )
            return native_after_attention + mlp_output

        sources_before_attn = completed_sources
        if partial_block is not None:
            sources_before_attn = sources_before_attn + (partial_block,)

        z_attn = attnres_aggregate(
            sources_before_attn,
            self.attn_pseudo_query,
            self.attn_key_norm,
        )
        if native_hidden is not None:
            z_attn = native_hidden + self.attn_alpha * (z_attn - native_hidden)
        if surrogate_attention_output is not None:
            if partial_block is not None:
                raise ValueError("Local surrogate requires a completed source history")
            z_mlp = attnres_aggregate(
                completed_sources + (surrogate_attention_output,),
                self.mlp_pseudo_query,
                self.mlp_key_norm,
            )
            return z_attn, z_mlp
        if position_ids is None or cache_position is None or position_embeddings is None:
            raise ValueError("Normal decoder execution requires position information")
        attn_output, _ = self.self_attn(
            hidden_states=self.input_layernorm(z_attn),
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=False,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )

        partial_after_attention = (
            attn_output
            if partial_block is None
            else partial_block + attn_output
        )
        z_mlp = attnres_aggregate(
            completed_sources + (partial_after_attention,),
            self.mlp_pseudo_query,
            self.mlp_key_norm,
        )
        native_after_attention = native_hidden + attn_output if native_hidden is not None else None
        if native_after_attention is not None:
            z_mlp = native_after_attention + self.mlp_alpha * (z_mlp - native_after_attention)
        mlp_output = self.mlp(self.post_attention_layernorm(z_mlp))
        next_partial = partial_after_attention + mlp_output
        layer_hidden = z_mlp + mlp_output
        if native_after_attention is not None:
            layer_hidden = native_after_attention + mlp_output
        return (
            next_partial,
            partial_after_attention,
            attn_output,
            mlp_output,
            z_attn,
            z_mlp,
            layer_hidden,
        )


class MoiraiQwen3Model(Qwen3PreTrainedModel):
    """Legacy Full AttnRes plus the post-Discovery formal Kimi runtime."""

    config_class = MoiraiQwen3Config

    def __init__(self, config: MoiraiQwen3Config) -> None:
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
        )
        self.layers = nn.ModuleList(
            [MoiraiQwen3DecoderLayer(config, index) for index in range(config.num_hidden_layers)]
        )
        self.final_pseudo_query = nn.Parameter(torch.zeros(config.hidden_size))
        self.final_key_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.final_alpha = nn.Parameter(torch.zeros(1))
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types
        self.post_init()
        self._reset_pseudo_queries()

    def _reset_pseudo_queries(self) -> None:
        for name, parameter in self.named_parameters():
            if "pseudo_query" in name:
                nn.init.zeros_(parameter)

    def _partition(self) -> MoiraiPartition | None:
        if self.config.attnres_execution == "full":
            return None
        lengths = self.config.moirai_partition
        if lengths is None:
            raise ValueError("moirai_partition is required for Moirai execution")
        return MoiraiPartition.from_lengths(
            lengths,
            task=self.config.moirai_task,
            num_transformer_blocks=self.config.num_hidden_layers,
            min_length=int(getattr(self.config, "moirai_min_block_length", 1)),
            max_length=int(getattr(self.config, "moirai_max_block_length", 4)),
            no_adjacent_singletons=bool(
                getattr(self.config, "moirai_no_adjacent_singletons", True)
            ),
        )

    def _run_layer(
        self,
        layer: MoiraiQwen3DecoderLayer,
        completed_sources: tuple[torch.Tensor, ...],
        partial_block: torch.Tensor | None,
        causal_mask: torch.Tensor | None,
        position_ids: torch.LongTensor,
        past_key_values: Cache | None,
        cache_position: torch.LongTensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if self.gradient_checkpointing and self.training:
            return checkpoint(
                layer,
                completed_sources,
                partial_block,
                causal_mask,
                position_ids,
                past_key_values,
                cache_position,
                position_embeddings,
                use_reentrant=False,
            )
        return layer(
            completed_sources,
            partial_block,
            causal_mask,
            position_ids,
            past_key_values,
            cache_position,
            position_embeddings,
        )

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        return_attnres_observations: bool = False,
        probe_layer0_only: bool = False,
        embedding_only: bool = False,
        local_surrogate_request: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> BaseModelOutputWithPast:
        if local_surrogate_request is not None:
            if self.config.attnres_execution != "full":
                raise ValueError("Local surrogate requires Full AttnRes execution")
            sources = tuple(local_surrogate_request["residual_sources"])
            attention_outputs = tuple(local_surrogate_request["attention_outputs"])
            start = int(local_surrogate_request["start"])
            end = int(local_surrogate_request["end"])
            interval_start = 1 + 2 * start
            interval_stop = 1 + 2 * (end + 1)
            block_summary = sum_block_sources(
                sources[interval_start:interval_stop]
            )

            def compressed_sources(available_count: int) -> tuple[torch.Tensor, ...]:
                return (
                    sources[:interval_start]
                    + (block_summary,)
                    + sources[interval_stop:available_count]
                )

            compared_sites: list[torch.Tensor] = []
            for layer_index in range(end + 1, self.config.num_hidden_layers):
                z_attn, z_mlp = self.layers[layer_index](
                    compressed_sources(1 + 2 * layer_index),
                    None,
                    surrogate_attention_output=attention_outputs[layer_index],
                )
                compared_sites.extend((z_attn, z_mlp))
            z_final = attnres_aggregate(
                compressed_sources(len(sources)),
                self.final_pseudo_query,
                self.final_key_norm,
            )
            compared_sites.append(z_final)
            output = BaseModelOutputWithPast(last_hidden_state=z_final, past_key_values=None)
            output.local_surrogate_sites = tuple(compared_sites)
            return output
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        if use_cache:
            raise ValueError("MoiraiBlock execution requires use_cache=False")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if embedding_only:
            return BaseModelOutputWithPast(
                last_hidden_state=inputs_embeds,
                past_key_values=None,
            )

        if cache_position is None:
            past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen,
                past_seen + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {
                "full_attention": _create_mask_compat(create_causal_mask, **mask_kwargs)
            }
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = _create_mask_compat(
                    create_sliding_window_causal_mask,
                    **mask_kwargs,
                )

        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        # Probe features must be task-independent.  Reuse the shared backbone's
        # native first block and do not touch partition, query, or alpha state.
        if probe_layer0_only:
            layer = self.layers[0]
            native_layer_output = layer(
                (),
                None,
                attention_mask=causal_mask_mapping[layer.attention_type],
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                native_hidden=inputs_embeds,
                native_only=True,
            )
            if not isinstance(native_layer_output, torch.Tensor):
                raise RuntimeError("Native-only Probe block returned routed outputs")
            return BaseModelOutputWithPast(
                last_hidden_state=native_layer_output,
                past_key_values=None,
            )

        partition = self._partition()
        boundary_ends = partition.boundary_ends if partition is not None else frozenset()

        # Embedding is always an independent source and is never part of the
        # Transformer-block partition.
        completed_sources: tuple[torch.Tensor, ...] = (inputs_embeds,)
        partial_block: torch.Tensor | None = None
        observations: list[torch.Tensor] = []
        residual_sources: list[torch.Tensor] = [inputs_embeds]
        attention_outputs: list[torch.Tensor] = []
        mlp_outputs: list[torch.Tensor] = []

        if self.config.attnres_execution == "formal":
            if partition is None:
                raise ValueError("Formal Block AttnRes requires a frozen task partition")
            native_hidden = inputs_embeds
            for layer_idx, layer in enumerate(self.layers):
                formal_args = (
                    completed_sources,
                    partial_block,
                    causal_mask_mapping[layer.attention_type],
                    position_ids,
                    past_key_values,
                    cache_position,
                    position_embeddings,
                    None,
                    native_hidden,
                )
                if self.gradient_checkpointing and self.training:
                    layer_outputs = checkpoint(
                        layer,
                        *formal_args,
                        use_reentrant=False,
                    )
                else:
                    layer_outputs = layer(*formal_args)
                (
                    next_partial,
                    partial_after_attention,
                    attn_output,
                    mlp_output,
                    z_attn,
                    z_mlp,
                    layer_hidden,
                ) = layer_outputs
                native_hidden = layer_hidden
                partial_block = next_partial
                if layer_idx in boundary_ends:
                    completed_sources = completed_sources + (partial_block,)
                    partial_block = None
                if return_attnres_observations:
                    observations.extend((z_attn, z_mlp))
                    attention_outputs.append(attn_output)
                    mlp_outputs.append(mlp_output)
            if partial_block is not None:
                raise RuntimeError("Formal partition ended with an unfinished partial block")
            routed_final = attnres_aggregate(
                completed_sources,
                self.final_pseudo_query,
                self.final_key_norm,
            )
            z_final = native_hidden + self.final_alpha * (routed_final - native_hidden)
            hidden_states = self.norm(z_final)
            output = BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=None,
            )
            if return_attnres_observations:
                output.attnres_observations = tuple(observations + [z_final])
                output.attnres_sources = tuple((inputs_embeds, *completed_sources[1:]))
                output.attnres_source_kinds = (
                    "embedding",
                    *("kimi_block_sum" for _ in completed_sources[1:]),
                )
                output.attnres_attention_outputs = tuple(attention_outputs)
                output.attnres_mlp_outputs = tuple(mlp_outputs)
            return output

        for layer_idx, layer in enumerate(self.layers):
            (
                next_partial,
                partial_after_attention,
                attn_output,
                mlp_output,
                z_attn,
                z_mlp,
                layer_hidden,
            ) = self._run_layer(
                layer,
                completed_sources,
                partial_block,
                causal_mask_mapping[layer.attention_type],
                position_ids,
                past_key_values,
                cache_position,
                position_embeddings,
            )
            if return_attnres_observations:
                observations.extend((z_attn, z_mlp))
                attention_outputs.append(attn_output)
                mlp_outputs.append(mlp_output)
            if partition is None:
                # Full AttnRes preserves Attention and MLP outputs as two
                # independent sources in execution order.
                completed_sources = completed_sources + (attn_output, mlp_output)
                residual_sources.extend((attn_output, mlp_output))
                partial_block = None
            else:
                partial_block = next_partial
                if layer_idx in boundary_ends:
                    completed_sources = completed_sources + (partial_block,)
                    residual_sources.append(partial_block)
                    partial_block = None

        if partial_block is not None:
            raise RuntimeError("Partition ended with an unfinished partial block")

        z_final = attnres_aggregate(
            completed_sources,
            self.final_pseudo_query,
            self.final_key_norm,
        )
        hidden_states = self.norm(z_final)
        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=None,
        )
        if return_attnres_observations:
            output.attnres_observations = tuple(observations + [z_final])
            output.attnres_sources = tuple(residual_sources)
            if partition is None:
                source_kinds = tuple(
                    kind
                    for _ in range(self.config.num_hidden_layers)
                    for kind in ("attention_output", "mlp_output")
                )
            else:
                source_kinds = tuple(
                    "moirai_block_sum" for _ in residual_sources[1:]
                )
            output.attnres_source_kinds = ("embedding", *source_kinds)
            output.attnres_attention_outputs = tuple(attention_outputs)
            output.attnres_mlp_outputs = tuple(mlp_outputs)
        return output


class MoiraiQwen3ForCausalLM(Qwen3PreTrainedModel, GenerationMixin):
    config_class = MoiraiQwen3Config
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: MoiraiQwen3Config) -> None:
        super().__init__(config)
        self.model = MoiraiQwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        return_attnres_observations: bool = False,
        probe_layer0_only: bool = False,
        embedding_only: bool = False,
        local_surrogate_request: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            return_attnres_observations=return_attnres_observations,
            probe_layer0_only=probe_layer0_only,
            embedding_only=embedding_only,
            local_surrogate_request=local_surrogate_request,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        if embedding_only:
            output = CausalLMOutputWithPast(
                logits=hidden_states.new_empty((hidden_states.shape[0], 0, self.vocab_size)),
                past_key_values=None,
            )
            output.input_embeddings = hidden_states
            return output
        if local_surrogate_request is not None:
            output = CausalLMOutputWithPast(
                logits=hidden_states.new_empty((hidden_states.shape[0], 0, self.vocab_size)),
                past_key_values=None,
            )
            output.local_surrogate_sites = outputs.local_surrogate_sites
            return output
        if probe_layer0_only:
            output = CausalLMOutputWithPast(
                logits=hidden_states.new_empty((hidden_states.shape[0], 0, self.vocab_size)),
                past_key_values=None,
            )
            output.probe_hidden_state = hidden_states
            return output
        slice_index = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int) and logits_to_keep > 0
            else logits_to_keep
            if not isinstance(logits_to_keep, int)
            else slice(None)
        )
        logits = self.lm_head(hidden_states[:, slice_index, :])
        loss = None
        if labels is not None:
            if logits.shape[:2] != labels.shape:
                raise ValueError(
                    "Moirai labels must already be causally shifted and align "
                    "one-to-one with logits"
                )
            loss = F.cross_entropy(
                logits.float().reshape(-1, self.config.vocab_size),
                labels.reshape(-1),
                ignore_index=-100,
            )

        output = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
        )
        if return_attnres_observations:
            output.attnres_observations = outputs.attnres_observations
            output.attnres_sources = outputs.attnres_sources
            output.attnres_source_kinds = outputs.attnres_source_kinds
            output.attnres_attention_outputs = outputs.attnres_attention_outputs
            output.attnres_mlp_outputs = outputs.attnres_mlp_outputs
        return output


AutoConfig.register(
    MoiraiQwen3Config.model_type,
    MoiraiQwen3Config,
    exist_ok=True,
)
AutoModelForCausalLM.register(
    MoiraiQwen3Config,
    MoiraiQwen3ForCausalLM,
    exist_ok=True,
)
