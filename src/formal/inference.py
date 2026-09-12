from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist

from src.distributed.fsdp_utils import named_parameters as distributed_named_parameters
from src.formal.runtime import parameter_hash, task_routing_parameter_names
from src.formal.task_banks import TaskBank


FORMAL_TASKS = ("math", "multihop", "code")


@dataclass(frozen=True)
class FormalProbePrediction:
    predicted_task: str
    logits: tuple[float, ...]
    probabilities: tuple[float, ...]


@dataclass(frozen=True)
class FormalInferenceResult:
    probe: FormalProbePrediction
    selected_task: str
    generated_ids: torch.LongTensor
    partition_sha256: str
    query_sha256: str
    alpha_sha256: str
    alpha_statistics: dict[str, float]


class FormalInferenceEngine:
    """Three-way Probe plus atomic task-bank routing for a formal checkpoint."""

    def __init__(
        self,
        *,
        model,
        banks: dict[str, TaskBank],
        probe_head: torch.nn.Linear,
        device: torch.device,
        probe_task: str = "math",
    ) -> None:
        if set(banks) != set(FORMAL_TASKS):
            raise ValueError("Formal inference requires math, multihop, and code banks")
        if probe_task not in banks:
            raise ValueError(f"Unknown formal Probe task: {probe_task}")
        if probe_head.out_features != len(FORMAL_TASKS):
            raise ValueError("Formal Probe head must be three-way")
        model_config = model.module.config if hasattr(model, "module") else model.config
        if probe_head.in_features != int(model_config.hidden_size):
            raise ValueError("Formal Probe head width must come from the model config")
        self.model = model
        self.banks = banks
        self.probe_head = probe_head.to(device).eval()
        self.device = device
        self.probe_task = probe_task

    def _alpha_statistics(self) -> dict[str, float]:
        names = task_routing_parameter_names(self.model)
        named = dict(distributed_named_parameters(self.model))
        alpha_parameters = [
            parameter.detach().float()
            for name, parameter in named.items()
            if name in names and "alpha" in name
        ]
        if not alpha_parameters:
            raise RuntimeError("Formal inference model has no alpha parameters")
        device = alpha_parameters[0].device
        sum_abs = torch.zeros((), dtype=torch.float32, device=device)
        max_abs = torch.zeros((), dtype=torch.float32, device=device)
        nonzero = torch.zeros((), dtype=torch.float32, device=device)
        total = 0
        for value in alpha_parameters:
            if value.numel() == 0:
                continue
            sum_abs = sum_abs + value.abs().sum()
            max_abs = torch.maximum(max_abs, value.abs().max())
            nonzero = nonzero + torch.count_nonzero(value).to(dtype=torch.float32)
            total += int(value.numel())
        if dist.is_initialized():
            totals = torch.stack((sum_abs, torch.tensor(float(total), device=device), nonzero))
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
            dist.all_reduce(max_abs, op=dist.ReduceOp.MAX)
            sum_abs, total_value, nonzero = totals
            total = max(1, int(total_value.item()))
        else:
            total_value = torch.tensor(float(total), device=device)
        return {
            "mean_abs": float((sum_abs / total_value.clamp_min(1.0)).cpu()),
            "max_abs": float(max_abs.cpu()),
            "nonzero_fraction": float((nonzero / total_value.clamp_min(1.0)).cpu()),
        }

    @torch.no_grad()
    def classify(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
    ) -> FormalProbePrediction:
        if input_ids.shape[0] != 1:
            raise ValueError("Formal inference currently routes one case at a time")
        output = self.model(
            input_ids=input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
            use_cache=False,
            probe_layer0_only=True,
        )
        hidden = output.probe_hidden_state
        mask = attention_mask.to(self.device).unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        logits = self.probe_head(pooled).float()
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Formal Probe logits contain NaN or Inf")
        probabilities = torch.softmax(logits, dim=-1)
        index = int(logits.argmax(dim=-1).item())
        prediction = FormalProbePrediction(
            predicted_task=FORMAL_TASKS[index],
            logits=tuple(float(value) for value in logits[0].cpu()),
            probabilities=tuple(float(value) for value in probabilities[0].cpu()),
        )
        del hidden, pooled, output
        return prediction

    @torch.no_grad()
    def _generate_fixed_task(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        *,
        task: str,
        maximum_new_tokens: int,
        eos_token_id: int | None,
    ) -> torch.LongTensor:
        routed = self.banks[task]
        routed.activate(self.model)
        before = parameter_hash(self.model, task_routing_parameter_names(self.model))
        generated = input_ids.to(self.device).clone()
        mask = attention_mask.to(self.device).clone()
        prompt_length = generated.shape[1]
        for _ in range(maximum_new_tokens):
            output = self.model(
                input_ids=generated,
                attention_mask=mask,
                use_cache=False,
                logits_to_keep=1,
            )
            model_config = self.model.module.config if hasattr(self.model, "module") else self.model.config
            if model_config.moirai_task != task:
                raise RuntimeError("Formal task route changed during generation")
            next_token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
            generated = torch.cat((generated, next_token), dim=1)
            mask = torch.cat((mask, torch.ones_like(next_token)), dim=1)
            if eos_token_id is not None and bool(torch.all(next_token == eos_token_id)):
                break
        after = parameter_hash(self.model, task_routing_parameter_names(self.model))
        if before != after:
            raise RuntimeError("Formal routed bank changed during inference")
        return generated[:, prompt_length:]

    @torch.no_grad()
    def infer(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        *,
        maximum_new_tokens: int,
        eos_token_id: int | None = None,
    ) -> FormalInferenceResult:
        original_ids = input_ids.detach().clone()
        original_mask = attention_mask.detach().clone()
        probe = self.classify(original_ids, original_mask)
        selected = probe.predicted_task
        bank = self.banks[selected]
        generated = self._generate_fixed_task(
            original_ids,
            original_mask,
            task=selected,
            maximum_new_tokens=maximum_new_tokens,
            eos_token_id=eos_token_id,
        )
        bank.activate(self.model)
        names = dict(distributed_named_parameters(self.model))
        query_names = [name for name in names if "pseudo_query" in name]
        alpha_names = [name for name in names if "alpha" in name]
        return FormalInferenceResult(
            probe=probe,
            selected_task=selected,
            generated_ids=generated,
            partition_sha256=bank.partition_sha256,
            query_sha256=parameter_hash(self.model, query_names),
            alpha_sha256=parameter_hash(self.model, alpha_names),
            alpha_statistics=self._alpha_statistics(),
        )
