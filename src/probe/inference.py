from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from src.common import load_yaml, sha256_file, tokenizer_sha256
from src.modeling.config_bundle import MoiraiConfigBundle
from src.modeling.full_attnres import MoiraiQwen3ForCausalLM
from src.probe.extract_features import CLASS_TO_TASK, load_task_bundle
from src.training.checkpointing import validate_post_training_base_manifest


class MoiraiInferenceError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProbePrediction:
    predicted_task: str
    logits: tuple[float, float]
    probabilities: tuple[float, float]


@dataclass(frozen=True)
class InferenceResult:
    probe: ProbePrediction
    selected_config: str
    generated_ids: torch.LongTensor
    partition_sha256: str
    query_sha256: str


ALLOWED_PROBE_CLASSES = frozenset({"math", "multihop"})
ALLOWED_SELECTED_CONFIGS = frozenset({"math", "multihop"})


def select_probe_config(prediction: ProbePrediction) -> str:
    """Select exactly the Math or Multihop config predicted by the binary probe."""
    assert prediction.predicted_task in ALLOWED_PROBE_CLASSES, (
        "Probe produced an illegal class: " f"{prediction.predicted_task!r}"
    )
    selected_config = prediction.predicted_task
    assert selected_config in ALLOWED_SELECTED_CONFIGS, (
        "Probe selected an illegal formal config: " f"{selected_config!r}"
    )
    assert selected_config == prediction.predicted_task
    return selected_config


class MoiraiInferenceEngine:
    def __init__(
        self,
        *,
        model,
        tokenizer,
        bundles: dict[str, MoiraiConfigBundle],
        probe_head: torch.nn.Linear,
        device: torch.device,
    ) -> None:
        if set(bundles) != ALLOWED_SELECTED_CONFIGS:
            raise ValueError("Inference requires exactly math and multihop bundles")
        self.model = model
        self.tokenizer = tokenizer
        self.bundles = bundles
        self.probe_head = probe_head
        self.device = device
        self.shallow_forward_count = 0
        self.formal_forward_count = 0

    @classmethod
    def from_config(
        cls,
        config_path: str | Path,
        *,
        device: torch.device | str | None = None,
        model=None,
        tokenizer=None,
    ) -> "MoiraiInferenceEngine":
        config = load_yaml(config_path)
        checkpoint = Path(config["base_checkpoint"])
        weights = sorted(checkpoint.glob("model*.safetensors"))
        if len(weights) != 1:
            raise RuntimeError("Inference base checkpoint must have one weight file")
        checkpoint_manifest_path = checkpoint / "checkpoint_manifest.json"
        if not checkpoint_manifest_path.is_file():
            raise FileNotFoundError(
                f"Inference checkpoint manifest is missing: {checkpoint_manifest_path}"
            )
        checkpoint_manifest = json.loads(
            checkpoint_manifest_path.read_text(encoding="utf-8")
        )
        validate_post_training_base_manifest(checkpoint_manifest)
        base_hash = sha256_file(weights[0])
        if checkpoint_manifest.get("model_weights_sha256") != base_hash:
            raise ValueError("Inference base checkpoint hash mismatch")
        bundles = {
            task: load_task_bundle(
                config,
                task=task,
                base_checkpoint_hash=base_hash,
            )
            for task in ("math", "multihop")
        }
        probe_dir = Path(config["output_dir"])
        probe_manifest = json.loads(
            (probe_dir / "probe_manifest.json").read_text(encoding="utf-8")
        )
        probe_entry_bundle = bundles["math"]
        if (
            probe_manifest.get("base_checkpoint_sha256") != base_hash
            or probe_manifest.get("partition_sha256")
            != probe_entry_bundle.partition.sha256
            or probe_manifest.get("query_sha256") != probe_entry_bundle.query_sha256
        ):
            raise ValueError("Probe manifest is not bound to Config_math")
        head_path = probe_dir / "probe_head.safetensors"
        if sha256_file(head_path) != probe_manifest["probe_head_sha256"]:
            raise ValueError("Probe head hash mismatch")
        class_mapping = probe_manifest.get("class_mapping")
        if class_mapping != {str(index): task for index, task in CLASS_TO_TASK.items()}:
            raise ValueError("Probe class mapping is not Math/Multihop")
        device = torch.device(
            device
            if device is not None
            else ("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        if model is None:
            model = MoiraiQwen3ForCausalLM.from_pretrained(
                checkpoint,
                local_files_only=True,
                torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
            ).to(device)
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(
                checkpoint,
                local_files_only=True,
                use_fast=True,
            )
        if tokenizer_sha256(tokenizer) != checkpoint_manifest.get("tokenizer_sha256"):
            raise ValueError("Inference checkpoint tokenizer hash mismatch")
        head = torch.nn.Linear(model.config.hidden_size, 2).to(device)
        head.load_state_dict(load_file(head_path, device=str(device)))
        head.eval()
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return cls(
            model=model,
            tokenizer=tokenizer,
            bundles=bundles,
            probe_head=head,
            device=device,
        )

    @torch.no_grad()
    def classify(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
    ) -> ProbePrediction:
        try:
            self.bundles["math"].apply_to_model(self.model)
            if self.model.config.moirai_task != "math":
                raise RuntimeError("Probe entry is not Config_math")
            shallow_output = self.model.model(
                input_ids=input_ids.to(self.device),
                attention_mask=attention_mask.to(self.device),
                use_cache=False,
                probe_layer0_only=True,
            )
            self.shallow_forward_count += 1
            hidden = shallow_output.last_hidden_state
            mask = attention_mask.to(self.device).unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            pooled = pooled.to(self.probe_head.weight.dtype)
            logits = self.probe_head(pooled).float()
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Probe logits contain NaN or Inf")
            class_index = int(logits.argmax(dim=-1).item())
            probabilities = torch.softmax(logits, dim=-1)
            prediction = ProbePrediction(
                predicted_task=CLASS_TO_TASK[class_index],
                logits=tuple(float(value) for value in logits[0].cpu()),
                probabilities=tuple(
                    float(value) for value in probabilities[0].cpu()
                ),
            )
            del (
                shallow_output,
                hidden,
                mask,
                pooled,
                logits,
                probabilities,
            )
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            return prediction
        except Exception as exc:
            raise MoiraiInferenceError(f"Probe classification failed: {exc}") from exc

    @torch.no_grad()
    def _greedy_generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        *,
        maximum_new_tokens: int,
    ) -> torch.LongTensor:
        generated = input_ids.to(self.device).clone()
        mask = attention_mask.to(self.device).clone()
        prompt_length = generated.shape[1]
        for _ in range(maximum_new_tokens):
            outputs = self.model(
                input_ids=generated,
                attention_mask=mask,
                use_cache=False,
                logits_to_keep=1,
            )
            self.formal_forward_count += 1
            next_token = outputs.logits[:, -1].argmax(dim=-1, keepdim=True)
            generated = torch.cat((generated, next_token), dim=1)
            mask = torch.cat((mask, torch.ones_like(next_token)), dim=1)
            if bool(torch.all(next_token == self.tokenizer.eos_token_id)):
                break
        return generated[:, prompt_length:]

    def select_config(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
    ) -> tuple[ProbePrediction, str, MoiraiConfigBundle]:
        probe = self.classify(input_ids, attention_mask)
        selected_config = select_probe_config(probe)
        assert selected_config in ALLOWED_SELECTED_CONFIGS
        assert selected_config == probe.predicted_task
        bundle = self.bundles[selected_config]
        bundle.apply_to_model(self.model)
        if self.model.config.moirai_task != selected_config:
            raise MoiraiInferenceError(
                "Applied model config does not match the binary probe prediction: "
                f"predicted={probe.predicted_task!r}, "
                f"model_config={self.model.config.moirai_task!r}"
            )
        return probe, selected_config, bundle

    def infer(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        *,
        maximum_new_tokens: int,
    ) -> InferenceResult:
        original_ids = input_ids.detach().clone()
        original_mask = attention_mask.detach().clone()
        probe, selected_config, bundle = self.select_config(
            original_ids,
            original_mask,
        )
        try:
            generated = self._greedy_generate(
                original_ids,
                original_mask,
                maximum_new_tokens=maximum_new_tokens,
            )
            return InferenceResult(
                probe=probe,
                selected_config=selected_config,
                generated_ids=generated,
                partition_sha256=bundle.partition.sha256,
                query_sha256=bundle.query_sha256,
            )
        except Exception as exc:
            raise MoiraiInferenceError(
                "Formal MoiraiBlock inference failed for selected config "
                f"{selected_config}: {exc}"
            ) from exc
