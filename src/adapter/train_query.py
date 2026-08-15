from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from transformers import AutoTokenizer

from src.common import (
    config_sha256,
    load_yaml,
    sha256_file,
    sha256_json,
    tokenizer_sha256,
)
from src.data.format_tasks import (
    SUPERVISED_TASKS,
    TASK_TO_SOURCE,
    collate_target_examples,
    encode_prompt_target,
    load_dataset_pool,
    load_manifest,
)
from src.distributed.fsdp_utils import (
    DistributedContext,
    all_reduce_sum,
    barrier,
    broadcast_object,
    clip_grad_norm,
    destroy_distributed,
    full_optimizer_state,
    init_distributed,
    load_full_optimizer_state,
    selected_parameter_sha256,
    selected_parameter_state,
    trainable_parameter_names,
    wrap_qwen3_fsdp,
)
from src.modeling.full_attnres import MoiraiQwen3DecoderLayer, MoiraiQwen3ForCausalLM
from src.modeling.partition import MoiraiPartition
from src.training.checkpointing import (
    pseudo_query_sha256,
    validate_post_training_base_manifest,
)


TRAINING_TOKEN_UNIT = "nonpadding_input"


def query_parameter_names(model) -> tuple[str, ...]:
    return tuple(
        sorted(name for name, _ in model.named_parameters() if "pseudo_query" in name)
    )


def freeze_for_query_training(model) -> tuple[str, ...]:
    names = query_parameter_names(model)
    if not names:
        raise ValueError("Model has no pseudo-query parameters")
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in names)
    actual_trainable = tuple(
        sorted(name for name, parameter in model.named_parameters() if parameter.requires_grad)
    )
    if actual_trainable != names:
        raise RuntimeError("Failed to isolate pseudo-query trainable parameters")
    return names


def zero_initialize_pseudo_queries(model) -> tuple[str, ...]:
    names = query_parameter_names(model)
    if not names:
        raise ValueError("Model has no pseudo-query parameters")
    named = dict(model.named_parameters())
    with torch.no_grad():
        for name in names:
            named[name].zero_()
    return names


def frozen_backbone_sha256(model) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        if "pseudo_query" in name:
            continue
        value = parameter.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def save_query_checkpoint(
    model,
    output_dir: str | Path,
    *,
    task: str,
    partition: MoiraiPartition,
    base_checkpoint_sha256: str,
    trained_tokens: int,
    processed_nonpadding_tokens: int,
    training_token_unit: str,
    validation_loss: float | None = None,
    backbone_before_sha256: str | None = None,
    backbone_after_sha256: str | None = None,
    frozen_backbone: bool | None = None,
    seed: int = 42,
    optimizer_steps: int | None = None,
    query_before_sha256: str | None = None,
    query_after_sha256: str | None = None,
    consumed_stable_ids: list[str] | None = None,
    training_passes: int = 1,
    unique_training_examples: int | None = None,
    processed_training_examples: int | None = None,
    query_state_override: dict[str, torch.Tensor] | None = None,
    trainable_override: tuple[str, ...] | None = None,
) -> dict:
    if partition.task != task:
        raise ValueError("Task and partition task do not match")
    if query_state_override is None:
        trainable = freeze_for_query_training(model)
        query_state = {
            name: parameter.detach().cpu().contiguous()
            for name, parameter in model.named_parameters()
            if name in trainable
        }
    else:
        query_state = query_state_override
        trainable = trainable_override or tuple(sorted(query_state))
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    query_file = output_path / "final_query.safetensors"
    save_file(query_state, query_file)
    manifest = {
        "task": task,
        "base_checkpoint_sha256": base_checkpoint_sha256,
        "partition_sha256": partition.sha256,
        "query_sha256": sha256_file(query_file),
        "query_file": query_file.name,
        "trainable_parameters": list(trainable),
        "trained_tokens": trained_tokens,
        "processed_nonpadding_tokens": processed_nonpadding_tokens,
        "training_token_unit": training_token_unit,
        "validation_loss": validation_loss,
        "backbone_before_sha256": backbone_before_sha256,
        "backbone_after_sha256": backbone_after_sha256,
        "frozen_backbone": frozen_backbone,
        "seed": seed,
        "optimizer_steps": optimizer_steps,
        "query_before_sha256": query_before_sha256,
        "query_after_sha256": query_after_sha256,
        "query_changed": (
            query_before_sha256 != query_after_sha256
            if query_before_sha256 is not None and query_after_sha256 is not None
            else None
        ),
        "consumed_stable_ids": consumed_stable_ids,
        "training_passes": training_passes,
        "unique_training_examples": unique_training_examples,
        "processed_training_examples": processed_training_examples,
    }
    (output_path / "query_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


class AdapterTokenScheduler:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        base_learning_rate: float,
        maximum_tokens: int,
        warmup_ratio: float,
    ) -> None:
        self.optimizer = optimizer
        self.base_learning_rate = base_learning_rate
        self.maximum_tokens = maximum_tokens
        self.warmup_tokens = int(maximum_tokens * warmup_ratio)
        self.trained_tokens = 0

    def step(self, trained_tokens: int) -> None:
        self.trained_tokens = int(trained_tokens)
        if trained_tokens < self.warmup_tokens:
            ratio = trained_tokens / max(1, self.warmup_tokens)
        else:
            progress = min(
                1.0,
                (trained_tokens - self.warmup_tokens)
                / max(1, self.maximum_tokens - self.warmup_tokens),
            )
            ratio = 0.5 * (1.0 + math.cos(math.pi * progress))
        for group in self.optimizer.param_groups:
            group["lr"] = self.base_learning_rate * ratio

    def state_dict(self) -> dict[str, int]:
        return {"trained_tokens": self.trained_tokens}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.step(int(state["trained_tokens"]))


def _checkpoint_weight(checkpoint: Path) -> tuple[Path, str, dict[str, Any]]:
    candidates = sorted(checkpoint.glob("model*.safetensors"))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one base model file, found {candidates}")
    weight = candidates[0]
    manifest = json.loads(
        (checkpoint / "checkpoint_manifest.json").read_text(encoding="utf-8")
    )
    actual = sha256_file(weight)
    if actual != manifest.get("model_weights_sha256"):
        raise ValueError("Base checkpoint hash mismatch")
    validate_post_training_base_manifest(manifest)
    return weight, actual, manifest


def _setup_distributed() -> tuple[int, int, int, torch.device]:
    context = init_distributed()
    return context.rank, context.world_size, context.local_rank, context.device


def _load_task_examples(
    *,
    task: str,
    split_name: str,
    records: list[dict[str, Any]],
    data_config: dict[str, Any],
    tokenizer,
    expected_count: int | None = None,
):
    source_name = TASK_TO_SOURCE[task]
    source = data_config["sources"][source_name]
    selected = [
        record
        for record in records
        if record["dataset"] == source["dataset_name"]
        and record["task"] == task
        and record["assigned_split"] == split_name
    ]
    selected.sort(key=lambda record: record["split_key"])
    if expected_count is not None and len(selected) != expected_count:
        raise RuntimeError(
            f"{task}/{split_name} requires exactly {expected_count} cases, "
            f"found {len(selected)}"
        )
    pool = load_dataset_pool(data_config, str(source["dataset_name"]))
    examples = []
    invalid: list[dict[str, str]] = []
    for record in selected:
        try:
            dataset, field_mapping = pool[str(record["official_split"])]
            examples.append(
                encode_prompt_target(
                    tokenizer,
                    task=task,
                    row=dataset[int(record["row_index"])],
                    field_mapping=field_mapping,
                    stable_id=record["stable_id"],
                    max_length=2048,
                )
            )
        except ValueError as exc:
            if "exceeds" not in str(exc):
                raise
            invalid.append({"stable_id": record["stable_id"], "reason": str(exc)})
    if not examples:
        raise RuntimeError(f"No valid {task} examples in {split_name}")
    if expected_count is not None and len(examples) != expected_count:
        raise RuntimeError(
            f"{task}/{split_name} has {len(examples)} valid cases after encoding; "
            f"expected exactly {expected_count}; invalid={invalid[:3]}"
        )
    return tuple(examples), invalid


def _loss_sum(
    logits: torch.Tensor,
    labels: torch.LongTensor,
    target_mask: torch.BoolTensor,
) -> tuple[torch.Tensor, int]:
    effective = labels.masked_fill(~target_mask, -100)
    count = int((effective != -100).sum().item())
    if count == 0:
        return logits.float().sum() * 0.0, 0
    value = F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        effective.reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )
    return value, count


def nonpadding_token_count(attention_mask: torch.Tensor) -> int:
    if attention_mask.ndim != 2:
        raise ValueError("Adapter attention_mask must have shape [batch, tokens]")
    count = int(attention_mask.to(dtype=torch.long).sum().item())
    if count <= 0:
        raise RuntimeError("Adapter batch has no non-padding input tokens")
    return count


def _write_training_progress(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


@torch.no_grad()
def _validation_loss(
    model,
    examples,
    *,
    tokenizer,
    context: DistributedContext,
) -> tuple[float, int]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for case_index, example in enumerate(examples):
        batch = collate_target_examples(
            [example],
            pad_token_id=tokenizer.pad_token_id,
        )
        target_mask = batch.pop("target_mask").to(context.device)
        labels = batch.pop("labels").to(context.device)
        if context.rank != case_index % context.world_size:
            target_mask.zero_()
        inputs = {key: value.to(context.device) for key, value in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(**inputs, use_cache=False).logits
        loss, count = _loss_sum(logits, labels, target_mask)
        reduced_loss = all_reduce_sum(loss.detach(), context)
        reduced_count = all_reduce_sum(
            torch.tensor(count, dtype=torch.long, device=context.device),
            context,
        )
        total_loss += float(reduced_loss.cpu())
        total_tokens += int(reduced_count.item())
    return total_loss / total_tokens, total_tokens


def _load_query_state(model, state: dict[str, torch.Tensor]) -> None:
    named = dict(model.named_parameters())
    expected = set(query_parameter_names(model))
    if set(state) != expected:
        raise ValueError("Resume query parameter names do not match the model")
    with torch.no_grad():
        for name, value in state.items():
            named[name].copy_(value.to(device=named[name].device, dtype=named[name].dtype))


def train_query_partition(
    *,
    task: str,
    partition: MoiraiPartition,
    output_dir: Path,
    checkpoint: Path,
    base_checkpoint_hash: str,
    expected_q_full_hash: str,
    expected_q_full_names: list[str],
    config: dict[str, Any],
    run_config_hash: str,
    train_examples,
    validation_examples,
    tokenizer,
    context: DistributedContext,
    resume: bool,
) -> dict[str, Any] | None:
    rank = context.rank
    world_size = context.world_size
    device = context.device
    if not resume and (output_dir / "query_manifest.json").exists():
        raise FileExistsError(
            f"Refusing to overwrite completed adapter without --resume: {output_dir}"
        )
    model = MoiraiQwen3ForCausalLM.from_pretrained(
        checkpoint,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    actual_query_names, actual_q_full_hash = pseudo_query_sha256(model) if rank == 0 else (None, None)
    actual_query_names = broadcast_object(actual_query_names, context)
    actual_q_full_hash = broadcast_object(actual_q_full_hash, context)
    if actual_query_names != expected_q_full_names:
        raise ValueError("Adapter base Q_full parameter names mismatch")
    if actual_q_full_hash != expected_q_full_hash:
        raise ValueError("Adapter base Q_full hash mismatch")
    model.config.attnres_execution = "moirai"
    model.config.moirai_partition = list(partition.lengths)
    model.config.moirai_task = task
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    if not resume:
        zero_initialize_pseudo_queries(model)
    freeze_for_query_training(model)
    _, query_before_hash = pseudo_query_sha256(model) if rank == 0 else (None, None)
    query_before_hash = broadcast_object(query_before_hash, context)
    backbone_before = broadcast_object(
        frozen_backbone_sha256(model) if rank == 0 else None,
        context,
    )
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    barrier(context)
    resume_state = None
    if resume:
        state_path = output_dir / "last_state.pt"
        if not state_path.is_file():
            raise FileNotFoundError(f"Adapter resume state is missing: {state_path}")
        resume_state = torch.load(state_path, map_location="cpu", weights_only=False)
        if resume_state["run_config_sha256"] != run_config_hash:
            raise ValueError("Adapter resume config hash mismatch")
        if resume_state["base_checkpoint_sha256"] != base_checkpoint_hash:
            raise ValueError("Adapter resume base checkpoint hash mismatch")
        if resume_state["partition_sha256"] != partition.sha256:
            raise ValueError("Adapter resume partition hash mismatch")
        _load_query_state(model, resume_state["query_state"])

    model = wrap_qwen3_fsdp(
        model,
        context,
        decoder_layer_classes=(MoiraiQwen3DecoderLayer,),
    )
    if trainable_parameter_names(model) != tuple(expected_q_full_names):
        raise RuntimeError("FSDP optimizer parameter set is not exactly pseudo-query")

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(config["learning_rate"]),
        betas=tuple(config["betas"]),
        eps=float(config["eps"]),
        weight_decay=float(config["weight_decay"]),
    )
    pass_examples = list(train_examples)
    random.Random(int(config["seed"])).shuffle(pass_examples)
    stable_ids = [example.stable_id for example in pass_examples]
    if len(stable_ids) != len(set(stable_ids)):
        raise RuntimeError("One-pass query training examples contain duplicate stable IDs")
    maximum_examples = len(pass_examples)
    if maximum_examples <= 0:
        raise RuntimeError("One-pass query training requires at least one example")
    maximum_tokens = sum(
        int(example.attention_mask.to(dtype=torch.long).sum().item())
        for example in pass_examples
    )
    if maximum_tokens <= 0:
        raise RuntimeError("One-pass query training has no non-padding input tokens")
    scheduler = AdapterTokenScheduler(
        optimizer,
        base_learning_rate=float(config["learning_rate"]),
        maximum_tokens=maximum_tokens,
        warmup_ratio=float(config["warmup_ratio"]),
    )
    trained_tokens = 0
    processed_nonpadding_tokens = 0
    global_step = 0
    checkpoint_interval_steps = int(config["checkpoint_interval_steps"])
    if resume_state is not None:
        load_full_optimizer_state(model, optimizer, resume_state["optimizer_state"])
        if int(resume_state["maximum_training_examples"]) != maximum_examples:
            raise ValueError("Adapter resume one-pass example count differs")
        if int(resume_state["planned_training_tokens"]) != maximum_tokens:
            raise ValueError("Adapter resume one-pass token total differs")
        scheduler.load_state_dict(resume_state["scheduler"])
        trained_tokens = int(resume_state["trained_tokens"])
        if resume_state.get("training_token_unit") != TRAINING_TOKEN_UNIT:
            raise ValueError("Adapter resume training token unit mismatch")
        processed_nonpadding_tokens = int(
            resume_state["processed_nonpadding_tokens"]
        )
        global_step = int(resume_state["global_step"])
        if not 0 <= global_step <= maximum_examples:
            raise ValueError("Adapter resume step is outside the one-pass range")
        expected_prefix_tokens = sum(
            int(example.attention_mask.to(dtype=torch.long).sum().item())
            for example in pass_examples[:global_step]
        )
        if trained_tokens != expected_prefix_tokens:
            raise ValueError("Adapter resume token count is not a one-pass prefix")

    progress_interval_steps = int(config["progress_interval_steps"])
    progress_path = output_dir / "training_progress.json"
    last_gradient_norm = 0.0
    while global_step < maximum_examples:
        model.train()
        example = pass_examples[global_step]
        batch = collate_target_examples(
            [example],
            pad_token_id=tokenizer.pad_token_id,
        )
        target_mask = batch.pop("target_mask").to(device)
        labels = batch.pop("labels").to(device)
        active = rank == global_step % world_size
        if not active:
            target_mask.zero_()
        local_nonpadding_count = (
            nonpadding_token_count(batch["attention_mask"]) if active else 0
        )
        input_count_tensor = torch.tensor(
            local_nonpadding_count,
            device=device,
            dtype=torch.long,
        )
        all_reduce_sum(input_count_tensor, context)
        global_nonpadding_count = int(input_count_tensor.item())
        inputs = {key: value.to(device) for key, value in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(**inputs, use_cache=False).logits
        local_loss, local_count = _loss_sum(logits, labels, target_mask)
        count_tensor = torch.tensor(local_count, device=device, dtype=torch.long)
        all_reduce_sum(count_tensor, context)
        global_count = int(count_tensor.item())
        if global_count == 0:
            raise RuntimeError("Adapter batch has no target tokens")
        loss = local_loss * world_size / global_count
        loss.backward()
        gradient_norm = clip_grad_norm(
            model,
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            float(config["gradient_clip_norm"]),
        )
        if isinstance(gradient_norm, torch.Tensor):
            gradient_norm = gradient_norm.detach().float().cpu().item()
        last_gradient_norm = float(gradient_norm)
        if not math.isfinite(last_gradient_norm) or last_gradient_norm <= 0.0:
            raise FloatingPointError("Adapter gradient norm is not positive and finite")
        optimizer.step()
        processed_nonpadding_tokens += global_nonpadding_count
        trained_tokens += global_nonpadding_count
        global_step += 1
        scheduler.step(trained_tokens)
        if rank == 0 and (
            global_step % progress_interval_steps == 0
            or global_step >= maximum_examples
        ):
            _write_training_progress(
                progress_path,
                {
                    "task": task,
                    "training_token_unit": TRAINING_TOKEN_UNIT,
                    "trained_tokens": trained_tokens,
                    "processed_nonpadding_tokens": processed_nonpadding_tokens,
                    "planned_training_tokens": maximum_tokens,
                    "training_passes": 1,
                    "unique_training_examples": maximum_examples,
                    "processed_training_examples": global_step,
                    "optimizer_steps": global_step,
                    "last_gradient_norm": last_gradient_norm,
                    "progress_ratio": global_step / maximum_examples,
                },
            )
        if (
            global_step % checkpoint_interval_steps == 0
            and global_step < maximum_examples
        ):
            query_state = selected_parameter_state(
                model,
                context,
                lambda name, _parameter: "pseudo_query" in name,
            )
            optimizer_state = full_optimizer_state(model, optimizer, context)
            if rank == 0:
                assert query_state is not None and optimizer_state is not None
                torch.save(
                    {
                        "run_config_sha256": run_config_hash,
                        "base_checkpoint_sha256": base_checkpoint_hash,
                        "partition_sha256": partition.sha256,
                        "query_state": query_state,
                        "optimizer_state": optimizer_state,
                        "scheduler": scheduler.state_dict(),
                        "trained_tokens": trained_tokens,
                        "processed_nonpadding_tokens": processed_nonpadding_tokens,
                        "training_token_unit": TRAINING_TOKEN_UNIT,
                        "global_step": global_step,
                        "maximum_training_examples": maximum_examples,
                        "planned_training_tokens": maximum_tokens,
                    },
                    output_dir / "last_state.pt",
                )
            barrier(context)

    if global_step != maximum_examples:
        raise RuntimeError("One-pass query training did not consume every example")
    if trained_tokens != maximum_tokens:
        raise RuntimeError("One-pass query training token total changed during execution")

    validation_loss, _ = _validation_loss(
        model,
        validation_examples,
        tokenizer=tokenizer,
        context=context,
    )
    backbone_after = selected_parameter_sha256(
        model,
        context,
        lambda name, _parameter: "pseudo_query" not in name,
    )
    if backbone_after != backbone_before:
        raise RuntimeError("FAILED_FROZEN_BACKBONE_CHECK")
    query_state = selected_parameter_state(
        model,
        context,
        lambda name, _parameter: "pseudo_query" in name,
    )
    if rank == 0:
        assert query_state is not None
        query_after_hash = sha256_json(
            {
                name: tensor.detach().float().cpu().tolist()
                for name, tensor in query_state.items()
            }
        )
        if query_after_hash == query_before_hash:
            raise RuntimeError("Adapter optimizer steps did not change pseudo-query")
        return save_query_checkpoint(
            model,
            output_dir,
            task=task,
            partition=partition,
            base_checkpoint_sha256=base_checkpoint_hash,
            trained_tokens=trained_tokens,
            processed_nonpadding_tokens=processed_nonpadding_tokens,
            training_token_unit=TRAINING_TOKEN_UNIT,
            validation_loss=validation_loss,
            backbone_before_sha256=backbone_before,
            backbone_after_sha256=backbone_after,
            frozen_backbone=True,
            seed=int(config["seed"]),
            optimizer_steps=global_step,
            query_before_sha256=query_before_hash,
            query_after_sha256=query_after_hash,
            consumed_stable_ids=sorted(
                example.stable_id for example in train_examples
            ),
            training_passes=1,
            unique_training_examples=maximum_examples,
            processed_training_examples=global_step,
            query_state_override=query_state,
            trainable_override=tuple(expected_q_full_names),
        )
    return None


def validate_query_training_protocol(
    config: dict[str, Any],
    *,
    stage_name: str,
) -> None:
    expected = {
        "seed": 42,
        "precision": "bf16",
        "distributed": "torchrun_fsdp_full_shard",
        "use_cache": False,
        "trainable_parameters": "pseudo_query_only",
        "optimizer": "AdamW",
        "learning_rate": 1.0e-3,
        "betas": [0.9, 0.999],
        "eps": 1.0e-8,
        "weight_decay": 0.0,
        "gradient_clip_norm": 1.0,
        "scheduler": "cosine",
        "warmup_ratio": 0.02,
        "training_token_unit": TRAINING_TOKEN_UNIT,
        "training_passes": 1,
        "micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 1,
    }
    for key, expected_value in expected.items():
        if config.get(key) != expected_value:
            raise ValueError(
                f"{stage_name} config mismatch for {key}: "
                f"expected {expected_value!r}, got {config.get(key)!r}"
            )
    for key in (
        "checkpoint_interval_steps",
        "progress_interval_steps",
    ):
        if int(config.get(key, 0)) <= 0:
            raise ValueError(f"{stage_name} {key} must be positive")
    if "training_cases_per_task" not in config:
        raise ValueError(f"{stage_name} training_cases_per_task is missing")


def validate_adapter_config(config: dict[str, Any]) -> None:
    validate_query_training_protocol(config, stage_name="Stage 3")
    expected_case_counts = {"math": 1000, "multihop": 1000, "code": 200}
    if config.get("training_cases_per_task") != expected_case_counts:
        raise ValueError(
            "Task query training case counts must be "
            f"{expected_case_counts!r}"
        )
    for key in {
        "base_checkpoint",
        "data_manifest",
        "data_config",
        "partition_root",
        "output_root",
    }:
        if key not in config:
            raise ValueError(f"Stage 3 config is missing {key}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=SUPERVISED_TASKS)
    parser.add_argument("--config", default="configs/stage3_adapter.yaml")
    parser.add_argument("--resume", nargs="?", const="auto", default="")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--data-manifest", default="")
    parser.add_argument("--partition-root", default="")
    parser.add_argument("--output-root", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    validate_adapter_config(config)
    context = init_distributed()
    run_config_hash = config_sha256(args.config)
    checkpoint = Path(args.checkpoint or config["base_checkpoint"])
    checkpoint_identity = broadcast_object(
        _checkpoint_weight(checkpoint) if context.is_rank0 else None,
        context,
    )
    _, base_checkpoint_hash, checkpoint_manifest = checkpoint_identity
    partition_root = Path(args.partition_root or config["partition_root"])
    output_root = Path(args.output_root or config["output_root"])
    data_manifest_path = Path(args.data_manifest or config["data_manifest"])
    partition_path = partition_root / args.task / "partition.json"
    if not partition_path.is_file() or not data_manifest_path.is_file():
        raise FileNotFoundError(
            "FAILED: Stage 3 partition or data manifest is missing"
        )
    partition = MoiraiPartition.from_json(partition_path)
    if partition.task != args.task:
        raise ValueError("Requested task does not match partition task")
    data_config = load_yaml(config["data_config"])
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        local_files_only=True,
        use_fast=True,
    )
    if tokenizer_sha256(tokenizer) != checkpoint_manifest.get("tokenizer_sha256"):
        raise ValueError("Adapter checkpoint tokenizer hash mismatch")
    records = load_manifest(data_manifest_path)
    train_examples, _ = _load_task_examples(
        task=args.task,
        split_name="stage3_adapter_train",
        records=records,
        data_config=data_config,
        tokenizer=tokenizer,
        expected_count=int(config["training_cases_per_task"][args.task]),
    )
    validation_examples, _ = _load_task_examples(
        task=args.task,
        split_name="stage3_adapter_val",
        records=records,
        data_config=data_config,
        tokenizer=tokenizer,
    )
    random.seed(int(config["seed"]))
    torch.manual_seed(int(config["seed"]))
    torch.cuda.manual_seed_all(int(config["seed"]))
    output_dir = output_root / args.task
    state_path = output_dir / "last_state.pt"
    if args.resume == "auto":
        resume_task = state_path.is_file()
    elif args.resume:
        requested = Path(args.resume).resolve()
        resume_task = (
            requested == output_dir.resolve()
            or requested == state_path.resolve()
        )
    else:
        resume_task = False
    train_query_partition(
        task=args.task,
        partition=partition,
        output_dir=output_dir,
        checkpoint=checkpoint,
        base_checkpoint_hash=base_checkpoint_hash,
        expected_q_full_hash=checkpoint_manifest["q_full_sha256"],
        expected_q_full_names=checkpoint_manifest["q_full_parameter_names"],
        config=config,
        run_config_hash=run_config_hash,
        train_examples=train_examples,
        validation_examples=validation_examples,
        tokenizer=tokenizer,
        context=context,
        resume=resume_task,
    )
    barrier(context)
    gc.collect()
    torch.cuda.empty_cache()
    destroy_distributed(context)


if __name__ == "__main__":
    main()
