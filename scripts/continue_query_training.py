from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

from src.adapter.train_query import (
    TRAINING_TOKEN_UNIT,
    _load_task_examples,
    _loss_sum,
    _validation_loss,
    frozen_backbone_sha256,
    freeze_for_query_training,
    nonpadding_token_count,
    query_parameter_names,
    AdapterTokenScheduler,
)
from src.common import load_yaml, sha256_file, sha256_json, tokenizer_sha256
from src.data.format_tasks import collate_target_examples, load_dataset_pool, load_manifest
from src.distributed.fsdp_utils import (
    all_reduce_sum,
    barrier,
    broadcast_object,
    clip_grad_norm,
    destroy_distributed,
    init_distributed,
    selected_parameter_sha256,
    selected_parameter_state,
    trainable_parameter_names,
    wrap_qwen3_fsdp,
)
from src.modeling.full_attnres import MoiraiQwen3DecoderLayer, MoiraiQwen3ForCausalLM
from src.modeling.partition import MoiraiPartition


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/base/qwen3_14b_full_attnres"
DATA_CONFIG = ROOT / "configs/data.yaml"
DATA_MANIFEST = ROOT / "outputs/data/splits.json"
TASK_STEPS = {"math": 1000, "multihop": 1000}


def _load_cached_multihop(path: Path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != "moirai_multihop_target_causal_examples_v1":
        raise RuntimeError("Unexpected token-matched Multi-hop cache format")
    from src.data.format_tasks import TargetCausalExample

    return tuple(
        TargetCausalExample(
            input_ids=x["input_ids"].long(),
            attention_mask=x["attention_mask"].long(),
            labels=x["labels"].long(),
            target_mask=x["target_mask"].bool(),
            stable_id=str(x["stable_id"]),
        )
        for x in payload["samples"]
    )


def _query_hash(state: dict[str, torch.Tensor]) -> str:
    return sha256_json({name: value.float().cpu().tolist() for name, value in sorted(state.items())})


def _training_input_hash(examples) -> str:
    digest = hashlib.sha256()
    for example in examples:
        digest.update(example.stable_id.encode("utf-8"))
        for name in ("input_ids", "attention_mask", "labels", "target_mask"):
            value = getattr(example, name).cpu().contiguous()
            digest.update(name.encode("ascii"))
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(str(tuple(value.shape)).encode("ascii"))
            digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=tuple(TASK_STEPS), required=True)
    parser.add_argument("--start-query", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--multihop-cache", type=Path)
    args = parser.parse_args()
    context = init_distributed()
    task = args.task
    if task == "multihop" and args.multihop_cache is None:
        raise ValueError("Multi-hop continuation requires the token-matched cache")

    config = load_yaml(ROOT / "configs/stage3_adapter_lr3e-5.yaml")
    tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True, use_fast=True)
    records = load_manifest(DATA_MANIFEST)
    data_config = load_yaml(DATA_CONFIG)
    if task == "multihop":
        train_examples = _load_cached_multihop(args.multihop_cache)
    else:
        train_examples, invalid = _load_task_examples(
            task=task,
            split_name="stage3_adapter_train",
            records=records,
            data_config=data_config,
            tokenizer=tokenizer,
            expected_count=1000,
        )
        if invalid:
            raise RuntimeError(f"Unexpected invalid {task} examples: {invalid[:2]}")
    if len(train_examples) != 1000 or len({x.stable_id for x in train_examples}) != 1000:
        raise RuntimeError("Continuation requires 1000 unique training examples")
    pass_examples = list(train_examples)
    random.Random(42).shuffle(pass_examples)

    partition = MoiraiPartition.from_json(ROOT / "outputs/formal/discovery" / task / "partition.json")
    start_hash = sha256_file(args.start_query)
    if context.rank == 0:
        if args.output.exists() and any(args.output.iterdir()):
            raise FileExistsError(f"Refusing non-empty continuation output: {args.output}")
        args.output.mkdir(parents=True, exist_ok=True)
    barrier(context)
    model = MoiraiQwen3ForCausalLM.from_pretrained(
        BASE, local_files_only=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    model.config.attnres_execution = "moirai"
    model.config.moirai_partition = list(partition.lengths)
    model.config.moirai_task = task
    model.config.use_cache = False
    state = load_file(str(args.start_query), device="cpu")
    names = query_parameter_names(model)
    if set(state) != set(names):
        raise RuntimeError("Starting query parameter names do not match Moirai model")
    with torch.no_grad():
        named = dict(model.named_parameters())
        for name, value in state.items():
            named[name].copy_(value.to(dtype=named[name].dtype))
    freeze_for_query_training(model)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    backbone_before = frozen_backbone_sha256(model)
    model = wrap_qwen3_fsdp(
        model, context, decoder_layer_classes=(MoiraiQwen3DecoderLayer,),
        sync_module_states=False, device_id=None,
    )
    if context.distributed:
        model = model.to(context.device)
    if trainable_parameter_names(model) != tuple(names):
        raise RuntimeError("Continuation optimizer is not pseudo-query-only")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=3.0e-5, betas=(0.9, 0.999), eps=1.0e-8, weight_decay=0.0,
    )
    planned_tokens = sum(int(x.attention_mask.sum()) for x in pass_examples)
    scheduler = AdapterTokenScheduler(
        optimizer, base_learning_rate=3.0e-5, maximum_tokens=planned_tokens, warmup_ratio=0.02
    )
    trained_tokens = 0
    for step, example in enumerate(pass_examples, start=1):
        model.train()
        batch = collate_target_examples([example], pad_token_id=tokenizer.pad_token_id)
        target_mask = batch.pop("target_mask").to(context.device)
        labels = batch.pop("labels").to(context.device)
        active = context.rank == (step - 1) % context.world_size
        if not active:
            target_mask.zero_()
        local_input_count = nonpadding_token_count(batch["attention_mask"]) if active else 0
        input_count = torch.tensor(local_input_count, device=context.device, dtype=torch.long)
        all_reduce_sum(input_count, context)
        inputs = {key: value.to(context.device) for key, value in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(**inputs, use_cache=False).logits
        local_loss, local_count = _loss_sum(logits, labels, target_mask)
        count = torch.tensor(local_count, device=context.device, dtype=torch.long)
        all_reduce_sum(count, context)
        loss = local_loss * context.world_size / int(count.item())
        loss.backward()
        clip_grad_norm(model, (p for p in model.parameters() if p.requires_grad), 1.0)
        optimizer.step()
        trained_tokens += int(input_count.item())
        scheduler.step(trained_tokens)
        if context.rank == 0 and (step == 1 or step % 100 == 0):
            (args.output / "training_progress.json").write_text(json.dumps({
                "task": task, "additional_step": step, "additional_steps": 1000,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "trained_tokens": trained_tokens, "planned_tokens": planned_tokens,
            }, indent=2) + "\n")
        if step % 100 == 0:
            qstate = selected_parameter_state(model, context, lambda n, _: "pseudo_query" in n)
            if context.rank == 0:
                assert qstate is not None
                save_file(qstate, args.output / f"query_step_{1000 + step:04d}.safetensors")
            barrier(context)
        del logits, loss, labels, target_mask, inputs, batch, input_count, count

    backbone_after = selected_parameter_sha256(model, context, lambda n, _: "pseudo_query" not in n)
    if backbone_after != backbone_before:
        raise RuntimeError("FAILED_FROZEN_BACKBONE_CHECK")
    qstate = selected_parameter_state(model, context, lambda n, _: "pseudo_query" in n)
    if context.rank == 0:
        assert qstate is not None
        final_path = args.output / "final_query.safetensors"
        save_file(qstate, final_path)
        final_hash = sha256_file(final_path)
        (args.output / "training_manifest.json").write_text(json.dumps({
            "task": task, "continuation": True, "continuation_from": str(args.start_query),
            "start_query_sha256": start_hash, "final_query_sha256": final_hash,
            "optimizer_steps": 2000, "additional_optimizer_steps": 1000,
            "training_cases": 1000, "trained_tokens_additional": trained_tokens,
            "training_input_sha256": _training_input_hash(train_examples),
            "base_checkpoint": str(BASE), "base_checkpoint_sha256": "10e2c2d7c5ebc2366b0559e0b4cf79ab9f88bff99eef0b2e058f87695c2e265d",
            "partition_sha256": partition.sha256, "learning_rate": 3.0e-5,
            "optimizer": "AdamW", "scheduler": "cosine", "warmup_ratio": 0.02,
            "backbone_before_sha256": backbone_before, "backbone_after_sha256": backbone_after,
            "backbone_unchanged": True, "completion_status": "complete",
        }, indent=2) + "\n")
    barrier(context)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    destroy_distributed(context)


if __name__ == "__main__":
    main()
