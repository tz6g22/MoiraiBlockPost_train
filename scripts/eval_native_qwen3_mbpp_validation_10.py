from __future__ import annotations

import json
import os
import re
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist
from datasets import load_from_disk
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import AutoTokenizer, Qwen3ForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

from src.common import load_yaml
from src.data.format_tasks import nested_value
from src.distributed.fsdp_utils import destroy_distributed, init_distributed
from src.evaluation.task_metrics import extract_code, mbpp_accuracy


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "artifacts/models/Qwen3-14B"
MANIFEST = ROOT / "outputs/formal/evaluation_code_mbpp_validation_10_moirai/evaluation_manifest.jsonl"
OUTPUT = ROOT / "outputs/formal/evaluation_code_mbpp_validation_10_native"
DATA_CONFIG = ROOT / "configs/data.yaml"
MAX_NEW_TOKENS = 512
EOS_TOKEN_ID = 151645
REPEAT_PATTERN = re.compile(r"(.)\1{5,}")


def collapse(text: str) -> bool:
    compact = "".join(text.split())
    return bool(compact and REPEAT_PATTERN.search(compact))


def load_pool(config: dict) -> dict[str, tuple]:
    pool = {}
    source = config["validation_sources"]["code"]
    dataset = load_from_disk(str(ROOT / source["local_path"]))
    split = str(source["official_split"])
    if hasattr(dataset, "keys"):
        dataset = dataset[split]
    pool[split] = (dataset, source["field_mapping"])
    return pool


def make_model(context):
    model = Qwen3ForCausalLM.from_pretrained(
        MODEL,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if not context.distributed:
        return model.to(context.device)
    policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={Qwen3DecoderLayer})
    return FSDP(
        model,
        auto_wrap_policy=policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        use_orig_params=True,
        limit_all_gathers=True,
        sync_module_states=False,
        device_id=None,
    ).to(context.device)


@torch.inference_mode()
def generate(model, input_ids, attention_mask, device):
    generated = input_ids.clone().to(device)
    mask = attention_mask.clone().to(device)
    for _ in range(MAX_NEW_TOKENS):
        outputs = model(
            input_ids=generated,
            attention_mask=mask,
            use_cache=False,
            logits_to_keep=1,
        )
        next_token = outputs.logits[:, -1].argmax(dim=-1, keepdim=True)
        generated = torch.cat((generated, next_token), dim=1)
        mask = torch.cat((mask, torch.ones_like(next_token)), dim=1)
        if bool(torch.all(next_token == EOS_TOKEN_ID)):
            break
    return generated[:, input_ids.shape[1] :], generated.shape[1] - input_ids.shape[1] < MAX_NEW_TOKENS


def main() -> None:
    context = init_distributed()
    config = load_yaml(DATA_CONFIG)
    manifest = [json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines() if line.strip()]
    manifest.sort(key=lambda row: row["stable_id"])
    if len(manifest) != 10 or len({row["stable_id"] for row in manifest}) != 10:
        raise RuntimeError("Expected 10 unique MBPP validation cases")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, use_fast=True)
    if tokenizer.eos_token_id != EOS_TOKEN_ID:
        raise RuntimeError(f"Native EOS mismatch: {tokenizer.eos_token_id}")
    pool = load_pool(config)
    model = make_model(context)
    if context.is_rank0:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        print("FORMAL_EVAL_STARTED model=native_qwen3 task=code case=1/10", flush=True)
    if context.distributed:
        dist.barrier()
    rows = []
    for index, record in enumerate(manifest, start=1):
        dataset, mapping = pool[str(record["official_split"])]
        source = dataset[int(record["row_index"])]
        prompt = record["prompt"]
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        generated, stopped = generate(model, encoded["input_ids"], encoded["attention_mask"], context.device)
        generation = tokenizer.decode(generated[0].cpu().tolist(), skip_special_tokens=True)
        tests = list(nested_value(source, str(mapping["test_list"])))
        setup = str(nested_value(source, str(mapping["test_setup_code"])))
        passed = bool(mbpp_accuracy(generation, test_list=tests, test_setup_code=setup))
        rows.append(
            {
                "stable_id": record["stable_id"],
                "prompt": prompt,
                "gold": str(nested_value(source, str(mapping["target"]))),
                "generation": generation,
                "extracted_code": extract_code(generation),
                "correct": passed,
                "passed": passed,
                "generation_token_count": int(generated.shape[1]),
                "stop_reason": f"eos_{EOS_TOKEN_ID}" if stopped else "max_new_tokens",
                "truncated": not stopped,
                "collapse": collapse(generation),
                "empty_output": not bool(generation.strip()),
                "execution_error": "",
            }
        )
        if context.is_rank0:
            print(f"FORMAL_EVAL_PROGRESS case={index}/10", flush=True)
        del generated, encoded
    if context.is_rank0:
        (OUTPUT / "predictions.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
        )
        (OUTPUT / "evaluation_results.json").write_text(
            json.dumps(
                {
                    "model": "Native Qwen3-14B",
                    "model_path": str(MODEL),
                    "task": "code",
                    "total": 10,
                    "correct": sum(int(row["correct"]) for row in rows),
                    "accuracy": sum(int(row["correct"]) for row in rows) / 10,
                    "collapse": sum(int(row["collapse"]) for row in rows),
                    "truncation": sum(int(row["truncated"]) for row in rows),
                    "empty_output": sum(int(row["empty_output"]) for row in rows),
                    "do_sample": False,
                    "num_beams": 1,
                    "use_cache": False,
                    "logits_to_keep": 1,
                    "max_new_tokens": MAX_NEW_TOKENS,
                    "eos_token_id": EOS_TOKEN_ID,
                    "metric": "mbpp_execution",
                    "attnres_loaded": False,
                    "pseudo_query_loaded": False,
                    "partition_loaded": False,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    destroy_distributed(context)


if __name__ == "__main__":
    main()
