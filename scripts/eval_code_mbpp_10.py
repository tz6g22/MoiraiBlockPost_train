from __future__ import annotations

import json
import os
import re
from pathlib import Path

import torch
from transformers import AutoTokenizer

from src.common import load_yaml
from src.data.format_tasks import encode_prompt_only, format_task_prompt, load_dataset_pool, nested_value
from src.data.format_tasks import load_manifest
from src.distributed.fsdp_utils import barrier, destroy_distributed, init_distributed, broadcast_object, wrap_qwen3_fsdp
from src.evaluation.run_evaluation import _base_checkpoint_hash, _greedy_generate, _load_bundle, GENERATION_LIMITS
from src.evaluation.task_metrics import extract_code, mbpp_accuracy
from src.modeling.full_attnres import MoiraiQwen3DecoderLayer, MoiraiQwen3ForCausalLM


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/base/qwen3_14b_full_attnres"
DATA_CONFIG = ROOT / "configs/data.yaml"
MANIFEST = Path(os.environ.get("CODE_EVAL_MANIFEST", ROOT / "outputs/data/splits.json"))
INTERFACE_MANIFEST = os.environ.get("CODE_EVAL_INTERFACE_MANIFEST")
OUT = Path(os.environ.get("CODE_EVAL_OUT", ROOT / "outputs/formal/evaluation_code_mbpp_10"))
EXPECTED_CASES = int(os.environ.get("CODE_EVAL_CASES", "10"))
PARTITION = ROOT / "outputs/formal/discovery/code/partition.json"
QUERY_MANIFEST = ROOT / "outputs/formal/adapter_lr3e-5/code/query_manifest.json"
MAX_NEW_TOKENS = GENERATION_LIMITS["code"]
EXPECTED_QUERY_HASH = ""
EXPECTED_PARTITION_HASH = ""
REPEAT_PATTERN = re.compile(r"(.)\1{5,}")


def collapse_flag(text: str) -> bool:
    compact = "".join(text.split())
    return bool(compact and REPEAT_PATTERN.search(compact))


def main() -> None:
    context = init_distributed()
    rank, device = context.rank, context.device
    data_config = load_yaml(DATA_CONFIG)
    if INTERFACE_MANIFEST:
        selected = [
            json.loads(line)
            for line in Path(INTERFACE_MANIFEST).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        manifest = load_manifest(MANIFEST)
        selected = sorted(
            (row for row in manifest if row.get("task") == "code" and row.get("assigned_split") == "stage4_final_eval"),
            key=lambda row: row["stable_id"],
        )
    if len(selected) != EXPECTED_CASES:
        raise RuntimeError(f"Code evaluation requires exactly {EXPECTED_CASES} cases, found {len(selected)}")
    pool = load_dataset_pool(data_config, "mbpp")
    base_hash = broadcast_object(_base_checkpoint_hash(BASE) if context.is_rank0 else None, context)
    tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True, use_fast=True)
    model = MoiraiQwen3ForCausalLM.from_pretrained(BASE, local_files_only=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    bundle = _load_bundle(partition_path=PARTITION, query_manifest_path=QUERY_MANIFEST, base_hash=base_hash)
    model = wrap_qwen3_fsdp(model, context, decoder_layer_classes=(MoiraiQwen3DecoderLayer,), sync_module_states=False, device_id=None).to(device)
    bundle.apply_to_model(model)
    model.eval()
    if rank == 0:
        OUT.mkdir(parents=True, exist_ok=True)
        print(f"FORMAL_EVAL_STARTED model=moiraiblock task=code case=1/{EXPECTED_CASES}", flush=True)
    barrier(context)
    output_rows: list[dict] = []
    for index, record in enumerate(selected):
        dataset, mapping = pool[str(record["official_split"])]
        row = dataset[int(record["row_index"])]
        prompt_string = record.get("prompt") or format_task_prompt("code", row, mapping)
        if record.get("prompt_token_ids") is not None:
            input_ids = torch.tensor(record["prompt_token_ids"], dtype=torch.long, device=device).unsqueeze(0)
            attention_mask = torch.ones_like(input_ids)
        else:
            prompt = encode_prompt_only(tokenizer, task="code", row=row, field_mapping=mapping, stable_id=record["stable_id"], max_length=2048 - MAX_NEW_TOKENS)
            input_ids = prompt.input_ids.unsqueeze(0).to(device)
            attention_mask = prompt.attention_mask.unsqueeze(0).to(device)
        generated_ids = _greedy_generate(model, input_ids, attention_mask, maximum_new_tokens=MAX_NEW_TOKENS, eos_token_id=tokenizer.eos_token_id)
        generation = tokenizer.decode(generated_ids[0].cpu(), skip_special_tokens=True)
        tests = list(nested_value(row, str(mapping["test_list"])))
        setup = str(nested_value(row, str(mapping["test_setup_code"])))
        passed = bool(mbpp_accuracy(generation, test_list=tests, test_setup_code=setup))
        output_rows.append({
            "stable_id": record["stable_id"], "prompt": prompt_string,
            "prompt_token_ids": record.get("prompt_token_ids"),
            "required_function_name": record.get("required_function_name"),
            "required_signature": record.get("required_signature"),
            "gold": str(nested_value(row, str(mapping["target"]))), "reference": str(nested_value(row, str(mapping["target"]))),
            "generation": generation, "extracted_code": extract_code(generation),
            "correct": passed, "passed": passed, "generation_token_count": int(generated_ids.shape[1]),
            "stop_reason": f"eos_{tokenizer.eos_token_id}" if generated_ids.shape[1] < MAX_NEW_TOKENS else "max_new_tokens",
            "truncated": int(generated_ids.shape[1]) >= MAX_NEW_TOKENS,
            "collapse": collapse_flag(generation), "empty_output": not bool(generation.strip()),
        })
        del generated_ids, input_ids, attention_mask
        if rank == 0:
            print(f"FORMAL_EVAL_PROGRESS case={index + 1}/{EXPECTED_CASES}", flush=True)
    if rank == 0:
        (OUT / "predictions.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output_rows), encoding="utf-8")
        (OUT / "evaluation_results.json").write_text(json.dumps({
            "model": "MoiraiBlock", "task": "code", "base_checkpoint": str(BASE), "base_checkpoint_hash": base_hash,
            "partition_path": str(PARTITION), "partition_hash": bundle.partition.sha256, "query_path": str(bundle.query_path), "query_hash": bundle.query_sha256,
            "total": len(output_rows), "correct": sum(int(row["correct"]) for row in output_rows), "accuracy": sum(int(row["correct"]) for row in output_rows) / len(output_rows),
            "collapse": sum(int(row["collapse"]) for row in output_rows), "empty_output": sum(int(row["empty_output"]) for row in output_rows), "truncation": sum(int(row["truncated"]) for row in output_rows),
            "prompt_format": "Problem: ...\\nCode:\\n", "do_sample": False, "num_beams": 1, "use_cache": False, "logits_to_keep": 1,
            "max_new_tokens": MAX_NEW_TOKENS, "eos_token_id": tokenizer.eos_token_id, "metric": "mbpp_execution",
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    destroy_distributed(context)


def prompt_text(tokenizer, token_ids: torch.Tensor) -> str:
    return tokenizer.decode(token_ids.cpu().tolist(), skip_special_tokens=False)


if __name__ == "__main__":
    main()
