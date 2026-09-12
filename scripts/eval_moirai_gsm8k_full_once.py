from __future__ import annotations

import json
import re
from pathlib import Path

import torch
from transformers import AutoTokenizer

from src.common import load_yaml
from src.data.format_tasks import encode_prompt_only
from src.distributed.fsdp_utils import barrier, destroy_distributed, init_distributed, broadcast_object, wrap_qwen3_fsdp
from src.evaluation.run_evaluation import _base_checkpoint_hash, _greedy_generate, _load_bundle
from src.evaluation.task_metrics import extract_math_answer
from src.modeling.full_attnres import MoiraiQwen3DecoderLayer, MoiraiQwen3ForCausalLM


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/base/qwen3_14b_full_attnres"
MANIFEST = ROOT / "outputs/formal/gsm8k_full_eval_manifest.jsonl"
OUT = ROOT / "outputs/formal/evaluation_gsm8k_full"
PARTITION = ROOT / "outputs/formal/discovery/math/partition.json"
QUERY_MANIFEST = ROOT / "outputs/formal/adapter_lr3e-5/math/query_manifest.json"
MAX_NEW_TOKENS = 256
NUMBER_REPEAT = re.compile(r"(.)\1{5,}")


def collapse(text: str) -> bool:
    compact = "".join(text.split())
    return bool(compact and NUMBER_REPEAT.search(compact))


def main() -> None:
    context = init_distributed()
    rank, device = context.rank, context.device
    rows = [json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines()]
    if len(rows) != 1319:
        raise RuntimeError(f"Expected 1319 GSM8K cases, found {len(rows)}")
    base_hash = broadcast_object(_base_checkpoint_hash(BASE) if context.is_rank0 else None, context)
    tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True, use_fast=True)
    model = MoiraiQwen3ForCausalLM.from_pretrained(
        BASE, local_files_only=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    bundle = _load_bundle(partition_path=PARTITION, query_manifest_path=QUERY_MANIFEST, base_hash=base_hash)
    model = wrap_qwen3_fsdp(
        model, context, decoder_layer_classes=(MoiraiQwen3DecoderLayer,),
        sync_module_states=False, device_id=None,
    ).to(device)
    bundle.apply_to_model(model)
    model.eval()
    if rank == 0:
        OUT.mkdir(parents=True, exist_ok=True)
        print("FORMAL_EVAL_STARTED model=moiraiblock case=1/1319", flush=True)
    barrier(context)
    output_rows = []
    for index, row in enumerate(rows):
        prompt = f"Question: {row['question'].strip()}\nAnswer:"
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        generated_ids = _greedy_generate(
            model, input_ids, attention_mask,
            maximum_new_tokens=MAX_NEW_TOKENS,
            eos_token_id=tokenizer.eos_token_id,
        )
        generation = tokenizer.decode(generated_ids[0].cpu(), skip_special_tokens=True)
        extracted = extract_math_answer(generation)
        gold = extract_math_answer(str(row["answer"]))
        generated_count = int(generated_ids.shape[1])
        record = {
            "stable_id": row["stable_id"], "prompt": prompt, "gold": row["answer"],
            "gold_extracted_answer": gold, "generation": generation,
            "extracted_answer": extracted, "correct": extracted == gold,
            "generation_token_count": generated_count,
            "stop_reason": f"eos_{tokenizer.eos_token_id}" if generated_count < MAX_NEW_TOKENS else "max_new_tokens",
            "truncated": generated_count >= MAX_NEW_TOKENS,
            "collapse": collapse(generation), "empty_output": not bool(generation.strip()),
        }
        if rank == 0:
            output_rows.append(record)
            with (OUT / "predictions.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
        del generated_ids, input_ids, attention_mask, encoded
        if rank == 0 and index + 1 < len(rows) and (index + 1) % 100 == 0:
            print(f"FORMAL_EVAL_PROGRESS case={index + 1}/1319", flush=True)
    if rank == 0:
        (OUT / "evaluation_results.json").write_text(json.dumps({
            "model": "MoiraiBlock", "base_checkpoint": str(BASE),
            "base_checkpoint_hash": base_hash, "partition_path": str(PARTITION),
            "partition_hash": bundle.partition.sha256, "query_path": str(bundle.query_path),
            "query_hash": bundle.query_sha256, "total": len(output_rows),
            "correct": sum(int(r["correct"]) for r in output_rows),
            "accuracy": sum(int(r["correct"]) for r in output_rows) / len(output_rows),
            "collapse": sum(int(r["collapse"]) for r in output_rows),
            "empty_output": sum(int(r["empty_output"]) for r in output_rows),
            "truncation": sum(int(r["truncated"]) for r in output_rows),
            "max_new_tokens": MAX_NEW_TOKENS, "eos_token_id": tokenizer.eos_token_id,
            "do_sample": False, "num_beams": 1, "use_cache": False, "logits_to_keep": 1,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    destroy_distributed(context)


if __name__ == "__main__":
    main()
