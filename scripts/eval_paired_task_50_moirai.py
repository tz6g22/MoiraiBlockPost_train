from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from transformers import AutoTokenizer

from scripts.eval_comparison_100 import select_evaluation_records
from src.common import load_yaml, sha256_json
from src.data.format_tasks import (
    encode_prompt_only,
    format_task_prompt,
    format_task_target,
    load_local_split,
    load_manifest,
)
from src.distributed.fsdp_utils import (
    barrier,
    destroy_distributed,
    init_distributed,
    wrap_qwen3_fsdp,
)
from src.evaluation.run_evaluation import (
    _base_checkpoint_hash,
    _greedy_generate,
    _load_bundle,
    GENERATION_LIMITS,
)
from src.evaluation.task_metrics import (
    clutrr_metrics,
    extract_code,
    extract_math_answer,
    task_score,
)
from src.modeling.full_attnres import MoiraiQwen3DecoderLayer, MoiraiQwen3ForCausalLM


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/base/qwen3_14b_full_attnres"
DATA_CONFIG = ROOT / "configs/data.yaml"
SPLIT_MANIFEST = ROOT / "outputs/data/splits.json"
GSM8K_MANIFEST = ROOT / "outputs/formal/gsm8k_full_eval_manifest.jsonl"
HUMANEVAL_MANIFEST = ROOT.parent / "data/humaneval/evaluation_manifest.jsonl"
QUERY_ROOT = ROOT / "outputs/formal/adapter_lr3e-5"
PARTITION_ROOT = ROOT / "outputs/formal/discovery"
CASE_COUNT = 50
REPEAT_PATTERN = re.compile(r"(.)\1{5,}")


def collapse(text: str) -> bool:
    compact = "".join(text.split())
    return bool(compact and REPEAT_PATTERN.search(compact))


def _error_type(stderr: str, returncode: int) -> str:
    if returncode == 0:
        return ""
    for name in ("NameError", "TypeError", "AssertionError", "SyntaxError"):
        if name in stderr:
            return name
    return "other"


def run_humaneval(prediction: str, *, entry_point: str, test: str) -> tuple[bool, str]:
    code = extract_code(prediction)
    source = "\n".join((code, test, f"check({entry_point})"))
    try:
        with tempfile.TemporaryDirectory(prefix="moiraiblock-humaneval-") as directory:
            completed = subprocess.run(
                [sys.executable, "-I", "-c", source],
                cwd=directory,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                timeout=5.0,
                check=False,
            )
    except subprocess.TimeoutExpired:
        return False, "Timeout"
    except OSError:
        return False, "other"
    return completed.returncode == 0, _error_type(completed.stderr or "", completed.returncode)


def _load_humaneval_rows() -> list[dict]:
    rows = [json.loads(line) for line in HUMANEVAL_MANIFEST.read_text(encoding="utf-8").splitlines()]
    if len(rows) != 164 or len({row["stable_id"] for row in rows}) != 164:
        raise RuntimeError("HumanEval manifest must contain 164 unique rows")
    return rows[:CASE_COUNT]


def _prompt_record(
    tokenizer,
    *,
    task: str,
    stable_id: str,
    prompt: str,
    gold: str,
    extra: dict,
) -> dict:
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    token_ids = encoded["input_ids"][0].tolist()
    if len(token_ids) > 2048 - GENERATION_LIMITS[task]:
        raise RuntimeError(f"{task} prompt exceeds evaluation budget: {stable_id}")
    return {
        "stable_id": stable_id,
        "task": task,
        "prompt": prompt,
        "prompt_token_ids": token_ids,
        "gold": gold,
        **extra,
    }


def select_cases(task: str, tokenizer, data_config: dict) -> list[dict]:
    if task == "math":
        rows = [json.loads(line) for line in GSM8K_MANIFEST.read_text(encoding="utf-8").splitlines()]
        if len(rows) < CASE_COUNT:
            raise RuntimeError("GSM8K canonical manifest has fewer than 50 rows")
        return [
            _prompt_record(
                tokenizer,
                task=task,
                stable_id=row["stable_id"],
                prompt=f"Question: {str(row['question']).strip()}\nAnswer:",
                gold=str(row["answer"]),
                extra={"dataset": "gsm8k", "official_split": row["official_split"]},
            )
            for row in rows[:CASE_COUNT]
        ]
    if task == "code":
        return [
            _prompt_record(
                tokenizer,
                task=task,
                stable_id=row["stable_id"],
                prompt=str(row["prompt"]),
                gold=str(row["canonical_solution"]),
                extra={
                    "dataset": "humaneval",
                    "official_split": "test",
                    "task_id": row["task_id"],
                    "entry_point": row["entry_point"],
                    "test": row["test"],
                    "canonical_solution": row["canonical_solution"],
                },
            )
            for row in _load_humaneval_rows()
        ]
    if task != "multihop":
        raise ValueError(f"Unsupported task: {task}")
    prior_manifest = load_manifest(SPLIT_MANIFEST)
    selected, dataset, mapping = select_evaluation_records(
        task="multihop",
        data_config=data_config,
        manifest=prior_manifest,
    )
    selected = selected[:CASE_COUNT]
    records: list[dict] = []
    for record in selected:
        row = dataset[int(record["row_index"])]
        prompt_example = encode_prompt_only(
            tokenizer,
            task=task,
            row=row,
            field_mapping=mapping,
            stable_id=record["stable_id"],
            max_length=2048 - GENERATION_LIMITS[task],
        )
        prompt = tokenizer.decode(prompt_example.input_ids.tolist(), skip_special_tokens=False)
        records.append(
            _prompt_record(
                tokenizer,
                task=task,
                stable_id=record["stable_id"],
                prompt=prompt,
                gold=format_task_target(task, row, mapping),
                extra={
                    "dataset": "clutrr",
                    "official_split": record["official_split"],
                    "selection_source": "eval_comparison_100 canonical selector, first 50",
                    "row_index": int(record["row_index"]),
                },
            )
        )
        # Use the canonical token IDs, including its context truncation.
        records[-1]["prompt_token_ids"] = prompt_example.input_ids.tolist()
    return records


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("math", "multihop", "code"), required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    context = init_distributed()
    data_config = load_yaml(DATA_CONFIG)
    tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True, use_fast=True)
    if args.manifest:
        cases = [
            json.loads(line)
            for line in args.manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        cases = select_cases(args.task, tokenizer, data_config)
    stable_ids = [case["stable_id"] for case in cases]
    if len(cases) != CASE_COUNT or len(set(stable_ids)) != CASE_COUNT:
        raise RuntimeError(f"{args.task} selection is not 50 unique cases")
    output_dir = args.output_dir or ROOT / "outputs/formal" / f"evaluation_{args.task}_50_taskmatched"
    manifest_path = output_dir / "evaluation_manifest.jsonl"
    moirai_path = output_dir / "moirai_predictions.jsonl"
    if context.is_rank0:
        allowed_existing = {"evaluation_manifest.jsonl", "leakage_audit.json"}
        existing = {path.name for path in output_dir.iterdir()} if output_dir.exists() else set()
        if existing - allowed_existing:
            raise FileExistsError(f"Refusing to overwrite non-empty output: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        if args.manifest:
            if manifest_path.resolve() != args.manifest.resolve():
                _write_jsonl(manifest_path, cases)
        else:
            _write_jsonl(manifest_path, cases)
        (output_dir / "selection_manifest.json").write_text(
            json.dumps(
                {
                    "task": args.task,
                    "cases": CASE_COUNT,
                    "stable_id_sha256": sha256_json(stable_ids),
                    "stable_ids": stable_ids,
                    "generation": {
                        "do_sample": False,
                        "num_beams": 1,
                        "use_cache": False,
                        "logits_to_keep": 1,
                        "max_new_tokens": GENERATION_LIMITS[args.task],
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    barrier(context)

    base_hash = _base_checkpoint_hash(BASE) if context.is_rank0 else None
    from src.distributed.fsdp_utils import broadcast_object

    base_hash = broadcast_object(base_hash, context)
    bundle = _load_bundle(
        partition_path=PARTITION_ROOT / args.task / "partition.json",
        query_manifest_path=QUERY_ROOT / args.task / "query_manifest.json",
        base_hash=base_hash,
    )
    model = MoiraiQwen3ForCausalLM.from_pretrained(
        BASE,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model = wrap_qwen3_fsdp(
        model,
        context,
        decoder_layer_classes=(MoiraiQwen3DecoderLayer,),
        sync_module_states=False,
        device_id=None,
    ).to(context.device)
    bundle.apply_to_model(model)
    model.eval()
    predictions: list[dict] = []
    if context.is_rank0:
        print(
            f"FORMAL_EVAL_STARTED model=moiraiblock task={args.task} case=1/{CASE_COUNT}",
            flush=True,
        )
    barrier(context)
    for index, case in enumerate(cases, start=1):
        input_ids = torch.tensor(case["prompt_token_ids"], dtype=torch.long, device=context.device).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)
        generated_ids = _greedy_generate(
            model,
            input_ids,
            attention_mask,
            maximum_new_tokens=GENERATION_LIMITS[args.task],
            eos_token_id=tokenizer.eos_token_id,
        )
        generation = tokenizer.decode(generated_ids[0].cpu(), skip_special_tokens=True)
        if args.task == "math":
            metrics = task_score(args.task, generation, gold=case["gold"])
            extracted = extract_math_answer(generation)
            execution_error = ""
        elif args.task == "multihop":
            metrics = clutrr_metrics(generation, case["gold"])
            extracted = generation
            execution_error = ""
        else:
            passed, execution_error = run_humaneval(
                generation,
                entry_point=case["entry_point"],
                test=case["test"],
            )
            metrics = {"accuracy": float(passed)}
            extracted = extract_code(generation)
        record = {
            **case,
            "generation": generation,
            "extracted_answer": extracted,
            "correct": bool(metrics["accuracy"]),
            "metrics": metrics,
            "generation_token_count": int(generated_ids.shape[1]),
            "stop_reason": (
                f"eos_{tokenizer.eos_token_id}"
                if generated_ids.shape[1] < GENERATION_LIMITS[args.task]
                else "max_new_tokens"
            ),
            "truncated": int(generated_ids.shape[1]) >= GENERATION_LIMITS[args.task],
            "collapse": collapse(generation),
            "empty_output": not bool(generation.strip()),
            "execution_error": execution_error,
            "query_sha256": bundle.query_sha256,
            "partition_sha256": bundle.partition.sha256,
            "base_checkpoint_sha256": base_hash,
        }
        if context.is_rank0:
            predictions.append(record)
            with moirai_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
            if index % 10 == 0:
                print(f"FORMAL_EVAL_PROGRESS model=moiraiblock task={args.task} case={index}/{CASE_COUNT}", flush=True)
        del generated_ids, input_ids, attention_mask
    if context.is_rank0:
        (output_dir / "moirai_manifest.json").write_text(
            json.dumps(
                {
                    "task": args.task,
                    "cases": CASE_COUNT,
                    "stable_id_sha256": sha256_json(stable_ids),
                    "query_path": str(bundle.query_path),
                    "query_sha256": bundle.query_sha256,
                    "partition_sha256": bundle.partition.sha256,
                    "base_checkpoint": str(BASE),
                    "base_checkpoint_sha256": base_hash,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    del model, bundle
    gc.collect()
    torch.cuda.empty_cache()
    barrier(context)
    destroy_distributed(context)


if __name__ == "__main__":
    main()
