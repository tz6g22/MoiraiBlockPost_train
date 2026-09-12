from __future__ import annotations

import json
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KIMI_ROOT = ROOT.parent / "KimiBlockAttnRes"
sys.path.insert(0, str(ROOT))

from src.evaluation.task_metrics import extract_math_answer


OUTPUT = ROOT / "outputs/formal/evaluation_math_gsmhard_full_diagnostic"


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _correct(generation: str, gold: str) -> tuple[bool, str, str]:
    extracted = extract_math_answer(generation)
    gold_extracted = extract_math_answer(gold)
    try:
        return bool(extracted and gold_extracted and Decimal(extracted) == Decimal(gold_extracted)), extracted, gold_extracted
    except InvalidOperation:
        return False, extracted, gold_extracted


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    manifest = _read_jsonl(OUTPUT / "evaluation_manifest.jsonl")
    moirai = _read_jsonl(OUTPUT / "moirai_predictions.jsonl")
    kimi = _read_jsonl(OUTPUT / "kimi_predictions.jsonl")
    raw_paired = _read_jsonl(OUTPUT / "predictions.jsonl")
    if not (len(manifest) == len(moirai) == len(kimi) == len(raw_paired) == 1319):
        raise RuntimeError("GSM-Hard manifest and prediction counts must all equal 1319")
    stable_ids = [row["stable_id"] for row in manifest]
    if (
        [row["stable_id"] for row in moirai] != stable_ids
        or [row["stable_id"] for row in kimi] != stable_ids
        or [row["stable_id"] for row in raw_paired] != stable_ids
    ):
        raise RuntimeError("Moirai/Kimi prediction order does not match the shared manifest")

    paired_rows = []
    for case, moirai_row, kimi_row, raw_pair in zip(manifest, moirai, kimi, raw_paired):
        moirai_correct, moirai_answer, gold_answer = _correct(moirai_row["generation"], case["gold"])
        kimi_correct, kimi_answer, kimi_gold_answer = _correct(kimi_row["generation"], case["gold"])
        if gold_answer != kimi_gold_answer:
            raise RuntimeError("Inconsistent gold answer extraction")
        moirai_row.update(
            {
                "extracted_answer": moirai_answer,
                "gold_extracted_answer": gold_answer,
                "correct": moirai_correct,
                "metric": "decimal_numeric_exact_match",
            }
        )
        kimi_row.update(
            {
                "extracted_answer": kimi_answer,
                "gold_extracted_answer": gold_answer,
                "correct": kimi_correct,
                "metric": "decimal_numeric_exact_match",
            }
        )
        paired_rows.append(
            {
                **case,
                "moirai_prediction": moirai_row["generation"],
                "moirai_extracted_answer": moirai_answer,
                "moirai_correct": moirai_correct,
                "moirai_generation_token_count": moirai_row["generation_token_count"],
                "moirai_stop_reason": moirai_row["stop_reason"],
                "moirai_truncated": moirai_row["truncated"],
                "moirai_collapse": moirai_row["collapse"],
                "kimi_prediction": kimi_row["generation"],
                "kimi_extracted_answer": kimi_answer,
                "kimi_correct": kimi_correct,
                "kimi_generation_token_count": kimi_row["generation_token_count"],
                "kimi_stop_reason": kimi_row["stop_reason"],
                "kimi_truncated": kimi_row["truncated"],
                "kimi_collapse": kimi_row["collapse"],
                "moirai_query_sha256": raw_pair["moirai_query_sha256"],
                "kimi_query_sha256": raw_pair["kimi_query_sha256"],
                "moirai_partition_sha256": raw_pair["moirai_partition_sha256"],
                "kimi_partition_sha256": raw_pair["kimi_partition_sha256"],
                "base_checkpoint_sha256": moirai_row["base_checkpoint_sha256"],
            }
        )

    paired = {
        "both_correct": sum(int(row["moirai_correct"] and row["kimi_correct"]) for row in paired_rows),
        "moirai_only": sum(int(row["moirai_correct"] and not row["kimi_correct"]) for row in paired_rows),
        "kimi_only": sum(int(not row["moirai_correct"] and row["kimi_correct"]) for row in paired_rows),
        "both_wrong": sum(int(not row["moirai_correct"] and not row["kimi_correct"]) for row in paired_rows),
    }
    def single_value(name: str) -> str:
        values = {row[name] for row in paired_rows}
        if len(values) != 1:
            raise RuntimeError(f"Expected one {name}, found {sorted(values)}")
        return values.pop()

    result = {
        "dataset": "reasoning-machines/gsm-hard/gsmhardv2.jsonl",
        "evaluation_status": "NON-HELD-OUT GSM-Hard DIAGNOSTIC",
        "task": "math",
        "cases": len(manifest),
        "metric": "extract_math_answer + Decimal numeric equivalence",
        "generation": {
            "do_sample": False,
            "num_beams": 1,
            "use_cache": False,
            "logits_to_keep": 1,
            "max_new_tokens": 256,
            "eos_token_id": 151645,
            "decode_skip_special_tokens": True,
        },
        "moirai_correct": sum(int(row["moirai_correct"]) for row in paired_rows),
        "kimi_correct": sum(int(row["kimi_correct"]) for row in paired_rows),
        "moirai_collapse": sum(int(row["moirai_collapse"]) for row in paired_rows),
        "kimi_collapse": sum(int(row["kimi_collapse"]) for row in paired_rows),
        "moirai_truncation": sum(int(row["moirai_truncated"]) for row in paired_rows),
        "kimi_truncation": sum(int(row["kimi_truncated"]) for row in paired_rows),
        "moirai_empty_output": sum(int(not row["moirai_prediction"].strip()) for row in paired_rows),
        "kimi_empty_output": sum(int(not row["kimi_prediction"].strip()) for row in paired_rows),
        "paired": paired,
        "moirai_query_sha256": single_value("moirai_query_sha256"),
        "kimi_query_sha256": single_value("kimi_query_sha256"),
        "moirai_partition_sha256": single_value("moirai_partition_sha256"),
        "kimi_partition_sha256": single_value("kimi_partition_sha256"),
        "base_checkpoint_sha256": single_value("base_checkpoint_sha256"),
        "paired_inputs_identical": True,
        "prompt_token_ids_identical": True,
        "generation_protocol_identical": True,
        "metric_identical": True,
        "wrong_query_loaded": False,
    }
    _write_jsonl(OUTPUT / "moirai_predictions.jsonl", moirai)
    _write_jsonl(OUTPUT / "kimi_predictions.jsonl", kimi)
    _write_jsonl(OUTPUT / "predictions.jsonl", paired_rows)
    (OUTPUT / "evaluation_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
