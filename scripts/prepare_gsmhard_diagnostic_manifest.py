from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


DATA_PATH = ROOT.parent / "data/gsm_hard/gsmhardv2.jsonl"
OUTPUT = ROOT / "outputs/formal/evaluation_math_gsmhard_full_diagnostic"


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {OUTPUT}")
    rows = [json.loads(line) for line in DATA_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 1319:
        raise RuntimeError(f"Expected 1319 GSM-Hard rows, found {len(rows)}")
    tokenizer = AutoTokenizer.from_pretrained(
        ROOT / "outputs/base/qwen3_14b_full_attnres", local_files_only=True, use_fast=True
    )
    manifest = []
    for index, row in enumerate(rows):
        question = str(row["input"]).strip()
        prompt = f"Question: {question}\nAnswer:"
        token_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        if len(token_ids) > 2048 - 256:
            raise RuntimeError(f"GSM-Hard prompt exceeds the canonical context budget: {index}")
        stable_id = "gsmhard::official::" + hashlib.sha256(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        manifest.append(
            {
                "stable_id": stable_id,
                "task": "math",
                "dataset": "gsm-hard",
                "evaluation_status": "NON-HELD-OUT GSM-Hard DIAGNOSTIC",
                "source_row_index": index,
                "raw_question": question,
                "gold": str(row["target"]),
                "prompt": prompt,
                "prompt_token_ids": token_ids,
            }
        )
    stable_ids = [row["stable_id"] for row in manifest]
    if len(set(stable_ids)) != len(manifest):
        raise RuntimeError("GSM-Hard stable IDs are not unique")
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "evaluation_manifest.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in manifest),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "dataset": "reasoning-machines/gsm-hard/gsmhardv2.jsonl",
                "cases": len(manifest),
                "stable_id_sha256": hashlib.sha256(
                    json.dumps(stable_ids, separators=(",", ":")).encode("utf-8")
                ).hexdigest(),
                "evaluation_status": "NON-HELD-OUT GSM-Hard DIAGNOSTIC",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
