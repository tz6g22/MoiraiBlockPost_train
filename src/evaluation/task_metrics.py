from __future__ import annotations

import collections
import re
import string
from typing import Any


def normalize_answer(text: str) -> str:
    value = text.lower()
    value = "".join(character for character in value if character not in string.punctuation)
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return " ".join(value.split())


def clutrr_metrics(prediction: str, gold: str) -> dict[str, float]:
    normalized_prediction = normalize_answer(prediction)
    normalized_gold = normalize_answer(gold)
    exact_match = float(normalized_prediction == normalized_gold)
    prediction_tokens = normalized_prediction.split()
    gold_tokens = normalized_gold.split()
    if not prediction_tokens and not gold_tokens:
        return {
            "accuracy": 1.0,
            "em": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
        }
    if not prediction_tokens or not gold_tokens:
        return {
            "accuracy": exact_match,
            "em": exact_match,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }
    common = collections.Counter(prediction_tokens) & collections.Counter(gold_tokens)
    overlap = sum(common.values())
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(gold_tokens)
    f1 = 2.0 * precision * recall / (precision + recall) if overlap else 0.0
    return {
        "accuracy": exact_match,
        "em": exact_match,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


_NUMBER_PATTERN = re.compile(r"[-+]?(?:\d[\d,]*\.?\d*|\.\d+)")


def _canonical_number(value: str) -> str:
    value = value.replace(",", "").replace(" ", "").rstrip(".")
    if value.startswith("+"):
        value = value[1:]
    return value


def extract_math_answer(text: str) -> str:
    if "####" in text:
        candidate = text.rsplit("####", 1)[1]
        matches = _NUMBER_PATTERN.findall(candidate)
        if matches:
            return _canonical_number(matches[-1])
    matches = _NUMBER_PATTERN.findall(text)
    return _canonical_number(matches[-1]) if matches else ""


def math_exact_match(prediction: str, gold: str) -> float:
    return float(
        bool(extract_math_answer(prediction))
        and extract_math_answer(prediction) == extract_math_answer(gold)
    )


def task_score(
    task: str,
    prediction: str,
    *,
    gold: str,
) -> dict[str, float]:
    if task == "math":
        return {"accuracy": math_exact_match(prediction, gold)}
    if task == "multihop":
        return clutrr_metrics(prediction, gold)
    raise ValueError(f"Unknown task: {task}")


def mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        raise ValueError("Cannot average an empty metric collection")
    keys = set(rows[0])
    if any(set(row) != keys for row in rows):
        raise ValueError("Metric rows have inconsistent keys")
    return {
        key: sum(float(row[key]) for row in rows) / len(rows)
        for key in sorted(keys)
    }
