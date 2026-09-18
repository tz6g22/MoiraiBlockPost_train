from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.common import load_yaml
from src.discovery.dynamic_programming import solve_similarity_partition
from src.discovery.linear_cka import (
    _partition_payload,
    interval_similarity_matrix,
    sweep_saved_similarity,
)


def _threshold_label(value: float) -> str:
    return format(float(value), "g")


def _breakpoint_analysis(
    *,
    source_dir: Path,
    output_dir: Path,
    tasks: list[str],
) -> dict[str, object]:
    analysis: dict[str, object] = {}
    for task in tasks:
        layer_cka = np.load(source_dir / task / "similarity_matrix.npy")
        interval = interval_similarity_matrix(layer_cka, reduction="min")
        values = sorted(
            {
                float(value)
                for start in range(interval.shape[0])
                for end in range(start + 1, interval.shape[1])
                for value in (interval[start, end],)
                if np.isfinite(value)
            }
        )
        regimes: list[dict[str, object]] = []
        for threshold in values:
            result = solve_similarity_partition(
                interval,
                similarity_threshold=threshold,
                task=task,
            )
            payload = _partition_payload(
                task=task,
                partition=result.partition,
                interval_similarity=interval,
                similarity_threshold=threshold,
                reduction="min",
            )
            threshold_dir = output_dir / task / "breakpoints" / f"threshold_{_threshold_label(threshold)}"
            threshold_dir.mkdir(parents=True, exist_ok=True)
            threshold_dir.joinpath("partition.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            signature = tuple(result.partition.lengths)
            if regimes and regimes[-1]["block_sizes"] == list(signature):
                regimes[-1]["through_similarity"] = threshold
            else:
                regimes.append(
                    {
                        "from_similarity": threshold,
                        "through_similarity": threshold,
                        "block_sizes": list(signature),
                        "num_moirai_blocks": len(signature),
                    }
                )
        analysis[task] = {
            "pairwise_breakpoint_count": len(values),
            "regimes": regimes,
        }
    return analysis


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qwen3_1.7b_discovery_linear_cka_min.yaml")
    args = parser.parse_args()
    config = load_yaml(args.config)
    source_dir = Path(config["source_output_dir"])
    output_dir = Path(config["output_dir"])
    tasks = list(config["tasks"])
    thresholds = [float(value) for value in config["similarity_thresholds"]]
    results = sweep_saved_similarity(
        output_dir,
        source_dir=source_dir,
        tasks=tasks,
        thresholds=thresholds,
        reduction=str(config["cka_interval_reduction"]),
    )
    breakpoint_analysis = _breakpoint_analysis(
        source_dir=source_dir,
        output_dir=output_dir,
        tasks=tasks,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "breakpoint_analysis.json").write_text(
        json.dumps(breakpoint_analysis, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "coarse_sensitivity.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
