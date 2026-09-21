from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import numpy as np
import yaml

from src.common import sha256_file, sha256_json
from src.discovery.dynamic_programming import solve_similarity_partition
from src.discovery.linear_cka import _partition_payload


ROOT = Path(__file__).resolve().parents[1]
BASE_RUN = ROOT / "outputs/formal_retrain/main_tasklocal_lr_retry_1589170"
BASE_DISCOVERY = ROOT / "outputs/formal_retrain/main/discovery"
CANDIDATE_ROOT = ROOT / "outputs/formal_retrain/main_math_partition_tau045_seed42"
TAU = 0.45
MAX_STEPS = 2706


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def config_diff(before, after, path=()):
    if isinstance(before, dict) and isinstance(after, dict):
        diffs = []
        for key in sorted(set(before) | set(after)):
            if key not in before or key not in after:
                diffs.append((path + (key,), before.get(key), after.get(key)))
            else:
                diffs.extend(config_diff(before[key], after[key], path + (key,)))
        return diffs
    if before != after:
        return [(path, before, after)]
    return []


def main() -> None:
    if CANDIDATE_ROOT.exists() and any(
        (CANDIDATE_ROOT / name).exists()
        for name in ("candidate_config.yaml", "run_manifest.json", "candidate_control_comparison.json")
    ):
        raise RuntimeError(f"Candidate output already exists; refusing reuse: {CANDIDATE_ROOT}")

    checkpoint_manifest = json.loads(
        (BASE_RUN / "checkpoint/checkpoint_manifest.json").read_text()
    )
    base_config = checkpoint_manifest["config"]
    config = copy.deepcopy(base_config)
    pipeline = config["pipeline"]
    pipeline["run_root"] = str(CANDIDATE_ROOT)
    pipeline["discovery_output"] = str(CANDIDATE_ROOT / "discovery")
    pipeline["joint_checkpoint_output"] = str(CANDIDATE_ROOT / "checkpoint")
    pipeline["probe_output"] = str(CANDIDATE_ROOT / "probe")
    pipeline["evaluation_output"] = str(CANDIDATE_ROOT / "evaluation")
    config["discovery"]["similarity_thresholds"]["math"] = TAU

    allowed_paths = {
        ("pipeline", key)
        for key in (
            "run_root",
            "discovery_output",
            "joint_checkpoint_output",
            "probe_output",
            "evaluation_output",
        )
    }
    allowed_paths.add(("discovery", "similarity_thresholds", "math"))
    diffs = config_diff(base_config, config)
    unexpected = [diff for diff in diffs if diff[0] not in allowed_paths]
    if unexpected:
        raise RuntimeError(f"Unexpected control-variable config differences: {unexpected}")

    candidate_discovery = CANDIDATE_ROOT / "discovery"
    for task in ("math", "multihop"):
        source = BASE_DISCOVERY / task
        target = candidate_discovery / task
        target.mkdir(parents=True, exist_ok=True)
        for name in ("similarity_matrix.npy", "interval_similarity_matrix.npy", "statistics.json"):
            shutil.copy2(source / name, target / name)

    source_manifest = json.loads((BASE_DISCOVERY / "linear_cka_manifest.json").read_text())
    data_manifest_path = Path(pipeline["data_manifest"])
    source_manifest["similarity_thresholds"] = [TAU, 0.40]
    source_manifest["similarity_thresholds_by_task"] = {
        "math": [TAU],
        "multihop": [0.40],
    }
    source_manifest["candidate_generated_from_saved_cka"] = True
    write_json(candidate_discovery / "linear_cka_manifest.json", source_manifest)

    math_similarity = np.load(
        candidate_discovery / "math/interval_similarity_matrix.npy", allow_pickle=False
    )
    selected = solve_similarity_partition(
        math_similarity,
        similarity_threshold=TAU,
        task="math",
        candidate_lengths=range(1, math_similarity.shape[0] + 1),
    )
    provenance = {
        "base_checkpoint_sha256": source_manifest["base_checkpoint_sha256"],
        "data_manifest_sha256": source_manifest["data_manifest_sha256"],
        "interval_similarity_matrix_sha256": sha256_file(
            candidate_discovery / "math/interval_similarity_matrix.npy"
        ),
        "similarity_matrix_sha256": sha256_file(
            candidate_discovery / "math/similarity_matrix.npy"
        ),
        "run_id": pipeline["run_id"],
        "candidate_generated_from_saved_cka": True,
    }
    math_payload = _partition_payload(
        task="math",
        partition=selected.partition,
        interval_similarity=math_similarity,
        similarity_threshold=TAU,
        reduction="min",
        provenance=provenance,
    )
    math_payload["total_similarity"] = float(
        sum(block["similarity"] for block in math_payload["blocks"])
    )
    write_json(
        candidate_discovery / "math" / "threshold_0.45" / "partition.json",
        math_payload,
    )

    multihop_partition_target = candidate_discovery / "multihop/threshold_0.4/partition.json"
    multihop_partition_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        BASE_DISCOVERY / "multihop/threshold_0.4/partition.json",
        multihop_partition_target,
    )
    summary = json.loads(
        (BASE_DISCOVERY / "formal_pipeline_discovery_summary.json").read_text()
    )
    summary["discovery_rerun"] = False
    summary["forward_reused"] = True
    summary["candidate_generated_from_saved_cka"] = True
    summary["tasks"]["math"] = {
        "lengths": list(selected.partition.lengths),
        "num_moirai_blocks": len(selected.partition.blocks),
        "partition_sha256": math_payload["partition_sha256"],
        "similarity_threshold": TAU,
    }
    write_json(candidate_discovery / "formal_pipeline_discovery_summary.json", summary)

    CANDIDATE_ROOT.mkdir(parents=True, exist_ok=True)
    config_path = CANDIDATE_ROOT / "candidate_config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    loaded_config = yaml.safe_load(config_path.read_text())
    data_config_path = Path(loaded_config["pipeline"]["data_config"])
    data_hash = sha256_file(data_manifest_path)
    run_manifest = {
        "run_id": loaded_config["pipeline"]["run_id"],
        "config_sha256": sha256_json(loaded_config),
        "data_config_sha256": sha256_file(data_config_path),
        "data_manifest": str(data_manifest_path),
        "data_manifest_sha256": data_hash,
        "enabled_tasks": list(loaded_config["tasks"]["enabled"]),
        "stage2_discovery_cases": loaded_config["discovery"]["cases"],
        "token_budget": loaded_config["data"]["token_budget"],
        "status": "DATA_READY_CANDIDATE_MATH_ONLY",
        "candidate_threshold": TAU,
        "candidate_max_steps": MAX_STEPS,
    }
    write_json(CANDIDATE_ROOT / "run_manifest.json", run_manifest)
    write_json(
        CANDIDATE_ROOT / "candidate_control_comparison.json",
        {
            "status": "PASS",
            "allowed_config_differences": [list(path) for path in sorted(allowed_paths)],
            "observed_config_differences": [
                {"path": list(path), "before": before, "after": after}
                for path, before, after in diffs
            ],
            "base_run": str(BASE_RUN),
            "candidate_run": str(CANDIDATE_ROOT),
            "training_stable_ids_source": str(data_manifest_path),
            "validation_stable_ids_source": str(data_manifest_path),
            "math_token_budget": int(loaded_config["data"]["token_budget"]["math"]),
            "candidate_max_steps": MAX_STEPS,
            "candidate_partition": list(selected.partition.lengths),
            "candidate_threshold": TAU,
        },
    )
    print(json.dumps({
        "candidate_config": str(config_path),
        "candidate_partition": list(selected.partition.lengths),
        "candidate_n": len(selected.partition.blocks),
        "candidate_max_block": max(selected.partition.lengths),
        "math_budget": loaded_config["data"]["token_budget"]["math"],
        "max_steps": MAX_STEPS,
        "control_differences": len(diffs),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
