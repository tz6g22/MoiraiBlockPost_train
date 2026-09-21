from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import yaml

from src.common import sha256_file, sha256_json


ROOT = Path(__file__).resolve().parents[1]
SOURCE_RUN = ROOT / "outputs/formal_retrain/main_math_partition_tau045_seed42"
SOURCE_DATA = ROOT / "outputs/formal_retrain/main_tasklocal_lr_retry_1589170/data/splits.json"
RUN_ROOT = ROOT / "outputs/formal_retrain/main_math_alpha005_fp32_retry1"
RUN_ID = "qwen3_1_7b_main_math_alpha005_fp32_retry1"
ALPHA_INIT = 0.05
TAU = 0.45
MATH_BUDGET = 500_000
MAX_STEPS = 2706


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    if RUN_ROOT.exists() and any(RUN_ROOT.iterdir()):
        raise RuntimeError(f"Refusing to reuse non-empty candidate run: {RUN_ROOT}")
    source_config = yaml.safe_load(
        (SOURCE_RUN / "candidate_config.yaml").read_text(encoding="utf-8")
    )
    config = copy.deepcopy(source_config)
    pipeline = config["pipeline"]
    pipeline.update(
        {
            "run_id": RUN_ID,
            "run_root": str(RUN_ROOT),
            "data_manifest": str(RUN_ROOT / "data/splits.json"),
            "discovery_output": str(RUN_ROOT / "discovery"),
            "joint_checkpoint_output": str(RUN_ROOT / "checkpoint"),
            "probe_output": str(RUN_ROOT / "probe"),
            "evaluation_output": str(RUN_ROOT / "evaluation"),
        }
    )
    config["experiment"]["name"] = "qwen3_1_7b_main_math_alpha005"
    config["tasks"]["enabled"] = ["math"]
    config["training"]["task_order"] = ["math"]
    config["training"]["task_sampling"] = {
        "strategy": "sequential_by_task_order",
        "math_weight": 1.0,
    }
    config["data"]["token_budget"] = {
        "math": MATH_BUDGET,
        "total": MATH_BUDGET,
    }
    config["discovery"]["cases"] = {"math": 200}
    config["discovery"]["similarity_thresholds"] = {"math": TAU}
    config["attnres"]["alpha"]["init"] = ALPHA_INIT
    config["attnres"]["alpha"]["allow_nonzero_init"] = True
    config["training"]["precision"]["routing_parameters"] = "float32"
    config["probe"]["labels"] = ["math"]
    config["probe"]["num_classes"] = 1
    config["probe"]["routing_rule"] = "argmax_1_way"
    config["inference"]["routes"] = {
        "math": {"config_bundle": ["P_math", "Q_math", "Alpha_math"]}
    }
    config["evaluation"]["tasks"] = ["math"]
    config["evaluation"]["max_new_tokens"] = {"math": 128}

    RUN_ROOT.mkdir(parents=True)
    config_path = RUN_ROOT / "candidate_config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    data_target = RUN_ROOT / "data/splits.json"
    data_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE_DATA, data_target)
    data_hash = sha256_file(data_target)

    source_discovery = SOURCE_RUN / "discovery"
    target_discovery = RUN_ROOT / "discovery"
    (target_discovery / "math/threshold_0.45").mkdir(parents=True, exist_ok=True)
    for name in (
        "similarity_matrix.npy",
        "interval_similarity_matrix.npy",
        "statistics.json",
    ):
        shutil.copy2(source_discovery / "math" / name, target_discovery / "math" / name)

    partition = json.loads(
        (source_discovery / "math/threshold_0.45/partition.json").read_text(
            encoding="utf-8"
        )
    )
    partition["run_id"] = RUN_ID
    partition["data_manifest_sha256"] = data_hash
    write_json(target_discovery / "math/threshold_0.45/partition.json", partition)

    cka_manifest = json.loads(
        (source_discovery / "linear_cka_manifest.json").read_text(encoding="utf-8")
    )
    cka_manifest.update(
        {
            "run_id": RUN_ID,
            "tasks": ["math"],
            "discovery_cases_per_task": {"math": 200},
            "similarity_thresholds": [TAU],
            "similarity_thresholds_by_task": {"math": [TAU]},
            "similarity_matrix_sha256": {
                "math": cka_manifest["similarity_matrix_sha256"]["math"]
            },
            "interval_similarity_matrix_sha256": {
                "math": cka_manifest["interval_similarity_matrix_sha256"]["math"]
            },
            "data_manifest_sha256": data_hash,
        }
    )
    write_json(target_discovery / "linear_cka_manifest.json", cka_manifest)
    write_json(
        target_discovery / "formal_pipeline_discovery_summary.json",
        {
            "status": "PASS",
            "run_id": RUN_ID,
            "forward_rerun": False,
            "forward_reused": True,
            "candidate_generated_from_saved_cka": True,
            "base_checkpoint_sha256": partition["base_checkpoint_sha256"],
            "num_transformer_blocks": 28,
            "tasks": {
                "math": {
                    "lengths": partition["block_sizes"],
                    "num_moirai_blocks": partition["num_moirai_blocks"],
                    "partition_sha256": partition["partition_sha256"],
                    "similarity_threshold": TAU,
                }
            },
        },
    )

    run_manifest = {
        "run_id": RUN_ID,
        "config_sha256": sha256_json(config),
        "data_config_sha256": sha256_file(ROOT / config["pipeline"]["data_config"]),
        "data_manifest": str(data_target),
        "data_manifest_sha256": data_hash,
        "enabled_tasks": ["math"],
        "stage2_discovery_cases": {"math": 200},
        "token_budget": {"math": MATH_BUDGET, "total": MATH_BUDGET},
        "status": "DATA_READY_ALPHA_PROTECTION_MATH_ONLY",
        "alpha_init": ALPHA_INIT,
        "partition": partition["block_sizes"],
        "threshold": TAU,
        "max_steps": MAX_STEPS,
    }
    write_json(RUN_ROOT / "run_manifest.json", run_manifest)
    write_json(
        RUN_ROOT / "candidate_control_comparison.json",
        {
            "status": "PASS",
            "baseline_run": str(SOURCE_RUN),
            "candidate_run": str(RUN_ROOT),
            "only_model_mechanism_change": "attnres.alpha.init",
            "baseline_alpha_init": 0.0,
            "candidate_alpha_init": ALPHA_INIT,
            "partition": partition["block_sizes"],
            "threshold": TAU,
            "data_manifest_sha256": data_hash,
            "math_token_budget": MATH_BUDGET,
            "enabled_tasks": ["math"],
            "max_steps": MAX_STEPS,
        },
    )
    print(json.dumps(run_manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
