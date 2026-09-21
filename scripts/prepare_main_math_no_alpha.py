from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import prepare_main_math_alpha005 as base
from src.common import sha256_json


RUN_ROOT = ROOT / "outputs/formal_retrain/main_math_no_alpha_tau045_seed42"
RUN_ID = "qwen3_1_7b_main_math_no_alpha_tau045_seed42"


def main() -> None:
    base.RUN_ROOT = RUN_ROOT
    base.RUN_ID = RUN_ID
    base.ALPHA_INIT = 0.0
    base.main()

    config_path = RUN_ROOT / "candidate_config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["experiment"]["name"] = "qwen3_1_7b_main_math_no_alpha"
    config["attnres"]["alpha"].update(
        {
            "enabled": False,
            "init": 0.0,
            "allow_nonzero_init": False,
        }
    )
    config["training"]["train_alpha"] = False
    config["training"]["optimizer"]["parameter_groups"].pop("alpha", None)
    config["identity_test"]["compare"] = [
        "original_qwen3_1_7b",
        "converted_attnres_no_alpha",
    ]
    config["identity_test"]["num_examples"] = 8
    config["verification"]["require_alpha_open"] = False
    config["inference"]["routes"] = {
        "math": {"config_bundle": ["P_math", "Q_math"]}
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    manifest_path = RUN_ROOT / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "config_sha256": sha256_json(config),
            "status": "DATA_READY_NO_ALPHA_MATH_ONLY",
            "alpha_enabled": False,
            "alpha_init": 0.0,
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    comparison_path = RUN_ROOT / "candidate_control_comparison.json"
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    comparison.update(
        {
            "only_model_mechanism_change": "attnres.alpha.disabled",
            "alpha_enabled": False,
            "candidate_alpha_init": None,
        }
    )
    comparison_path.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
