from __future__ import annotations

import argparse

from src.common import load_yaml
from src.discovery.ordinary_residual import formal_discovery_status, require_defined_residual_cost


def main() -> None:
    parser = argparse.ArgumentParser(description="Formal task-adaptive pipeline gate")
    parser.add_argument("--config", default="qwen3_14b_config.yaml")
    parser.add_argument("--connectivity-only", action="store_true")
    args = parser.parse_args()
    config = load_yaml(args.config)
    status = formal_discovery_status(config.get("discovery", {}))
    print(status)
    # Do not run the historical Full AttnRes/replay Discovery by accident.
    require_defined_residual_cost(config.get("discovery", {}))
    raise RuntimeError(
        "Formal residual-only Discovery is defined, but its execution entry has "
        "not been wired in this checkout"
    )


if __name__ == "__main__":
    main()
