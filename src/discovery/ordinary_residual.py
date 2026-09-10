"""Guards for the formal ordinary-residual Discovery boundary.

The repository's historical Discovery cost is defined over Full AttnRes
observations and compressed replay.  That is not a valid implementation of
the new method.  Until a research-defined ordinary-residual cost is supplied,
the formal entry must stop rather than silently changing the objective.
"""

RESIDUAL_DISCOVERY_COST_UNDEFINED = "RESIDUAL_DISCOVERY_COST_UNDEFINED"


def require_defined_residual_cost(config: dict) -> None:
    if not bool(config.get("ordinary_residual_cost_defined", False)):
        raise RuntimeError(
            f"{RESIDUAL_DISCOVERY_COST_UNDEFINED}: the current repository only "
            "contains the legacy AttnRes/replay cost path"
        )


def formal_discovery_status(config: dict) -> dict[str, object]:
    return {
        "model_mode": "original_residual_only",
        "attnres_accessed": False,
        "query_accessed": False,
        "alpha_accessed": False,
        "true_moirai_replay": False,
        "cost_status": (
            "DEFINED" if bool(config.get("ordinary_residual_cost_defined", False))
            else RESIDUAL_DISCOVERY_COST_UNDEFINED
        ),
    }
