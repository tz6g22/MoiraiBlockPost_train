from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from src.common import load_yaml, sha256_json


FORMAL_TASKS = ("math", "multihop", "code")
_FORBIDDEN_FIXED_KEYS = (
    "P_fixed",
    "Q_fixed",
    "Alpha_fixed",
    "fixed_checkpoint",
    "fixed_fallback",
)


def load_formal_config(path: str | Path) -> dict[str, Any]:
    config = load_yaml(path)
    validate_formal_config(config)
    return config


def validate_formal_config(config: dict[str, Any]) -> None:
    experiment = config.get("experiment", {})
    if (
        experiment.get("model_family") != "qwen3"
        or experiment.get("model_size") != "14b"
        or experiment.get("mode") != "formal_posttraining"
    ):
        raise ValueError("Formal config must target Qwen3-14B formal post-training")

    model = config.get("model", {})
    if model.get("dtype") != "bfloat16":
        raise ValueError("Formal Qwen3-14B runtime must use bfloat16 parameters")
    if model.get("infer_architecture_from_checkpoint") is not True:
        raise ValueError("Formal runtime must infer architecture from the checkpoint")
    if model.get("discovery_model_mode") != "original_pretrained_transformer":
        raise ValueError("Formal config must use the original pretrained transformer for Discovery")
    if model.get("use_cache_train") is not False:
        raise ValueError("Formal training must set use_cache_train=false")
    if model.get("gradient_checkpointing") is not True:
        raise ValueError("Formal training must enable gradient checkpointing")

    pipeline = config.get("pipeline", {})
    for key in (
        "data_config",
        "data_manifest",
        "discovery_output",
        "joint_checkpoint_output",
        "probe_output",
        "evaluation_output",
    ):
        if not isinstance(pipeline.get(key), str) or not pipeline[key]:
            raise ValueError(f"Formal pipeline is missing {key}")
    if int(pipeline.get("max_sequence_length", 0)) <= 0:
        raise ValueError("Formal pipeline max_sequence_length must be positive")

    tasks = tuple(config.get("tasks", {}).get("enabled", ()))
    if tasks != FORMAL_TASKS:
        raise ValueError(f"Formal tasks must be exactly {FORMAL_TASKS}, got {tasks}")
    fixed = config.get("tasks", {}).get("fixed", {})
    if any(bool(fixed.get(key)) for key in ("enabled", "train", "checkpoint", "route", "fallback")):
        raise ValueError("Fixed mode is disabled in the formal Qwen3-14B pipeline")

    discovery = config.get("discovery", {})
    if discovery.get("enabled") is not True:
        raise ValueError("Formal Discovery must be enabled")
    required_discovery_flags = {
        "model_mode": "original_residual_only",
        "ordinary_residual_cost_defined": True,
        "forbid_attnres": True,
        "forbid_query": True,
        "forbid_alpha": True,
        "forbid_true_moirai_replay": True,
        "forbid_fixed_replay": True,
        "undefined_cost_behavior": "fail",
    }
    for key, expected in required_discovery_flags.items():
        if discovery.get(key) != expected:
            raise ValueError(f"Formal Discovery config mismatch for {key}: {discovery.get(key)!r}")
    if discovery.get("sample_from_final_task_mixture") is not True:
        raise ValueError("Formal Discovery must sample from the final task mixture")
    if discovery.get("exclude_from_posttraining") is not True:
        raise ValueError("Discovery cases must be excluded from post-training")
    discovery_cases = discovery.get("cases", {})
    if set(discovery_cases) != set(FORMAL_TASKS) or any(
        int(discovery_cases.get(task, 0)) <= 0 for task in FORMAL_TASKS
    ):
        raise ValueError("Formal Discovery must declare a positive case count for every task")
    partition = discovery.get("partition", {})
    if partition.get("unit") != "complete_transformer_block":
        raise ValueError("Formal Discovery partitions must use complete Transformer blocks")
    for key in (
        "continuous",
        "non_overlapping",
        "full_coverage",
        "block_length_constraints_from_partition_config",
        "candidate_block_count_from_partition_config",
        "no_adjacent_singletons",
        "deterministic_tie_break",
    ):
        if partition.get(key) is not True:
            raise ValueError(f"Formal Discovery partition contract requires {key}=true")
    if partition.get("fixed_block_size") != 4:
        raise ValueError("Formal Discovery fixed_block_size must be 4")
    if partition.get("num_blocks_policy") != "ceil_num_layers_over_fixed_block_size":
        raise ValueError("Formal Discovery must derive N from model depth and fixed block size")
    if partition.get("min_block_length") != 2 or partition.get("max_block_length") != 6:
        raise ValueError("Formal Qwen3-14B Discovery must use block lengths in [2, 6]")

    attnres = config.get("attnres", {})
    if attnres.get("enabled_after_discovery_only") is not True:
        raise ValueError("AttnRes must be disabled until Discovery has completed")
    if attnres.get("implementation") != "kimi_compatible_block_attnres":
        raise ValueError("Formal runtime must use Kimi-compatible Block AttnRes")
    if attnres.get("intra_block_residual_semantics") != "cumulative":
        raise ValueError("Formal runtime requires cumulative residual semantics")
    if attnres.get("embedding_is_independent_source") is not True:
        raise ValueError("Formal runtime must keep embedding as an independent source")
    if attnres.get("completed_block_representation") != "cumulative_residual":
        raise ValueError("Completed Block AttnRes history must use cumulative residuals")
    if attnres.get("partial_block_representation") != "cumulative_residual":
        raise ValueError("Partial Block AttnRes state must use cumulative residuals")
    if attnres.get("alpha", {}).get("init") != 0.0:
        raise ValueError("Formal Alpha must be zero-initialized")
    for key in (
        "forbid_mean_pooling",
        "forbid_learnable_block_summary",
        "forbid_projection_summary",
        "forbid_extra_block_encoder",
        "forbid_new_source_semantics",
    ):
        if attnres.get(key) is not True:
            raise ValueError(f"Formal config must forbid {key}")
    query = attnres.get("query", {})
    for key in ("site_specific", "task_specific", "input_independent_parameter"):
        if query.get(key) is not True:
            raise ValueError(f"Formal pseudo-query contract requires {key}=true")
    alpha = attnres.get("alpha", {})
    for key in ("site_specific", "task_specific"):
        if alpha.get(key) is not True:
            raise ValueError(f"Formal Alpha contract requires {key}=true")
    if alpha.get("purpose") != "identity_preserving_gate":
        raise ValueError("Formal Alpha must be an identity-preserving gate")
    for key in (
        "recency_bias",
        "delta_source_formulation",
        "null_source_routing",
        "manual_alpha_schedule",
    ):
        if attnres.get(key) is not False:
            raise ValueError(f"Formal AttnRes must disable {key}")

    data = config.get("data", {})
    if data.get("budget_unit") != "non_padding_tokens":
        raise ValueError("Formal budgets must count non-padding input tokens")
    if data.get("mixture_training") is not True:
        raise ValueError("Formal training must enable task/data mixture training")
    if data.get("sequential_dataset_training") is not False:
        raise ValueError("Formal training cannot use sequential dataset training")
    for key in ("shuffle", "interleave_across_sources", "interleave_across_tasks"):
        if data.get(key) is not True:
            raise ValueError(f"Formal data mixture must enable {key}")
    if data.get("source_policy") != "local_first_then_external_topup":
        raise ValueError("Formal data must use local_first_then_external_topup")
    for key in ("normalize_prompt_format", "normalize_target_format", "normalize_loss_mask"):
        if data.get(key) is not True:
            raise ValueError(f"Formal data normalization must enable {key}")
    isolation = data.get("isolation", {})
    for key in (
        "stable_ids",
        "hash_audit",
        "discovery_vs_train_disjoint",
        "probe_vs_train_disjoint",
        "eval_vs_all_training_disjoint",
        "exact_dedup",
        "normalized_text_dedup",
    ):
        if isolation.get(key) is not True:
            raise ValueError(f"Formal data isolation requires {key}=true")
    token_budget = data.get("token_budget", {})
    if any(int(token_budget.get(task, 0)) <= 0 for task in FORMAL_TASKS):
        raise ValueError("Each formal task must have a positive token budget")
    if int(token_budget.get("total", 0)) != sum(int(token_budget[task]) for task in FORMAL_TASKS):
        raise ValueError("Formal total token budget does not equal task budgets")

    training = config.get("training", {})
    if training.get("type") != "full_parameter_joint_posttraining":
        raise ValueError("Formal training must be full-parameter joint post-training")
    for key in ("shared_backbone", "train_backbone", "train_query", "train_alpha"):
        if training.get(key) is not True:
            raise ValueError(f"Formal training flag {key} must be true")
    if training.get("train_partition") is not False:
        raise ValueError("Formal partitions must be frozen during training")
    if training.get("task_sampling", {}).get("strategy") != "balanced_by_token_budget":
        raise ValueError("Formal task sampling must be balanced_by_token_budget")
    batching = training.get("batching", {})
    if int(batching.get("micro_batch_size", 0)) != 1:
        raise ValueError("The formal trainer currently requires micro_batch_size=1")
    if int(batching.get("gradient_accumulation_steps", 0)) != 1:
        raise ValueError("The formal trainer currently requires gradient_accumulation_steps=1")
    if batching.get("dynamic_padding") is not True:
        raise ValueError("Formal batching must enable dynamic padding")
    if batching.get("count_non_padding_tokens") is not True:
        raise ValueError("Formal batching must count non-padding tokens")
    loss = training.get("loss", {})
    if loss.get("type") != "causal_lm_cross_entropy":
        raise ValueError("Formal training must use causal LM cross-entropy")
    if loss.get("supervise_prompt") is not False:
        raise ValueError("Formal loss must not supervise prompt tokens")
    for key in ("supervise_target", "supervise_eos", "ignore_padding"):
        if loss.get(key) is not True:
            raise ValueError(f"Formal loss contract requires {key}=true")
    stopping = training.get("stopping", {})
    if stopping.get("primary") != "token_budget" or stopping.get("stop_when_all_task_budgets_reached") is not True:
        raise ValueError("Formal training must stop on complete per-task token budgets")
    checkpointing = training.get("checkpointing", {})
    for key in ("save_final", "save_optimizer_state", "save_scheduler_state", "save_rng_state"):
        if checkpointing.get(key) is not True:
            raise ValueError(f"Formal checkpointing requires {key}=true")

    distributed = config.get("distributed", {})
    if int(distributed.get("discovery_processes_per_node", 0)) <= 0:
        raise ValueError("Formal Discovery requires a positive process count")
    discovery_port = int(distributed.get("discovery_master_port", 0))
    if not 1024 <= discovery_port <= 65535:
        raise ValueError("Formal Discovery master port must be in [1024, 65535]")
    if int(distributed.get("formal_processes_per_node", 0)) <= 0:
        raise ValueError("Formal stage requires a positive process count")
    formal_port = int(distributed.get("formal_master_port", 0))
    if not 1024 <= formal_port <= 65535:
        raise ValueError("Formal stage master port must be in [1024, 65535]")
    if distributed.get("mutually_exclusive_backend") is not True:
        raise ValueError("Formal distributed backends must be mutually exclusive")
    if distributed.get("strategy") != "fsdp":
        raise ValueError(
            "The formal Qwen3-14B implementation requires distributed.strategy=fsdp"
        )
    if distributed.get("fsdp", {}).get("enabled_if_selected") is not True:
        raise ValueError("Formal FSDP backend must be enabled")
    if distributed.get("deepspeed", {}).get("enabled_if_selected") is not False:
        raise ValueError("Unsupported DeepSpeed/ZeRO backend must remain disabled")
    if distributed.get("activation_checkpointing") is not True:
        raise ValueError("Formal distributed activation checkpointing must be enabled")

    optimizer = training.get("optimizer", {})
    if optimizer.get("name") != "adamw" or tuple(optimizer.get("betas", ())) != (0.9, 0.95):
        raise ValueError("Formal optimizer must be AdamW with betas=(0.9,0.95)")
    if float(optimizer.get("eps", 0.0)) != 1.0e-8:
        raise ValueError("Formal optimizer eps must be 1e-8")
    groups = optimizer.get("parameter_groups", {})
    expected_groups = {
        "backbone": (3.0e-6, 0.1),
        "attnres": (3.0e-5, 0.0),
    }
    for name, (lr, weight_decay) in expected_groups.items():
        group = groups.get(name, {})
        if float(group.get("lr", 0.0)) != lr or float(group.get("weight_decay", -1.0)) != weight_decay:
            raise ValueError(f"Formal optimizer group {name} does not match the 14B config")
    scheduler = training.get("scheduler", {})
    if scheduler.get("type") != "cosine" or float(scheduler.get("warmup_ratio", -1.0)) != 0.03:
        raise ValueError("Formal scheduler must be cosine with warmup_ratio=0.03")
    if float(scheduler.get("min_lr_ratio", -1.0)) != 0.1:
        raise ValueError("Formal scheduler min_lr_ratio must be 0.1")
    if float(training.get("clipping", {}).get("max_grad_norm", -1.0)) != 1.0:
        raise ValueError("Formal gradient clipping must be 1.0")
    if training.get("precision", {}).get("compute") != "bfloat16":
        raise ValueError("Formal compute precision must be BF16")
    precision = training.get("precision", {})
    if precision.get("parameters") != "bfloat16":
        raise ValueError("Formal parameter precision must be BF16")
    if precision.get("loss_accumulation") != "float32":
        raise ValueError("Formal loss accumulation must be FP32")

    identity = config.get("identity_test", {})
    if identity.get("required") is not True or identity.get("before_training") is not True:
        raise ValueError("Formal training requires a pre-training identity test")
    if identity.get("compare") != ["original_qwen3_14b", "converted_attnres_alpha_zero"]:
        raise ValueError("Formal identity test must compare original Qwen3 and zero-alpha conversion")
    if identity.get("fail_status") != "IDENTITY_CONVERSION_FAILED":
        raise ValueError("Formal identity test fail status is not configured")
    if identity.get("metrics") != ["max_abs_logit_diff", "mean_abs_logit_diff"]:
        raise ValueError("Formal identity test metrics are incomplete")
    verification = config.get("verification", {})
    for key in (
        "require_backbone_update",
        "require_query_update",
        "require_alpha_open",
        "require_partition_unchanged",
        "require_inactive_task_bank_unchanged",
    ):
        if verification.get(key) is not True:
            raise ValueError(f"Formal verification requires {key}=true")

    probe = config.get("probe", {})
    if probe.get("labels") != list(FORMAL_TASKS):
        raise ValueError("Formal Probe labels must be exactly math/multihop/code")
    if probe.get("routing_rule") != "argmax_three_way":
        raise ValueError("Formal Probe must use argmax_three_way")
    high_confidence_threshold = float(probe.get("high_confidence_threshold", -1.0))
    if not 0.0 <= high_confidence_threshold <= 1.0:
        raise ValueError("Formal Probe high_confidence_threshold must be in [0, 1]")
    if probe.get("fixed_label") or probe.get("fixed_fallback"):
        raise ValueError("Formal Probe cannot contain a Fixed fallback")
    for key in (
        "discard_hidden_after_routing",
        "discard_kv_cache_after_routing",
        "discard_all_probe_state",
    ):
        if probe.get(key) is not True:
            raise ValueError(f"Formal Probe must enforce {key}")

    inference = config.get("inference", {})
    expected_routes = {
        "math": ["P_math", "Q_math", "Alpha_math"],
        "multihop": ["P_multihop", "Q_multihop", "Alpha_multihop"],
        "code": ["P_code", "Q_code", "Alpha_code"],
    }
    if inference.get("routes") != {
        task: {"config_bundle": bundle} for task, bundle in expected_routes.items()
    }:
        raise ValueError("Formal inference routes must bind each task to its P/Q/Alpha bundle")
    if inference.get("fixed_route") is not False:
        raise ValueError("Formal inference cannot use a Fixed route")
    for key in (
        "restart_original_input_from_layer0",
        "refeed_original_tokens",
        "keep_selected_route_for_full_generation",
    ):
        if inference.get(key) is not True:
            raise ValueError(f"Formal inference requires {key}=true")
    for key in ("reprobe_per_token", "allow_mid_generation_route_switch", "reuse_probe_cache", "reuse_cross_case_cache"):
        if inference.get(key) is not False:
            raise ValueError(f"Formal inference must disable {key}")

    evaluation = config.get("evaluation", {})
    if evaluation.get("tasks") != list(FORMAL_TASKS):
        raise ValueError("Formal evaluation tasks must be exactly math/multihop/code")
    if evaluation.get("evaluate_fixed") is not False:
        raise ValueError("Formal evaluation cannot include a Fixed route")
    required_case_fields = {
        "case_id",
        "true_task",
        "probe_prediction",
        "selected_mode",
        "partition_hash",
        "query_hash",
        "alpha_statistics",
        "answer",
        "correctness",
        "latency",
        "peak_memory",
    }
    if not required_case_fields.issubset(set(evaluation.get("per_case_record", ()))):
        raise ValueError("Formal evaluation per-case record is incomplete")
    required_aggregate_fields = {
        "routing_accuracy",
        "end_to_end_accuracy",
        "task_specific_accuracy",
        "latency",
        "peak_memory",
    }
    if not required_aggregate_fields.issubset(set(evaluation.get("aggregate", ()))):
        raise ValueError("Formal evaluation aggregate record is incomplete")

    checkpoint = config.get("checkpoint", {})
    required_checkpoint_fields = {
        "base_model_hash",
        "shared_backbone_hash",
        "enabled_tasks",
        "partition_per_task",
        "partition_hash_per_task",
        "query_hash_per_task",
        "alpha_hash_per_task",
        "optimizer_config",
        "scheduler_config",
        "token_budget",
        "consumed_tokens_per_task",
        "data_manifest_hash",
        "seed",
    }
    if not required_checkpoint_fields.issubset(set(checkpoint.get("save", ()) )):
        raise ValueError("Formal checkpoint save contract is incomplete")
    forbidden = set(checkpoint.get("forbid_keys", ()))
    if not set(_FORBIDDEN_FIXED_KEYS).issubset(forbidden):
        raise ValueError("Formal checkpoint must forbid all Fixed keys")


def formal_config_sha256(config: dict[str, Any]) -> str:
    return sha256_json(config)


def resolve_base_model_path(config: dict[str, Any]) -> Path:
    raw = str(config["model"]["base_model_path"])
    resolved = os.path.expandvars(raw)
    if "$" in resolved:
        raise RuntimeError(
            "Qwen3-14B checkpoint path is unresolved; set QWEN3_14B_PATH before formal execution"
        )
    path = Path(resolved)
    if not path.exists():
        raise FileNotFoundError(f"Qwen3-14B checkpoint does not exist: {path}")
    return path
