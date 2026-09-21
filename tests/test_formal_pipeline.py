from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from src.data.format_tasks import TargetCausalExample
from src.formal.checkpoint import converted_config_sha256, load_joint_checkpoint, save_joint_checkpoint
from src.formal.config import enabled_tasks, load_formal_config, validate_formal_config
from src.formal.conversion import formal_config_from_qwen
from src.data.format_tasks import canonical_content_sha256, canonical_stable_id
from src.data.source_provenance import validate_manifest_row_identity
from src.discovery.run_all import validate_discovery_config
from src.formal.data import audit_formal_source_provenance, validate_formal_source_policy
from src.formal.inference import FormalInferenceEngine
from src.formal.pipeline import (
    _discovery_partition_paths,
    _load_formal_runtime,
    _load_partitions,
    _pipeline_paths,
    _validate_only,
)
from src.formal.runtime import build_joint_optimizer, parameter_hash
from src.formal.task_banks import TaskBank
from src.formal.train_joint import (
    FormalTokenScheduler,
    _fit_example_to_token_budget,
    train_token_budget_sequential,
)
from src.data.prepare_post_data import _weighted_counts
from src.modeling.full_attnres import MoiraiQwen3ForCausalLM
from src.modeling.partition import MoiraiPartition


def test_formal_validation_accepts_current_manifest_source_provenance(monkeypatch) -> None:
    monkeypatch.setenv("QWEN3_14B_PATH", "artifacts/models/Qwen3-14B")
    config = load_formal_config("qwen3_14b_config.yaml")
    result = _validate_only(config, Path("artifacts/models/Qwen3-14B"))
    resolved = result["resolved"]
    assert resolved["training_mode"] == "sequential_task_training"
    assert resolved["formal_train_entrypoint"].endswith("train_token_budget_sequential")
    assert resolved["mixture_trainer_reachable"] is False
    assert resolved["task_order"] == ["math", "multihop", "code"]


def test_formal_pipeline_does_not_import_legacy_mixture_trainer() -> None:
    import src.formal.pipeline as formal_pipeline

    assert hasattr(formal_pipeline, "train_token_budget_sequential")
    assert not hasattr(formal_pipeline, "train_token_budget_mixture")


def test_formal_enabled_tasks_drive_two_and_three_task_modes() -> None:
    two_task_config = load_formal_config("qwen3_1.7b_config.yaml")
    three_task_config = load_formal_config("qwen3_14b_config.yaml")
    assert enabled_tasks(two_task_config) == ("math", "multihop")
    assert enabled_tasks(three_task_config) == ("math", "multihop", "code")
    assert two_task_config["discovery"]["method"] == "linear_cka_min"
    assert two_task_config["discovery"]["similarity_thresholds"] == {
        "math": 0.4,
        "multihop": 0.4,
    }
    for config in (two_task_config, three_task_config):
        assert config["data"]["mixture_training"] is False
        if config["discovery"].get("method") == "linear_cka_min":
            assert config["data"]["sequential_task_training"] is True
        else:
            assert config["data"]["sequential_dataset_training"] is True
        assert tuple(config["training"]["task_order"]) == enabled_tasks(config)


def test_formal_cka_run_paths_and_source_quota_are_dynamic() -> None:
    config = load_formal_config("qwen3_1.7b_config.yaml")
    paths = _pipeline_paths(config)
    partition_paths = _discovery_partition_paths(config)
    assert "discovery_partitions" not in config["pipeline"]
    assert str(paths["run_root"]) in str(paths["data_manifest"])
    assert all(str(paths["discovery_output"]) in str(path) for path in partition_paths.values())
    assert _weighted_counts(
        1000,
        ("svamp", "gsm8k", "math_train", "openmathinstruct2"),
        {"svamp": 0.25, "gsm8k": 0.25, "math_train": 0.25, "openmathinstruct2": 0.25},
    ) == {"svamp": 250, "gsm8k": 250, "math_train": 250, "openmathinstruct2": 250}


def test_formal_runtime_rejects_legacy_cosine_partition_path() -> None:
    config = load_formal_config("qwen3_14b_config.yaml")
    with pytest.raises(ValueError, match="Linear CKA min"):
        _load_partitions(config, Path("artifacts/models/Qwen3-14B"))


def test_formal_partition_loader_rejects_legacy_cosine_artifact(tmp_path, monkeypatch) -> None:
    config = copy.deepcopy(load_formal_config("qwen3_14b_config.yaml"))
    config["discovery"]["merge_cost_threshold"] = 0.2
    config["discovery"]["cases"] = {"math": 1, "multihop": 1}
    config["pipeline"]["discovery_output"] = str(tmp_path / "discovery")
    root = Path(config["pipeline"]["discovery_output"])
    base_hash = "base-hash"
    data_hash = "data-hash"
    monkeypatch.setattr("src.formal.pipeline._native_weight_hash", lambda _path: base_hash)
    monkeypatch.setattr(
        "src.formal.pipeline._native_config",
        lambda _path: type("Config", (), {"num_hidden_layers": 2})(),
    )
    for task in ("math", "multihop"):
        task_dir = root / task
        task_dir.mkdir(parents=True)
        partition = MoiraiPartition.from_lengths(
            [2],
            task=task,
            num_transformer_blocks=2,
            min_length=1,
            max_length=2,
            no_adjacent_singletons=False,
        )
        payload = partition.to_dict()
        payload.update(
            {
                "discovery_checkpoint_sha256": base_hash,
                "data_manifest_sha256": data_hash,
                "cost_method": "ordinary_residual_pairwise_directional_v1",
                "merge_cost_threshold": 0.2,
            }
        )
        (task_dir / "partition.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
        cost_path = task_dir / "cost_mean.npy"
        np.save(cost_path, np.array([[0.0, 0.3], [np.inf, 0.0]]))
        from src.common import sha256_file

        (task_dir / "stage2_manifest.json").write_text(
            json.dumps(
                {
                    "task": task,
                    "case_count": 1,
                    "base_checkpoint_sha256": base_hash,
                    "data_manifest_sha256": data_hash,
                    "cost_method": "ordinary_residual_pairwise_directional_v1",
                    "partition_sha256": partition.sha256,
                    "cost_mean_sha256": sha256_file(cost_path),
                }
            ),
            encoding="utf-8",
        )
    with pytest.raises(ValueError, match="Linear CKA min"):
        _load_partitions(
            config,
            Path("unused-checkpoint"),
            expected_data_manifest_sha256=data_hash,
        )


def test_inactive_supported_task_records_are_not_grouped() -> None:
    from src.formal.data import records_by_task_and_stage

    records = [
        {"task": task, "assigned_split": "probe_train", "split_key": task}
        for task in ("math", "multihop", "code")
    ]
    grouped = records_by_task_and_stage(
        records,
        stage="probe_train",
        expected_counts={"math": 1, "multihop": 1},
        enabled_tasks=("math", "multihop"),
    )
    assert set(grouped) == {"math", "multihop"}
    assert all(record["task"] != "code" for values in grouped.values() for record in values)


def test_discovery_worker_accepts_only_enabled_task_case_counts() -> None:
    validate_discovery_config(
        {
            "tasks": ["math", "multihop"],
            "model_mode": "original_residual_only",
            "formal_discovery": True,
            "ordinary_residual_cost_defined": True,
            "merge_cost_threshold": 0.5,
            "discovery_cases_per_task": {"math": 2, "multihop": 2},
            "min_block_length": 1,
            "max_block_length": None,
            "base_checkpoint": "checkpoint",
            "data_manifest": "manifest",
            "data_config": "config",
            "output_dir": "output",
        }
    )


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("attnres", "query", "input_independent_parameter"), False),
        (("inference", "allow_mid_generation_route_switch"), True),
        (("data", "isolation", "probe_vs_discovery_disjoint"), True),
        (("checkpoint", "save"), ["base_model_hash"]),
    ],
)
def test_formal_config_rejects_non_enforced_contract(path, value) -> None:
    config = load_formal_config("qwen3_14b_config.yaml")
    mutated = copy.deepcopy(config)
    target = mutated
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        validate_formal_config(mutated)


def test_formal_source_provenance_binds_stages_to_configured_sections() -> None:
    data_config = {
        "sources": {
            "math_train": {
                "dataset_name": "gsm8k",
                "revision": "r1",
                "official_split": "train",
            },
        },
        "validation_sources": {},
        "probe_sources": {},
        "evaluation_sources": {
            "math_eval": {
                "dataset_name": "gsm8k",
                "revision": "r1",
                "official_split": "test",
            },
        },
    }
    valid_records = [
        {
            "task": "math",
            "assigned_split": "stage2_discovery",
            "dataset": "gsm8k",
            "dataset_revision": "r1",
            "official_split": "train",
            "stable_id": "train-row",
        },
        {
            "task": "math",
            "assigned_split": "stage4_final_eval",
            "dataset": "gsm8k",
            "dataset_revision": "r1",
            "official_split": "test",
            "stable_id": "test-row",
        },
    ]
    assert audit_formal_source_provenance(valid_records, data_config=data_config)["status"] == "PASS"

    invalid_records = [dict(valid_records[0]), dict(valid_records[1])]
    invalid_records[1]["official_split"] = "train"
    with pytest.raises(RuntimeError, match="FORMAL_DATA_SOURCE_PROVENANCE_MISMATCH"):
        audit_formal_source_provenance(invalid_records, data_config=data_config)


def test_formal_source_provenance_rejects_task_dataset_mismatch() -> None:
    data_config = {
        "sources": {
            "gsm8k": {
                "dataset_name": "gsm8k",
                "revision": "r1",
                "official_split": "train",
            },
            "mbpp": {
                "dataset_name": "mbpp",
                "revision": "r2",
                "official_split": "train",
            },
        },
        "discovery_sources": {
            "math": {"gsm8k": 1},
            "multihop": {"gsm8k": 1},
            "code": {"mbpp": 1},
        },
        "validation_sources": {},
        "probe_sources": {},
        "evaluation_sources": {},
    }
    record = {
        "task": "math",
        "assigned_split": "stage2_discovery",
        "dataset": "mbpp",
        "dataset_revision": "r2",
        "official_split": "train",
        "stable_id": "wrong-task-dataset",
    }
    with pytest.raises(RuntimeError, match="FORMAL_DATA_TASK_SOURCE_MISMATCH"):
        audit_formal_source_provenance([record], data_config=data_config)


def test_formal_source_policy_rejects_unregistered_external_topup() -> None:
    formal_config = {
        "data": {
            "source_policy": "local_first_then_external_topup",
            "math": {
                "local_sources": ["gsm8k"],
                "external_topup_priority": ["math_topup"],
            },
            "multihop": {
                "local_sources": ["clutrr"],
                "external_topup_priority": [],
            },
            "code": {
                "local_sources": ["mbpp"],
                "external_topup_priority": [],
            },
        }
    }
    data_config = {
        "sources": {"gsm8k": {}, "clutrr": {}, "mbpp": {}},
    }
    with pytest.raises(RuntimeError, match="CONFIG_DECLARED_BUT_NOT_ENFORCED"):
        validate_formal_source_policy(formal_config, data_config)


def test_formal_source_policy_rejects_extra_stage_reuse() -> None:
    formal_config = {
        "data": {
            "source_policy": "local_first_then_external_topup",
            "math": {"local_sources": ["gsm8k"], "external_topup_priority": []},
            "multihop": {"local_sources": ["clutrr"], "external_topup_priority": []},
            "code": {"local_sources": ["mbpp"], "external_topup_priority": []},
        }
    }
    data_config = {
        "sources": {"gsm8k": {}, "clutrr": {}, "mbpp": {}},
        "external_sources": {},
        "allowed_cross_stage_reuse": [
            ["stage2_discovery", "probe_train"],
            ["stage3_adapter_train", "probe_train"],
        ],
    }
    with pytest.raises(ValueError, match="only stage2_discovery/probe_train"):
        validate_formal_source_policy(formal_config, data_config)


def test_formal_runtime_identity_covers_all_task_partitions(tmp_path) -> None:
    native_config = Qwen3Config(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        layer_types=["full_attention"] * 6,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        use_cache=False,
    )
    native = Qwen3ForCausalLM(native_config)
    checkpoint = tmp_path / "native"
    native.save_pretrained(checkpoint, safe_serialization=True)
    partitions = {
        task: MoiraiPartition.from_lengths(
            lengths,
            task=task,
            num_transformer_blocks=6,
            min_length=2,
            max_length=4,
            no_adjacent_singletons=True,
        )
        for task, lengths in {
            "math": [2, 4],
            "multihop": [3, 3],
            "code": [4, 2],
        }.items()
    }
    examples = {
        task: (
            TargetCausalExample(
                input_ids=torch.tensor([3, 4, 5, 6]),
                labels=torch.tensor([4, 5, 6, 2]),
                attention_mask=torch.ones(4, dtype=torch.long),
                target_mask=torch.ones(4, dtype=torch.bool),
                stable_id=f"{task}-identity",
            ),
        )
        for task in ("math", "multihop", "code")
    }
    _, formal, banks, identity = _load_formal_runtime(
        {"model": {"gradient_checkpointing": False}},
        checkpoint=checkpoint,
        partitions=partitions,
        device=torch.device("cpu"),
        identity_examples=examples,
    )
    assert identity["status"] == "PASS"
    assert set(identity["per_task"]) == {"math", "multihop", "code"}
    assert all(
        value["status"] == "PASS" for value in identity["per_task"].values()
    )
    assert banks["math"].partition_sha256 == partitions["math"].sha256
    assert formal.config.moirai_task == "math"


def test_formal_manifest_row_identity_is_recomputed_from_loaded_row() -> None:
    mapping = {
        "id": "id",
        "question": "question",
        "target": "answer",
    }
    row = {"id": "q-1", "question": "2+2", "answer": "4"}
    record = {
        "dataset": "gsm8k",
        "official_split": "train",
        "row_index": 0,
        "stable_id": canonical_stable_id("gsm8k", "train", row, mapping),
        "content_sha256": canonical_content_sha256("gsm8k", row, mapping),
    }

    validate_manifest_row_identity(record, row=row, field_mapping=mapping)
    row["answer"] = "5"
    with pytest.raises(RuntimeError, match="FORMAL_DATA_ROW_IDENTITY_MISMATCH"):
        validate_manifest_row_identity(record, row=row, field_mapping=mapping)


def test_probe_layer0_hidden_is_independent_of_task_routed_parameters() -> None:
    config = Qwen3Config(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        layer_types=["full_attention"] * 4,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        use_cache=False,
    )
    native = Qwen3ForCausalLM(config).eval()
    formal = MoiraiQwen3ForCausalLM(
        formal_config_from_qwen(
            config,
            min_block_length=2,
            max_block_length=2,
            no_adjacent_singletons=True,
        )
    ).eval()
    incompatible = formal.load_state_dict(native.state_dict(), strict=False)
    assert not incompatible.unexpected_keys
    assert all(
        "pseudo_query" in name or "alpha" in name or "key_norm" in name
        for name in incompatible.missing_keys
    )
    formal.config.attnres_execution = "formal"
    formal.config.moirai_partition = [2, 2]
    formal.config.moirai_task = "math"
    input_ids = torch.tensor([[3, 4, 5, 6, 7]])
    attention_mask = torch.ones_like(input_ids)
    first = formal(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        probe_layer0_only=True,
    ).probe_hidden_state
    with torch.no_grad():
        for name, parameter in formal.named_parameters():
            if any(token in name for token in ("pseudo_query", "alpha", "key_norm")):
                parameter.normal_()
    second = formal(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        probe_layer0_only=True,
    ).probe_hidden_state
    torch.testing.assert_close(first, second)


@pytest.mark.parametrize(
    "task_names",
    [("math", "multihop"), ("math", "multihop", "code")],
)
def test_formal_checkpoint_round_trip_restores_shared_and_task_state(
    tmp_path, task_names
) -> None:
    native_config = Qwen3Config(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        layer_types=["full_attention"] * 4,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        use_cache=False,
    )
    formal_config = formal_config_from_qwen(
        native_config,
        min_block_length=2,
        max_block_length=2,
        no_adjacent_singletons=True,
    )
    model = MoiraiQwen3ForCausalLM(formal_config)
    partitions = {
        task: MoiraiPartition.from_lengths(
            [2, 2],
            task=task,
            num_transformer_blocks=4,
            min_length=2,
            max_length=2,
        )
        for task in task_names
    }
    banks = {}
    for task, partition in partitions.items():
        model.config.moirai_partition = list(partition.lengths)
        model.config.moirai_task = task
        banks[task] = TaskBank.from_model(
            model,
            task=task,
            partition_sha256=partition.sha256,
        )
        banks[task].partition_lengths = partition.lengths
        assert all("key_norm" not in name for name in banks[task].state)
    banks[task_names[0]].activate(model)
    optimizer = build_joint_optimizer(
        model,
        backbone_lr=3e-6,
        attnres_lr=3e-5,
        backbone_weight_decay=0.1,
    )
    names_by_parameter = {id(parameter): name for name, parameter in model.named_parameters()}
    attnres_group_names = {
        names_by_parameter[id(parameter)] for parameter in optimizer.param_groups[1]["params"]
    }
    assert any("key_norm" in name for name in attnres_group_names)
    assert all("key_norm" not in name for name in banks["math"].state)
    loss = sum(parameter.float().sum() for parameter in model.parameters())
    loss.backward()
    optimizer.step()
    banks["math"].capture(model)
    banks["math"].capture_optimizer_state(
        optimizer,
        model,
        tuple(name for name in banks["math"].state if "pseudo_query" in name or "alpha" in name),
    )
    scheduler = FormalTokenScheduler(
        optimizer,
        maximum_tokens=300,
        warmup_ratio=0.03,
        min_lr_ratio=0.1,
    )
    scheduler.step(17)
    config = {
        "data": {"token_budget": {"math": 10, "multihop": 10, "code": 10}},
        "training": {
            "optimizer": {"name": "adamw", "lr": 3.0e-6},
            "scheduler": {"type": "cosine", "warmup_ratio": 0.03},
        },
    }
    banks[task_names[-1]].activate(model)
    token_values = {task: index + 1 for index, task in enumerate(task_names)}
    token_budgets = {task: 10 for task in task_names}
    config["data"]["token_budget"] = token_budgets
    training_progress = {
        "mode": "sequential_task_training",
        "task_order": list(task_names),
        "current_task_index": 1,
        "current_task": task_names[1],
        "completed_tasks": [task_names[0]],
        "global_step": 3,
        "task_steps": {task: 1 for task in task_names},
        "task_cumulative_tokens": token_values,
        "global_cumulative_tokens": sum(token_values.values()),
        "task_positions": {task: 1 for task in task_names},
        "task_epochs": {task: 0 for task in task_names},
    }
    manifest = save_joint_checkpoint(
        tmp_path,
        model=model,
        banks=banks,
        partitions={task: partition.to_dict() for task, partition in partitions.items()},
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        base_checkpoint_sha256="base-hash",
        data_manifest_sha256="data-hash",
        consumed_tokens=token_values,
        seed=42,
        identity_test={
            "status": "PASS",
            "base_checkpoint_sha256": "base-hash",
            "max_abs_logit_diff": 0.0,
            "mean_abs_logit_diff": 0.0,
            "converted_config_sha256": converted_config_sha256(model),
        },
        training_progress=training_progress,
    )

    restored = MoiraiQwen3ForCausalLM(formal_config_from_qwen(
        native_config,
        min_block_length=2,
        max_block_length=2,
        no_adjacent_singletons=True,
    ))
    restored_banks = {}
    for task, partition in partitions.items():
        restored.config.moirai_partition = list(partition.lengths)
        restored.config.moirai_task = task
        restored_banks[task] = TaskBank.from_model(
            restored,
            task=task,
            partition_sha256=partition.sha256,
        )
        restored_banks[task].partition_lengths = partition.lengths
    restored_banks[task_names[0]].activate(restored)
    restored_optimizer = build_joint_optimizer(
        restored,
        backbone_lr=3e-6,
        attnres_lr=3e-5,
        backbone_weight_decay=0.1,
    )
    restored_scheduler = FormalTokenScheduler(
        restored_optimizer,
        maximum_tokens=300,
        warmup_ratio=0.03,
        min_lr_ratio=0.1,
    )
    loaded = load_joint_checkpoint(
        tmp_path,
        model=restored,
        banks=restored_banks,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        active_task=task_names[0],
        restore_rng=False,
        expected_base_checkpoint_sha256="base-hash",
        expected_data_manifest_sha256="data-hash",
        expected_config_sha256=manifest["config_sha256"],
    )
    assert loaded["enabled_tasks"] == list(task_names)
    assert loaded["consumed_tokens_per_task"] == token_values
    assert set(loaded["partition_per_task"]) == set(task_names)
    assert set(loaded["query_hash_per_task"]) == set(task_names)
    assert set(loaded["alpha_hash_per_task"]) == set(task_names)
    assert loaded["base_model_hash"] == "base-hash"
    assert loaded["data_manifest_hash"] == "data-hash"
    assert loaded["training_progress"] == training_progress
    if "code" not in task_names:
        assert "code" not in loaded["enabled_tasks"]
        assert not (tmp_path / "code").exists()
    for task in partitions:
        assert restored_banks[task].state_hash() == banks[task].state_hash()
    assert restored_scheduler.trained_tokens == 17


def test_formal_final_budget_slice_preserves_supervised_tail() -> None:
    example = TargetCausalExample(
        input_ids=torch.arange(8),
        labels=torch.arange(1, 9),
        attention_mask=torch.ones(8, dtype=torch.long),
        target_mask=torch.tensor([False] * 5 + [True] * 3),
        stable_id="budget-boundary",
    )
    fitted = _fit_example_to_token_budget(example, 3)
    assert fitted.input_ids.tolist() == [5, 6, 7]
    assert fitted.labels.tolist() == [6, 7, 8]
    assert fitted.target_mask.tolist() == [True, True, True]
    assert _fit_example_to_token_budget(example, 8) is example


@pytest.mark.parametrize(
    "task_order",
    [("math", "multihop"), ("multihop", "math")],
)
def test_formal_sequential_transition_preserves_backbone_and_isolates_banks(task_order) -> None:
    config = Qwen3Config(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        layer_types=["full_attention"] * 4,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        use_cache=False,
    )
    model = MoiraiQwen3ForCausalLM(
        formal_config_from_qwen(
            config,
            min_block_length=2,
            max_block_length=2,
            no_adjacent_singletons=True,
        )
    )
    partitions = {
        task: MoiraiPartition.from_lengths(
            [2, 2],
            task=task,
            num_transformer_blocks=4,
            min_length=2,
            max_length=2,
        )
        for task in task_order
    }
    banks = {}
    for task, partition in partitions.items():
        model.config.moirai_partition = list(partition.lengths)
        model.config.moirai_task = task
        banks[task] = TaskBank.from_model(
            model,
            task=task,
            partition_sha256=partition.sha256,
        )
        banks[task].partition_lengths = partition.lengths
    banks[task_order[0]].activate(model)
    initial_bank_hashes = {task: banks[task].state_hash() for task in banks}
    initial_backbone_hash = parameter_hash(model)
    example = TargetCausalExample(
        input_ids=torch.tensor([3, 4, 5]),
        labels=torch.tensor([4, 5, 6]),
        attention_mask=torch.ones(3, dtype=torch.long),
        target_mask=torch.tensor([False, True, True]),
        stable_id="tiny",
    )
    training_config = {
        "optimizer": {
            "parameter_groups": {
                "backbone": {"lr": 3e-6, "weight_decay": 0.1},
                "attnres": {"lr": 3e-5, "weight_decay": 0.0},
            },
            "betas": [0.9, 0.95],
            "eps": 1e-8,
        },
        "scheduler": {"warmup_ratio": 0.03, "min_lr_ratio": 0.1},
        "clipping": {"max_grad_norm": 1.0},
        "batching": {"micro_batch_size": 1, "gradient_accumulation_steps": 1},
    }
    tokenizer = type("TinyTokenizer", (), {"pad_token_id": 0})()
    records = []
    boundary_backbone_hashes = []
    optimizer, scheduler, consumed, results, progress = train_token_budget_sequential(
        model,
        banks=banks,
        examples_by_task={
            task: (TargetCausalExample(
                input_ids=example.input_ids,
                labels=example.labels,
                attention_mask=example.attention_mask,
                target_mask=example.target_mask,
                stable_id=task,
            ),)
            for task in task_order
        },
        tokenizer=tokenizer,
        training_config=training_config,
        token_budgets={task: 5 for task in task_order},
        task_order=task_order,
        device=torch.device("cpu"),
        max_steps=4,
        seed=42,
        shuffle=False,
        on_step=records.append,
        on_task_boundary=lambda _progress, _optimizer, _scheduler: boundary_backbone_hashes.append(
            parameter_hash(model)
        ),
    )
    del optimizer, scheduler
    assert len(results) == 2 * len(task_order)
    assert [record["task"] for record in records] == [
        task_order[0], task_order[0], task_order[1], task_order[1]
    ]
    assert all(
        {
            "stage",
            "global_step",
            "task_step",
            "loss",
            "learning_rate_backbone",
            "learning_rate_query",
            "learning_rate_alpha",
            "non_padding_tokens_this_step",
            "task_cumulative_tokens",
            "global_cumulative_tokens",
        }.issubset(record)
        for record in records
    )
    assert consumed == {task: 5 for task in task_order}
    assert progress["completed_tasks"] == list(task_order)
    assert parameter_hash(model) != initial_backbone_hash
    assert len(boundary_backbone_hashes) == 2
    assert boundary_backbone_hashes[0] != initial_backbone_hash
    assert boundary_backbone_hashes[1] != boundary_backbone_hashes[0]
    assert banks[task_order[0]].state_hash() != initial_bank_hashes[task_order[0]]
    assert banks[task_order[1]].state_hash() != initial_bank_hashes[task_order[1]]
    assert all(banks[task].partition_lengths == (2, 2) for task in banks)


@pytest.mark.parametrize(
    "task_names",
    [("math", "multihop"), ("math", "multihop", "code")],
)
def test_formal_inference_probe_and_generation_use_one_task_bundle(task_names) -> None:
    config = Qwen3Config(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        layer_types=["full_attention"] * 2,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        use_cache=False,
    )
    model = MoiraiQwen3ForCausalLM(
        formal_config_from_qwen(
            config,
            min_block_length=2,
            max_block_length=2,
            no_adjacent_singletons=True,
        )
    )
    banks = {}
    for task in task_names:
        partition = MoiraiPartition.from_lengths(
            [2],
            task=task,
            num_transformer_blocks=2,
            min_length=2,
            max_length=2,
        )
        model.config.moirai_partition = [2]
        model.config.moirai_task = task
        banks[task] = TaskBank.from_model(
            model,
            task=task,
            partition_sha256=partition.sha256,
        )
        banks[task].partition_lengths = partition.lengths
    banks["math"].activate(model)
    probe_head = torch.nn.Linear(32, len(task_names))
    with torch.no_grad():
        probe_head.weight.zero_()
        probe_head.bias.zero_()
        probe_head.bias[-1] = 1.0
    engine = FormalInferenceEngine(
        model=model,
        banks=banks,
        probe_head=probe_head,
        device=torch.device("cpu"),
    )
    input_ids = torch.tensor([[3, 4, 5]])
    attention_mask = torch.ones_like(input_ids)
    result = engine.infer(
        input_ids,
        attention_mask,
        maximum_new_tokens=1,
        eos_token_id=None,
    )
    assert result.probe.predicted_task == task_names[-1]
    assert result.selected_task == task_names[-1]
    assert result.generated_ids.shape == (1, 1)
    assert result.partition_sha256 == banks[task_names[-1]].partition_sha256
    assert model.config.moirai_task == task_names[-1]
