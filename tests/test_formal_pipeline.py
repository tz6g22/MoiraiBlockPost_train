from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from src.data.format_tasks import TargetCausalExample
from src.formal.checkpoint import converted_config_sha256, load_joint_checkpoint, save_joint_checkpoint
from src.formal.config import load_formal_config, validate_formal_config
from src.formal.conversion import formal_config_from_qwen
from src.data.format_tasks import canonical_content_sha256, canonical_stable_id
from src.data.source_provenance import validate_manifest_row_identity
from src.formal.data import audit_formal_source_provenance, validate_formal_source_policy
from src.formal.inference import FormalInferenceEngine
from src.formal.pipeline import _load_formal_runtime, _validate_only
from src.formal.runtime import build_joint_optimizer, parameter_hash
from src.formal.task_banks import TaskBank
from src.formal.train_joint import FormalTokenScheduler, train_token_budget_mixture
from src.modeling.full_attnres import MoiraiQwen3ForCausalLM
from src.modeling.partition import MoiraiPartition


def test_formal_validation_rejects_old_manifest_with_wrong_source_provenance(monkeypatch) -> None:
    monkeypatch.setenv("QWEN3_14B_PATH", "artifacts/models/Qwen3-14B")
    config = load_formal_config("qwen3_14b_config.yaml")
    with pytest.raises(RuntimeError, match="FORMAL_DATA_SOURCE_PROVENANCE_MISMATCH"):
        _validate_only(config, Path("artifacts/models/Qwen3-14B"))


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("attnres", "query", "input_independent_parameter"), False),
        (("inference", "allow_mid_generation_route_switch"), True),
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


def test_formal_checkpoint_round_trip_restores_shared_and_task_state(tmp_path) -> None:
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
        for task in ("math", "multihop", "code")
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
    banks["math"].activate(model)
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
    banks["code"].activate(model)
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
        consumed_tokens={"math": 3, "multihop": 2, "code": 1},
        seed=42,
        identity_test={
            "status": "PASS",
            "base_checkpoint_sha256": "base-hash",
            "max_abs_logit_diff": 0.0,
            "mean_abs_logit_diff": 0.0,
            "converted_config_sha256": converted_config_sha256(model),
        },
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
    restored_banks["math"].activate(restored)
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
        active_task="math",
        restore_rng=False,
        expected_base_checkpoint_sha256="base-hash",
        expected_data_manifest_sha256="data-hash",
        expected_config_sha256=manifest["config_sha256"],
    )
    assert loaded["enabled_tasks"] == ["math", "multihop", "code"]
    assert loaded["consumed_tokens_per_task"] == {"math": 3, "multihop": 2, "code": 1}
    assert set(loaded["partition_per_task"]) == {"math", "multihop", "code"}
    assert set(loaded["query_hash_per_task"]) == {"math", "multihop", "code"}
    assert set(loaded["alpha_hash_per_task"]) == {"math", "multihop", "code"}
    assert loaded["base_model_hash"] == "base-hash"
    assert loaded["data_manifest_hash"] == "data-hash"
    for task in partitions:
        assert restored_banks[task].state_hash() == banks[task].state_hash()
    assert restored_scheduler.trained_tokens == 17


def test_formal_mixture_updates_only_active_task_bank() -> None:
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
        for task in ("math", "multihop", "code")
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
    banks["math"].activate(model)
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
    optimizer, scheduler, consumed, results = train_token_budget_mixture(
        model,
        banks=banks,
        examples_by_task={task: (example,) for task in banks},
        tokenizer=tokenizer,
        training_config=training_config,
        token_budgets={task: 6 for task in banks},
        device=torch.device("cpu"),
        max_steps=6,
        seed=42,
        shuffle=False,
    )
    del optimizer, scheduler
    assert len(results) == 6
    assert consumed == {"math": 6, "multihop": 6, "code": 6}
    assert parameter_hash(model) != initial_backbone_hash
    assert all(banks[task].state_hash() != initial_bank_hashes[task] for task in banks)
    assert all(banks[task].partition_lengths == (2, 2) for task in banks)


def test_formal_inference_probe_and_generation_use_one_task_bundle() -> None:
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
    for task in ("math", "multihop", "code"):
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
    probe_head = torch.nn.Linear(32, 3)
    with torch.no_grad():
        probe_head.weight.zero_()
        probe_head.bias.copy_(torch.tensor([0.0, 0.0, 1.0]))
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
    assert result.probe.predicted_task == "code"
    assert result.selected_task == "code"
    assert result.generated_ids.shape == (1, 1)
    assert result.partition_sha256 == banks["code"].partition_sha256
    assert model.config.moirai_task == "code"
