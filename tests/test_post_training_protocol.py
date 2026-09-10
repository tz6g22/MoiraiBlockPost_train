from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from src.adapter.train_query import nonpadding_token_count, validate_adapter_config
from src.adapter.train_fixed_query import validate_fixed_adapter_config
from src.common import load_yaml
from src.discovery.dynamic_programming import solve_partition
from src.discovery.run_all import validate_discovery_config
from src.evaluation.run_evaluation import validate_evaluation_config
from src.evaluation.task_metrics import task_score
from src.data.format_tasks import format_task_prompt, format_task_target, load_manifest
from src.data.leakage_audit import audit_manifest
from src.modeling.full_attnres import MoiraiQwen3ForCausalLM
from src.modeling.partition import MoiraiPartition, fixed_kimi_partition
from src.modeling.prepare_hf_checkpoint import _custom_config
from src.probe.extract_features import validate_probe_config
from src.probe.inference import (
    MoiraiInferenceEngine,
    ProbePrediction,
    select_probe_config,
)
from src.training.checkpointing import validate_post_training_base_manifest


def test_requested_case_counts_and_preserved_training_protocol() -> None:
    data = load_yaml("configs/data.yaml")
    assert data["sources"]["gsm8k"]["dataset_name"] == "gsm8k"
    assert set(data["sources"]) == {"clutrr", "gsm8k", "svamp", "mbpp"}
    assert data["sources"]["clutrr"]["repo_id"] == "CLUTRR/v1"
    assert data["sources"]["clutrr"]["task_name"] == "task_1.2"
    assert data["sources"]["clutrr"]["official_split"] == "train"
    assert data["validation_sources"]["multihop"]["official_split"] == (
        "validation"
    )
    assert data["evaluation_sources"]["multihop"]["official_split"] == "test"
    assert data["sources"]["mbpp"]["official_split"] == "train"
    assert data["validation_sources"]["code"]["official_split"] == "validation"
    assert data["evaluation_sources"]["code"]["official_split"] == "test"
    assert set(data["evaluation_sources"]) == {"math", "multihop", "code"}
    assert data["probe_sources"]["gsm8k_main_train"]["dataset_name"] == "gsm8k"
    assert data["counts"]["stage2_discovery"] == {
        "math": 500,
        "multihop": 500,
        "code": 200,
    }
    assert data["discovery_sources"]["math"] == {"gsm8k": 250, "svamp": 250}
    assert data["discovery_sources"]["multihop"] == {"clutrr": 500}
    assert data["discovery_sources"]["code"] == {"mbpp": 200}
    assert data["counts"]["stage3_adapter_train"] == 1000
    assert data["task_count_overrides"]["code"]["stage3_adapter_train"] == 200
    assert data["allowed_cross_stage_reuse"] == [
        ["stage2_discovery", "stage3_adapter_train"]
    ]
    assert data["counts"]["probe_train"] == 200
    assert data["counts"]["probe_val"] == 500
    assert data["counts"]["stage4_final_eval"] == 10

    discovery = load_yaml("configs/stage2_discovery.yaml")
    adapter = load_yaml("configs/stage3_adapter.yaml")
    fixed_adapter = load_yaml("configs/stage3_fixed_adapter.yaml")
    probe = load_yaml("configs/probe.yaml")
    evaluation = load_yaml("configs/evaluation.yaml")
    validate_discovery_config(discovery)
    validate_adapter_config(adapter)
    with pytest.raises(RuntimeError, match="Fixed mode is disabled"):
        validate_fixed_adapter_config(fixed_adapter)
    validate_probe_config(probe)
    validate_evaluation_config(evaluation)

    assert discovery["tasks"] == ["math", "multihop", "code"]
    assert discovery["discovery_cases_per_task"] == {
        "math": 500,
        "multihop": 500,
        "code": 200,
    }
    assert probe["classes"] == {0: "math", 1: "multihop", 2: "code"}
    assert probe["classifier"] == "Linear(5120,3)"
    assert probe["confidence_threshold"] == 0.5
    assert discovery["num_moirai_blocks"] == list(range(10, 17))
    assert discovery["boundary_refinement_sweeps"] == 5
    assert adapter["training_token_unit"] == "nonpadding_input"
    assert adapter["training_passes"] == 1
    assert adapter["checkpoint_interval_steps"] == 100
    assert adapter["progress_interval_steps"] == 100
    assert adapter["trainable_parameters"] == "pseudo_query_only"
    assert adapter["training_cases_per_task"] == {
        "math": 1000,
        "multihop": 1000,
        "code": 200,
    }
    assert "source_tasks" not in adapter
    assert "num_transformer_blocks" not in adapter
    assert fixed_adapter["source_tasks"] == ["math", "multihop", "code"]
    assert fixed_adapter["trainable_parameters"] == "pseudo_query_only"
    assert fixed_adapter["training_token_unit"] == "nonpadding_input"
    assert fixed_adapter["training_passes"] == 1
    assert fixed_adapter["checkpoint_interval_steps"] == 100
    assert fixed_adapter["training_cases_per_task"] == {
        "math": 1000,
        "multihop": 1000,
        "code": 200,
    }
    assert fixed_adapter["progress_interval_steps"] == 100
    assert "partition_root" not in fixed_adapter
    assert evaluation["evaluation_examples_per_task"] == 10
    assert evaluation["primary_metric"] == "accuracy"


def test_query_budget_counts_full_nonpadding_input() -> None:
    attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
    assert nonpadding_token_count(attention_mask) == 5


def test_code_manifest_uses_one_pool_and_keeps_final_evaluation_isolated() -> None:
    records = [
        record
        for record in load_manifest("outputs/data/splits.json")
        if record["task"] == "code"
    ]
    by_stage = {
        stage: [row for row in records if row["assigned_split"] == stage]
        for stage in {
            "stage2_discovery",
            "stage3_adapter_train",
            "stage3_adapter_val",
            "probe_train",
            "probe_val",
            "stage4_final_eval",
        }
    }
    assert {stage: len(rows) for stage, rows in by_stage.items()} == {
        "stage2_discovery": 200,
        "stage3_adapter_train": 200,
        "stage3_adapter_val": 45,
        "probe_train": 100,
        "probe_val": 45,
        "stage4_final_eval": 10,
    }
    assert {row["dataset"] for row in records} == {"mbpp"}
    assert {row["official_split"] for row in records} == {
        "train",
        "validation",
        "test",
    }
    assert {
        row["official_split"] for row in by_stage["stage3_adapter_train"]
    } == {"train", "validation", "test"}
    assert {
        row["official_split"] for row in by_stage["stage4_final_eval"]
    } == {"train", "validation", "test"}
    for rows in by_stage.values():
        stable_ids = [row["stable_id"] for row in rows]
        content_hashes = [row["content_sha256"] for row in rows]
        assert len(stable_ids) == len(set(stable_ids))
        assert len(content_hashes) == len(set(content_hashes))

    discovery_ids = {row["stable_id"] for row in by_stage["stage2_discovery"]}
    query_ids = {row["stable_id"] for row in by_stage["stage3_adapter_train"]}
    assert discovery_ids == query_ids
    for isolated_stage in {
        "stage3_adapter_val",
        "probe_train",
        "probe_val",
        "stage4_final_eval",
    }:
        isolated_ids = {row["stable_id"] for row in by_stage[isolated_stage]}
        assert not isolated_ids & discovery_ids
        assert not isolated_ids & query_ids

    final_ids = {row["stable_id"] for row in by_stage["stage4_final_eval"]}
    prior_ids = {
        row["stable_id"]
        for stage, rows in by_stage.items()
        if stage != "stage4_final_eval"
        for row in rows
    }
    assert not final_ids & prior_ids

    audit = audit_manifest(
        "outputs/data/splits.json",
        allowed_cross_stage_reuse=(
            ("stage2_discovery", "stage3_adapter_train"),
        ),
    )
    assert audit["status"] == "PASS"
    assert audit["id_intersections"] == []
    assert audit["content_hash_intersections"] == []


def test_every_stage_is_unique_and_final_eval_is_globally_unused() -> None:
    records = load_manifest("outputs/data/splits.json")
    expected = {
        "math": {"stage2_discovery": 500, "stage3_adapter_train": 1000},
        "multihop": {"stage2_discovery": 500, "stage3_adapter_train": 1000},
        "code": {"stage2_discovery": 200, "stage3_adapter_train": 200},
    }
    for task, stage_counts in expected.items():
        task_rows = [row for row in records if row["task"] == task]
        for stage, count in stage_counts.items():
            stage_rows = [
                row for row in task_rows if row["assigned_split"] == stage
            ]
            assert len(stage_rows) == count
            assert len({row["stable_id"] for row in stage_rows}) == count
            assert len({row["content_sha256"] for row in stage_rows}) == count

        final_rows = [
            row
            for row in task_rows
            if row["assigned_split"] == "stage4_final_eval"
        ]
        prior_rows = [
            row
            for row in task_rows
            if row["assigned_split"] != "stage4_final_eval"
        ]
        assert len(final_rows) == len({row["stable_id"] for row in final_rows}) == 10
        assert not {row["stable_id"] for row in final_rows} & {
            row["stable_id"] for row in prior_rows
        }
        assert not {row["content_sha256"] for row in final_rows} & {
            row["content_sha256"] for row in prior_rows
        }


def test_accuracy_is_the_primary_metric_for_all_evaluation_tasks() -> None:
    assert task_score("math", "The answer is 42", gold="#### 42")["accuracy"] == 1.0
    multihop_correct = task_score("multihop", "grandmother", gold="grandmother")
    multihop_wrong = task_score("multihop", "mother", gold="grandmother")
    assert multihop_correct["accuracy"] == multihop_correct["em"] == 1.0
    assert multihop_wrong["accuracy"] == multihop_wrong["em"] == 0.0
    passing_code = "def add(a, b):\n    return a + b"
    failing_code = "def add(a, b):\n    return a - b"
    tests = ["assert add(2, 3) == 5"]
    assert task_score("code", passing_code, gold="", test_list=tests)["accuracy"] == 1.0
    assert task_score("code", failing_code, gold="", test_list=tests)["accuracy"] == 0.0


def test_clutrr_multihop_format_uses_only_story_query_and_target_text() -> None:
    row = {
        "story": "[A] is [B]'s mother.",
        "query": "('A', 'B')",
        "target_text": "mother",
        "proof_state": "MUST_NOT_APPEAR",
        "edge_types": "MUST_NOT_APPEAR",
    }
    mapping = {
        "story": "story",
        "query": "query",
        "target": "target_text",
    }
    prompt = format_task_prompt("multihop", row, mapping)
    assert row["story"] in prompt
    assert row["query"] in prompt
    assert "MUST_NOT_APPEAR" not in prompt
    assert format_task_target("multihop", row, mapping) == "mother"


def test_mbpp_code_format_uses_only_validated_prompt_and_target_fields() -> None:
    row = {
        "prompt": "Write a function that doubles an integer.",
        "target": "def double(value):\n    return value * 2",
        "test_list": ["assert double(3) == 6"],
        "challenge_test_list": ["MUST_NOT_APPEAR"],
    }
    mapping = {"prompt": "prompt", "target": "target"}
    prompt = format_task_prompt("code", row, mapping)
    assert row["prompt"] in prompt
    assert "MUST_NOT_APPEAR" not in prompt
    assert format_task_target("code", row, mapping) == row["target"]


def test_qwen3_14b_depth_uses_same_partition_rules() -> None:
    fixed = fixed_kimi_partition(task="fixed", num_transformer_blocks=40)
    assert fixed.task == "fixed"
    assert fixed.lengths == (4, 4, 4, 4, 4, 4, 4, 4, 4, 4)

    costs = np.full((40, 40), np.inf, dtype=np.float64)
    for start in range(40):
        for end in range(start, min(40, start + 4)):
            costs[start, end] = float(end - start + 1)
    for block_count in range(10, 17):
        result = solve_partition(costs, num_blocks=block_count, task="math")
        assert len(result.partition.blocks) == block_count
        assert sum(result.partition.lengths) == 40
        result.partition.validate()


def test_three_class_probe_routes_by_argmax_without_fallback() -> None:
    math_prediction = ProbePrediction(
        predicted_task="math",
        logits=(2.0, -1.0, -2.0),
        probabilities=(0.90, 0.07, 0.03),
    )
    assert select_probe_config(math_prediction) == "math"

    multihop_prediction = ProbePrediction(
        predicted_task="multihop",
        logits=(-1.0, 1.0, -2.0),
        probabilities=(0.15, 0.80, 0.05),
    )
    assert select_probe_config(multihop_prediction) == "multihop"

    code_prediction = ProbePrediction(
        predicted_task="code",
        logits=(-1.0, -2.0, 2.0),
        probabilities=(0.05, 0.05, 0.90),
    )
    assert select_probe_config(code_prediction) == "code"

    low_confidence = ProbePrediction(
        predicted_task="math",
        logits=(0.1, 0.0, -0.1),
        probabilities=(0.37, 0.33, 0.30),
    )
    assert select_probe_config(low_confidence) == "math"

    illegal_prediction = ProbePrediction(
        predicted_task="fixed",
        logits=(3.0, -2.0, -3.0),
        probabilities=(0.99, 0.005, 0.005),
    )
    with np.testing.assert_raises(AssertionError):
        select_probe_config(illegal_prediction)


def test_inference_applies_code_as_an_atomic_bundle() -> None:
    class FakeBundle:
        def __init__(self, task: str) -> None:
            self.partition = MoiraiPartition.from_lengths(
                [4],
                task=task,
                num_transformer_blocks=4,
            )
            self.query_sha256 = f"{task}-query"

        def apply_to_model(self, model) -> None:
            model.applied_task = self.partition.task
            model.config.moirai_task = self.partition.task

    model = SimpleNamespace(
        applied_task=None,
        config=SimpleNamespace(moirai_task=None),
    )
    engine = MoiraiInferenceEngine(
        model=model,
        tokenizer=SimpleNamespace(eos_token_id=1),
        bundles={
            task: FakeBundle(task)
            for task in ("math", "multihop", "code")
        },
        probe_head=torch.nn.Linear(4, 3),
        device=torch.device("cpu"),
    )
    prediction = ProbePrediction(
        predicted_task="code",
        logits=(-0.2, -0.1, 0.8),
        probabilities=(0.15, 0.15, 0.70),
    )
    engine.classify = lambda _ids, _mask: prediction
    engine._greedy_generate = lambda _ids, _mask, maximum_new_tokens: torch.tensor(
        [[7]], dtype=torch.long
    )
    result = engine.infer(
        torch.tensor([[2]], dtype=torch.long),
        torch.ones((1, 1), dtype=torch.long),
        maximum_new_tokens=1,
    )

    assert model.applied_task == "code"
    assert result.probe.predicted_task == "code"
    assert result.selected_config == "code"
    assert result.partition_sha256 == engine.bundles["code"].partition.sha256
    assert result.query_sha256 == "code-query"



def test_hf_backbone_keys_load_without_changing_backbone_weights() -> None:
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        tie_word_embeddings=True,
        pad_token_id=0,
        eos_token_id=1,
        use_cache=False,
    )
    torch.manual_seed(7)
    base = Qwen3ForCausalLM(config)
    converted = MoiraiQwen3ForCausalLM(_custom_config(config))
    incompatible = converted.load_state_dict(base.state_dict(), strict=False)
    assert not incompatible.unexpected_keys
    assert incompatible.missing_keys
    assert all(
        any(
            fragment in name
            for fragment in (
                "pseudo_query",
                "attn_key_norm",
                    "mlp_key_norm",
                    "final_key_norm",
                    "alpha",
            )
        )
        for name in incompatible.missing_keys
    )
    converted_state = converted.state_dict()
    for name, value in base.state_dict().items():
        assert torch.equal(value, converted_state[name])

    output = converted(
        input_ids=torch.tensor([[2, 3, 4]], dtype=torch.long),
        attention_mask=torch.ones((1, 3), dtype=torch.long),
        use_cache=False,
        return_attnres_observations=True,
    )
    assert output.logits.shape == (1, 3, 64)
    assert len(output.attnres_observations) == 9


def test_hf_bootstrap_is_the_post_training_base() -> None:
    bootstrap = {
            "architecture": "Full AttnRes",
            "checkpoint_origin": "huggingface_post_training_bootstrap",
            "stage1_skipped": True,
            "source_repo_id": "Qwen/Qwen3-14B",
            "conversion": (
                "copy_qwen3_backbone_and_zero_initialize_attnres_parameters"
            ),
            "q_full_parameter_names": ["model.final_pseudo_query"],
            "q_full_sha256": "test",
    }
    validate_post_training_base_manifest(bootstrap)
