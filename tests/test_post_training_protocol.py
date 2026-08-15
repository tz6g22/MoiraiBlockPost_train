from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from src.adapter.train_query import nonpadding_token_count, validate_adapter_config
from src.adapter.train_fixed_query import validate_fixed_adapter_config
from src.common import load_yaml
from src.discovery.dynamic_programming import solve_partition
from src.discovery.run_all import validate_discovery_config
from src.evaluation.run_evaluation import validate_evaluation_config
from src.evaluation.task_metrics import task_score
from src.data.format_tasks import format_task_prompt, format_task_target
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
    assert set(data["sources"]) == {"clutrr", "gsm8k", "svamp"}
    assert data["sources"]["clutrr"]["repo_id"] == "CLUTRR/v1"
    assert data["sources"]["clutrr"]["task_name"] == "task_1.2"
    assert data["sources"]["clutrr"]["official_split"] == "train"
    assert data["validation_sources"]["multihop"]["official_split"] == (
        "validation"
    )
    assert data["evaluation_sources"]["multihop"]["official_split"] == "test"
    assert set(data["evaluation_sources"]) == {"math", "multihop"}
    assert data["probe_sources"]["gsm8k_main_train"]["dataset_name"] == "gsm8k"
    assert data["counts"]["stage2_discovery"] == {
        "math": 200,
        "multihop": 200,
    }
    assert data["discovery_sources"]["math"] == {"gsm8k": 100, "svamp": 100}
    assert data["discovery_sources"]["multihop"] == {"clutrr": 200}
    assert data["counts"]["stage3_adapter_train"] == 200
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
    validate_fixed_adapter_config(fixed_adapter)
    validate_probe_config(probe)
    validate_evaluation_config(evaluation)

    assert discovery["tasks"] == ["math", "multihop"]
    assert discovery["discovery_cases_per_task"] == {"math": 200, "multihop": 200}
    assert probe["classes"] == {0: "math", 1: "multihop"}
    assert probe["classifier"] == "Linear(1024,2)"
    assert discovery["num_moirai_blocks"] == list(range(9, 17))
    assert discovery["boundary_refinement_sweeps"] == 5
    assert adapter["training_token_unit"] == "nonpadding_input"
    assert adapter["training_passes"] == 1
    assert adapter["checkpoint_interval_steps"] == 100
    assert adapter["progress_interval_steps"] == 100
    assert adapter["trainable_parameters"] == "pseudo_query_only"
    assert adapter["training_cases_per_task"] == 200
    assert "source_tasks" not in adapter
    assert "num_transformer_blocks" not in adapter
    assert fixed_adapter["source_tasks"] == ["math", "multihop"]
    assert fixed_adapter["trainable_parameters"] == "pseudo_query_only"
    assert fixed_adapter["training_token_unit"] == "nonpadding_input"
    assert fixed_adapter["training_passes"] == 1
    assert fixed_adapter["checkpoint_interval_steps"] == 100
    assert fixed_adapter["training_cases_per_task"] == 100
    assert fixed_adapter["progress_interval_steps"] == 100
    assert "partition_root" not in fixed_adapter
    assert evaluation["evaluation_examples_per_task"] == 10
    assert evaluation["primary_metric"] == "accuracy"


def test_query_budget_counts_full_nonpadding_input() -> None:
    attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
    assert nonpadding_token_count(attention_mask) == 5


def test_accuracy_is_the_primary_metric_for_both_evaluation_tasks() -> None:
    assert task_score("math", "The answer is 42", gold="#### 42")["accuracy"] == 1.0
    multihop_correct = task_score("multihop", "grandmother", gold="grandmother")
    multihop_wrong = task_score("multihop", "mother", gold="grandmother")
    assert multihop_correct["accuracy"] == multihop_correct["em"] == 1.0
    assert multihop_wrong["accuracy"] == multihop_wrong["em"] == 0.0


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


def test_qwen3_0_6b_depth_uses_same_partition_rules() -> None:
    fixed = fixed_kimi_partition(task="fixed", num_transformer_blocks=28)
    assert fixed.task == "fixed"
    assert fixed.lengths == (4, 4, 4, 4, 4, 4, 4)

    costs = np.full((28, 28), np.inf, dtype=np.float64)
    for start in range(28):
        for end in range(start, min(28, start + 4)):
            costs[start, end] = float(end - start + 1)
    for block_count in range(9, 17):
        result = solve_partition(costs, num_blocks=block_count, task="math")
        assert len(result.partition.blocks) == block_count
        assert sum(result.partition.lengths) == 28
        result.partition.validate()


def test_binary_probe_selects_exactly_its_argmax_task_config() -> None:
    math_prediction = ProbePrediction(
        predicted_task="math",
        logits=(2.0, -1.0),
        probabilities=(0.95, 0.05),
    )
    assert select_probe_config(math_prediction) == "math"

    multihop_prediction = ProbePrediction(
        predicted_task="multihop",
        logits=(-1.0, 1.0),
        probabilities=(0.25, 0.75),
    )
    assert select_probe_config(multihop_prediction) == "multihop"

    illegal_prediction = ProbePrediction(
        predicted_task="fixed",
        logits=(3.0, -2.0),
        probabilities=(0.99, 0.01),
    )
    with np.testing.assert_raises(AssertionError):
        select_probe_config(illegal_prediction)


def test_inference_applies_the_exact_binary_probe_config() -> None:
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
        bundles={task: FakeBundle(task) for task in ("math", "multihop")},
        probe_head=torch.nn.Linear(4, 2),
        device=torch.device("cpu"),
    )
    prediction = ProbePrediction(
        predicted_task="multihop",
        logits=(0.2, -0.2),
        probabilities=(0.6, 0.4),
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

    assert model.applied_task == "multihop"
    assert result.probe.predicted_task == "multihop"
    assert result.selected_config == "multihop"
    assert result.partition_sha256 == engine.bundles["multihop"].partition.sha256
    assert result.query_sha256 == "multihop-query"


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
            "source_repo_id": "Qwen/Qwen3-0.6B",
            "conversion": (
                "copy_qwen3_backbone_and_zero_initialize_attnres_parameters"
            ),
            "q_full_parameter_names": ["model.final_pseudo_query"],
            "q_full_sha256": "test",
    }
    validate_post_training_base_manifest(bootstrap)
