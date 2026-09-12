from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

from src.common import load_yaml
from src.data.format_tasks import (
    encode_prompt_only,
    load_dataset_pool,
    load_manifest,
)
from src.discovery.dynamic_programming import (
    brute_force_partition,
    count_feasible_partitions,
    solve_partition,
)
from src.discovery.fixed_policy import resolve_fixed_num_blocks
from src.discovery.ordinary_residual import (
    collect_ordinary_residual_reference,
    pairwise_directional_interval_cost,
    pairwise_directional_interval_costs,
)
from src.modeling.partition import fixed_kimi_partition, MoiraiPartition


def _tiny_model(vocab_size: int = 97) -> Qwen3ForCausalLM:
    config = Qwen3Config(
        vocab_size=vocab_size,
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
    return Qwen3ForCausalLM(config).eval()


def test_pairwise_directional_cost_zero_for_collinear_contributions() -> None:
    base = torch.tensor([1.0, 2.0, -1.0]).view(1, 1, 1, 3)
    values = torch.cat((base, 2.0 * base, 0.5 * base), dim=0)
    mask = torch.ones(1, 1, dtype=torch.long)
    cost = pairwise_directional_interval_cost(values, mask, start=0, end=2)
    assert float(cost) < 1.0e-6


def test_pairwise_directional_cost_increases_for_orthogonal_contributions() -> None:
    values = torch.eye(3).view(3, 1, 1, 3)
    mask = torch.ones(1, 1, dtype=torch.long)
    cost = pairwise_directional_interval_cost(values, mask, start=0, end=2)
    assert float(cost) > 0.5


def test_pairwise_directional_cost_is_robust_to_uniform_scale() -> None:
    values = torch.tensor(
        [
            [[[1.0, 2.0, 0.0]]],
            [[[0.0, 1.0, 3.0]]],
            [[[-1.0, 0.0, 2.0]]],
        ]
    )
    mask = torch.ones(1, 1, dtype=torch.long)
    original = pairwise_directional_interval_cost(values, mask, start=0, end=2)
    scaled = pairwise_directional_interval_cost(17.0 * values, mask, start=0, end=2)
    torch.testing.assert_close(original, scaled, rtol=1.0e-5, atol=1.0e-6)


def test_pairwise_directional_cost_masks_padding_representation() -> None:
    values = torch.tensor(
        [
            [[[1.0, 0.0], [100.0, 100.0]]],
            [[[0.0, 1.0], [-200.0, 50.0]]],
            [[[1.0, 1.0], [300.0, -400.0]]],
        ]
    )
    mask = torch.tensor([[1, 0]], dtype=torch.long)
    masked = pairwise_directional_interval_cost(values, mask, start=0, end=2)
    changed_padding = values.clone()
    changed_padding[:, :, 1] *= 1000.0
    torch.testing.assert_close(
        masked,
        pairwise_directional_interval_cost(changed_padding, mask, start=0, end=2),
    )


def test_native_qwen_reference_is_forward_only_and_task_inputs_differ() -> None:
    torch.manual_seed(7)
    model = _tiny_model()
    first = torch.tensor([[3, 4, 5, 6, 7]])
    second = torch.tensor([[31, 29, 27, 25, 23]])
    mask = torch.ones_like(first)
    first_reference = collect_ordinary_residual_reference(
        model,
        input_ids=first,
        attention_mask=mask,
    )
    second_reference = collect_ordinary_residual_reference(
        model,
        input_ids=second,
        attention_mask=mask,
    )
    assert first_reference.residual_contributions.shape == (4, 1, 5, 32)
    assert len(first_reference.block_inputs) == len(first_reference.block_outputs) == 4
    assert not hasattr(model.config, "attnres_execution")
    assert not any("query" in name.lower() or "alpha" in name.lower() for name, _ in model.named_parameters())
    intervals = tuple((start, end) for start in range(4) for end in range(start, 4))
    first_costs = pairwise_directional_interval_costs(first_reference, intervals)
    second_costs = pairwise_directional_interval_costs(second_reference, intervals)
    assert any(abs(first_costs[key] - second_costs[key]) > 1.0e-8 for key in intervals)


def test_dp_consumes_residual_cost_matrix_without_replay() -> None:
    matrix = np.full((6, 6), np.inf, dtype=np.float64)
    for start in range(6):
        for length in (1, 2):
            end = start + length - 1
            if end < 6:
                matrix[start, end] = float(length * 10 + start)
    dp = solve_partition(matrix, num_blocks=3, task="tiny", candidate_lengths=[1, 2])
    brute = brute_force_partition(matrix, num_blocks=3, task="tiny", candidate_lengths=[1, 2])
    assert dp.partition.sha256 == brute.partition.sha256
    assert dp.cost == brute.cost


def test_qwen3_partition_search_space_is_non_degenerate() -> None:
    feasible = count_feasible_partitions(
        40,
        num_blocks=10,
        candidate_lengths=[2, 3, 4, 5, 6],
    )
    assert feasible == 856945
    assert feasible > 1


def test_synthetic_costs_can_move_qwen3_boundaries() -> None:
    allowed_lengths = (2, 3, 4, 5, 6)
    targets = (
        (4,) * 10,
        (3, 5, 4, 4, 2, 6, 3, 5, 4, 4),
        (5, 3, 3, 5, 4, 4, 6, 2, 4, 4),
    )

    results = []
    for target in targets:
        matrix = np.full((40, 40), np.inf, dtype=np.float64)
        start = 0
        for length in allowed_lengths:
            for position in range(40 - length + 1):
                matrix[position, position + length - 1] = 1.0
        for length in target:
            matrix[start, start + length - 1] = 0.0
            start += length
        result = solve_partition(
            matrix,
            num_blocks=10,
            task="synthetic",
            candidate_lengths=allowed_lengths,
        )
        assert result.partition.lengths == target
        assert len(result.partition.blocks) == 10
        assert sum(result.partition.lengths) == 40
        assert min(result.partition.lengths) >= 2
        assert max(result.partition.lengths) <= 6
        assert result.cost == 0.0
        results.append(result.partition)

    assert results[0].boundary_ends != results[1].boundary_ends
    assert results[1].boundary_ends != results[2].boundary_ends
    assert results[1].lengths != (4,) * 10
    assert results[2].lengths != (4,) * 10


def test_partition_metadata_round_trip_preserves_2_to_6_constraints() -> None:
    partition = MoiraiPartition.from_lengths(
        [6, 4],
        task="math",
        num_transformer_blocks=10,
        min_length=2,
        max_length=6,
        no_adjacent_singletons=True,
    )
    restored = MoiraiPartition.from_dict(partition.to_dict())
    assert restored == partition
    assert restored.sha256 == partition.sha256
    assert restored.lengths == (6, 4)


def test_fixed_block_count_policy_uses_dynamic_model_depth() -> None:
    assert resolve_fixed_num_blocks(
        28, fixed_block_size=4, policy="ceil_num_layers_over_fixed_block_size"
    ) == 7
    assert resolve_fixed_num_blocks(
        32, fixed_block_size=4, policy="ceil_num_layers_over_fixed_block_size"
    ) == 8
    assert resolve_fixed_num_blocks(
        36, fixed_block_size=4, policy="ceil_num_layers_over_fixed_block_size"
    ) == 9


def test_tasks_share_n_and_fixed_baseline_has_same_completed_block_count() -> None:
    task_n = {
        task: resolve_fixed_num_blocks(
            40,
            fixed_block_size=4,
            policy="ceil_num_layers_over_fixed_block_size",
        )
        for task in ("math", "multihop", "code")
    }
    assert task_n == {"math": 10, "multihop": 10, "code": 10}

    fixed = fixed_kimi_partition(
        task="fixed", num_transformer_blocks=40, block_size=4
    )
    adaptive = MoiraiPartition.from_lengths(
        [3, 5, 4, 4, 2, 6, 3, 5, 4, 4],
        task="math",
        num_transformer_blocks=40,
        min_length=2,
        max_length=6,
    )
    assert len(fixed.blocks) == len(adaptive.blocks) == 10
    assert fixed.boundary_ends != adaptive.boundary_ends
    assert fixed.lengths == (4,) * 10
    assert min(adaptive.lengths) >= 2 and max(adaptive.lengths) <= 6
    assert sum(adaptive.lengths) == sum(fixed.lengths) == 40


def test_fixed_kimi_allows_short_final_block() -> None:
    partition = fixed_kimi_partition(
        task="fixed", num_transformer_blocks=30, block_size=4
    )
    assert partition.lengths == (4, 4, 4, 4, 4, 4, 4, 2)
    assert len(partition.blocks) == resolve_fixed_num_blocks(
        30, fixed_block_size=4, policy="ceil_num_layers_over_fixed_block_size"
    )


def test_local_math_multihop_code_cases_are_task_specific() -> None:
    tokenizer_path = Path("/iridisfs/scratch/tz6g22/kimi/data/tokenizers/qwen3-0.6b")
    if not (tokenizer_path / "tokenizer.json").is_file():
        raise AssertionError("The local tiny tokenizer required by the sanity test is missing")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, use_fast=True)
    data_config = load_yaml("configs/data.yaml")
    manifest = load_manifest("outputs/data/splits.json")
    model = _tiny_model(vocab_size=tokenizer.vocab_size)
    matrices: dict[str, tuple[float, ...]] = {}
    for task in ("math", "multihop", "code"):
        source_name = next(iter(data_config["discovery_sources"][task]))
        source = data_config["sources"][source_name]
        records = [
            row for row in manifest
            if row["task"] == task
            and row["dataset"] == source["dataset_name"]
            and row["assigned_split"] == "stage2_discovery"
        ][:2]
        pool = load_dataset_pool(data_config, str(source["dataset_name"]))
        examples = []
        for record in records:
            dataset, mapping = pool[str(record["official_split"])]
            examples.append(
                encode_prompt_only(
                    tokenizer,
                    task=task,
                    row=dataset[int(record["row_index"])],
                    field_mapping=mapping,
                    stable_id=str(record["stable_id"]),
                    max_length=128,
                )
            )
        values: list[float] = []
        for example in examples:
            reference = collect_ordinary_residual_reference(
                model,
                input_ids=example.input_ids.unsqueeze(0),
                attention_mask=example.attention_mask.unsqueeze(0),
            )
            values.append(
                float(
                    pairwise_directional_interval_cost(
                        reference.residual_contributions,
                        reference.attention_mask,
                        start=0,
                        end=2,
                    )
                )
            )
        matrices[task] = tuple(values)
    assert len(set(matrices.values())) == 3
