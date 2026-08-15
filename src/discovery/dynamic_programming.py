from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.modeling.partition import MoiraiPartition


TIE_TOLERANCE = 1.0e-12


@dataclass(frozen=True)
class DPResult:
    partition: MoiraiPartition
    cost: float
    predecessor: dict[tuple[int, int, int], tuple[int, int, int, int]]


@dataclass(frozen=True)
class _State:
    cost: float
    lengths: tuple[int, ...]


def _boundary_sequence(lengths: tuple[int, ...]) -> tuple[int, ...]:
    total = 0
    boundaries: list[int] = []
    for length in lengths[:-1]:
        total += length
        boundaries.append(total - 1)
    return tuple(boundaries)


def _prefer(candidate: _State, current: _State | None) -> bool:
    if current is None:
        return True
    if candidate.cost < current.cost - TIE_TOLERANCE:
        return True
    if abs(candidate.cost - current.cost) > TIE_TOLERANCE:
        return False

    candidate_key = (
        _boundary_sequence(candidate.lengths),
        -candidate.lengths[-1],
        candidate.lengths,
    )
    current_key = (
        _boundary_sequence(current.lengths),
        -current.lengths[-1],
        current.lengths,
    )
    return candidate_key < current_key


def solve_partition(
    cost_matrix: np.ndarray,
    *,
    num_blocks: int,
    task: str,
) -> DPResult:
    """Solve specification section 17 with deterministic backtracking."""
    if cost_matrix.ndim != 2 or cost_matrix.shape[0] != cost_matrix.shape[1]:
        raise ValueError("cost_matrix must be square")
    transformer_blocks = int(cost_matrix.shape[0])
    if not 9 <= num_blocks <= 16:
        raise ValueError("Formal Moirai search only permits N in [9, 16]")

    states: dict[tuple[int, int, int], _State] = {
        (0, 0, 0): _State(cost=0.0, lengths=())
    }
    predecessor: dict[tuple[int, int, int], tuple[int, int, int, int]] = {}

    for i in range(1, transformer_blocks + 1):
        for k in range(1, min(num_blocks, i) + 1):
            for length in (1, 2, 3, 4):
                previous_i = i - length
                if previous_i < 0:
                    continue
                interval_cost = float(cost_matrix[previous_i, i - 1])
                if not math.isfinite(interval_cost):
                    continue
                new_singleton = int(length == 1)
                for previous_singleton in (0, 1):
                    previous_key = (previous_i, k - 1, previous_singleton)
                    previous = states.get(previous_key)
                    if previous is None:
                        continue
                    if previous_singleton == 1 and new_singleton == 1:
                        continue
                    candidate = _State(
                        cost=previous.cost + interval_cost,
                        lengths=previous.lengths + (length,),
                    )
                    key = (i, k, new_singleton)
                    if _prefer(candidate, states.get(key)):
                        states[key] = candidate
                        predecessor[key] = (
                            previous_i,
                            k - 1,
                            previous_singleton,
                            length,
                        )

    terminal_candidates = [
        (key, state)
        for key, state in states.items()
        if key[0] == transformer_blocks and key[1] == num_blocks
    ]
    if not terminal_candidates:
        raise ValueError(
            f"No valid partition for R={transformer_blocks}, N={num_blocks}"
        )
    best_key, best_state = terminal_candidates[0]
    for key, state in terminal_candidates[1:]:
        if _prefer(state, best_state):
            best_key, best_state = key, state

    # Verify that predecessor recovery exactly reproduces the selected path.
    recovered: list[int] = []
    cursor = best_key
    while cursor != (0, 0, 0):
        if cursor not in predecessor:
            raise RuntimeError(f"Missing predecessor for DP state {cursor}")
        previous_i, previous_k, previous_z, length = predecessor[cursor]
        recovered.append(length)
        cursor = (previous_i, previous_k, previous_z)
    recovered.reverse()
    if tuple(recovered) != best_state.lengths:
        raise RuntimeError("DP predecessor backtracking disagrees with selected state")

    partition = MoiraiPartition.from_lengths(
        recovered,
        task=task,
        num_transformer_blocks=transformer_blocks,
    )
    if len(partition.blocks) != num_blocks:
        raise RuntimeError("DP returned the wrong number of MoiraiBlocks")
    return DPResult(
        partition=partition,
        cost=best_state.cost,
        predecessor=predecessor,
    )


def brute_force_partition(
    cost_matrix: np.ndarray,
    *,
    num_blocks: int,
    task: str,
) -> DPResult:
    """Small-problem oracle used only to verify DP correctness."""
    transformer_blocks = int(cost_matrix.shape[0])
    candidates: list[_State] = []

    def visit(lengths: tuple[int, ...], consumed: int) -> None:
        if len(lengths) == num_blocks:
            if consumed == transformer_blocks:
                cost = 0.0
                start = 0
                for length in lengths:
                    interval_cost = float(cost_matrix[start, start + length - 1])
                    if not math.isfinite(interval_cost):
                        return
                    cost += interval_cost
                    start += length
                candidates.append(_State(cost=cost, lengths=lengths))
            return
        for length in (1, 2, 3, 4):
            if lengths and lengths[-1] == 1 and length == 1:
                continue
            if consumed + length <= transformer_blocks:
                visit(lengths + (length,), consumed + length)

    visit((), 0)
    if not candidates:
        raise ValueError("No brute-force partition exists")
    best = candidates[0]
    for candidate in candidates[1:]:
        if _prefer(candidate, best):
            best = candidate
    partition = MoiraiPartition.from_lengths(
        best.lengths,
        task=task,
        num_transformer_blocks=transformer_blocks,
    )
    return DPResult(partition=partition, cost=best.cost, predecessor={})


def valid_interval_mask(transformer_blocks: int = 32) -> np.ndarray:
    mask = np.zeros((transformer_blocks, transformer_blocks), dtype=bool)
    for start in range(transformer_blocks):
        for end in range(start, min(transformer_blocks, start + 4)):
            mask[start, end] = True
    return mask
