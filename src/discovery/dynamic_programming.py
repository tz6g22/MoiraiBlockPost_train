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
    predecessor: dict[tuple[int, int], tuple[int, int, int]]


@dataclass(frozen=True)
class _State:
    cost: float
    lengths: tuple[int, ...]


@dataclass(frozen=True)
class _SimilarityState:
    similarity: float
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
    if len(candidate.lengths) != len(current.lengths):
        return len(candidate.lengths) < len(current.lengths)
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
    merge_cost_threshold: float,
    task: str,
    candidate_lengths: Sequence[int] | None = None,
    no_adjacent_singletons: bool = False,
) -> DPResult:
    """Minimize block count over complete partitions under a cost threshold."""
    if cost_matrix.ndim != 2 or cost_matrix.shape[0] != cost_matrix.shape[1]:
        raise ValueError("cost_matrix must be square")
    transformer_blocks = int(cost_matrix.shape[0])
    if not math.isfinite(float(merge_cost_threshold)) or merge_cost_threshold < 0:
        raise ValueError("merge_cost_threshold must be finite and non-negative")
    lengths = tuple(
        length
        for length in sorted(set(candidate_lengths or range(1, transformer_blocks + 1)))
        if length <= transformer_blocks
    )
    if not lengths or any(length <= 0 for length in lengths):
        raise ValueError("candidate_lengths must contain positive integers")

    states: dict[tuple[int, int], _State] = {(0, 0): _State(cost=0.0, lengths=())}
    predecessor: dict[tuple[int, int], tuple[int, int, int]] = {}

    for i in range(1, transformer_blocks + 1):
        for length in lengths:
            previous_i = i - length
            if previous_i < 0:
                continue
            interval_cost = float(cost_matrix[previous_i, i - 1])
            legal = length == 1 or (
                math.isfinite(interval_cost)
                and interval_cost <= merge_cost_threshold + TIE_TOLERANCE
            )
            if not legal:
                continue
            interval_cost = 0.0 if length == 1 else interval_cost
            new_singleton = int(length == 1)
            for previous_singleton in (0, 1):
                previous_key = (previous_i, previous_singleton)
                previous = states.get(previous_key)
                if previous is None:
                    continue
                if no_adjacent_singletons and previous_singleton == 1 and new_singleton == 1:
                    continue
                candidate = _State(
                    cost=previous.cost + interval_cost,
                    lengths=previous.lengths + (length,),
                )
                key = (i, new_singleton)
                if _prefer(candidate, states.get(key)):
                    states[key] = candidate
                    predecessor[key] = (previous_i, previous_singleton, length)

    terminal_candidates = [
        (key, state)
        for key, state in states.items()
        if key[0] == transformer_blocks
    ]
    if not terminal_candidates:
        raise ValueError(f"No valid partition for R={transformer_blocks}")
    best_key, best_state = terminal_candidates[0]
    for key, state in terminal_candidates[1:]:
        if _prefer(state, best_state):
            best_key, best_state = key, state

    # Verify that predecessor recovery exactly reproduces the selected path.
    recovered: list[int] = []
    cursor = best_key
    while cursor != (0, 0):
        if cursor not in predecessor:
            raise RuntimeError(f"Missing predecessor for DP state {cursor}")
        previous_i, previous_singleton, length = predecessor[cursor]
        recovered.append(length)
        cursor = (previous_i, previous_singleton)
    recovered.reverse()
    if tuple(recovered) != best_state.lengths:
        raise RuntimeError("DP predecessor backtracking disagrees with selected state")

    partition = MoiraiPartition.from_lengths(
        recovered,
        task=task,
        num_transformer_blocks=transformer_blocks,
        min_length=min(lengths),
        max_length=max(lengths),
        no_adjacent_singletons=no_adjacent_singletons,
    )
    return DPResult(
        partition=partition,
        cost=best_state.cost,
        predecessor=predecessor,
    )


def solve_similarity_partition(
    similarity_matrix: np.ndarray,
    *,
    similarity_threshold: float,
    task: str,
    candidate_lengths: Sequence[int] | None = None,
    no_adjacent_singletons: bool = False,
) -> DPResult:
    """Minimize block count under a direct interval-similarity threshold."""
    if similarity_matrix.ndim != 2 or similarity_matrix.shape[0] != similarity_matrix.shape[1]:
        raise ValueError("similarity_matrix must be square")
    transformer_blocks = int(similarity_matrix.shape[0])
    if not math.isfinite(float(similarity_threshold)) or not 0.0 <= similarity_threshold <= 1.0:
        raise ValueError("similarity_threshold must be finite and in [0, 1]")
    lengths = tuple(
        length
        for length in sorted(set(candidate_lengths or range(1, transformer_blocks + 1)))
        if length <= transformer_blocks
    )
    if not lengths or any(length <= 0 for length in lengths):
        raise ValueError("candidate_lengths must contain positive integers")

    states: dict[tuple[int, int], _SimilarityState] = {
        (0, 0): _SimilarityState(similarity=0.0, lengths=())
    }
    predecessor: dict[tuple[int, int], tuple[int, int, int]] = {}

    def prefer(candidate: _SimilarityState, current: _SimilarityState | None) -> bool:
        if current is None:
            return True
        if len(candidate.lengths) != len(current.lengths):
            return len(candidate.lengths) < len(current.lengths)
        if candidate.similarity > current.similarity + TIE_TOLERANCE:
            return True
        if abs(candidate.similarity - current.similarity) > TIE_TOLERANCE:
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

    for end_exclusive in range(1, transformer_blocks + 1):
        for length in lengths:
            start = end_exclusive - length
            if start < 0:
                continue
            interval_similarity = float(similarity_matrix[start, end_exclusive - 1])
            legal = length == 1 or (
                math.isfinite(interval_similarity)
                and interval_similarity + TIE_TOLERANCE >= similarity_threshold
            )
            if not legal:
                continue
            interval_similarity = 1.0 if length == 1 else interval_similarity
            new_singleton = int(length == 1)
            for previous_singleton in (0, 1):
                previous_key = (start, previous_singleton)
                previous = states.get(previous_key)
                if previous is None:
                    continue
                if no_adjacent_singletons and previous_singleton == 1 and new_singleton == 1:
                    continue
                candidate = _SimilarityState(
                    similarity=previous.similarity + interval_similarity,
                    lengths=previous.lengths + (length,),
                )
                key = (end_exclusive, new_singleton)
                if prefer(candidate, states.get(key)):
                    states[key] = candidate
                    predecessor[key] = (start, previous_singleton, length)

    terminal_candidates = [
        (key, state)
        for key, state in states.items()
        if key[0] == transformer_blocks
    ]
    if not terminal_candidates:
        raise ValueError(f"No valid similarity partition for R={transformer_blocks}")
    best_key, best_state = terminal_candidates[0]
    for key, state in terminal_candidates[1:]:
        if prefer(state, best_state):
            best_key, best_state = key, state

    recovered: list[int] = []
    cursor = best_key
    while cursor != (0, 0):
        if cursor not in predecessor:
            raise RuntimeError(f"Missing similarity predecessor for DP state {cursor}")
        previous_start, previous_singleton, length = predecessor[cursor]
        recovered.append(length)
        cursor = (previous_start, previous_singleton)
    recovered.reverse()
    if tuple(recovered) != best_state.lengths:
        raise RuntimeError("Similarity DP predecessor disagrees with selected state")

    partition = MoiraiPartition.from_lengths(
        recovered,
        task=task,
        num_transformer_blocks=transformer_blocks,
        min_length=min(lengths),
        max_length=max(lengths),
        no_adjacent_singletons=no_adjacent_singletons,
    )
    return DPResult(
        partition=partition,
        cost=float(-best_state.similarity),
        predecessor=predecessor,
    )


def brute_force_partition(
    cost_matrix: np.ndarray,
    *,
    merge_cost_threshold: float,
    task: str,
    candidate_lengths: Sequence[int] | None = None,
    no_adjacent_singletons: bool = False,
) -> DPResult:
    """Small-problem oracle used only to verify DP correctness."""
    transformer_blocks = int(cost_matrix.shape[0])
    if not math.isfinite(float(merge_cost_threshold)) or merge_cost_threshold < 0:
        raise ValueError("merge_cost_threshold must be finite and non-negative")
    allowed_lengths = tuple(
        length
        for length in sorted(set(candidate_lengths or range(1, transformer_blocks + 1)))
        if length <= transformer_blocks
    )
    if not allowed_lengths or any(length <= 0 for length in allowed_lengths):
        raise ValueError("candidate_lengths must contain positive integers")
    candidates: list[_State] = []

    def visit(lengths: tuple[int, ...], consumed: int) -> None:
        if consumed == transformer_blocks:
            cost = 0.0
            start = 0
            for length in lengths:
                interval_cost = float(cost_matrix[start, start + length - 1])
                if length > 1 and (
                    not math.isfinite(interval_cost)
                    or interval_cost > merge_cost_threshold + TIE_TOLERANCE
                ):
                    return
                cost += 0.0 if length == 1 else interval_cost
                start += length
            candidates.append(_State(cost=cost, lengths=lengths))
            return
        for length in allowed_lengths:
            if no_adjacent_singletons and lengths and lengths[-1] == 1 and length == 1:
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
        min_length=min(allowed_lengths),
        max_length=max(allowed_lengths),
        no_adjacent_singletons=no_adjacent_singletons,
    )
    return DPResult(partition=partition, cost=best.cost, predecessor={})


def valid_interval_mask(
    transformer_blocks: int = 32,
    candidate_lengths: Sequence[int] | None = None,
) -> np.ndarray:
    lengths = tuple(sorted(set(candidate_lengths or (1, 2, 3, 4))))
    if not lengths or any(length <= 0 for length in lengths):
        raise ValueError("candidate_lengths must contain positive integers")
    mask = np.zeros((transformer_blocks, transformer_blocks), dtype=bool)
    for start in range(transformer_blocks):
        for length in lengths:
            end = start + length - 1
            if end < transformer_blocks:
                mask[start, end] = True
    return mask
