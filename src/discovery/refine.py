from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from src.modeling.partition import MoiraiPartition


@dataclass(frozen=True)
class RefinementRecord:
    sweep: int
    boundary_index: int
    direction: str
    old_lengths: tuple[int, ...]
    candidate_lengths: tuple[int, ...]
    old_score: float
    candidate_score: float
    accepted: bool

    def to_dict(self) -> dict:
        return {
            "sweep": self.sweep,
            "boundary_index": self.boundary_index,
            "direction": self.direction,
            "old_lengths": list(self.old_lengths),
            "candidate_lengths": list(self.candidate_lengths),
            "old_score": self.old_score,
            "candidate_score": self.candidate_score,
            "accepted": self.accepted,
        }


@dataclass(frozen=True)
class RefinementResult:
    partition: MoiraiPartition
    score: float
    records: tuple[RefinementRecord, ...]


def boundary_neighbors(
    partition: MoiraiPartition,
) -> tuple[tuple[int, str, MoiraiPartition], ...]:
    lengths = list(partition.lengths)
    candidates: list[tuple[int, str, MoiraiPartition]] = []
    for boundary_index in range(len(lengths) - 1):
        for direction, delta in (("left", -1), ("right", 1)):
            candidate_lengths = list(lengths)
            candidate_lengths[boundary_index] += delta
            candidate_lengths[boundary_index + 1] -= delta
            try:
                candidate = MoiraiPartition.from_lengths(
                    candidate_lengths,
                    task=partition.task,
                    num_transformer_blocks=partition.num_transformer_blocks,
                )
            except ValueError:
                continue
            candidates.append((boundary_index, direction, candidate))
    return tuple(candidates)


def refine_partition(
    initial: MoiraiPartition,
    score_partition: Callable[[MoiraiPartition], float],
    *,
    initial_score: float | None = None,
    maximum_sweeps: int = 5,
) -> RefinementResult:
    if maximum_sweeps <= 0:
        raise ValueError("maximum_sweeps must be positive")
    current = initial
    current_score = (
        float(initial_score)
        if initial_score is not None
        else float(score_partition(initial))
    )
    records: list[RefinementRecord] = []

    for sweep in range(maximum_sweeps):
        scored: list[tuple[float, int, int, str, MoiraiPartition]] = []
        for boundary_index, direction, candidate in boundary_neighbors(current):
            score = float(score_partition(candidate))
            direction_order = 0 if direction == "left" else 1
            scored.append(
                (score, boundary_index, direction_order, direction, candidate)
            )
        if not scored:
            break
        best_score, best_boundary, best_direction_order, best_direction, best_partition = (
            scored[0]
        )
        for candidate in scored[1:]:
            score, boundary, direction_order, direction, partition = candidate
            if score < best_score - 1.0e-12 or (
                abs(score - best_score) <= 1.0e-12
                and (boundary, direction_order, partition.lengths)
                < (
                    best_boundary,
                    best_direction_order,
                    best_partition.lengths,
                )
            ):
                (
                    best_score,
                    best_boundary,
                    best_direction_order,
                    best_direction,
                    best_partition,
                ) = candidate
        required_improvement = max(1.0e-8, 1.0e-6 * current_score)
        accepted = current_score - best_score > required_improvement
        for score, boundary, _, direction, candidate in scored:
            records.append(
                RefinementRecord(
                    sweep=sweep,
                    boundary_index=boundary,
                    direction=direction,
                    old_lengths=current.lengths,
                    candidate_lengths=candidate.lengths,
                    old_score=current_score,
                    candidate_score=score,
                    accepted=(
                        accepted
                        and boundary == best_boundary
                        and direction == best_direction
                        and candidate.lengths == best_partition.lengths
                    ),
                )
            )
        if not accepted:
            break
        current = best_partition
        current_score = best_score

    return RefinementResult(
        partition=current,
        score=current_score,
        records=tuple(records),
    )
