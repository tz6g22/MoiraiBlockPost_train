from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from src.common import canonical_json, sha256_json


@dataclass(frozen=True)
class MoiraiBlock:
    block_id: int
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start + 1

    def to_dict(self) -> dict[str, int]:
        return {
            "block_id": self.block_id,
            "start": self.start,
            "end": self.end,
            "length": self.length,
        }


@dataclass(frozen=True)
class MoiraiPartition:
    task: str
    num_transformer_blocks: int
    blocks: tuple[MoiraiBlock, ...]
    min_block_length: int = 1
    max_block_length: int = 4
    no_adjacent_singletons: bool = True

    @classmethod
    def from_lengths(
        cls,
        lengths: Iterable[int],
        *,
        task: str,
        num_transformer_blocks: int = 32,
        min_length: int = 1,
        max_length: int = 4,
        no_adjacent_singletons: bool = True,
    ) -> "MoiraiPartition":
        blocks: list[MoiraiBlock] = []
        start = 0
        for block_id, length in enumerate(lengths):
            if not isinstance(length, int):
                raise TypeError("Partition lengths must be integers")
            end = start + length - 1
            blocks.append(MoiraiBlock(block_id=block_id, start=start, end=end))
            start = end + 1
        partition = cls(
            task=task,
            num_transformer_blocks=num_transformer_blocks,
            blocks=tuple(blocks),
            min_block_length=min_length,
            max_block_length=max_length,
            no_adjacent_singletons=no_adjacent_singletons,
        )
        partition.validate()
        return partition

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MoiraiPartition":
        required = {"task", "num_transformer_blocks", "blocks"}
        missing = sorted(required - payload.keys())
        if missing:
            raise ValueError(f"Partition JSON is missing keys: {missing}")
        raw_blocks = payload["blocks"]
        if not isinstance(raw_blocks, list):
            raise TypeError("partition.blocks must be a list")
        constraints = payload.get("constraints", {})
        if not isinstance(constraints, dict):
            raise TypeError("partition.constraints must be a mapping")
        min_length = int(constraints.get("min_length", 1))
        max_length = int(constraints.get("max_length", 4))
        no_adjacent_singletons = bool(
            constraints.get("no_adjacent_singletons", True)
        )
        blocks: list[MoiraiBlock] = []
        for raw in raw_blocks:
            if not isinstance(raw, dict):
                raise TypeError("Each partition block must be a mapping")
            expected = {"block_id", "start", "end", "length"}
            if set(raw) != expected:
                raise ValueError(
                    "Each partition block must contain exactly "
                    f"{sorted(expected)}, got {sorted(raw)}"
                )
            block = MoiraiBlock(
                block_id=int(raw["block_id"]),
                start=int(raw["start"]),
                end=int(raw["end"]),
            )
            if int(raw["length"]) != block.length:
                raise ValueError(f"Incorrect length for block {block.block_id}")
            blocks.append(block)
        partition = cls(
            task=str(payload["task"]),
            num_transformer_blocks=int(payload["num_transformer_blocks"]),
            blocks=tuple(blocks),
            min_block_length=min_length,
            max_block_length=max_length,
            no_adjacent_singletons=no_adjacent_singletons,
        )
        partition.validate()
        declared_hash = payload.get("partition_sha256")
        if declared_hash is not None and declared_hash != partition.sha256:
            raise ValueError("partition_sha256 does not match partition content")
        return partition

    @classmethod
    def from_json(cls, path: str | Path) -> "MoiraiPartition":
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise TypeError("Partition JSON root must be an object")
        return cls.from_dict(payload)

    @property
    def lengths(self) -> tuple[int, ...]:
        return tuple(block.length for block in self.blocks)

    @property
    def boundary_ends(self) -> frozenset[int]:
        return frozenset(block.end for block in self.blocks)

    def validate(
        self,
        *,
        min_length: int | None = None,
        max_length: int | None = None,
        no_adjacent_singletons: bool | None = None,
    ) -> None:
        min_length = self.min_block_length if min_length is None else min_length
        max_length = self.max_block_length if max_length is None else max_length
        no_adjacent_singletons = (
            self.no_adjacent_singletons
            if no_adjacent_singletons is None
            else no_adjacent_singletons
        )
        if min_length <= 0 or max_length < min_length:
            raise ValueError("Partition length constraints are invalid")
        if not self.task:
            raise ValueError("Partition task must be non-empty")
        if self.num_transformer_blocks <= 0:
            raise ValueError("num_transformer_blocks must be positive")
        if not self.blocks:
            raise ValueError("Partition must contain at least one block")

        expected_start = 0
        previous_length: int | None = None
        for expected_id, block in enumerate(self.blocks):
            if block.block_id != expected_id:
                raise ValueError("block_id values must be contiguous and zero-based")
            if block.start != expected_start:
                raise ValueError("Partition must be continuous and non-overlapping")
            if not min_length <= block.length <= max_length:
                raise ValueError(
                    f"Block {block.block_id} length {block.length} is outside "
                    f"[{min_length}, {max_length}]"
                )
            if no_adjacent_singletons and previous_length == 1 and block.length == 1:
                raise ValueError("Adjacent singleton blocks are forbidden")
            expected_start = block.end + 1
            previous_length = block.length

        if expected_start != self.num_transformer_blocks:
            raise ValueError(
                "Partition does not fully cover transformer blocks: "
                f"covered 0..{expected_start - 1}, expected 0..{self.num_transformer_blocks - 1}"
            )

    def hash_payload(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "num_transformer_blocks": self.num_transformer_blocks,
            "blocks": [block.to_dict() for block in self.blocks],
            "constraints": {
                "min_length": self.min_block_length,
                "max_length": self.max_block_length,
                "no_adjacent_singletons": self.no_adjacent_singletons,
                "continuous": True,
            },
        }

    @property
    def sha256(self) -> str:
        return sha256_json(self.hash_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.hash_payload()
        payload.update(
            {
                "num_moirai_blocks": len(self.blocks),
                "num_moirai_blocks_note": "must equal len(blocks), not a fixed constant",
                "partition_sha256": self.sha256,
            }
        )
        return payload

    def save_json(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            canonical_json(self.to_dict()) + "\n",
            encoding="utf-8",
        )


def fixed_kimi_partition(
    *,
    task: str = "fixed",
    num_transformer_blocks: int = 32,
    block_size: int = 4,
) -> MoiraiPartition:
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    full_blocks, remainder = divmod(num_transformer_blocks, block_size)
    lengths = [block_size] * full_blocks
    if remainder:
        lengths.append(remainder)
    return MoiraiPartition.from_lengths(
        lengths,
        task=task,
        num_transformer_blocks=num_transformer_blocks,
    )
