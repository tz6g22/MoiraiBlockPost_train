from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from safetensors.torch import load_file

from src.common import sha256_file
from src.distributed.fsdp_utils import load_selected_parameter_state
from src.modeling.partition import MoiraiPartition


@dataclass(frozen=True)
class MoiraiConfigBundle:
    task: str
    base_checkpoint_sha256: str
    partition: MoiraiPartition
    query_path: Path
    query_sha256: str
    trainable_parameters: tuple[str, ...]
    alpha_path: Path | None = None
    alpha_sha256: str | None = None

    @classmethod
    def load(
        cls,
        *,
        partition_path: str | Path,
        query_manifest_path: str | Path,
        expected_base_checkpoint_sha256: str,
    ) -> "MoiraiConfigBundle":
        partition = MoiraiPartition.from_json(partition_path)
        manifest_path = Path(query_manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        required = {
            "task",
            "base_checkpoint_sha256",
            "partition_sha256",
            "query_sha256",
            "query_file",
            "trainable_parameters",
            "trained_tokens",
            "processed_nonpadding_tokens",
            "training_token_unit",
            "seed",
        }
        missing = sorted(required - manifest.keys())
        if missing:
            raise ValueError(f"Query manifest is missing keys: {missing}")
        if manifest["task"] != partition.task:
            raise ValueError("Partition and query tasks do not match")
        if manifest["partition_sha256"] != partition.sha256:
            raise ValueError("Partition hash does not match query manifest")
        if manifest["base_checkpoint_sha256"] != expected_base_checkpoint_sha256:
            raise ValueError("Base checkpoint hash does not match query manifest")
        if manifest["training_token_unit"] != "nonpadding_input":
            raise ValueError("Query manifest uses an unsupported training token unit")
        trainable_parameters = manifest["trainable_parameters"]
        if (
            not isinstance(trainable_parameters, list)
            or not trainable_parameters
            or not all(isinstance(name, str) and name for name in trainable_parameters)
        ):
            raise ValueError(
                "Query manifest trainable_parameters must be a non-empty string list"
            )
        if any("query" not in name for name in trainable_parameters):
            raise ValueError(
                "Query manifest contains a trainable parameter outside pseudo-query"
            )

        query_path = manifest_path.parent / manifest["query_file"]
        if not query_path.is_file():
            raise FileNotFoundError(f"Query file is missing: {query_path}")
        actual_query_hash = sha256_file(query_path)
        if actual_query_hash != manifest["query_sha256"]:
            raise ValueError("Query file hash does not match query manifest")
        alpha_path = None
        alpha_hash = None
        if manifest.get("alpha_file") is not None:
            alpha_path = manifest_path.parent / str(manifest["alpha_file"])
            if not alpha_path.is_file():
                raise FileNotFoundError(f"Alpha checkpoint is missing: {alpha_path}")
            alpha_hash = sha256_file(alpha_path)
            if alpha_hash != manifest.get("alpha_sha256"):
                raise ValueError("Alpha checkpoint hash does not match query manifest")
        return cls(
            task=partition.task,
            base_checkpoint_sha256=expected_base_checkpoint_sha256,
            partition=partition,
            query_path=query_path,
            query_sha256=actual_query_hash,
            trainable_parameters=tuple(trainable_parameters),
            alpha_path=alpha_path,
            alpha_sha256=alpha_hash,
        )

    def apply_to_model(self, model) -> None:
        state = load_file(self.query_path)
        expected = set(self.trainable_parameters)
        if set(state) != expected:
            raise ValueError("Query checkpoint tensors do not match its manifest")
        load_selected_parameter_state(
            model,
            state,
            lambda name, _parameter: "pseudo_query" in name,
        )
        if self.alpha_path is not None:
            alpha_state = load_file(self.alpha_path)
            expected_alpha = {
                name for name, _ in model.named_parameters() if "alpha" in name
            }
            if set(alpha_state) != expected_alpha:
                raise ValueError("Alpha checkpoint tensors do not match the model")
            load_selected_parameter_state(
                model,
                alpha_state,
                lambda name, _parameter: "alpha" in name,
            )
        model.config.attnres_execution = "formal" if self.alpha_path else "moirai"
        model.config.moirai_partition = list(self.partition.lengths)
        model.config.moirai_task = self.task
        model.config.use_cache = False


def require_matching_bundle(
    partition: MoiraiPartition,
    bundle: MoiraiConfigBundle,
) -> None:
    if partition.task != bundle.task or partition.sha256 != bundle.partition.sha256:
        raise ValueError("Partition and pseudo-query must be switched atomically")
