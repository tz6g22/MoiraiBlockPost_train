from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig,
    StateDictType,
)
from safetensors.torch import load_file, save_file

from src.common import sha256_file, sha256_json
from src.distributed.fsdp_utils import load_full_optimizer_state
from src.formal.runtime import task_routing_parameter_names
from src.formal.task_banks import TaskBank
from src.modeling.partition import MoiraiPartition


FORBIDDEN_FIXED_KEYS = ("P_fixed", "Q_fixed", "Alpha_fixed", "fixed_checkpoint", "fixed_fallback")


def _model_config(model):
    """Return the native config whether the model is FSDP-wrapped or not."""
    return model.module.config if isinstance(model, FSDP) else model.config


def converted_config_sha256(model) -> str:
    """Hash immutable conversion settings, excluding the active task route."""
    payload = _model_config(model).to_dict()
    payload["moirai_partition"] = None
    payload["moirai_task"] = "unassigned"
    return sha256_json(payload)


def _save_state(path: Path, state: dict[str, torch.Tensor]) -> str:
    if not state:
        raise ValueError(f"Cannot save an empty state: {path}")
    save_file(
        {name: value.detach().cpu().contiguous() for name, value in state.items()},
        path,
    )
    return sha256_file(path)


def _state_hash(state: dict[str, torch.Tensor]) -> str:
    return sha256_json(
        {
            name: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": sha256_json(value.detach().cpu().contiguous().tolist()),
            }
            for name, value in sorted(state.items())
        }
    )


def save_joint_checkpoint(
    output_dir: str | Path,
    *,
    model,
    banks: dict[str, TaskBank],
    partitions: dict[str, dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    scheduler,
    config: dict[str, Any],
    base_checkpoint_sha256: str,
    data_manifest_sha256: str,
    consumed_tokens: dict[str, int],
    seed: int,
    identity_test: dict[str, Any],
    training_progress: dict[str, Any] | None = None,
    metrics_files: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Save the complete shared-backbone plus the enabled task-bank state."""
    tasks = tuple(banks)
    if not tasks or set(partitions) != set(tasks):
        raise ValueError("Formal checkpoint task banks and partitions do not match")
    if any(key in config for key in FORBIDDEN_FIXED_KEYS):
        raise ValueError("Fixed keys cannot enter a formal checkpoint config")
    if identity_test.get("status") != "PASS":
        raise ValueError("Formal checkpoint requires a passing identity test")
    if identity_test.get("base_checkpoint_sha256") != base_checkpoint_sha256:
        raise ValueError("Formal checkpoint identity/base hash mismatch")
    if identity_test.get("converted_config_sha256") != converted_config_sha256(model):
        raise ValueError("Formal checkpoint identity/converted-config hash mismatch")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    # Only Q/Alpha are task-specific. Shared key normalization remains in the
    # backbone-side checkpoint state and is not duplicated per task.
    routing = set(task_routing_parameter_names(model))
    distributed = isinstance(model, FSDP)
    is_rank0 = not dist.is_initialized() or dist.get_rank() == 0
    if distributed:
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            full_state = model.state_dict()
        shared_state = {
            name: value
            for name, value in full_state.items()
            if name not in routing
        }
    else:
        shared_state = {
            name: parameter.detach().cpu().clone()
            for name, parameter in model.named_parameters()
            if name not in routing
        }
    shared_path = root / "shared_backbone.safetensors"
    shared_file_hash = _save_state(shared_path, shared_state) if is_rank0 else ""
    if dist.is_initialized():
        dist.barrier()
    task_records: dict[str, Any] = {}
    for task in tasks:
        bank = banks[task]
        if bank.task != task or bank.partition_sha256 != partitions[task]["partition_sha256"]:
            raise ValueError(f"Task bank and partition metadata disagree for {task}")
        task_dir = root / task
        task_dir.mkdir(parents=True, exist_ok=True)
        query_state = {name: value for name, value in bank.state.items() if "pseudo_query" in name}
        alpha_state = {name: value for name, value in bank.state.items() if "alpha" in name}
        query_hash = _save_state(task_dir / "query.safetensors", query_state) if is_rank0 else ""
        alpha_file = f"{task}/alpha.safetensors" if alpha_state else None
        alpha_hash = (
            _save_state(task_dir / "alpha.safetensors", alpha_state)
            if is_rank0 and alpha_state
            else None
        )
        task_records[task] = {
            "partition": partitions[task],
            "partition_sha256": bank.partition_sha256,
            "query_file": f"{task}/query.safetensors",
            "query_sha256": query_hash,
            "alpha_file": alpha_file,
            "alpha_sha256": alpha_hash,
            "query_state_sha256": _state_hash(query_state),
            "alpha_state_sha256": _state_hash(alpha_state),
        }
    if distributed:
        from src.distributed.fsdp_utils import full_optimizer_state

        optimizer_state = full_optimizer_state(model, optimizer, _distributed_context())
    else:
        optimizer_state = optimizer.state_dict()
    if is_rank0:
        if distributed and optimizer_state is None:
            raise RuntimeError("Rank 0 did not receive the full FSDP optimizer state")
        torch.save(optimizer_state, root / "optimizer.pt")
        (root / "scheduler.json").write_text(
            json.dumps(scheduler.state_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if distributed:
        task_optimizer_path = root / f"task_optimizer_rank{dist.get_rank()}.pt"
        torch.save(
            {task: banks[task].optimizer_state for task in tasks},
            task_optimizer_path,
        )
        rng_state = {"torch": torch.get_rng_state()}
        if torch.cuda.is_available():
            rng_state["cuda"] = torch.cuda.get_rng_state_all()
        rng_path = root / f"rng_rank{dist.get_rank()}.pt"
        torch.save(rng_state, rng_path)
        local_task_optimizer_hash = sha256_file(task_optimizer_path)
        local_rng_hash = sha256_file(rng_path)
        task_optimizer_hashes: list[str | None] = [None] * dist.get_world_size()
        rng_hashes: list[str | None] = [None] * dist.get_world_size()
        dist.all_gather_object(task_optimizer_hashes, local_task_optimizer_hash)
        dist.all_gather_object(rng_hashes, local_rng_hash)
        if is_rank0:
            torch.save(
                {task: banks[task].optimizer_state for task in tasks},
                root / "task_optimizer.pt",
            )
            rng_state = {"torch": torch.get_rng_state()}
            if torch.cuda.is_available():
                rng_state["cuda"] = torch.cuda.get_rng_state_all()
            torch.save(rng_state, root / "rng.pt")
    else:
        torch.save(
            {task: banks[task].optimizer_state for task in tasks},
            root / "task_optimizer.pt",
        )
        rng_state = {"torch": torch.get_rng_state()}
        if torch.cuda.is_available():
            rng_state["cuda"] = torch.cuda.get_rng_state_all()
        torch.save(rng_state, root / "rng.pt")
        task_optimizer_hashes = [sha256_file(root / "task_optimizer.pt")]
        rng_hashes = [sha256_file(root / "rng.pt")]
    if dist.is_initialized():
        dist.barrier()
    manifest = {
        "checkpoint_kind": "formal_shared_backbone",
        "base_model_hash": base_checkpoint_sha256,
        "base_checkpoint_sha256": base_checkpoint_sha256,
        "shared_backbone_file": shared_path.name,
        "shared_backbone_hash": shared_file_hash,
        "shared_backbone_sha256": shared_file_hash,
        "enabled_tasks": list(tasks),
        "task_banks": task_records,
        "partition_per_task": {
            task: partitions[task] for task in tasks
        },
        "partition_hash_per_task": {
            task: task_records[task]["partition_sha256"] for task in tasks
        },
        "query_hash_per_task": {
            task: task_records[task]["query_sha256"] for task in tasks
        },
        "alpha_hash_per_task": {
            task: task_records[task]["alpha_sha256"] for task in tasks
        },
        "optimizer_config": config["training"]["optimizer"],
        "scheduler_config": config["training"]["scheduler"],
        "optimizer_file": "optimizer.pt",
        "task_optimizer_file": "task_optimizer.pt",
        "task_optimizer_files": (
            [f"task_optimizer_rank{rank}.pt" for rank in range(dist.get_world_size())]
            if distributed
            else ["task_optimizer.pt"]
        ),
        "task_optimizer_sha256": task_optimizer_hashes,
        "scheduler_file": "scheduler.json",
        "rng_file": "rng.pt",
        "rng_files": (
            [f"rng_rank{rank}.pt" for rank in range(dist.get_world_size())]
            if distributed
            else ["rng.pt"]
        ),
        "rng_sha256": rng_hashes,
        "distributed_world_size": dist.get_world_size() if distributed else 1,
        "config": config,
        "config_sha256": sha256_json(config),
        "data_manifest_hash": data_manifest_sha256,
        "data_manifest_sha256": data_manifest_sha256,
        "identity_test": identity_test,
        "token_budget": config["data"]["token_budget"],
        "consumed_tokens_per_task": {task: int(consumed_tokens[task]) for task in tasks},
        "seed": int(seed),
        "training_progress": training_progress or {},
        "metrics_files": metrics_files or {},
        "forbidden_fixed_keys": list(FORBIDDEN_FIXED_KEYS),
    }
    manifest["manifest_sha256"] = sha256_json(manifest)
    if is_rank0:
        (root / "checkpoint_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if dist.is_initialized():
        payload = [manifest if is_rank0 else None]
        dist.broadcast_object_list(payload, src=0)
        manifest = payload[0]
    return manifest


def _distributed_context():
    """Provide the small context required by full optimizer-state collection."""
    from src.distributed.fsdp_utils import DistributedContext

    if not dist.is_initialized():
        raise RuntimeError("FSDP checkpoint collection requires an initialized process group")
    return DistributedContext(
        rank=dist.get_rank(),
        local_rank=int(torch.cuda.current_device()),
        world_size=dist.get_world_size(),
        device=torch.device("cuda", torch.cuda.current_device()),
    )


def load_joint_checkpoint(
    checkpoint_dir: str | Path,
    *,
    model,
    banks: dict[str, TaskBank],
    optimizer: torch.optim.Optimizer | None = None,
    scheduler=None,
    active_task: str | None = None,
    restore_rng: bool = True,
    expected_base_checkpoint_sha256: str | None = None,
    expected_data_manifest_sha256: str | None = None,
    expected_config_sha256: str | None = None,
) -> dict[str, Any]:
    """Restore and validate every formal shared/task state before training or inference."""
    root = Path(checkpoint_dir)
    manifest_path = root / "checkpoint_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"STALE_CHECKPOINT_MISMATCH: formal checkpoint manifest is missing: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    declared_manifest_hash = manifest.get("manifest_sha256")
    manifest_payload = dict(manifest)
    manifest_payload.pop("manifest_sha256", None)
    if declared_manifest_hash != sha256_json(manifest_payload):
        raise ValueError("Formal checkpoint manifest hash mismatch")
    tasks = tuple(banks)
    if tuple(manifest.get("enabled_tasks", ())) != tasks:
        raise ValueError("TASK_ORDER_MISMATCH: formal checkpoint task set does not match enabled tasks")
    if manifest.get("base_model_hash") != manifest.get("base_checkpoint_sha256"):
        raise ValueError("MODEL_IDENTITY_MISMATCH: formal checkpoint base model hash aliases disagree")
    if manifest.get("shared_backbone_hash") != manifest.get("shared_backbone_sha256"):
        raise ValueError("STALE_CHECKPOINT_MISMATCH: formal checkpoint shared-backbone hash aliases disagree")
    if manifest.get("data_manifest_hash") != manifest.get("data_manifest_sha256"):
        raise ValueError("STALE_CHECKPOINT_MISMATCH: formal checkpoint data-manifest hash aliases disagree")
    if set(manifest.get("partition_per_task", ())) != set(tasks):
        raise ValueError("STALE_PARTITION_MISMATCH: formal checkpoint partition_per_task is incomplete")
    task_banks = manifest.get("task_banks", {})
    if set(task_banks) != set(tasks):
        raise ValueError("TASK_ORDER_MISMATCH: formal checkpoint task_banks do not match enabled tasks")
    for task in tasks:
        task_record = task_banks.get(task, {})
        if manifest["partition_hash_per_task"].get(task) != task_record.get("partition_sha256"):
            raise ValueError(f"STALE_PARTITION_MISMATCH: formal checkpoint partition hash index mismatch for {task}")
        if manifest["query_hash_per_task"].get(task) != task_record.get("query_sha256"):
            raise ValueError(f"STALE_CHECKPOINT_MISMATCH: formal checkpoint query hash index mismatch for {task}")
        if manifest["alpha_hash_per_task"].get(task) != task_record.get("alpha_sha256"):
            raise ValueError(f"STALE_CHECKPOINT_MISMATCH: formal checkpoint alpha hash index mismatch for {task}")
    checkpoint_config = manifest.get("config", {})
    if manifest.get("optimizer_config") != checkpoint_config.get("training", {}).get("optimizer"):
        raise ValueError("STALE_CHECKPOINT_MISMATCH: formal checkpoint optimizer config is not self-consistent")
    if manifest.get("scheduler_config") != checkpoint_config.get("training", {}).get("scheduler"):
        raise ValueError("STALE_CHECKPOINT_MISMATCH: formal checkpoint scheduler config is not self-consistent")
    identity_test = manifest.get("identity_test")
    if not isinstance(identity_test, dict) or identity_test.get("status") != "PASS":
        raise ValueError("MODEL_IDENTITY_MISMATCH: formal checkpoint identity test is missing or failed")
    if (
        expected_base_checkpoint_sha256 is not None
        and identity_test.get("base_checkpoint_sha256") != expected_base_checkpoint_sha256
    ):
        raise ValueError("MODEL_IDENTITY_MISMATCH: formal checkpoint identity/base hash mismatch")
    if identity_test.get("converted_config_sha256") != converted_config_sha256(model):
        raise ValueError("MODEL_IDENTITY_MISMATCH: formal checkpoint identity/converted-config hash mismatch")
    if isinstance(model, FSDP):
        declared_world_size = int(manifest.get("distributed_world_size", 0))
        if not dist.is_initialized() or declared_world_size != dist.get_world_size():
            raise ValueError(
                "STALE_CHECKPOINT_MISMATCH: formal checkpoint world size does not match the active FSDP process group"
            )
    expected_identity = {
        "base_checkpoint_sha256": expected_base_checkpoint_sha256,
        "data_manifest_sha256": expected_data_manifest_sha256,
        "config_sha256": expected_config_sha256,
    }
    for field, expected in expected_identity.items():
        if expected is not None and manifest.get(field) != expected:
            status = "MODEL_IDENTITY_MISMATCH" if field == "base_checkpoint_sha256" else "STALE_CHECKPOINT_MISMATCH"
            raise ValueError(f"{status}: formal checkpoint {field} mismatch")
    if set(manifest.get("forbidden_fixed_keys", ())) != set(FORBIDDEN_FIXED_KEYS):
        raise ValueError("STALE_CHECKPOINT_MISMATCH: formal checkpoint Fixed-key policy mismatch")
    if any(
        "fixed" in str(key).lower()
        for key in manifest
        if key != "forbidden_fixed_keys"
    ):
        raise ValueError("STALE_CHECKPOINT_MISMATCH: formal checkpoint manifest contains a Fixed entry")
    shared_path = root / manifest["shared_backbone_file"]
    if sha256_file(shared_path) != manifest["shared_backbone_sha256"]:
        raise ValueError("STALE_CHECKPOINT_MISMATCH: shared backbone checkpoint hash mismatch")
    shared = load_file(shared_path, device="cpu")
    if isinstance(model, FSDP):
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
        ):
            incompatible = model.load_state_dict(shared, strict=False)
    else:
        incompatible = model.load_state_dict(shared, strict=False)
    # Only the task-routed Q/Alpha tensors are intentionally absent from the
    # shared backbone file.  Every native Qwen3 parameter, including lm_head,
    # must be present so a partial backbone cannot pass reload validation.
    allowed_missing = set(task_routing_parameter_names(model))
    if set(incompatible.missing_keys) - allowed_missing or incompatible.unexpected_keys:
        raise ValueError(f"STALE_CHECKPOINT_MISMATCH: shared backbone state mismatch: {incompatible}")
    loaded_banks: dict[str, TaskBank] = {}
    for task in tasks:
        if task not in banks or task not in task_banks:
            raise ValueError(f"TASK_ORDER_MISMATCH: formal checkpoint is missing task {task}")
        record = task_banks[task]
        partition_record = record.get("partition", {})
        if partition_record.get("task") != task:
            raise ValueError(f"STALE_PARTITION_MISMATCH: formal checkpoint partition task mismatch for {task}")
        partition = MoiraiPartition.from_dict(partition_record)
        if partition.num_transformer_blocks != int(_model_config(model).num_hidden_layers):
            raise ValueError(f"MODEL_IDENTITY_MISMATCH: formal checkpoint depth mismatch for {task}")
        if partition.sha256 != record["partition_sha256"]:
            raise ValueError(f"STALE_PARTITION_MISMATCH: formal checkpoint partition hash mismatch for {task}")
        state: dict[str, torch.Tensor] = {}
        for field in ("query_file", "alpha_file"):
            relative_path = record.get(field)
            if relative_path is None:
                continue
            path = root / relative_path
            if sha256_file(path) != record.get(field.replace("file", "sha256")):
                raise ValueError(f"STALE_CHECKPOINT_MISMATCH: {task} {field} hash mismatch")
            state.update(load_file(path, device="cpu"))
        query_state = {name: value for name, value in state.items() if "pseudo_query" in name}
        alpha_state = {name: value for name, value in state.items() if "alpha" in name}
        if _state_hash(query_state) != record.get("query_state_sha256"):
            raise ValueError(f"STALE_CHECKPOINT_MISMATCH: {task} query state hash mismatch")
        if _state_hash(alpha_state) != record.get("alpha_state_sha256"):
            raise ValueError(f"STALE_CHECKPOINT_MISMATCH: {task} alpha state hash mismatch")
        bank = banks[task]
        if bank.partition_sha256 != record["partition_sha256"] or set(bank.state) != set(state):
            raise ValueError(f"STALE_CHECKPOINT_MISMATCH: task bank metadata mismatch for {task}")
        bank.partition_lengths = tuple(partition.lengths)
        bank.state = {name: value.clone() for name, value in state.items()}
        bank.optimizer_state = {}
        loaded_banks[task] = bank
    if isinstance(model, FSDP):
        files = manifest.get("task_optimizer_files")
        hashes = manifest.get("task_optimizer_sha256")
        rank = dist.get_rank()
        if not isinstance(files, list) or len(files) != dist.get_world_size():
            raise ValueError("STALE_CHECKPOINT_MISMATCH: formal checkpoint lacks per-rank task optimizer states")
        task_optimizer_path = root / files[rank]
        if isinstance(hashes, list) and sha256_file(task_optimizer_path) != hashes[rank]:
            raise ValueError("STALE_CHECKPOINT_MISMATCH: formal task optimizer state hash mismatch")
    else:
        task_optimizer_path = root / manifest["task_optimizer_file"]
    task_optimizer = torch.load(
        task_optimizer_path,
        map_location="cpu",
        weights_only=False,
    )
    if set(task_optimizer) != set(tasks):
        raise ValueError("STALE_CHECKPOINT_MISMATCH: formal checkpoint task optimizer state is incomplete")
    for task in tasks:
        loaded_banks[task].optimizer_state = task_optimizer[task]
    if optimizer is not None:
        optimizer_state = torch.load(
            root / manifest["optimizer_file"],
            map_location="cpu",
            weights_only=False,
        ) if not isinstance(model, FSDP) or dist.get_rank() == 0 else None
        if isinstance(model, FSDP):
            load_full_optimizer_state(model, optimizer, optimizer_state)
        else:
            optimizer.load_state_dict(optimizer_state)
    if scheduler is not None:
        scheduler.load_state_dict(json.loads((root / manifest["scheduler_file"]).read_text(encoding="utf-8")))
    if active_task is not None:
        if active_task not in loaded_banks:
            raise ValueError(f"Unknown active formal task: {active_task}")
        loaded_banks[active_task].activate(model)
        if optimizer is not None:
            loaded_banks[active_task].restore_optimizer_state(
                optimizer,
                model,
                task_routing_parameter_names(model),
            )
    if restore_rng:
        if isinstance(model, FSDP):
            files = manifest.get("rng_files")
            hashes = manifest.get("rng_sha256")
            rank = dist.get_rank()
            if not isinstance(files, list) or len(files) != dist.get_world_size():
                raise ValueError("STALE_CHECKPOINT_MISMATCH: formal checkpoint lacks per-rank RNG states")
            rng_path = root / files[rank]
            if isinstance(hashes, list) and sha256_file(rng_path) != hashes[rank]:
                raise ValueError("STALE_CHECKPOINT_MISMATCH: formal RNG state hash mismatch")
        else:
            rng_path = root / manifest["rng_file"]
        rng_state = torch.load(rng_path, map_location="cpu", weights_only=False)
        torch.set_rng_state(rng_state["torch"])
        if torch.cuda.is_available() and "cuda" in rng_state:
            torch.cuda.set_rng_state_all(rng_state["cuda"])
    return manifest


def save_task_bank(
    output_dir: str | Path,
    *,
    bank: TaskBank,
    base_checkpoint_sha256: str,
    partition_sha256: str,
    data_manifest_sha256: str,
    optimizer_config: dict[str, Any],
    scheduler_config: dict[str, Any],
    consumed_tokens: int,
) -> dict[str, Any]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    query_state = {name: value for name, value in bank.state.items() if "pseudo_query" in name}
    alpha_state = {name: value for name, value in bank.state.items() if "alpha" in name}
    query_path = root / "query.safetensors"
    alpha_path = root / "alpha.safetensors"
    save_file(query_state, query_path)
    save_file(alpha_state, alpha_path)
    manifest = {
        "task": bank.task,
        "base_checkpoint_sha256": base_checkpoint_sha256,
        "partition_sha256": partition_sha256,
        "query_file": query_path.name,
        "query_sha256": sha256_file(query_path),
        "alpha_file": alpha_path.name,
        "alpha_sha256": sha256_file(alpha_path),
        "data_manifest_sha256": data_manifest_sha256,
        "optimizer_config": optimizer_config,
        "scheduler_config": scheduler_config,
        "consumed_tokens": int(consumed_tokens),
        "trainable_parameters": sorted(bank.state),
        "checkpoint_kind": "task_specific_q_alpha",
    }
    manifest["manifest_sha256"] = sha256_json(manifest)
    (root / "training_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
