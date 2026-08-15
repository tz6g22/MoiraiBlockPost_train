from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
import transformers
from huggingface_hub import scan_cache_dir, snapshot_download
from transformers import AutoTokenizer, Qwen3ForCausalLM

from src.common import config_sha256, load_yaml, sha256_file, tokenizer_sha256
from src.modeling.full_attnres import MoiraiQwen3Config, MoiraiQwen3ForCausalLM
from src.training.checkpointing import pseudo_query_sha256


def _custom_config(base_config) -> MoiraiQwen3Config:
    payload: dict[str, Any] = base_config.to_dict()
    for key in (
        "architectures",
        "model_type",
        "transformers_version",
        "_name_or_path",
    ):
        payload.pop(key, None)
    payload.update(
        {
            "attnres_execution": "full",
            "moirai_partition": None,
            "moirai_task": "full_reference",
            "use_cache": False,
        }
    )
    return MoiraiQwen3Config(**payload)


def _usable_cached_snapshot(repo_id: str, revision: str) -> Path | None:
    required = {
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
    }
    for repository in scan_cache_dir().repos:
        if repository.repo_type != "model" or repository.repo_id != repo_id:
            continue
        candidates = (
            [repository.refs[revision]]
            if revision in repository.refs
            else list(repository.revisions)
        )
        for cached_revision in candidates:
            present = {item.file_name for item in cached_revision.files}
            if required.issubset(present):
                return Path(cached_revision.snapshot_path)
    return None


def convert_hf_model(
    source: str | Path,
    *,
    dtype: torch.dtype,
) -> tuple[MoiraiQwen3ForCausalLM, Any, list[str]]:
    tokenizer = AutoTokenizer.from_pretrained(
        source,
        local_files_only=True,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = Qwen3ForCausalLM.from_pretrained(
        source,
        local_files_only=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model = MoiraiQwen3ForCausalLM(_custom_config(base.config)).to(dtype=dtype)
    incompatible = model.load_state_dict(base.state_dict(), strict=False)
    del base

    allowed_missing_fragments = (
        "pseudo_query",
        "attn_key_norm",
        "mlp_key_norm",
        "final_key_norm",
    )
    disallowed_missing = [
        name
        for name in incompatible.missing_keys
        if not any(fragment in name for fragment in allowed_missing_fragments)
    ]
    if disallowed_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "HF Qwen3 weights do not match the Moirai backbone: "
            f"missing={disallowed_missing}, unexpected={incompatible.unexpected_keys}"
        )
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "pseudo_query" in name:
                parameter.zero_()
    model.config._attn_implementation = "sdpa"
    model.config.use_cache = False
    model.tie_weights()
    return model, tokenizer, sorted(incompatible.missing_keys)


def prepare(config_path: str | Path, *, force: bool = False) -> dict[str, Any]:
    config = load_yaml(config_path)
    repo_id = str(config["repo_id"])
    revision = str(config["revision"])
    local_dir = Path(config["local_dir"])
    output_dir = Path(config["converted_checkpoint"])
    manifest_path = output_dir / "checkpoint_manifest.json"
    if manifest_path.is_file() and not force:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_config_hash = config_sha256(config_path)
        if (
            existing.get("source_repo_id") != repo_id
            or existing.get("source_revision") != revision
            or existing.get("run_config_sha256") != expected_config_hash
        ):
            raise ValueError(
                "Existing HF bootstrap checkpoint does not match the current "
                "model_source config; rerun with --force"
            )
        return existing
    if output_dir.exists() and any(output_dir.iterdir()):
        if not force:
            raise RuntimeError(f"Output directory is not empty: {output_dir}")
        shutil.rmtree(output_dir)

    try:
        resolved_snapshot = Path(
            snapshot_download(
                repo_id=repo_id,
                revision=revision,
                local_dir=local_dir,
            )
        )
    except Exception as online_error:
        resolved_snapshot = _usable_cached_snapshot(repo_id, revision)
        if resolved_snapshot is None:
            raise RuntimeError(
                f"Unable to download {repo_id}@{revision} and no complete local "
                "Hugging Face model/tokenizer snapshot is available"
            ) from online_error
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[str(config["torch_dtype"])]
    model, tokenizer, initialized_parameters = convert_hf_model(
        resolved_snapshot,
        dtype=dtype,
    )
    if int(model.config.num_hidden_layers) != 28 or int(model.config.hidden_size) != 1024:
        raise ValueError(
            "Qwen/Qwen3-0.6B is expected to have 28 layers and hidden size 1024; "
            f"got {model.config.num_hidden_layers} and {model.config.hidden_size}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(
        output_dir,
        safe_serialization=True,
        max_shard_size="5GB",
    )
    tokenizer.save_pretrained(output_dir)
    weights = sorted(output_dir.glob("model*.safetensors"))
    if len(weights) != 1:
        raise RuntimeError(
            f"Expected one Qwen3-0.6B weight file after conversion, got {weights}"
        )
    query_names, query_hash = pseudo_query_sha256(model)
    snapshot_commit = resolved_snapshot.resolve().name
    manifest = {
        "architecture": "Full AttnRes",
        "checkpoint_origin": "huggingface_post_training_bootstrap",
        "stage1_skipped": True,
        "source_repo_id": repo_id,
        "source_revision": revision,
        "source_snapshot": str(resolved_snapshot),
        "source_snapshot_commit": snapshot_commit,
        "conversion": "copy_qwen3_backbone_and_zero_initialize_attnres_parameters",
        "initialized_parameters": initialized_parameters,
        "model_weights_sha256": sha256_file(weights[0]),
        "model_config_sha256": sha256_file(output_dir / "config.json"),
        "tokenizer_sha256": tokenizer_sha256(tokenizer),
        "q_full_parameter_names": query_names,
        "q_full_sha256": query_hash,
        "num_hidden_layers": int(model.config.num_hidden_layers),
        "hidden_size": int(model.config.hidden_size),
        "run_config_sha256": config_sha256(config_path),
        "seed": 42,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "environment.json").write_text(
        json.dumps(
            {
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/model_source.yaml")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = prepare(args.config, force=args.force)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
