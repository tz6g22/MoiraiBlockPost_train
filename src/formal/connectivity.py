from __future__ import annotations

import argparse

import torch
from transformers import Qwen3ForCausalLM
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from src.formal.conversion import formal_config_from_qwen
from src.formal.runtime import build_joint_optimizer, identity_test, parameter_hash, trainability_audit
from src.modeling.full_attnres import MoiraiQwen3ForCausalLM


def run_tiny_connectivity() -> dict[str, object]:
    config = Qwen3Config(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        layer_types=["full_attention"] * 4,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        use_cache=False,
    )
    native = Qwen3ForCausalLM(config)
    converted = MoiraiQwen3ForCausalLM(formal_config_from_qwen(config))
    incompatible = converted.load_state_dict(native.state_dict(), strict=False)
    unexpected_missing = [
        name
        for name in incompatible.missing_keys
        if "pseudo_query" not in name and "alpha" not in name and "key_norm" not in name
    ]
    if unexpected_missing or incompatible.unexpected_keys:
        raise RuntimeError(f"Conversion failed: {incompatible}")
    converted.config.attnres_execution = "formal"
    converted.config.moirai_partition = [2, 2]
    converted.config.moirai_task = "connectivity"
    converted.eval()
    native.eval()
    input_ids = torch.randint(0, config.vocab_size, (1, 8))
    attention_mask = torch.ones_like(input_ids)
    identity = identity_test(native, converted, input_ids, attention_mask)
    if identity["status"] != "PASS":
        raise RuntimeError("IDENTITY_CONVERSION_FAILED")
    before = parameter_hash(converted)
    audit = trainability_audit(converted)
    query_before = parameter_hash(converted, audit["query"])
    alpha_before = parameter_hash(converted, audit["alpha"])
    optimizer = build_joint_optimizer(
        converted,
        backbone_lr=3.0e-6,
        attnres_lr=3.0e-5,
        backbone_weight_decay=0.1,
    )
    converted.train()
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        logits = converted(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            logits_to_keep=0,
        ).logits
        logits.float().mean().backward()
        torch.nn.utils.clip_grad_norm_(converted.parameters(), 1.0)
        optimizer.step()
    after = parameter_hash(converted)
    query_after = parameter_hash(converted, audit["query"])
    alpha_after = parameter_hash(converted, audit["alpha"])
    if before == after:
        raise RuntimeError("Formal optimizer step did not update parameters")
    if query_before == query_after:
        raise RuntimeError("Formal query bank did not update after alpha opened")
    if alpha_before == alpha_after:
        raise RuntimeError("Formal alpha bank did not update")
    return {
        "identity": identity,
        "trainable_parameter_counts": {key: len(value) for key, value in audit.items() if isinstance(value, tuple)},
        "backbone_updated": True,
        "query_updated": True,
        "alpha_updated": True,
        "query_present": bool(audit["query"]),
        "alpha_present": bool(audit["alpha"]),
        "partition_trainable": audit["partition_trainable"],
        "status": "PASS",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiny", action="store_true")
    args = parser.parse_args()
    if not args.tiny:
        raise SystemExit("Use --tiny for the local connectivity test; no formal 14B training is started.")
    print(run_tiny_connectivity())


if __name__ == "__main__":
    main()
