from __future__ import annotations

import platform
import statistics
import sys
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
import transformers


def runtime_environment(commit_hash: str) -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "gpu_model": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "gpu_count": torch.cuda.device_count(),
        "cuda": torch.version.cuda,
        "pytorch": torch.__version__,
        "transformers": transformers.__version__,
        "commit_hash": commit_hash,
        "precision": "bfloat16",
        "attention_backend": "sdpa",
    }


def benchmark_cuda(
    operation: Callable[[], Any],
    *,
    warmup_runs: int,
    timed_runs: int,
    processed_nonpadding_tokens: int,
) -> dict[str, float | int]:
    if not torch.cuda.is_available():
        raise RuntimeError("Formal efficiency measurement requires CUDA")
    if warmup_runs < 0 or timed_runs <= 0:
        raise ValueError("Invalid benchmark run counts")
    if processed_nonpadding_tokens <= 0:
        raise ValueError("processed_nonpadding_tokens must be positive")
    for _ in range(warmup_runs):
        operation()
    torch.cuda.synchronize()
    timings: list[float] = []
    for _ in range(timed_runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        operation()
        end.record()
        torch.cuda.synchronize()
        timings.append(float(start.elapsed_time(end)))
    mean_ms = statistics.fmean(timings)
    return {
        "median_ms": statistics.median(timings),
        "mean_ms": mean_ms,
        "p95_ms": float(np.quantile(np.asarray(timings), 0.95)),
        "processed_nonpadding_tokens": processed_nonpadding_tokens,
        "tokens_per_second": processed_nonpadding_tokens / (mean_ms / 1000.0),
    }


def measure_peak_memory(operation: Callable[[], Any]) -> dict[str, int]:
    if not torch.cuda.is_available():
        raise RuntimeError("Formal memory measurement requires CUDA")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    operation()
    torch.cuda.synchronize()
    return {
        "allocated_peak_bytes": int(torch.cuda.max_memory_allocated()),
        "reserved_peak_bytes": int(torch.cuda.max_memory_reserved()),
    }


def source_storage_counts(
    *,
    num_transformer_blocks: int,
    partition_lengths: tuple[int, ...] | None,
) -> dict[str, int]:
    if partition_lengths is None:
        return {
            "embedding_sources": 1,
            "attention_sources": num_transformer_blocks,
            "mlp_sources": num_transformer_blocks,
            "current_partial_sources": 0,
            "final_source_tensors": 1 + 2 * num_transformer_blocks,
            "maximum_live_source_tensors": 1 + 2 * num_transformer_blocks,
        }
    if sum(partition_lengths) != num_transformer_blocks:
        raise ValueError("Partition does not cover the Transformer depth")
    completed = len(partition_lengths)
    return {
        "embedding_sources": 1,
        "completed_block_representations": completed,
        "maximum_completed_while_partial": max(0, completed - 1),
        "current_partial_sources": 1,
        "final_source_tensors": 1 + completed,
        "maximum_live_source_tensors": 1 + completed,
    }

