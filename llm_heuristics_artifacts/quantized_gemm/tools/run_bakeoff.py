"""Q0 autotuner bake-off: LFBOTreeSearch vs LLMSeededLFBOTreeSearch.

Runs each autotuner on one shape of ``examples.matmul.matmul`` at
``HELION_AUTOTUNE_EFFORT=full``, captures wall-clock time, number of
benchmarked configs, and best measured perf. Designed to be invoked
per (autotuner, shape) so each run is in an isolated subprocess.

Usage:

    HELION_AUTOTUNER=LFBOTreeSearch \
    python run_bakeoff.py --kernel matmul --M 2048 --K 2048 --N 2048 \
        --out-json /tmp/bakeoff/lfbo_2048.json

    HELION_AUTOTUNER=LLMSeededLFBOTreeSearch \
    HELION_LLM_PROVIDER=bedrock HELION_LLM_MODEL=us.anthropic.claude-opus-4-7 \
    python run_bakeoff.py --kernel matmul --M 2048 --K 2048 --N 2048 \
        --out-json /tmp/bakeoff/llmseeded_2048.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from helion.autotuner import benchmark_provider as _bp_module


def _build_matmul_args(M: int, K: int, N: int, dtype_name: str):
    from examples.matmul import matmul

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype_name]
    x = torch.randn(M, K, device="cuda", dtype=dtype)
    y = torch.randn(K, N, device="cuda", dtype=dtype)
    return matmul, (x, y)


def _install_hook():
    """Return (configs_tried, restore) where configs_tried is a list that
    grows as benchmarks run."""
    configs_tried: list[tuple[float, str]] = []  # (perf_ms, config_hash)
    original = _bp_module.LocalBenchmarkProvider.benchmark

    def wrapped(self, configs, *, desc="Benchmarking"):
        results = original(self, configs, desc=desc)
        for r in results:
            if r.status != "ok" or not math.isfinite(r.perf):
                continue
            configs_tried.append((float(r.perf), ""))
        return results

    _bp_module.LocalBenchmarkProvider.benchmark = wrapped

    def restore():
        _bp_module.LocalBenchmarkProvider.benchmark = original

    return configs_tried, restore


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", default="matmul", choices=["matmul"])
    ap.add_argument("--M", type=int, required=True)
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    ap.add_argument("--out-json", required=True, type=Path)
    args = ap.parse_args()

    # The autotuner selection comes from HELION_AUTOTUNER. effort=full from env.
    autotuner_name = os.environ.get("HELION_AUTOTUNER", "LFBOTreeSearch")
    assert os.environ.get("HELION_AUTOTUNE_EFFORT") == "full", (
        "Set HELION_AUTOTUNE_EFFORT=full before invoking"
    )

    kernel_fn, kern_args = _build_matmul_args(args.M, args.K, args.N, args.dtype)
    bound = kernel_fn.bind(kern_args)
    bound.settings.autotune_effort = "full"

    configs_tried, restore = _install_hook()
    t0 = time.perf_counter()
    err = None
    try:
        winner = bound.autotune(kern_args, force=True)
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"
        winner = None
    finally:
        restore()
    elapsed = time.perf_counter() - t0

    # Stats on what got benchmarked
    perfs = [p for p, _ in configs_tried]
    stats = {
        "autotuner": autotuner_name,
        "kernel": args.kernel,
        "M": args.M, "K": args.K, "N": args.N,
        "dtype": args.dtype,
        "autotune_effort": os.environ.get("HELION_AUTOTUNE_EFFORT"),
        "llm_provider": os.environ.get("HELION_LLM_PROVIDER"),
        "llm_model": os.environ.get("HELION_LLM_MODEL"),
        "wall_elapsed_s": elapsed,
        "configs_benchmarked_ok": len(perfs),
        "best_perf_ms": min(perfs) if perfs else None,
        "median_perf_ms": (
            sorted(perfs)[len(perfs) // 2] if perfs else None
        ),
        "winner_config": dict(winner) if winner is not None else None,
        "error": err,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(stats, indent=2))
    print(json.dumps({k: v for k, v in stats.items() if k != "winner_config"}, indent=2))
    return 0 if err is None else 1


if __name__ == "__main__":
    sys.exit(main())
