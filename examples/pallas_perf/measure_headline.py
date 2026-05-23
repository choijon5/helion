"""Single-shape headline measurement for the bf16 1024x1024x1024 matmul.

Runs the Helion matmul kernel from ``matmul_helion.py`` on the bf16 1024^3
headline shape exactly once, mirroring the harness timing convention
(20 iters x 5 repeats, warmup excluded, ``synchronize_device`` between
calls).  Prints a parseable line ``helion_bf16_1024x1024x1024: median=<us>
us`` for the per-cycle hill-climb log.
"""

from __future__ import annotations

import os
import sys
import timeit

# Force full autotuning effort to mirror the production harness; set before
# importing helion so the value is picked up at autotuner initialization.
os.environ.setdefault("HELION_AUTOTUNE_EFFORT", "full")

import numpy as np
import torch

from helion._testing import DEVICE
from helion.autotuner.benchmarking import synchronize_device

# Import the kernel from matmul_helion so any kernel-side change is picked
# up by both the full harness and this single-shape probe.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from matmul_helion import helion_matmul_kernel  # pyrefly: ignore [missing-import]


def main() -> None:
    m, k, n = 1024, 1024, 1024
    torch.manual_seed(0)
    x = torch.randn((m, k), dtype=torch.bfloat16, device=DEVICE)
    torch.manual_seed(1)
    y = torch.randn((k, n), dtype=torch.bfloat16, device=DEVICE)

    bound = helion_matmul_kernel.bind((x, y))
    best_config = bound.autotune((x, y), force=True)
    print(f"Optimal autotuned config: {best_config}", file=sys.stderr)
    compiled_fn = bound.compile_config(best_config, allow_print=False)

    def _run() -> None:
        out = compiled_fn(x, y)
        synchronize_device(out)

    _run()  # warmup excluded from timing
    samples = np.array(timeit.repeat(_run, repeat=5, number=20)) / 20
    median_us = float(np.median(samples)) * 1e6
    print(f"helion_bf16_{m}x{k}x{n}: median={median_us:.2f} us")


if __name__ == "__main__":
    main()
