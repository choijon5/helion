"""Shared configuration collections for matrix multiplication benchmarks."""

from __future__ import annotations

# Standard shapes suite covering tall/skinny, short/wide, square, matvec, and
# outer products. Format: (M, K, N) where A is M x K, B is K x N, C is M x N.
SHAPES: list[tuple[int, int, int]] = [
    (1024, 1, 1024),
    (1024, 1024, 1024),
    (1024, 128, 1024),
    (128, 1024, 1024),
    (1, 1024, 1024),
    (1, 1, 1024),
    (1024, 1024, 1),
]

# Large-shape extension (manager refinement 2026-05-25 — see plan.md §1 G7
# device-us block). The original ``SHAPES`` matrix above carries only one
# compute-bound shape (``1024^3``); the rest are skinny / vector /
# degenerate shapes that are structurally dispatch-bounded (KFLOP-MFLOP
# range theoretical_min_us — they cannot exercise the MXU peak no matter
# how good the kernel is). The shapes below are sized so the chip's MXU
# actually has to sustain peak FLOPS for tens-to-hundreds of us, giving
# G7-prefetch / G7-launch-fusion / G7-Mosaic substeps a clean
# compute-bound signal that's not buried under dispatch noise. They are
# NOT part of the original cota matrix — kept here as an opt-in list so
# the canonical 14-row table stays comparable across cycles.
#
# Per-shape FLOPs (bf16 on TPU v7, peak 1155 TFLOPS/s):
#   - 2048×2048×2048: 17.18 GFLOP → 14.87 us theoretical_min_us
#   - 4096×4096×4096: 137.44 GFLOP → 119.00 us theoretical_min_us
#
# DR#7 Track 3 device baseline on these shapes (cycle-DR#7 ad-hoc probe,
# subsumed by ``measure_headline.py --device-us-calls`` since cycle 36):
#   - 2048³: JAX 22.54 us / Pallas 64.71 us / Helion-kernel 24.25 us
#   - 4096³: not yet measured (manager-added under cycle 36)
LARGE_SHAPES: list[tuple[int, int, int]] = [
    (2048, 2048, 2048),
    (4096, 4096, 4096),
]

# Standard static block dimension configurations.
# Format: (bm, bk, bn) where bm is the block size along M, bk along K, bn along N.
BLOCK_CONFIGS: list[tuple[int, int, int]] = [
    (512, 512, 512),
    (128, 128, 128),
]

# Supported data types for benchmarking.
DTYPES: list[str] = ["bfloat16", "float32"]


def main() -> None:
    """Configuration module; nothing to run directly."""
    print("matmul_configs is a configuration module; import it from a benchmark.")


if __name__ == "__main__":
    main()
