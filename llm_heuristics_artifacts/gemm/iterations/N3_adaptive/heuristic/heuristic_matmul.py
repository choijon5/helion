"""Hand-coded round-0 heuristic for matmul that adapts tile sizes to M/K/N.

Built from the archived B200 tuned_configs oracle. The archive only contains
square shapes, so we observe that good configs satisfy (approximately):

- block_m grows with D up to 128 (256 for D >= 2560)
- block_n grows with D up to 256 (or 512 for very large D)
- block_k ∈ {32, 64, 128}; 32 dominates for large D with big block_n
- num_warps = 4 is the dominant safe choice; 8 for big block_n
- pid_type = "flat"; l2_groupings picks are noisy so we use 1-8

For skewed shapes we clamp block_m <= M, block_n <= N, block_k <= K.
Features used: arg0.shape (M, K), arg1.shape (K, N), dtypes. No kernel
name strings and no config/run hashes. See plan.md feature-audit gate.
"""

from __future__ import annotations

import torch


def _clamp_pot(value: int, hi: int, lo: int = 16) -> int:
    """Clamp ``value`` to the largest power of two <= min(value, hi), at least lo."""
    v = min(int(value), int(hi))
    if v < lo:
        return lo
    # largest POT <= v
    out = 1
    while (out << 1) <= v:
        out <<= 1
    return max(out, lo)


def _matmul_block_sizes(M: int, K: int, N: int) -> tuple[int, int, int]:
    """Pick (block_m, block_n, block_k) from shape.

    Heuristic rules:
    - Small D (<= 512): block_m = 32, block_n = 64, block_k = 128 (tl.dot minima)
    - Medium D (<= 1408): block_m = 64, block_n = 128, block_k = 64
    - Large D (<= 2560): block_m = 128, block_n = 256, block_k = 32
    - Huge D (> 2560): block_m = 256, block_n = 256, block_k = 32
    Then clamp each block size to its axis dim (but stay ≥ tl.dot minima).
    """
    D = max(M, N)
    if D <= 512:
        bm, bn, bk = 32, 64, 128
    elif D <= 1408:
        bm, bn, bk = 64, 128, 64
    elif D <= 2560:
        bm, bn, bk = 128, 256, 32
    else:
        bm, bn, bk = 256, 256, 32
    bm = _clamp_pot(bm, M, lo=16)
    bn = _clamp_pot(bn, N, lo=16)
    bk = _clamp_pot(bk, K, lo=32)
    return bm, bn, bk


def _matmul_runtime(M: int, K: int, N: int) -> tuple[int, int, int, str]:
    """Pick (num_warps, num_stages, l2_groupings, pid_type)."""
    # Prefer more warps when block_n large.
    total_out = M * N
    if total_out <= 512 * 512:
        num_warps = 4
        num_stages = 5
    elif total_out <= 1536 * 1536:
        num_warps = 4
        num_stages = 6
    else:
        num_warps = 8
        num_stages = 4
    # l2_groupings helps L2 reuse when D >= 1024, otherwise small.
    D = max(M, N)
    if D <= 512:
        l2g = 1
    elif D <= 1024:
        l2g = 4
    elif D <= 2048:
        l2g = 8
    else:
        l2g = 16
    return num_warps, num_stages, l2g, "flat"


def autotune_matmul(*args) -> dict:
    """Return a Config dict for the given matmul args (x, y, [epilogue]).

    x: [M, K], y: [K, N]. No kernel name strings are used.
    """
    x = args[0]
    y = args[1]
    M = int(x.shape[0])
    K = int(x.shape[1])
    N = int(y.shape[1])

    bm, bn, bk = _matmul_block_sizes(M, K, N)
    nw, ns, l2g, pid = _matmul_runtime(M, K, N)

    return {
        "block_sizes": [bm, bn, bk],
        "loop_orders": [[0, 1]],
        "l2_groupings": [l2g],
        "range_unroll_factors": [0, 0],
        "range_warp_specializes": [None, None],
        "range_num_stages": [0, 0],
        "range_multi_buffers": [None, None],
        "range_flattens": [None, None],
        "load_eviction_policies": ["", ""],
        "num_warps": nw,
        "num_stages": ns,
        "indexing": ["pointer", "pointer", "pointer"],
        "atomic_indexing": [],
        "pid_type": pid,
    }


def key_matmul(*args) -> int:
    """Return a cache-key integer; the returned int indexes into a virtual
    config set built on the fly, so we just hash the block_sizes triple.
    """
    x = args[0]; y = args[1]
    bm, bn, bk = _matmul_block_sizes(int(x.shape[0]), int(x.shape[1]), int(y.shape[1]))
    # Pack into a stable int; not used for dispatch in this backend.
    return (bm.bit_length() << 16) | (bn.bit_length() << 8) | bk.bit_length()
