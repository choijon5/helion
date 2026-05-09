"""Hand-coded round-0 heuristic for fp8_gemm that adapts tile sizes to M/K/N.

Same structure as heuristic_matmul; differences for fp8:
- fp8_gemm uses hl.dot and accumulates in fp32; its sweet-spot blocks
  are similar but block_k tends larger because fp8 halves memory
  pressure vs fp16.
- ``static_shapes=True`` on the kernel means per-shape compile, so we
  pick fewer unique configs across the shape range and lean on
  Helion's defaults for unused fields.
- Features used are kernel-arg-only (no kernel name string leakage).
"""

from __future__ import annotations

import torch


def _clamp_pot(value: int, hi: int, lo: int = 16) -> int:
    v = min(int(value), int(hi))
    if v < lo:
        return lo
    out = 1
    while (out << 1) <= v:
        out <<= 1
    return max(out, lo)


def _fp8_block_sizes(M: int, K: int, N: int) -> tuple[int, int, int]:
    D = max(M, N)
    if D <= 512:
        bm, bn, bk = 32, 64, 128
    elif D <= 1408:
        bm, bn, bk = 64, 128, 128
    elif D <= 2560:
        bm, bn, bk = 128, 256, 64
    else:
        bm, bn, bk = 128, 256, 64
    bm = _clamp_pot(bm, M, lo=16)
    bn = _clamp_pot(bn, N, lo=16)
    bk = _clamp_pot(bk, K, lo=32)
    return bm, bn, bk


def _fp8_runtime(M: int, K: int, N: int) -> tuple[int, int, int, str]:
    total_out = M * N
    if total_out <= 512 * 512:
        num_warps = 4
        num_stages = 5
    elif total_out <= 1536 * 1536:
        num_warps = 4
        num_stages = 5
    else:
        num_warps = 8
        num_stages = 3
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


def autotune_fp8_gemm(*args) -> dict:
    x = args[0]
    y = args[1]
    M = int(x.shape[0])
    K = int(x.shape[1])
    N = int(y.shape[1])

    bm, bn, bk = _fp8_block_sizes(M, K, N)
    nw, ns, l2g, pid = _fp8_runtime(M, K, N)

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


def key_fp8_gemm(*args) -> int:
    x = args[0]; y = args[1]
    bm, bn, bk = _fp8_block_sizes(int(x.shape[0]), int(x.shape[1]), int(y.shape[1]))
    return (bm.bit_length() << 16) | (bn.bit_length() << 8) | bk.bit_length()
