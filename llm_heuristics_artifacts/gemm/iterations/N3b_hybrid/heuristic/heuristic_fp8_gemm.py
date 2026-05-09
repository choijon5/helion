"""Hybrid round-0 heuristic for fp8_gemm.

Same structure as N3b matmul: archive decision-tree for in-range square
shapes, adaptive fallback for skewed or out-of-range.
"""

from __future__ import annotations

import torch


_ARCHIVE_CONFIGS = [
    {'block_sizes': [256, 256, 64], 'loop_orders': [[1, 0]], 'l2_groupings': [4], 'range_unroll_factors': [0, 2], 'range_warp_specializes': [None, None], 'range_num_stages': [0, 1], 'range_multi_buffers': [None, None], 'range_flattens': [None, True], 'load_eviction_policies': ['first', 'first'], 'num_warps': 8, 'num_stages': 6, 'indexing': ['tensor_descriptor', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [128, 32, 128], 'loop_orders': [[1, 0]], 'l2_groupings': [1], 'range_unroll_factors': [0, 1], 'range_warp_specializes': [None, False], 'range_num_stages': [0, 4], 'range_multi_buffers': [None, False], 'range_flattens': [None, None], 'load_eviction_policies': ['last', 'last'], 'num_warps': 4, 'num_stages': 5, 'indexing': ['pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [128, 128, 128], 'loop_orders': [[0, 1]], 'l2_groupings': [8], 'range_unroll_factors': [0, 2], 'range_warp_specializes': [None, False], 'range_num_stages': [0, 0], 'range_multi_buffers': [None, False], 'range_flattens': [None, None], 'load_eviction_policies': ['last', ''], 'num_warps': 4, 'num_stages': 5, 'indexing': ['pointer', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat', 'epilogue_subtile': 2},
    {'block_sizes': [128, 256, 128], 'loop_orders': [[0, 1]], 'l2_groupings': [32], 'range_unroll_factors': [0, 2], 'range_warp_specializes': [None, False], 'range_num_stages': [0, 3], 'range_multi_buffers': [None, None], 'range_flattens': [None, False], 'load_eviction_policies': ['last', 'last'], 'num_warps': 8, 'num_stages': 4, 'indexing': ['pointer', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat', 'epilogue_subtile': 2},
    {'block_sizes': [16, 32, 256], 'loop_orders': [[0, 1]], 'l2_groupings': [16], 'range_unroll_factors': [0, 1], 'range_warp_specializes': [None, None], 'range_num_stages': [0, 3], 'range_multi_buffers': [None, None], 'range_flattens': [None, True], 'load_eviction_policies': ['first', ''], 'num_warps': 4, 'num_stages': 1, 'indexing': ['pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
]


def _archive_dispatch_index(M: int) -> int:
    """Decision tree copied from N2 fp8 heuristic (keyed on M)."""
    if M <= 1536:
        if M <= 256:
            return 4
        if M <= 1024:
            return 1
        return 2
    if M <= 2048:
        return 3
    return 0


def _clamp_pot(value: int, hi: int, lo: int = 16) -> int:
    v = min(int(value), int(hi))
    if v < lo:
        return lo
    out = 1
    while (out << 1) <= v:
        out <<= 1
    return max(out, lo)


def _adaptive_config(M: int, K: int, N: int) -> dict:
    D = max(M, N)
    total_out = M * N
    # Gate block sizes on the dominant dim; gate num_warps/stages on
    # total output size so M-skinny / N-skinny shapes don't get too many
    # warps for their workload.
    if D <= 512:
        bm, bn, bk = 32, 64, 128
    elif D <= 1408:
        bm, bn, bk = 64, 128, 128
    else:
        bm, bn, bk = 128, 256, 64
    if total_out <= 512 * 512:
        nw, ns = 4, 5
    elif total_out <= 1536 * 1536:
        nw, ns = 4, 5
    else:
        nw, ns = 8, 3
    bm = _clamp_pot(bm, M, lo=16)
    bn = _clamp_pot(bn, N, lo=16)
    bk = _clamp_pot(bk, K, lo=32)
    if D <= 512:
        l2g = 1
    elif D <= 1024:
        l2g = 4
    elif D <= 2048:
        l2g = 8
    else:
        l2g = 16
    return {
        'block_sizes': [bm, bn, bk],
        'loop_orders': [[0, 1]],
        'l2_groupings': [l2g],
        'range_unroll_factors': [0, 0],
        'range_warp_specializes': [None, None],
        'range_num_stages': [0, 0],
        'range_multi_buffers': [None, None],
        'range_flattens': [None, None],
        'load_eviction_policies': ['', ''],
        'num_warps': nw,
        'num_stages': ns,
        'indexing': ['pointer', 'pointer', 'pointer'],
        'atomic_indexing': [],
        'pid_type': 'flat',
    }


def _is_in_archive_range(M: int, K: int, N: int) -> bool:
    aspect = max(M, N, K) / max(1, min(M, N, K))
    if aspect > 1.25:
        return False
    if not (256 <= M <= 4096 and 256 <= K <= 4096 and 256 <= N <= 4096):
        return False
    return True


def autotune_fp8_gemm(*args) -> dict:
    x = args[0]; y = args[1]
    M, K = int(x.shape[0]), int(x.shape[1])
    N = int(y.shape[1])
    if _is_in_archive_range(M, K, N):
        idx = _archive_dispatch_index(M)
        cfg = dict(_ARCHIVE_CONFIGS[idx])
        bm, bn, bk = cfg['block_sizes']
        cfg = dict(cfg)
        cfg['block_sizes'] = [
            _clamp_pot(bm, M, lo=16),
            _clamp_pot(bn, N, lo=16),
            _clamp_pot(bk, K, lo=32),
        ]
        return cfg
    return _adaptive_config(M, K, N)


def key_fp8_gemm(*args) -> int:
    x = args[0]; y = args[1]
    M = int(x.shape[0]); K = int(x.shape[1]); N = int(y.shape[1])
    if _is_in_archive_range(M, K, N):
        return _archive_dispatch_index(M)
    cfg = _adaptive_config(M, K, N)
    bs = cfg['block_sizes']
    return 1000 + (bs[0].bit_length() << 16) | (bs[1].bit_length() << 8) | bs[2].bit_length()
