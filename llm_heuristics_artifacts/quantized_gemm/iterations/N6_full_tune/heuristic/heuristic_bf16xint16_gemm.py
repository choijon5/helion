"""Quantized-GEMM heuristic dispatcher for ``_bf16xint16_gemm``.

x[M,K] bf16 * w[K,N] int16 -> out[M,N] bf16.

Unlike int4/fp4, the weight operand is *not* packed, so the B tile
footprint is 2 bytes/element (int16). The kernel casts int16 to bf16
before dot, so in shmem the operand is int16 (still 2 bytes) —
budget math is the same as a bf16-bf16 matmul.
"""

from __future__ import annotations

import torch

import importlib.util
import sys
from pathlib import Path as _Path

_CORE_PATH = _Path(__file__).resolve().parent / "_dispatcher_core.py"
if "_quantized_dispatcher_core" in sys.modules:
    core = sys.modules["_quantized_dispatcher_core"]
else:
    _spec = importlib.util.spec_from_file_location(
        "_quantized_dispatcher_core", _CORE_PATH
    )
    core = importlib.util.module_from_spec(_spec)
    sys.modules["_quantized_dispatcher_core"] = core
    _spec.loader.exec_module(core)


_KERNEL_CLASS = "matmul_int16"

_FALLBACK_TABLE = {
    'small_m': {
        'atomic_indexing': [], 'block_sizes': [64, 32, 256],
        'indexing': ['pointer', 'pointer', 'pointer'],
        'l2_groupings': [1], 'load_eviction_policies': ['', ''],
        'loop_orders': [[0, 1]], 'num_stages': 3, 'num_warps': 8,
        'pid_type': 'flat', 'range_flattens': [None, False],
        'range_multi_buffers': [None, None], 'range_num_stages': [0, 0],
        'range_unroll_factors': [0, 0], 'range_warp_specializes': [None, None],
    },
    'small_n': {
        'atomic_indexing': [], 'block_sizes': [128, 32, 128],
        'indexing': ['pointer', 'pointer', 'tensor_descriptor'],
        'l2_groupings': [2], 'load_eviction_policies': ['last', 'first'],
        'loop_orders': [[0, 1]], 'num_stages': 3, 'num_warps': 8,
        'pid_type': 'flat', 'range_flattens': [None, None],
        'range_multi_buffers': [None, None], 'range_num_stages': [0, 0],
        'range_unroll_factors': [0, 0], 'range_warp_specializes': [None, None],
    },
    'small_k': {
        'atomic_indexing': [], 'block_sizes': [128, 128, 64],
        'indexing': ['pointer', 'pointer', 'tensor_descriptor'],
        'l2_groupings': [4], 'load_eviction_policies': ['', ''],
        'loop_orders': [[0, 1]], 'num_stages': 3, 'num_warps': 8,
        'pid_type': 'flat', 'range_flattens': [None, None],
        'range_multi_buffers': [None, None], 'range_num_stages': [0, 0],
        'range_unroll_factors': [0, 0], 'range_warp_specializes': [None, None],
    },
    'balanced': {
        'atomic_indexing': [], 'block_sizes': [128, 64, 64],
        'indexing': ['pointer', 'pointer', 'tensor_descriptor'],
        'l2_groupings': [4], 'load_eviction_policies': ['', ''],
        'loop_orders': [[0, 1]], 'num_stages': 3, 'num_warps': 8,
        'pid_type': 'flat', 'range_flattens': [None, None],
        'range_multi_buffers': [None, None], 'range_num_stages': [0, 0],
        'range_unroll_factors': [0, 0], 'range_warp_specializes': [None, None],
    },
    'rect': {
        'atomic_indexing': [], 'block_sizes': [256, 256, 64],
        'indexing': ['pointer', 'pointer', 'tensor_descriptor'],
        'l2_groupings': [8], 'load_eviction_policies': ['last', 'last'],
        'loop_orders': [[0, 1]], 'num_stages': 3, 'num_warps': 8,
        'pid_type': 'flat', 'range_flattens': [None, None],
        'range_multi_buffers': [None, False], 'range_num_stages': [0, 0],
        'range_unroll_factors': [0, 0], 'range_warp_specializes': [None, None],
    },
}


def _shape_from_args(args):
    x = args[0]
    w = args[1]
    M = int(x.shape[0])
    K = int(x.shape[1])
    # w is [K, N] int16, unpacked.
    N = int(w.shape[1])
    return M, K, N


def autotune_bf16xint16_gemm(*args) -> dict:
    M, K, N = _shape_from_args(args)
    a_dtype_size = int(args[0].element_size())
    b_dtype_size = int(args[1].element_size())
    return core.dispatch(
        _KERNEL_CLASS, M, K, N, a_dtype_size, b_dtype_size,
        _FALLBACK_TABLE,
    )


def key_bf16xint16_gemm(*args) -> int:
    M, K, N = _shape_from_args(args)
    return core.dispatch_key(_KERNEL_CLASS, M, K, N)
