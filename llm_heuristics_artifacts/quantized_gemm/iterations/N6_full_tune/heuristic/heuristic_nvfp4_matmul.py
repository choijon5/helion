"""Quantized-GEMM heuristic dispatcher for ``nvfp4_matmul``.

A[M,K] bf16 * B_packed[K//2, N] int8-packed-fp4_e2m1 -> out[M,N] bf16.

The B tile is stored as nibble-packed int8 but the compiler
materializes fp32 after dequant; we charge bf16 (2 bytes) for the
B tile in shmem to match measured resource accounting.
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


_KERNEL_CLASS = "matmul_fp4"

_FALLBACK_TABLE = {
    'small_m': {
        'atomic_indexing': [], 'block_sizes': [64, 64, 16],
        'indexing': ['pointer', 'tensor_descriptor', 'tensor_descriptor'],
        'l2_groupings': [1], 'load_eviction_policies': ['', 'first'],
        'loop_orders': [[0, 1]], 'num_stages': 3, 'num_warps': 2,
        'pid_type': 'flat', 'range_flattens': [None, None],
        'range_multi_buffers': [None, None], 'range_num_stages': [0, 0],
        'range_unroll_factors': [0, 0], 'range_warp_specializes': [None, None],
    },
    'small_n': {
        'atomic_indexing': [], 'block_sizes': [64, 128, 16],
        'indexing': ['pointer', 'pointer', 'pointer'],
        'l2_groupings': [4], 'load_eviction_policies': ['last', 'last'],
        'loop_orders': [[0, 1]], 'num_stages': 3, 'num_warps': 8,
        'pid_type': 'flat', 'range_flattens': [None, None],
        'range_multi_buffers': [None, None], 'range_num_stages': [0, 1],
        'range_unroll_factors': [0, 0], 'range_warp_specializes': [None, None],
    },
    'small_k': {
        'atomic_indexing': [], 'block_sizes': [8, 128, 128],
        'indexing': ['tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor'],
        'l2_groupings': [1], 'load_eviction_policies': ['', 'first'],
        'loop_orders': [[0, 1]], 'num_stages': 3, 'num_warps': 4,
        'pid_type': 'flat', 'range_flattens': [None, None],
        'range_multi_buffers': [None, None], 'range_num_stages': [0, 0],
        'range_unroll_factors': [0, 1], 'range_warp_specializes': [None, None],
    },
    'balanced': {
        'atomic_indexing': [], 'block_sizes': [64, 128, 32],
        'indexing': ['pointer', 'pointer', 'tensor_descriptor'],
        'l2_groupings': [1], 'load_eviction_policies': ['last', ''],
        'loop_orders': [[0, 1]], 'num_stages': 1, 'num_warps': 8,
        'pid_type': 'flat', 'range_flattens': [None, None],
        'range_multi_buffers': [None, True], 'range_num_stages': [0, 0],
        'range_unroll_factors': [0, 0], 'range_warp_specializes': [None, False],
    },
    'rect': {
        'atomic_indexing': [], 'block_sizes': [8, 256, 128],
        'indexing': ['tensor_descriptor', 'pointer', 'tensor_descriptor'],
        'l2_groupings': [16], 'load_eviction_policies': ['', ''],
        'loop_orders': [[0, 1]], 'num_stages': 6, 'num_warps': 4,
        'pid_type': 'flat', 'range_flattens': [None, False],
        'range_multi_buffers': [None, True], 'range_num_stages': [0, 4],
        'range_unroll_factors': [0, 3], 'range_warp_specializes': [None, False],
    },
}


def _shape_from_args(args):
    A = args[0]
    B = args[1]
    M = int(A.shape[0])
    K = int(A.shape[1])
    # B is [K//2, N] packed fp4.
    N = int(B.shape[1])
    return M, K, N


def autotune_nvfp4_matmul(*args) -> dict:
    M, K, N = _shape_from_args(args)
    a_dtype_size = int(args[0].element_size())  # 2 for bf16
    # Charge bf16 for the B tile (same reasoning as int4 kernel).
    b_dtype_size = 2
    return core.dispatch(
        _KERNEL_CLASS, M, K, N, a_dtype_size, b_dtype_size,
        _FALLBACK_TABLE,
    )


def key_nvfp4_matmul(*args) -> int:
    M, K, N = _shape_from_args(args)
    return core.dispatch_key(_KERNEL_CLASS, M, K, N)
