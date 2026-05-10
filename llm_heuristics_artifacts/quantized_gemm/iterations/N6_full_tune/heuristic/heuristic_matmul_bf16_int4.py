"""Quantized-GEMM heuristic dispatcher for ``matmul_bf16_int4``.

A[M,K] bf16 * B[K//2, N] int8-packed-int4 -> out[M,N] bf16.

Shmem footprint assumes packed-int4 B tile (0.5 byte/element in
shmem), i.e. b_dtype_size=1 only counts the loaded byte; but the
compiler materializes bf16 after unpack, so we charge bf16 (2 bytes)
for the B tile — matches measured OutOfResources thresholds.
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


_KERNEL_CLASS = "matmul_int4"

# Exemplars from the Q2 archive (median-perf winner in each shape
# group). See llm_heuristics_artifacts/quantized_gemm/iterations/Q3_heuristic
# for how these were derived.
_FALLBACK_TABLE = {
    'small_m': {
        'atomic_indexing': [], 'block_sizes': [128, 16, 32],
        'indexing': ['pointer', 'pointer', 'pointer'],
        'l2_groupings': [4], 'load_eviction_policies': ['last', ''],
        'loop_orders': [[0, 1]], 'num_stages': 5, 'num_warps': 4,
        'pid_type': 'flat', 'range_flattens': [None, None],
        'range_multi_buffers': [None, None], 'range_num_stages': [0, 0],
        'range_unroll_factors': [0, 0], 'range_warp_specializes': [None, None],
    },
    'small_n': {
        'atomic_indexing': [], 'block_sizes': [64, 64, 16],
        'indexing': ['pointer', 'tensor_descriptor', 'pointer'],
        'l2_groupings': [1], 'load_eviction_policies': ['', ''],
        'loop_orders': [[0, 1]], 'num_stages': 3, 'num_warps': 2,
        'pid_type': 'flat', 'range_flattens': [None, None],
        'range_multi_buffers': [None, None], 'range_num_stages': [0, 0],
        'range_unroll_factors': [0, 3], 'range_warp_specializes': [None, None],
    },
    'small_k': {
        'atomic_indexing': [], 'block_sizes': [16, 128, 256],
        'indexing': ['pointer', 'tensor_descriptor', 'tensor_descriptor'],
        'l2_groupings': [1], 'load_eviction_policies': ['', ''],
        'loop_orders': [[1, 0]], 'num_stages': 4, 'num_warps': 8,
        'pid_type': 'flat', 'range_flattens': [None, None],
        'range_multi_buffers': [None, True], 'range_num_stages': [0, 0],
        'range_unroll_factors': [0, 2], 'range_warp_specializes': [None, None],
    },
    'balanced': {
        'atomic_indexing': [], 'block_sizes': [64, 128, 64],
        'indexing': ['tensor_descriptor', 'pointer', 'tensor_descriptor'],
        'l2_groupings': [1], 'load_eviction_policies': ['', ''],
        'loop_orders': [[0, 1]], 'num_stages': 3, 'num_warps': 8,
        'pid_type': 'flat', 'range_flattens': [None, None],
        'range_multi_buffers': [None, False], 'range_num_stages': [0, 2],
        'range_unroll_factors': [0, 0], 'range_warp_specializes': [None, None],
    },
    'rect': {
        'atomic_indexing': [], 'block_sizes': [16, 128, 128],
        'indexing': ['pointer', 'tensor_descriptor', 'tensor_descriptor'],
        'l2_groupings': [2], 'load_eviction_policies': ['', 'first'],
        'loop_orders': [[0, 1]], 'num_stages': 3, 'num_warps': 4,
        'pid_type': 'flat', 'range_flattens': [None, None],
        'range_multi_buffers': [None, None], 'range_num_stages': [0, 0],
        'range_unroll_factors': [0, 3], 'range_warp_specializes': [None, None],
    },
}


def _shape_from_args(args):
    A = args[0]
    B = args[1]
    M = int(A.shape[0])
    K = int(A.shape[1])
    # B is [K//2, N] packed int4.
    N = int(B.shape[1])
    return M, K, N


def autotune_matmul_bf16_int4(*args) -> dict:
    M, K, N = _shape_from_args(args)
    a_dtype_size = int(args[0].element_size())  # 2 for bf16
    # Charge bf16 for B tile because unpacked operand is what sits in
    # shmem after the loader's dequant.
    b_dtype_size = 2
    return core.dispatch(
        _KERNEL_CLASS, M, K, N, a_dtype_size, b_dtype_size,
        _FALLBACK_TABLE,
    )


def key_matmul_bf16_int4(*args) -> int:
    M, K, N = _shape_from_args(args)
    return core.dispatch_key(_KERNEL_CLASS, M, K, N)
