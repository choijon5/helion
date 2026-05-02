"""
Auto-generated heuristic for kernel: grouped_gemm
Backend: decision_tree

Provides:
- key_grouped_gemm(*args): Returns config index (cache key)
- autotune_grouped_gemm(*args): Returns config dict for the given arguments
"""

import torch


def key_grouped_gemm(*args) -> int:
    """Select config index for the given arguments (also serves as cache key)."""
    _arg0_dim2 = int(args[0].shape[2]) if len(args) > 0 and isinstance(args[0], torch.Tensor) and args[0].ndim > 2 else 0
    if _arg0_dim2 <= 256.0:
        return 1
    else:
        return 0


def autotune_grouped_gemm(*args) -> dict:
    """Select the optimal config for the given arguments."""
    _C = [
        {'block_sizes': [1, 128, 128, 64], 'loop_orders': [[1, 0, 2]], 'l2_groupings': [32], 'range_unroll_factors': [0, 3], 'range_warp_specializes': [None, False], 'range_num_stages': [0, 1], 'range_multi_buffers': [None, True], 'range_flattens': [None, None], 'load_eviction_policies': ['', 'last'], 'num_warps': 4, 'num_stages': 3, 'indexing': ['pointer', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [1, 16, 64, 64], 'loop_orders': [[0, 1, 2]], 'l2_groupings': [4], 'range_unroll_factors': [0, 0], 'range_warp_specializes': [None, None], 'range_num_stages': [0, 0], 'range_multi_buffers': [None, None], 'range_flattens': [None, None], 'load_eviction_policies': ['', 'last'], 'num_warps': 4, 'num_stages': 3, 'indexing': ['tensor_descriptor', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    ]
    return _C[key_grouped_gemm(*args)]
