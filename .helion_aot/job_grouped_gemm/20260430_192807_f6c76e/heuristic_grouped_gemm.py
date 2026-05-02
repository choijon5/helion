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
    _arg0_dim1 = int(args[0].shape[1]) if len(args) > 0 and isinstance(args[0], torch.Tensor) and args[0].ndim > 1 else 0
    _arg0_dim2 = int(args[0].shape[2]) if len(args) > 0 and isinstance(args[0], torch.Tensor) and args[0].ndim > 2 else 0
    if _arg0_dim2 <= 256.0:
        if _arg0_dim1 <= 128.0:
            return 1
        else:
            return 2
    else:
        return 0


def autotune_grouped_gemm(*args) -> dict:
    """Select the optimal config for the given arguments."""
    _C = [
        {'block_sizes': [1, 128, 128, 64], 'loop_orders': [[1, 2, 0]], 'l2_groupings': [2], 'range_unroll_factors': [0, 0], 'range_warp_specializes': [None, None], 'range_num_stages': [0, 2], 'range_multi_buffers': [None, False], 'range_flattens': [None, True], 'load_eviction_policies': ['first', 'first'], 'num_warps': 4, 'num_stages': 4, 'indexing': ['pointer', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [1, 64, 8, 64], 'loop_orders': [[1, 2, 0]], 'l2_groupings': [8], 'range_unroll_factors': [0, 1], 'range_warp_specializes': [None, None], 'range_num_stages': [0, 0], 'range_multi_buffers': [None, False], 'range_flattens': [None, True], 'load_eviction_policies': ['first', ''], 'num_warps': 8, 'num_stages': 6, 'indexing': ['pointer', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [1, 128, 16, 64], 'loop_orders': [[1, 2, 0]], 'l2_groupings': [8], 'range_unroll_factors': [0, 1], 'range_warp_specializes': [None, None], 'range_num_stages': [0, 0], 'range_multi_buffers': [None, False], 'range_flattens': [None, True], 'load_eviction_policies': ['first', ''], 'num_warps': 8, 'num_stages': 6, 'indexing': ['pointer', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    ]
    return _C[key_grouped_gemm(*args)]
