"""
Auto-generated heuristic for kernel: attention
Backend: decision_tree

Provides:
- key_attention(*args): Returns config index (cache key)
- autotune_attention(*args): Returns config dict for the given arguments
"""

import torch


def key_attention(*args) -> int:
    """Select config index for the given arguments (also serves as cache key)."""
    _arg0_dim3 = int(args[0].shape[3]) if len(args) > 0 and isinstance(args[0], torch.Tensor) and args[0].ndim > 3 else 0
    if _arg0_dim3 <= 64.0:
        return 0
    else:
        return 1


def autotune_attention(*args) -> dict:
    """Select the optimal config for the given arguments."""
    _C = [
        {'block_sizes': [1, 128, 128], 'loop_orders': [[0, 1]], 'l2_groupings': [4], 'range_unroll_factors': [0, 3], 'range_warp_specializes': [None, False], 'range_num_stages': [0, 4], 'range_multi_buffers': [None, None], 'range_flattens': [None, False], 'load_eviction_policies': ['', '', 'last'], 'num_warps': 4, 'num_stages': 3, 'indexing': ['pointer', 'pointer', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [1, 128, 64], 'loop_orders': [[1, 0]], 'l2_groupings': [8], 'range_unroll_factors': [0, 3], 'range_warp_specializes': [None, False], 'range_num_stages': [0, 1], 'range_multi_buffers': [None, True], 'range_flattens': [None, None], 'load_eviction_policies': ['', '', 'first'], 'num_warps': 8, 'num_stages': 3, 'indexing': ['tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    ]
    return _C[key_attention(*args)]
