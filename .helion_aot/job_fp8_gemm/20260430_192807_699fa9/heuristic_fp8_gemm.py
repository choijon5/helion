"""
Auto-generated heuristic for kernel: fp8_gemm
Backend: decision_tree

Provides:
- key_fp8_gemm(*args): Returns config index (cache key)
- autotune_fp8_gemm(*args): Returns config dict for the given arguments
"""

import torch


def key_fp8_gemm(*args) -> int:
    """Select config index for the given arguments (also serves as cache key)."""
    _arg0_dim0 = int(args[0].shape[0]) if len(args) > 0 and isinstance(args[0], torch.Tensor) and args[0].ndim > 0 else 0
    if _arg0_dim0 <= 1536.0:
        if _arg0_dim0 <= 256.0:
            return 5
        else:
            if _arg0_dim0 <= 768.0:
                if _arg0_dim0 <= 512.0:
                    return 1
                else:
                    return 3
            else:
                return 1
    else:
        if _arg0_dim0 <= 2176.0:
            if _arg0_dim0 <= 2048.0:
                return 2
            else:
                return 4
        else:
            if _arg0_dim0 <= 3072.0:
                return 0
            else:
                if _arg0_dim0 <= 3712.0:
                    return 2
                else:
                    return 0


def autotune_fp8_gemm(*args) -> dict:
    """Select the optimal config for the given arguments."""
    _C = [
        {'block_sizes': [256, 256, 128], 'loop_orders': [[1, 0]], 'l2_groupings': [2], 'range_unroll_factors': [0, 2], 'range_warp_specializes': [None, False], 'range_num_stages': [0, 1], 'range_multi_buffers': [None, None], 'range_flattens': [None, False], 'load_eviction_policies': ['', 'first'], 'num_warps': 8, 'num_stages': 3, 'indexing': ['pointer', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat', 'epilogue_subtile': 2},
        {'block_sizes': [128, 64, 128], 'loop_orders': [[1, 0]], 'l2_groupings': [16], 'range_unroll_factors': [0, 0], 'range_warp_specializes': [None, None], 'range_num_stages': [0, 4], 'range_multi_buffers': [None, False], 'range_flattens': [None, False], 'load_eviction_policies': ['first', ''], 'num_warps': 4, 'num_stages': 4, 'indexing': ['pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [128, 256, 128], 'loop_orders': [[1, 0]], 'l2_groupings': [2], 'range_unroll_factors': [0, 3], 'range_warp_specializes': [None, False], 'range_num_stages': [0, 1], 'range_multi_buffers': [None, True], 'range_flattens': [None, None], 'load_eviction_policies': ['first', ''], 'num_warps': 4, 'num_stages': 4, 'indexing': ['pointer', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat', 'epilogue_subtile': 2},
        {'block_sizes': [64, 64, 256], 'loop_orders': [[0, 1]], 'l2_groupings': [2], 'range_unroll_factors': [0, 0], 'range_warp_specializes': [None, False], 'range_num_stages': [0, 3], 'range_multi_buffers': [None, True], 'range_flattens': [None, False], 'load_eviction_policies': ['last', 'first'], 'num_warps': 4, 'num_stages': 6, 'indexing': ['pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [128, 128, 128], 'loop_orders': [[1, 0]], 'l2_groupings': [4], 'range_unroll_factors': [0, 3], 'range_warp_specializes': [None, None], 'range_num_stages': [0, 1], 'range_multi_buffers': [None, False], 'range_flattens': [None, False], 'load_eviction_policies': ['first', ''], 'num_warps': 4, 'num_stages': 3, 'indexing': ['pointer', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [64, 8, 64], 'loop_orders': [[1, 0]], 'l2_groupings': [16], 'range_unroll_factors': [0, 4], 'range_warp_specializes': [None, None], 'range_num_stages': [0, 3], 'range_multi_buffers': [None, True], 'range_flattens': [None, None], 'load_eviction_policies': ['last', 'first'], 'num_warps': 4, 'num_stages': 6, 'indexing': ['pointer', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    ]
    return _C[key_fp8_gemm(*args)]
