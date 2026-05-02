"""
Auto-generated heuristic for kernel: softmax
Backend: decision_tree

Provides:
- key_softmax(*args): Returns config index (cache key)
- autotune_softmax(*args): Returns config dict for the given arguments
"""

import torch


def key_softmax(*args) -> int:
    """Select config index for the given arguments (also serves as cache key)."""
    _arg0_dim1 = int(args[0].shape[1]) if len(args) > 0 and isinstance(args[0], torch.Tensor) and args[0].ndim > 1 else 0
    if _arg0_dim1 <= 3968.0:
        if _arg0_dim1 <= 1024.0:
            if _arg0_dim1 <= 384.0:
                return 3
            else:
                return 2
        else:
            if _arg0_dim1 <= 2816.0:
                return 1
            else:
                if _arg0_dim1 <= 3200.0:
                    if _arg0_dim1 <= 2944.0:
                        return 0
                    else:
                        if _arg0_dim1 <= 3072.0:
                            return 1
                        else:
                            return 0
                else:
                    return 1
    else:
        return 0


def autotune_softmax(*args) -> dict:
    """Select the optimal config for the given arguments."""
    _C = [
        {'block_sizes': [1], 'reduction_loops': [None], 'range_unroll_factors': [0], 'range_warp_specializes': [None], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['', 'first', 'last', 'last'], 'num_warps': 8, 'num_stages': 5, 'indexing': ['tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [1], 'reduction_loops': [None], 'range_unroll_factors': [0], 'range_warp_specializes': [None], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['', '', 'last', ''], 'num_warps': 4, 'num_stages': 2, 'indexing': ['pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [1], 'reduction_loops': [None], 'range_unroll_factors': [0], 'range_warp_specializes': [None], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['first', 'last', 'last', ''], 'num_warps': 1, 'num_stages': 7, 'indexing': ['pointer', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [16], 'reduction_loops': [None], 'range_unroll_factors': [0], 'range_warp_specializes': [None], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['last', 'first', '', 'last'], 'num_warps': 16, 'num_stages': 3, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    ]
    return _C[key_softmax(*args)]
