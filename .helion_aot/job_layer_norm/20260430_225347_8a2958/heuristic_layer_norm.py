"""
Auto-generated heuristic for kernel: layer_norm
Backend: decision_tree

Provides:
- key_layer_norm(*args): Returns config index (cache key)
- autotune_layer_norm(*args): Returns config dict for the given arguments
"""

import torch


def key_layer_norm(*args) -> int:
    """Select config index for the given arguments (also serves as cache key)."""
    _arg0_dim1 = int(args[0].shape[1]) if len(args) > 0 and isinstance(args[0], torch.Tensor) and args[0].ndim > 1 else 0
    if _arg0_dim1 <= 4096.0:
        if _arg0_dim1 <= 2560.0:
            return 1
        else:
            if _arg0_dim1 <= 3072.0:
                return 3
            else:
                return 1
    else:
        if _arg0_dim1 <= 12288.0:
            if _arg0_dim1 <= 5120.0:
                if _arg0_dim1 <= 4608.0:
                    return 0
                else:
                    return 3
            else:
                if _arg0_dim1 <= 8192.0:
                    if _arg0_dim1 <= 7168.0:
                        return 0
                    else:
                        return 2
                else:
                    return 0
        else:
            return 2


def autotune_layer_norm(*args) -> dict:
    """Select the optimal config for the given arguments."""
    _C = [
        {'block_sizes': [1], 'reduction_loops': [1024], 'range_unroll_factors': [0], 'range_warp_specializes': [False], 'range_num_stages': [0], 'range_multi_buffers': [True], 'range_flattens': [None], 'load_eviction_policies': ['', '', 'first', '', '', 'first', 'first', 'first'], 'num_warps': 4, 'num_stages': 1, 'indexing': ['tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'persistent_interleaved', 'num_sm_multiplier': 64, 'maxnreg': 64},
        {'block_sizes': [2], 'reduction_loops': [None], 'range_unroll_factors': [0], 'range_warp_specializes': [None], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['', 'last', '', '', '', '', 'first', ''], 'num_warps': 4, 'num_stages': 2, 'indexing': ['pointer', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [1], 'reduction_loops': [None], 'range_unroll_factors': [0], 'range_warp_specializes': [None], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['', '', '', '', '', '', '', ''], 'num_warps': 16, 'num_stages': 1, 'indexing': ['pointer', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [1], 'reduction_loops': [1024], 'range_unroll_factors': [0], 'range_warp_specializes': [None], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['', '', '', '', '', '', '', ''], 'num_warps': 1, 'num_stages': 7, 'indexing': ['pointer', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    ]
    return _C[key_layer_norm(*args)]
