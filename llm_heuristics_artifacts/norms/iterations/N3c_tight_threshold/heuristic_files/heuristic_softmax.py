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
    _arg0_dim0 = int(args[0].shape[0]) if len(args) > 0 and isinstance(args[0], torch.Tensor) and args[0].ndim > 0 else 0
    _arg0_dim1 = int(args[0].shape[1]) if len(args) > 0 and isinstance(args[0], torch.Tensor) and args[0].ndim > 1 else 0
    if _arg0_dim1 <= 4096.0:
        if _arg0_dim1 <= 1024.0:
            if _arg0_dim1 <= 512.0:
                if _arg0_dim1 <= 256.0:
                    return 10
                else:
                    if _arg0_dim1 <= 384.0:
                        return 1
                    else:
                        return 9
            else:
                if _arg0_dim1 <= 768.0:
                    return 4
                else:
                    return 4
        else:
            if _arg0_dim1 <= 2304.0:
                if _arg0_dim1 <= 1536.0:
                    return 1
                else:
                    if _arg0_dim1 <= 2048.0:
                        if _arg0_dim1 <= 1920.0:
                            return 1
                        else:
                            return 1
                    else:
                        return 1
            else:
                if _arg0_dim1 <= 3712.0:
                    if _arg0_dim1 <= 3328.0:
                        if _arg0_dim1 <= 3072.0:
                            return 1
                        else:
                            return 1
                    else:
                        return 1
                else:
                    if _arg0_dim1 <= 3968.0:
                        if _arg0_dim1 <= 3840.0:
                            return 1
                        else:
                            return 1
                    else:
                        return 0
    else:
        if _arg0_dim1 <= 6656.0:
            if _arg0_dim1 <= 5760.0:
                if _arg0_dim1 <= 5504.0:
                    if _arg0_dim1 <= 4864.0:
                        if _arg0_dim1 <= 4480.0:
                            return 0
                        else:
                            return 0
                    else:
                        if _arg0_dim1 <= 5120.0:
                            return 0
                        else:
                            return 0
                else:
                    if _arg0_dim1 <= 5632.0:
                        return 3
                    else:
                        return 3
            else:
                return 0
        else:
            if _arg0_dim1 <= 12544.0:
                if _arg0_dim1 <= 8320.0:
                    if _arg0_dim1 <= 7936.0:
                        if _arg0_dim1 <= 7040.0:
                            return 0
                        else:
                            return 0
                    else:
                        return 0
                else:
                    if _arg0_dim1 <= 8448.0:
                        return 2
                    else:
                        if _arg0_dim1 <= 9984.0:
                            return 0
                        else:
                            return 0
            else:
                if _arg0_dim0 <= 2048.0:
                    return 12
                else:
                    if _arg0_dim1 <= 12672.0:
                        return 0
                    else:
                        return 0


def autotune_softmax(*args) -> dict:
    """Select the optimal config for the given arguments."""
    _C = [
        {'block_sizes': [1], 'reduction_loops': [None], 'range_unroll_factors': [0], 'range_warp_specializes': [None], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['', 'last', 'last', 'first'], 'num_warps': 8, 'num_stages': 4, 'indexing': ['pointer', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [1], 'reduction_loops': [None], 'range_unroll_factors': [0], 'range_warp_specializes': [None], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['last', '', '', 'first'], 'num_warps': 4, 'num_stages': 1, 'indexing': ['pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [1], 'reduction_loops': [None], 'range_unroll_factors': [0], 'range_warp_specializes': [None], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['', '', '', ''], 'num_warps': 8, 'num_stages': 8, 'indexing': ['tensor_descriptor', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [1], 'reduction_loops': [None], 'range_unroll_factors': [0], 'range_warp_specializes': [None], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['', 'first', 'first', ''], 'num_warps': 8, 'num_stages': 8, 'indexing': ['tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [1], 'reduction_loops': [None], 'range_unroll_factors': [0], 'range_warp_specializes': [None], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['last', 'first', 'last', ''], 'num_warps': 2, 'num_stages': 4, 'indexing': ['pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [1], 'reduction_loops': [None], 'range_unroll_factors': [0], 'range_warp_specializes': [None], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['', 'first', 'last', ''], 'num_warps': 4, 'num_stages': 4, 'indexing': ['pointer', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [1], 'reduction_loops': [None], 'range_unroll_factors': [0], 'range_warp_specializes': [None], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['', '', 'last', 'last'], 'num_warps': 8, 'num_stages': 7, 'indexing': ['pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [1], 'reduction_loops': [None], 'range_unroll_factors': [0], 'range_warp_specializes': [None], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['', 'last', 'last', ''], 'num_warps': 1, 'num_stages': 7, 'indexing': ['pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [1], 'reduction_loops': [None], 'range_unroll_factors': [0], 'range_warp_specializes': [None], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['', 'last', '', 'last'], 'num_warps': 8, 'num_stages': 7, 'indexing': ['pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [1], 'reduction_loops': [None], 'range_unroll_factors': [0], 'range_warp_specializes': [None], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['', 'first', '', ''], 'num_warps': 2, 'num_stages': 1, 'indexing': ['pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [4], 'reduction_loops': [None], 'range_unroll_factors': [0], 'range_warp_specializes': [None], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['', '', '', 'last'], 'num_warps': 4, 'num_stages': 7, 'indexing': ['pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [16], 'reduction_loops': [None], 'range_unroll_factors': [0], 'range_warp_specializes': [None], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['last', 'first', '', 'last'], 'num_warps': 16, 'num_stages': 3, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [1], 'reduction_loops': [None], 'range_unroll_factors': [0], 'range_warp_specializes': [None], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['', 'first', 'first', 'first'], 'num_warps': 32, 'num_stages': 6, 'indexing': ['pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [1], 'reduction_loops': [None], 'range_unroll_factors': [0], 'range_warp_specializes': [None], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['first', '', '', ''], 'num_warps': 8, 'num_stages': 5, 'indexing': ['tensor_descriptor', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    ]
    return _C[key_softmax(*args)]
