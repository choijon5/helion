"""
Auto-generated combined heuristic for tutorial_kernels.py on cuda sm100.
Per-kernel decision-tree heuristics built from the AOT pretune campaign.

Provides per-kernel:
  - key_<kernel>(*args)      : returns config index
  - autotune_<kernel>(*args) : returns config dict
"""

import torch


# ---------------- vector_add (run 20260430_225347_f4807c) ----------------
def key_vector_add(*args) -> int:
    """Select config index for the given arguments (also serves as cache key)."""
    # No features needed
    return 0


def autotune_vector_add(*args) -> dict:
    """Select the optimal config for the given arguments."""
    _C = [
        {'block_sizes': [2048], 'range_unroll_factors': [0], 'range_warp_specializes': [None], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['first', ''], 'num_warps': 8, 'num_stages': 4, 'indexing': ['pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    ]
    return _C[key_vector_add(*args)]

# ---------------- matmul (run 20260430_192807_2dc612) ----------------
def key_matmul(*args) -> int:
    """Select config index for the given arguments (also serves as cache key)."""
    _arg0_dim0 = int(args[0].shape[0]) if len(args) > 0 and isinstance(args[0], torch.Tensor) and args[0].ndim > 0 else 0
    if _arg0_dim0 <= 1536.0:
        if _arg0_dim0 <= 768.0:
            if _arg0_dim0 <= 512.0:
                return 3
            else:
                return 6
        else:
            if _arg0_dim0 <= 1024.0:
                return 4
            else:
                return 2
    else:
        if _arg0_dim0 <= 2176.0:
            if _arg0_dim0 <= 2048.0:
                return 1
            else:
                return 5
        else:
            if _arg0_dim0 <= 3072.0:
                return 0
            else:
                if _arg0_dim0 <= 3712.0:
                    return 1
                else:
                    return 0


def autotune_matmul(*args) -> dict:
    """Select the optimal config for the given arguments."""
    _C = [
        {'block_sizes': [128, 256, 32], 'loop_orders': [[1, 0]], 'l2_groupings': [2], 'range_unroll_factors': [0, 2], 'range_warp_specializes': [None, None], 'range_num_stages': [0, 4], 'range_multi_buffers': [None, False], 'range_flattens': [None, None], 'load_eviction_policies': ['first', 'last'], 'num_warps': 4, 'num_stages': 4, 'indexing': ['pointer', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [128, 256, 64], 'loop_orders': [[0, 1]], 'l2_groupings': [2], 'range_unroll_factors': [0, 2], 'range_warp_specializes': [None, None], 'range_num_stages': [0, 3], 'range_multi_buffers': [None, False], 'range_flattens': [None, None], 'load_eviction_policies': ['last', ''], 'num_warps': 4, 'num_stages': 4, 'indexing': ['pointer', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [128, 128, 64], 'loop_orders': [[0, 1]], 'l2_groupings': [64], 'range_unroll_factors': [0, 1], 'range_warp_specializes': [None, False], 'range_num_stages': [0, 4], 'range_multi_buffers': [None, None], 'range_flattens': [None, True], 'load_eviction_policies': ['first', ''], 'num_warps': 4, 'num_stages': 6, 'indexing': ['pointer', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [16, 32, 128], 'loop_orders': [[0, 1]], 'l2_groupings': [8], 'range_unroll_factors': [0, 0], 'range_warp_specializes': [None, False], 'range_num_stages': [0, 3], 'range_multi_buffers': [None, None], 'range_flattens': [None, False], 'load_eviction_policies': ['first', ''], 'num_warps': 2, 'num_stages': 5, 'indexing': ['pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [64, 128, 64], 'loop_orders': [[0, 1]], 'l2_groupings': [32], 'range_unroll_factors': [0, 0], 'range_warp_specializes': [None, None], 'range_num_stages': [0, 4], 'range_multi_buffers': [None, None], 'range_flattens': [None, False], 'load_eviction_policies': ['', 'first'], 'num_warps': 4, 'num_stages': 7, 'indexing': ['pointer', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [128, 128, 64], 'loop_orders': [[0, 1]], 'l2_groupings': [32], 'range_unroll_factors': [0, 3], 'range_warp_specializes': [None, False], 'range_num_stages': [0, 3], 'range_multi_buffers': [None, True], 'range_flattens': [None, None], 'load_eviction_policies': ['last', 'first'], 'num_warps': 4, 'num_stages': 3, 'indexing': ['pointer', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [64, 64, 128], 'loop_orders': [[0, 1]], 'l2_groupings': [1], 'range_unroll_factors': [0, 4], 'range_warp_specializes': [None, None], 'range_num_stages': [0, 2], 'range_multi_buffers': [None, None], 'range_flattens': [None, None], 'load_eviction_policies': ['last', 'last'], 'num_warps': 8, 'num_stages': 5, 'indexing': ['pointer', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    ]
    return _C[key_matmul(*args)]

# ---------------- softmax (run 20260501_105410_7a2c56) ----------------
def key_softmax(*args) -> int:
    """Select config index for the given arguments (also serves as cache key)."""
    _arg0_dim1 = int(args[0].shape[1]) if len(args) > 0 and isinstance(args[0], torch.Tensor) and args[0].ndim > 1 else 0
    if _arg0_dim1 <= 4096.0:
        if _arg0_dim1 <= 1024.0:
            if _arg0_dim1 <= 384.0:
                return 1
            else:
                return 2
        else:
            if _arg0_dim1 <= 2048.0:
                if _arg0_dim1 <= 1920.0:
                    return 1
                else:
                    return 2
            else:
                return 1
    else:
        return 0


def autotune_softmax(*args) -> dict:
    """Select the optimal config for the given arguments."""
    _C = [
        {'block_sizes': [1], 'reduction_loops': [None], 'range_unroll_factors': [0], 'range_warp_specializes': [None], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['', '', '', ''], 'num_warps': 8, 'num_stages': 3, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [1], 'reduction_loops': [None], 'range_unroll_factors': [0], 'range_warp_specializes': [None], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['', '', '', ''], 'num_warps': 4, 'num_stages': 3, 'indexing': ['tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [1], 'reduction_loops': [None], 'range_unroll_factors': [0], 'range_warp_specializes': [None], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['last', 'first', 'last', ''], 'num_warps': 2, 'num_stages': 4, 'indexing': ['pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    ]
    return _C[key_softmax(*args)]

# ---------------- layer_norm (run 20260501_013839_c93350) ----------------
def key_layer_norm(*args) -> int:
    """Select config index for the given arguments (also serves as cache key)."""
    _arg0_dim1 = int(args[0].shape[1]) if len(args) > 0 and isinstance(args[0], torch.Tensor) and args[0].ndim > 1 else 0
    if _arg0_dim1 <= 11776.0:
        if _arg0_dim1 <= 3584.0:
            if _arg0_dim1 <= 1536.0:
                return 2
            else:
                if _arg0_dim1 <= 3072.0:
                    return 0
                else:
                    return 2
        else:
            if _arg0_dim1 <= 8192.0:
                if _arg0_dim1 <= 6144.0:
                    if _arg0_dim1 <= 4096.0:
                        return 3
                    else:
                        return 0
                else:
                    return 3
            else:
                return 0
    else:
        return 1


def autotune_layer_norm(*args) -> dict:
    """Select the optimal config for the given arguments."""
    _C = [
        {'block_sizes': [1], 'reduction_loops': [1024], 'range_unroll_factors': [0], 'range_warp_specializes': [None], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['', '', 'first', '', '', 'first', '', ''], 'num_warps': 4, 'num_stages': 1, 'indexing': ['pointer', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [1], 'reduction_loops': [None], 'range_unroll_factors': [0], 'range_warp_specializes': [None], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['first', 'last', 'last', '', '', 'last', 'last', 'last'], 'num_warps': 16, 'num_stages': 2, 'indexing': ['pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [2], 'reduction_loops': [None], 'range_unroll_factors': [0], 'range_warp_specializes': [None], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['first', '', 'last', '', 'last', 'last', 'first', ''], 'num_warps': 4, 'num_stages': 4, 'indexing': ['pointer', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [1], 'reduction_loops': [None], 'range_unroll_factors': [0], 'range_warp_specializes': [None], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['', '', '', 'last', '', 'last', '', 'last'], 'num_warps': 8, 'num_stages': 1, 'indexing': ['pointer', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    ]
    return _C[key_layer_norm(*args)]

# ---------------- attention (run 20260501_060054_3a4222) ----------------
def key_attention(*args) -> int:
    """Select config index for the given arguments (also serves as cache key)."""
    _arg0_dim3 = int(args[0].shape[3]) if len(args) > 0 and isinstance(args[0], torch.Tensor) and args[0].ndim > 3 else 0
    if _arg0_dim3 <= 64.0:
        return 1
    else:
        return 0


def autotune_attention(*args) -> dict:
    """Select the optimal config for the given arguments."""
    _C = [
        {'block_sizes': [1, 128, 64], 'loop_orders': [[0, 1]], 'l2_groupings': [16], 'range_unroll_factors': [1, 0], 'range_warp_specializes': [None, None], 'range_num_stages': [0, 3], 'range_multi_buffers': [True, False], 'range_flattens': [False, False], 'load_eviction_policies': ['', 'last', 'first'], 'num_warps': 4, 'num_stages': 4, 'indexing': ['pointer', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'persistent_interleaved', 'num_sm_multiplier': 2},
        {'block_sizes': [1, 128, 128], 'loop_orders': [[0, 1]], 'l2_groupings': [32], 'range_unroll_factors': [0, 4], 'range_warp_specializes': [None, None], 'range_num_stages': [0, 2], 'range_multi_buffers': [None, False], 'range_flattens': [None, True], 'load_eviction_policies': ['', '', 'first'], 'num_warps': 4, 'num_stages': 3, 'indexing': ['pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    ]
    return _C[key_attention(*args)]

# ---------------- grouped_gemm (run 20260501_032357_62ca6d) ----------------
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
        {'block_sizes': [1, 128, 128, 64], 'loop_orders': [[0, 1, 2]], 'l2_groupings': [16], 'range_unroll_factors': [0, 0], 'range_warp_specializes': [None, False], 'range_num_stages': [0, 1], 'range_multi_buffers': [None, False], 'range_flattens': [None, True], 'load_eviction_policies': ['first', 'first'], 'num_warps': 4, 'num_stages': 4, 'indexing': ['pointer', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [1, 32, 64, 64], 'loop_orders': [[2, 1, 0]], 'l2_groupings': [8], 'range_unroll_factors': [0, 2], 'range_warp_specializes': [None, None], 'range_num_stages': [0, 1], 'range_multi_buffers': [None, False], 'range_flattens': [None, False], 'load_eviction_policies': ['', ''], 'num_warps': 4, 'num_stages': 5, 'indexing': ['tensor_descriptor', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    ]
    return _C[key_grouped_gemm(*args)]

# ---------------- fp8_gemm (run 20260501_220932_cb0d7e) ----------------
def key_fp8_gemm(*args) -> int:
    """Select config index for the given arguments (also serves as cache key)."""
    _arg0_dim0 = int(args[0].shape[0]) if len(args) > 0 and isinstance(args[0], torch.Tensor) and args[0].ndim > 0 else 0
    if _arg0_dim0 <= 2176.0:
        if _arg0_dim0 <= 1536.0:
            if _arg0_dim0 <= 768.0:
                if _arg0_dim0 <= 384.0:
                    if _arg0_dim0 <= 256.0:
                        return 5
                    else:
                        return 1
                else:
                    return 3
            else:
                return 1
        else:
            if _arg0_dim0 <= 2048.0:
                return 2
            else:
                return 4
    else:
        return 0


def autotune_fp8_gemm(*args) -> dict:
    """Select the optimal config for the given arguments."""
    _C = [
        {'block_sizes': [128, 256, 64], 'loop_orders': [[1, 0]], 'l2_groupings': [4], 'range_unroll_factors': [0, 0], 'range_warp_specializes': [None, False], 'range_num_stages': [0, 2], 'range_multi_buffers': [None, False], 'range_flattens': [None, False], 'load_eviction_policies': ['', ''], 'num_warps': 4, 'num_stages': 4, 'indexing': ['pointer', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [128, 128, 128], 'loop_orders': [[0, 1]], 'l2_groupings': [2], 'range_unroll_factors': [0, 1], 'range_warp_specializes': [None, None], 'range_num_stages': [0, 0], 'range_multi_buffers': [None, False], 'range_flattens': [None, False], 'load_eviction_policies': ['', 'last'], 'num_warps': 4, 'num_stages': 4, 'indexing': ['pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [128, 256, 128], 'loop_orders': [[1, 0]], 'l2_groupings': [32], 'range_unroll_factors': [0, 4], 'range_warp_specializes': [None, False], 'range_num_stages': [0, 2], 'range_multi_buffers': [None, None], 'range_flattens': [None, True], 'load_eviction_policies': ['first', ''], 'num_warps': 4, 'num_stages': 4, 'indexing': ['pointer', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat', 'epilogue_subtile': 2},
        {'block_sizes': [128, 32, 256], 'loop_orders': [[0, 1]], 'l2_groupings': [8], 'range_unroll_factors': [0, 0], 'range_warp_specializes': [None, False], 'range_num_stages': [0, 4], 'range_multi_buffers': [None, None], 'range_flattens': [None, None], 'load_eviction_policies': ['last', 'last'], 'num_warps': 8, 'num_stages': 3, 'indexing': ['pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [128, 128, 64], 'loop_orders': [[0, 1]], 'l2_groupings': [8], 'range_unroll_factors': [0, 0], 'range_warp_specializes': [None, False], 'range_num_stages': [0, 0], 'range_multi_buffers': [None, True], 'range_flattens': [None, False], 'load_eviction_policies': ['last', 'first'], 'num_warps': 4, 'num_stages': 6, 'indexing': ['pointer', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [16, 32, 128], 'loop_orders': [[1, 0]], 'l2_groupings': [8], 'range_unroll_factors': [0, 0], 'range_warp_specializes': [None, None], 'range_num_stages': [0, 3], 'range_multi_buffers': [None, False], 'range_flattens': [None, False], 'load_eviction_policies': ['', ''], 'num_warps': 4, 'num_stages': 6, 'indexing': ['pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    ]
    return _C[key_fp8_gemm(*args)]
