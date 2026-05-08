"""
Auto-generated heuristic for kernel: softmax
Backend: nearest_neighbor

Provides:
- key_softmax(*args): Returns config index (cache key)
- autotune_softmax(*args): Returns config dict for the given arguments

Matching strategy:
1. Match categorical features exactly (dtype, etc.)
2. Find highest numeric values <= query (prefer lower/safer configs)
3. Fall back to lowest numeric values > query if no lower match
"""

import torch

# Training data: (categorical_values, numeric_values, config_idx)
_TRAIN_SOFTMAX = [
    ((), (4096.0, 10112.0), 0),
    ((), (4096.0, 10880.0), 0),
    ((), (4096.0, 12288.0), 0),
    ((), (4096.0, 10240.0), 0),
    ((), (4096.0, 640.0), 2),
    ((), (4096.0, 8832.0), 0),
    ((), (4096.0, 1536.0), 1),
    ((), (4096.0, 5888.0), 0),
    ((), (4096.0, 2176.0), 3),
    ((), (4096.0, 11776.0), 0),
    ((), (4096.0, 1920.0), 1),
    ((), (4096.0, 5376.0), 0),
    ((), (4096.0, 5376.0), 0),
    ((), (4096.0, 7936.0), 0),
    ((), (4096.0, 4480.0), 0),
    ((), (4096.0, 6144.0), 0),
    ((), (4096.0, 1792.0), 1),
    ((), (4096.0, 4352.0), 0),
    ((), (4096.0, 10496.0), 0),
    ((), (4096.0, 10496.0), 0),
    ((), (4096.0, 8192.0), 0),
    ((), (4096.0, 9472.0), 0),
    ((), (4096.0, 8960.0), 0),
    ((), (4096.0, 5504.0), 0),
    ((), (4096.0, 3072.0), 1),
    ((), (4096.0, 7808.0), 0),
    ((), (4096.0, 3200.0), 1),
    ((), (4096.0, 2048.0), 1),
    ((), (4096.0, 11136.0), 0),
    ((), (4096.0, 10112.0), 0),
    ((), (4096.0, 1152.0), 1),
    ((), (4096.0, 4480.0), 0),
    ((), (4096.0, 4096.0), 3),
    ((), (4096.0, 512.0), 5),
    ((), (4096.0, 1152.0), 1),
    ((), (4096.0, 10880.0), 0),
    ((), (4096.0, 10240.0), 0),
    ((), (4096.0, 384.0), 1),
    ((), (4096.0, 6144.0), 0),
    ((), (4096.0, 6016.0), 0),
    ((), (4096.0, 2816.0), 3),
    ((), (4096.0, 7424.0), 0),
    ((), (4096.0, 4608.0), 0),
    ((), (4096.0, 9728.0), 0),
    ((), (4096.0, 4608.0), 0),
    ((), (4096.0, 5120.0), 0),
    ((), (4096.0, 11648.0), 0),
    ((), (4096.0, 9728.0), 0),
    ((), (4096.0, 3968.0), 3),
    ((), (4096.0, 11520.0), 0),
    ((), (4096.0, 9984.0), 0),
    ((), (4096.0, 4864.0), 0),
    ((), (4096.0, 2432.0), 3),
    ((), (4096.0, 1408.0), 1),
    ((), (4096.0, 3072.0), 3),
    ((), (4096.0, 8448.0), 0),
    ((), (4096.0, 6272.0), 0),
    ((), (4096.0, 11520.0), 0),
    ((), (4096.0, 8576.0), 0),
    ((), (4096.0, 8064.0), 0),
    ((), (4096.0, 1280.0), 1),
    ((), (4096.0, 2944.0), 1),
    ((), (4096.0, 8832.0), 0),
    ((), (4096.0, 11648.0), 0),
    ((), (4096.0, 6912.0), 0),
    ((), (4096.0, 896.0), 2),
    ((), (4096.0, 7552.0), 0),
    ((), (4096.0, 3456.0), 3),
    ((), (4096.0, 1408.0), 1),
    ((), (4096.0, 4992.0), 0),
    ((), (4096.0, 7040.0), 0),
    ((), (4096.0, 11008.0), 0),
    ((), (4096.0, 9984.0), 0),
    ((), (4096.0, 1792.0), 1),
    ((), (4096.0, 6784.0), 0),
    ((), (4096.0, 2688.0), 3),
    ((), (4096.0, 12544.0), 0),
    ((), (4096.0, 7168.0), 0),
    ((), (4096.0, 7552.0), 0),
    ((), (4096.0, 9600.0), 0),
    ((), (4096.0, 12416.0), 0),
    ((), (4096.0, 7168.0), 0),
    ((), (4096.0, 11264.0), 0),
    ((), (4096.0, 11776.0), 0),
    ((), (4096.0, 9088.0), 0),
    ((), (4096.0, 1664.0), 1),
    ((), (4096.0, 12032.0), 0),
    ((), (4096.0, 9216.0), 0),
    ((), (4096.0, 7296.0), 0),
    ((), (4096.0, 10624.0), 0),
    ((), (4096.0, 896.0), 2),
    ((), (4096.0, 12288.0), 0),
    ((), (4096.0, 4352.0), 0),
    ((), (4096.0, 12672.0), 0),
    ((), (4096.0, 512.0), 2),
    ((), (4096.0, 5248.0), 0),
    ((), (4096.0, 12160.0), 0),
    ((), (4096.0, 8704.0), 0),
    ((), (4096.0, 3968.0), 1),
    ((), (4096.0, 5248.0), 0),
    ((), (4096.0, 3584.0), 1),
    ((), (4096.0, 256.0), 4),
    ((), (4096.0, 4224.0), 0),
    ((), (4096.0, 6272.0), 0),
    ((), (4096.0, 3712.0), 3),
    ((), (4096.0, 8320.0), 0),
    ((), (4096.0, 256.0), 5),
    ((), (4096.0, 7424.0), 0),
    ((), (4096.0, 2944.0), 0),
    ((), (4096.0, 3456.0), 1),
    ((), (4096.0, 12416.0), 0),
    ((), (4096.0, 1664.0), 2),
    ((), (4096.0, 5888.0), 0),
    ((), (4096.0, 2048.0), 2),
    ((), (4096.0, 9216.0), 0),
    ((), (4096.0, 5632.0), 0),
    ((), (4096.0, 12672.0), 0),
    ((), (4096.0, 6016.0), 0),
    ((), (4096.0, 6400.0), 0),
    ((), (4096.0, 12160.0), 0),
    ((), (4096.0, 5632.0), 0),
    ((), (4096.0, 7808.0), 0),
    ((), (4096.0, 11392.0), 0),
    ((), (4096.0, 3200.0), 1),
    ((), (4096.0, 10368.0), 0),
    ((), (4096.0, 3712.0), 1),
    ((), (4096.0, 11392.0), 0),
    ((), (4096.0, 9856.0), 0),
    ((), (4096.0, 9856.0), 0),
    ((), (4096.0, 9600.0), 0),
    ((), (4096.0, 1920.0), 1),
    ((), (4096.0, 6528.0), 0),
    ((), (4096.0, 768.0), 2),
    ((), (4096.0, 2560.0), 3),
    ((), (4096.0, 1280.0), 1),
    ((), (4096.0, 8704.0), 0),
    ((), (4096.0, 6400.0), 0),
    ((), (4096.0, 6784.0), 0),
    ((), (4096.0, 8320.0), 0),
    ((), (4096.0, 7936.0), 0),
    ((), (4096.0, 3328.0), 1),
    ((), (4096.0, 8960.0), 0),
    ((), (4096.0, 2304.0), 1),
    ((), (4096.0, 2176.0), 1),
    ((), (4096.0, 9344.0), 0),
    ((), (4096.0, 5120.0), 0),
    ((), (4096.0, 7040.0), 0),
    ((), (4096.0, 3840.0), 3),
    ((), (4096.0, 2432.0), 1),
    ((), (4096.0, 7680.0), 0),
    ((), (4096.0, 9472.0), 0),
    ((), (4096.0, 6656.0), 0),
    ((), (4096.0, 10368.0), 0),
    ((), (4096.0, 768.0), 2),
    ((), (4096.0, 4864.0), 0),
    ((), (4096.0, 12032.0), 0),
    ((), (4096.0, 640.0), 2),
    ((), (4096.0, 11136.0), 0),
    ((), (4096.0, 5504.0), 0),
    ((), (4096.0, 10624.0), 0),
    ((), (4096.0, 2688.0), 1),
    ((), (4096.0, 4992.0), 0),
    ((), (4096.0, 9088.0), 0),
    ((), (4096.0, 11904.0), 0),
    ((), (4096.0, 3328.0), 1),
    ((), (4096.0, 1024.0), 2),
    ((), (4096.0, 10752.0), 0),
    ((), (2048.0, 32768.0), 6),
    ((), (4096.0, 9344.0), 0),
    ((), (4096.0, 1024.0), 2),
    ((), (4096.0, 4096.0), 1),
    ((), (4096.0, 384.0), 2),
    ((), (4096.0, 11008.0), 0),
    ((), (4096.0, 6912.0), 0),
    ((), (4096.0, 7680.0), 0),
    ((), (4096.0, 7296.0), 0),
    ((), (4096.0, 11264.0), 0),
    ((), (4096.0, 16384.0), 0),
    ((), (4096.0, 2816.0), 1),
    ((), (4096.0, 4736.0), 0),
    ((), (4096.0, 2304.0), 3),
    ((), (4096.0, 3584.0), 3),
    ((), (4096.0, 11904.0), 0),
    ((), (4096.0, 5760.0), 0),
    ((), (4096.0, 8448.0), 0),
    ((), (4096.0, 8064.0), 0),
    ((), (4096.0, 10752.0), 0),
    ((), (4096.0, 1536.0), 1),
    ((), (4096.0, 3840.0), 1),
    ((), (4096.0, 2560.0), 1),
    ((), (4096.0, 5760.0), 0),
    ((), (4096.0, 4224.0), 0),
    ((), (4096.0, 6528.0), 0),
    ((), (4096.0, 8576.0), 0),
    ((), (4096.0, 8192.0), 0),
    ((), (4096.0, 6656.0), 0),
    ((), (4096.0, 4736.0), 0),
    ((), (4096.0, 12544.0), 0),
]

_CAT_FEATURES_SOFTMAX = []
_NUM_FEATURES_SOFTMAX = ['arg0_dim0', 'arg0_dim1']


def key_softmax(*args) -> int:
    """Select config index for the given arguments (also serves as cache key)."""
    _arg0_dim0 = int(args[0].shape[0]) if len(args) > 0 and isinstance(args[0], torch.Tensor) and args[0].ndim > 0 else 0
    _arg0_dim1 = int(args[0].shape[1]) if len(args) > 0 and isinstance(args[0], torch.Tensor) and args[0].ndim > 1 else 0

    # Build categorical and numeric feature tuples
    cat_vals = ()
    num_vals = (_arg0_dim0, _arg0_dim1,)

    # Find matching training points
    candidates = []
    for train_cat, train_num, config_idx in _TRAIN_SOFTMAX:
        if train_cat == cat_vals:
            candidates.append((train_num, config_idx))

    if not candidates:
        # No categorical match - return first config
        return _TRAIN_SOFTMAX[0][2] if _TRAIN_SOFTMAX else 0

    if not num_vals or len(num_vals) == 0:
        # No numeric features - return first match
        return candidates[0][1]

    # Find best numeric match: highest values <= query, else lowest > query
    best_lower_idx = -1
    best_lower_score = float("-inf")
    best_higher_idx = -1
    best_higher_score = float("inf")

    for train_num, config_idx in candidates:
        score = sum(train_num)
        all_lower = all(t <= q for t, q in zip(train_num, num_vals))
        if all_lower:
            if score > best_lower_score:
                best_lower_score = score
                best_lower_idx = config_idx
        else:
            if score < best_higher_score:
                best_higher_score = score
                best_higher_idx = config_idx

    if best_lower_idx >= 0:
        return best_lower_idx
    if best_higher_idx >= 0:
        return best_higher_idx
    return candidates[0][1]


def autotune_softmax(*args) -> dict:
    """Select the optimal config for the given arguments."""
    _C = [
        {'block_sizes': [1], 'reduction_loops': [None], 'range_unroll_factors': [0], 'range_warp_specializes': [None], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['', '', '', ''], 'num_warps': 8, 'num_stages': 2, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [1], 'reduction_loops': [None], 'range_unroll_factors': [0], 'range_warp_specializes': [None], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['last', '', '', 'first'], 'num_warps': 4, 'num_stages': 1, 'indexing': ['pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [1], 'reduction_loops': [None], 'range_unroll_factors': [0], 'range_warp_specializes': [None], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['last', 'first', 'last', ''], 'num_warps': 2, 'num_stages': 4, 'indexing': ['pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [1], 'reduction_loops': [None], 'range_unroll_factors': [0], 'range_warp_specializes': [None], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['', '', 'last', ''], 'num_warps': 4, 'num_stages': 2, 'indexing': ['pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [16], 'reduction_loops': [None], 'range_unroll_factors': [0], 'range_warp_specializes': [None], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['', 'first', 'last', ''], 'num_warps': 16, 'num_stages': 1, 'indexing': ['tensor_descriptor', 'pointer', 'pointer', 'pointer', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [4], 'reduction_loops': [None], 'range_unroll_factors': [0], 'range_warp_specializes': [None], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['', '', '', 'last'], 'num_warps': 4, 'num_stages': 7, 'indexing': ['pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [1], 'reduction_loops': [None], 'range_unroll_factors': [0], 'range_warp_specializes': [None], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['', 'first', 'first', 'first'], 'num_warps': 32, 'num_stages': 6, 'indexing': ['pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    ]
    return _C[key_softmax(*args)]
