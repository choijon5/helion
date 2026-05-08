"""
Auto-generated heuristic for kernel: rms_norm
Backend: nearest_neighbor

Provides:
- key_rms_norm(*args): Returns config index (cache key)
- autotune_rms_norm(*args): Returns config dict for the given arguments

Matching strategy:
1. Match categorical features exactly (dtype, etc.)
2. Find highest numeric values <= query (prefer lower/safer configs)
3. Fall back to lowest numeric values > query if no lower match
"""

import torch

# Training data: (categorical_values, numeric_values, config_idx)
_TRAIN_RMS_NORM = [
    ((), (4096.0, 12288.0, 50331648.0), 6),
    ((), (2048.0, 1536.0, 3145728.0), 0),
    ((), (2048.0, 2048.0, 4194304.0), 3),
    ((), (8192.0, 4096.0, 33554432.0), 5),
    ((), (4096.0, 7168.0, 29360128.0), 1),
    ((), (2048.0, 2047.0, 4192256.0), 0),
    ((), (8192.0, 8192.0, 67108864.0), 1),
    ((), (2048.0, 1023.0, 2095104.0), 0),
    ((), (2048.0, 6144.0, 12582912.0), 3),
    ((), (2048.0, 16384.0, 33554432.0), 4),
    ((), (2048.0, 4096.0, 8388608.0), 3),
    ((), (2048.0, 5120.0, 10485760.0), 3),
    ((), (2048.0, 3072.0, 6291456.0), 0),
    ((), (2048.0, 8192.0, 16777216.0), 1),
    ((), (2048.0, 32768.0, 67108864.0), 4),
    ((), (145956.0, 384.0, 56047104.0), 2),
    ((), (4096.0, 3584.0, 14680064.0), 5),
    ((), (4096.0, 8192.0, 33554432.0), 1),
    ((), (16384.0, 8192.0, 134217728.0), 1),
    ((), (2048.0, 96.0, 196608.0), 2),
    ((), (589824.0, 256.0, 150994944.0), 2),
    ((), (16384.0, 4096.0, 67108864.0), 5),
    ((), (4096.0, 4096.0, 16777216.0), 5),
    ((), (4096.0, 5120.0, 20971520.0), 1),
    ((), (2048.0, 127.0, 260096.0), 0),
    ((), (2048.0, 768.0, 1572864.0), 0),
    ((), (380668.0, 512.0, 194902016.0), 2),
    ((), (2048.0, 48.0, 98304.0), 0),
    ((), (2048.0, 1024.0, 2097152.0), 0),
    ((), (1179648.0, 256.0, 301989888.0), 2),
]

_CAT_FEATURES_RMS_NORM = []
_NUM_FEATURES_RMS_NORM = ['arg0_dim0', 'arg0_dim1', 'arg0_numel']


def key_rms_norm(*args) -> int:
    """Select config index for the given arguments (also serves as cache key)."""
    _arg0_dim0 = int(args[0].shape[0]) if len(args) > 0 and isinstance(args[0], torch.Tensor) and args[0].ndim > 0 else 0
    _arg0_dim1 = int(args[0].shape[1]) if len(args) > 0 and isinstance(args[0], torch.Tensor) and args[0].ndim > 1 else 0
    _arg0_numel = int(args[0].numel()) if len(args) > 0 and isinstance(args[0], torch.Tensor) else 0

    # Build categorical and numeric feature tuples
    cat_vals = ()
    num_vals = (_arg0_dim0, _arg0_dim1, _arg0_numel,)

    # Find matching training points
    candidates = []
    for train_cat, train_num, config_idx in _TRAIN_RMS_NORM:
        if train_cat == cat_vals:
            candidates.append((train_num, config_idx))

    if not candidates:
        # No categorical match - return first config
        return _TRAIN_RMS_NORM[0][2] if _TRAIN_RMS_NORM else 0

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


def autotune_rms_norm(*args) -> dict:
    """Select the optimal config for the given arguments."""
    _C = [
        {'block_sizes': [2], 'reduction_loops': [None], 'range_unroll_factors': [0], 'range_warp_specializes': [None], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['last', 'first', 'first', '', ''], 'num_warps': 4, 'num_stages': 6, 'indexing': ['pointer', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [1], 'reduction_loops': [None], 'range_unroll_factors': [0], 'range_warp_specializes': [None], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['first', 'last', 'last', '', 'last'], 'num_warps': 16, 'num_stages': 4, 'indexing': ['pointer', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [4], 'reduction_loops': [None], 'range_unroll_factors': [0], 'range_warp_specializes': [None], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['first', 'last', '', 'first', ''], 'num_warps': 2, 'num_stages': 3, 'indexing': ['pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [2], 'reduction_loops': [2048], 'range_unroll_factors': [0], 'range_warp_specializes': [None], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['last', 'first', 'last', 'first', 'first'], 'num_warps': 2, 'num_stages': 1, 'indexing': ['tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [1], 'reduction_loops': [8192], 'range_unroll_factors': [0], 'range_warp_specializes': [None], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['', 'first', 'last', '', ''], 'num_warps': 8, 'num_stages': 3, 'indexing': ['pointer', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [1], 'reduction_loops': [None], 'range_unroll_factors': [0], 'range_warp_specializes': [None], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['', 'last', 'last', 'first', ''], 'num_warps': 8, 'num_stages': 2, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [1], 'reduction_loops': [2048], 'range_unroll_factors': [0], 'range_warp_specializes': [None], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['', 'first', 'last', 'first', 'last'], 'num_warps': 8, 'num_stages': 8, 'indexing': ['pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    ]
    return _C[key_rms_norm(*args)]
