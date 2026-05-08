"""
Auto-generated heuristic for kernel: layer_norm
Backend: nearest_neighbor

Provides:
- key_layer_norm(*args): Returns config index (cache key)
- autotune_layer_norm(*args): Returns config dict for the given arguments

Matching strategy:
1. Match categorical features exactly (dtype, etc.)
2. Find highest numeric values <= query (prefer lower/safer configs)
3. Fall back to lowest numeric values > query if no lower match
"""

import torch

# Training data: (categorical_values, numeric_values, config_idx)
_TRAIN_LAYER_NORM = [
    ((), (4096.0, 26624.0, 109051904.0), 2),
    ((), (4096.0, 12800.0, 52428800.0), 0),
    ((), (4096.0, 11264.0, 46137344.0), 0),
    ((), (4096.0, 22528.0, 92274688.0), 2),
    ((), (4096.0, 11264.0, 46137344.0), 0),
    ((), (4096.0, 3584.0, 14680064.0), 6),
    ((), (4096.0, 12288.0, 50331648.0), 0),
    ((), (4096.0, 8192.0, 33554432.0), 7),
    ((), (4096.0, 14336.0, 58720256.0), 0),
    ((), (4096.0, 13312.0, 54525952.0), 0),
    ((), (1152.0, 36864.0, 42467328.0), 5),
    ((), (4096.0, 4608.0, 18874368.0), 8),
    ((), (4096.0, 20480.0, 83886080.0), 2),
    ((), (4096.0, 7680.0, 31457280.0), 7),
    ((), (4096.0, 4608.0, 18874368.0), 8),
    ((), (1024.0, 36864.0, 37748736.0), 5),
    ((), (4096.0, 1024.0, 4194304.0), 4),
    ((), (4096.0, 19456.0, 79691776.0), 2),
    ((), (4096.0, 15872.0, 65011712.0), 0),
    ((), (4096.0, 6656.0, 27262976.0), 1),
    ((), (4096.0, 13312.0, 54525952.0), 0),
    ((), (4096.0, 14848.0, 60817408.0), 0),
    ((), (4096.0, 15360.0, 62914560.0), 0),
    ((), (4096.0, 2560.0, 10485760.0), 3),
    ((), (4096.0, 18432.0, 75497472.0), 2),
    ((), (4096.0, 7168.0, 29360128.0), 7),
    ((), (4096.0, 14336.0, 58720256.0), 0),
    ((), (4096.0, 29696.0, 121634816.0), 2),
    ((), (4096.0, 6656.0, 27262976.0), 7),
    ((), (4096.0, 16384.0, 67108864.0), 0),
    ((), (4096.0, 13824.0, 56623104.0), 0),
    ((), (2048.0, 8192.0, 16777216.0), 7),
    ((), (4096.0, 5120.0, 20971520.0), 8),
    ((), (4096.0, 2048.0, 8388608.0), 4),
    ((), (4096.0, 3072.0, 12582912.0), 3),
    ((), (4096.0, 6144.0, 25165824.0), 1),
    ((), (4096.0, 1536.0, 6291456.0), 4),
    ((), (4096.0, 5632.0, 23068672.0), 8),
    ((), (4096.0, 23552.0, 96468992.0), 2),
    ((), (4096.0, 13824.0, 56623104.0), 0),
    ((), (4096.0, 4096.0, 16777216.0), 1),
    ((), (4096.0, 9216.0, 37748736.0), 0),
    ((), (4096.0, 2048.0, 8388608.0), 3),
    ((), (4096.0, 5632.0, 23068672.0), 1),
    ((), (4096.0, 9728.0, 39845888.0), 0),
    ((), (4096.0, 8192.0, 33554432.0), 1),
    ((), (4096.0, 1536.0, 6291456.0), 3),
    ((), (4096.0, 5120.0, 20971520.0), 8),
    ((), (4096.0, 3584.0, 14680064.0), 4),
    ((), (4096.0, 10752.0, 44040192.0), 0),
    ((), (4096.0, 25600.0, 104857600.0), 2),
    ((), (4096.0, 14848.0, 60817408.0), 0),
    ((), (4096.0, 6144.0, 25165824.0), 7),
    ((), (4096.0, 2560.0, 10485760.0), 4),
    ((), (4096.0, 16384.0, 67108864.0), 2),
    ((), (4096.0, 11776.0, 48234496.0), 0),
    ((), (4096.0, 9728.0, 39845888.0), 0),
    ((), (4096.0, 27648.0, 113246208.0), 2),
    ((), (4096.0, 10752.0, 44040192.0), 0),
    ((), (4096.0, 21504.0, 88080384.0), 2),
    ((), (4096.0, 17408.0, 71303168.0), 2),
    ((), (2048.0, 3584.0, 7340032.0), 7),
    ((), (8192.0, 4096.0, 33554432.0), 4),
    ((), (4096.0, 7680.0, 31457280.0), 1),
    ((), (4096.0, 28672.0, 117440512.0), 2),
    ((), (4096.0, 7168.0, 29360128.0), 1),
    ((), (4096.0, 10240.0, 41943040.0), 0),
    ((), (4096.0, 12288.0, 50331648.0), 0),
    ((), (4096.0, 10240.0, 41943040.0), 0),
    ((), (4096.0, 8704.0, 35651584.0), 0),
    ((), (4096.0, 15360.0, 62914560.0), 0),
    ((), (4096.0, 15872.0, 65011712.0), 0),
    ((), (4096.0, 9216.0, 37748736.0), 0),
    ((), (4096.0, 3072.0, 12582912.0), 4),
    ((), (4096.0, 4096.0, 16777216.0), 7),
    ((), (4096.0, 1024.0, 4194304.0), 3),
    ((), (4096.0, 12800.0, 52428800.0), 0),
    ((), (4096.0, 8704.0, 35651584.0), 0),
    ((), (4096.0, 11776.0, 48234496.0), 0),
    ((), (8192.0, 5120.0, 41943040.0), 8),
    ((), (8192.0, 7168.0, 58720256.0), 7),
    ((), (4096.0, 24576.0, 100663296.0), 2),
]

_CAT_FEATURES_LAYER_NORM = []
_NUM_FEATURES_LAYER_NORM = ['arg0_dim0', 'arg0_dim1', 'arg0_numel']


def key_layer_norm(*args) -> int:
    """Select config index for the given arguments (also serves as cache key)."""
    _arg0_dim0 = int(args[0].shape[0]) if len(args) > 0 and isinstance(args[0], torch.Tensor) and args[0].ndim > 0 else 0
    _arg0_dim1 = int(args[0].shape[1]) if len(args) > 0 and isinstance(args[0], torch.Tensor) and args[0].ndim > 1 else 0
    _arg0_numel = int(args[0].numel()) if len(args) > 0 and isinstance(args[0], torch.Tensor) else 0

    # Build categorical and numeric feature tuples
    cat_vals = ()
    num_vals = (_arg0_dim0, _arg0_dim1, _arg0_numel,)

    # Find matching training points
    candidates = []
    for train_cat, train_num, config_idx in _TRAIN_LAYER_NORM:
        if train_cat == cat_vals:
            candidates.append((train_num, config_idx))

    if not candidates:
        # No categorical match - return first config
        return _TRAIN_LAYER_NORM[0][2] if _TRAIN_LAYER_NORM else 0

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


def autotune_layer_norm(*args) -> dict:
    """Select the optimal config for the given arguments."""
    _C = [
        {'block_sizes': [1], 'reduction_loops': [2048], 'range_unroll_factors': [0], 'range_warp_specializes': [None], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['last', '', 'first', '', '', 'last', '', 'last'], 'num_warps': 8, 'num_stages': 4, 'indexing': ['tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [1], 'reduction_loops': [None], 'range_unroll_factors': [0], 'range_warp_specializes': [None], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['first', '', '', 'first', '', 'last', 'first', ''], 'num_warps': 8, 'num_stages': 4, 'indexing': ['pointer', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [1], 'reduction_loops': [4096], 'range_unroll_factors': [0], 'range_warp_specializes': [None], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['last', '', 'first', '', '', '', '', 'last'], 'num_warps': 16, 'num_stages': 5, 'indexing': ['pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor'], 'pid_type': 'flat'},
        {'block_sizes': [2], 'reduction_loops': [None], 'range_unroll_factors': [0], 'range_warp_specializes': [None], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['last', '', 'last', 'last', 'first', 'first', '', 'last'], 'num_warps': 4, 'num_stages': 1, 'indexing': ['pointer', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [1], 'reduction_loops': [None], 'range_unroll_factors': [0], 'range_warp_specializes': [None], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['first', '', 'last', '', 'last', 'last', 'last', ''], 'num_warps': 4, 'num_stages': 7, 'indexing': ['pointer', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [1], 'reduction_loops': [4096], 'range_unroll_factors': [0], 'range_warp_specializes': [None], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['last', '', 'first', '', '', 'last', '', ''], 'num_warps': 16, 'num_stages': 2, 'indexing': ['pointer', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [1], 'reduction_loops': [None], 'range_unroll_factors': [0], 'range_warp_specializes': [None], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['', '', 'last', '', 'last', 'first', 'last', 'first'], 'num_warps': 4, 'num_stages': 7, 'indexing': ['tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [1], 'reduction_loops': [None], 'range_unroll_factors': [0], 'range_warp_specializes': [None], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['first', 'last', 'last', 'last', 'first', 'last', 'first', 'first'], 'num_warps': 8, 'num_stages': 2, 'indexing': ['tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [1], 'reduction_loops': [1024], 'range_unroll_factors': [0], 'range_warp_specializes': [None], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['', 'last', 'first', '', 'last', 'first', 'last', 'last'], 'num_warps': 1, 'num_stages': 2, 'indexing': ['pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    ]
    return _C[key_layer_norm(*args)]
