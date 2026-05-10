"""N5 heuristic: observed-heuristics JSON lookup + N3b fallback.

Dispatch:
1. Bucket the query shape via the same `shape_bucket_for_class` logic
   used by ``scripts/llm_heuristics_research.py``.
2. If the bucket matches a rule in ``derived_general_heuristics.json``,
   return its highest-`win_count` template (merged with Helion defaults
   for fields the template omits).
3. Otherwise fall back to the N3b hybrid (archive tree for
   near-square in-range; adaptive block sizing otherwise).

No kernel name strings. No config hashes in the dispatch.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch

_RULES_PATH_ENV = "HELION_GEMM_OBSERVED_HEURISTICS_PATH"
_DEFAULT_RULES_PATH = (
    Path(__file__).resolve().parent.parent / "derived_general_heuristics.json"
)


def _load_rules() -> dict:
    path = Path(os.environ.get(_RULES_PATH_ENV) or _DEFAULT_RULES_PATH)
    return json.loads(path.read_text())


_RULES = _load_rules()


# --- Shape bucketing (mirrors scripts/llm_heuristics_research.py) -----

_MATMUL_DIM_BINS = (128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768)


def _bin_le(value: int, bins: tuple[int, ...]) -> str:
    for b in bins:
        if value <= b:
            return f"<={b}"
    return f">{bins[-1]}"


def _aspect_bucket(m: int, n: int, k: int) -> str:
    dims = [("m", m), ("n", n), ("k", k)]
    smallest = min(dims, key=lambda kv: kv[1])
    largest = max(dims, key=lambda kv: kv[1])
    if largest[1] == 0:
        return "degenerate"
    ratio = largest[1] / max(1, smallest[1])
    if ratio < 2:
        return "balanced"
    return f"skinny_{smallest[0]}"


def _matmul_bucket(dtype: str, M: int, K: int, N: int) -> dict:
    return {
        "aspect": _aspect_bucket(M, N, K),
        "dtype": dtype,
        "k_bin": _bin_le(K, _MATMUL_DIM_BINS),
        "m_bin": _bin_le(M, _MATMUL_DIM_BINS),
        "n_bin": _bin_le(N, _MATMUL_DIM_BINS),
    }


def _lookup_template(kernel_class: str, bucket: dict) -> dict | None:
    """Return the highest-win_count *compilable* template for this bucket,
    or None if no rule matches or every template exceeds the hardware
    resource budget for the caller-supplied query.

    Supports both the ``derived_general_heuristics.json`` schema (uses
    ``selected_templates``) and the ``runtime_observed_heuristics.json``
    schema (uses ``templates``).

    This function does not know the query shape; the ``_fits_budget``
    filter is applied in ``autotune_matmul`` after the dispatcher sees
    the live args.
    """
    key = frozenset(bucket.items())
    for rule in _RULES.get("rules", []):
        if rule.get("kernel_class") != kernel_class:
            continue
        if frozenset(rule["shape_bucket"].items()) != key:
            continue
        templates = rule.get("templates") or rule.get("selected_templates") or []
        if not templates:
            return None
        # Return *all* templates, ordered best-first, so the caller can
        # reject ones that would blow the shmem budget on its query.
        return sorted(
            templates,
            key=lambda t: (
                -t.get("win_count", 0),
                t.get("geomean_slowdown", 2.0),
            ),
        )
    return None


# --- Hardware resource budget -------------------------------------

# B200 shared-memory capacity per SM (bytes). Triton reserves some for
# the runtime (~2 KB observed on sm_100) and our estimator is
# approximate, so we use a safety margin.
_B200_SHMEM_LIMIT_BYTES = 232448
_SHMEM_SAFETY_MARGIN = 0.90  # stay under 90% of hardware limit


def _estimate_matmul_shmem_bytes(
    block_m: int, block_n: int, block_k: int,
    num_stages: int, dtype_size: int,
) -> int:
    """Estimate shared-memory bytes for a Helion matmul tile.

    Two operand tiles A[block_m, block_k] and B[block_k, block_n] are
    double-buffered per pipeline stage. The fp32 accumulator lives in
    registers, not shmem, so it doesn't count.

    This matches the OutOfResources message numbers observed on B200 in
    practice for the N6 heuristic's bad picks.
    """
    a_tile = block_m * block_k * dtype_size
    b_tile = block_k * block_n * dtype_size
    # num_stages stages of pipeline => num_stages copies of each operand.
    return (a_tile + b_tile) * num_stages


def _fits_shmem_budget(cfg: dict, M: int, K: int, N: int, dtype_size: int) -> bool:
    """True if ``cfg``'s tile sizes are estimated to fit B200 shmem."""
    bs = cfg.get("block_sizes") or [0, 0, 0]
    if len(bs) < 3:
        return True
    bm, bn, bk = int(bs[0]), int(bs[1]), int(bs[2])
    # Clamp to actual axis sizes — a [128, 512, 32] template on N=128
    # effectively runs as [128, 128, 32], which is smaller.
    bm = min(bm, M); bn = min(bn, N); bk = min(bk, K)
    ns = int(cfg.get("num_stages", 1)) or 1
    est = _estimate_matmul_shmem_bytes(bm, bn, bk, ns, dtype_size)
    return est <= int(_B200_SHMEM_LIMIT_BYTES * _SHMEM_SAFETY_MARGIN)


# --- N3b fallback (archive tree + adaptive) ------------------------

_ARCHIVE_CONFIGS = [
    {'block_sizes': [128, 512, 32], 'loop_orders': [[0, 1]], 'l2_groupings': [64], 'range_unroll_factors': [0, 4], 'range_warp_specializes': [None, False], 'range_num_stages': [0, 0], 'range_multi_buffers': [None, None], 'range_flattens': [None, None], 'load_eviction_policies': ['first', ''], 'num_warps': 8, 'num_stages': 5, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [64, 64, 128], 'loop_orders': [[1, 0]], 'l2_groupings': [2], 'range_unroll_factors': [0, 1], 'range_warp_specializes': [None, None], 'range_num_stages': [0, 3], 'range_multi_buffers': [None, None], 'range_flattens': [None, True], 'load_eviction_policies': ['first', 'first'], 'num_warps': 8, 'num_stages': 4, 'indexing': ['pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [128, 256, 64], 'loop_orders': [[0, 1]], 'l2_groupings': [2], 'range_unroll_factors': [0, 3], 'range_warp_specializes': [None, False], 'range_num_stages': [0, 4], 'range_multi_buffers': [None, False], 'range_flattens': [None, True], 'load_eviction_policies': ['last', ''], 'num_warps': 4, 'num_stages': 4, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat', 'epilogue_subtile': 2},
    {'block_sizes': [128, 128, 64], 'loop_orders': [[0, 1]], 'l2_groupings': [64], 'range_unroll_factors': [0, 1], 'range_warp_specializes': [None, False], 'range_num_stages': [0, 4], 'range_multi_buffers': [None, None], 'range_flattens': [None, True], 'load_eviction_policies': ['first', ''], 'num_warps': 4, 'num_stages': 6, 'indexing': ['pointer', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    {'block_sizes': [32, 32, 64], 'loop_orders': [[1, 0]], 'l2_groupings': [32], 'range_unroll_factors': [1, 1], 'range_warp_specializes': [False, None], 'range_num_stages': [0, 0], 'range_multi_buffers': [True, None], 'range_flattens': [False, None], 'load_eviction_policies': ['first', ''], 'num_warps': 4, 'num_stages': 5, 'indexing': ['pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'persistent_interleaved', 'num_sm_multiplier': 4, 'maxnreg': 64},
]


def _clamp_pot(value: int, hi: int, lo: int = 16) -> int:
    v = min(int(value), int(hi))
    if v < lo:
        return lo
    out = 1
    while (out << 1) <= v:
        out <<= 1
    return max(out, lo)


def _archive_dispatch_index(M: int) -> int:
    if M <= 1536:
        if M <= 256:
            return 4
        if M <= 1024:
            return 1
        return 3
    if M <= 2048:
        return 2
    return 0


def _is_in_archive_range(M: int, K: int, N: int) -> bool:
    aspect = max(M, N, K) / max(1, min(M, N, K))
    if aspect > 1.25:
        return False
    if not (256 <= M <= 4096 and 256 <= K <= 4096 and 256 <= N <= 4096):
        return False
    return True


def _fallback_config(M: int, K: int, N: int) -> dict:
    """N3b hybrid fallback: archive tree if in-range, adaptive otherwise."""
    if _is_in_archive_range(M, K, N):
        idx = _archive_dispatch_index(M)
        cfg = dict(_ARCHIVE_CONFIGS[idx])
        bm, bn, bk = cfg['block_sizes']
        cfg['block_sizes'] = [
            _clamp_pot(bm, M, lo=16),
            _clamp_pot(bn, N, lo=16),
            _clamp_pot(bk, K, lo=32),
        ]
        return cfg
    # Adaptive
    D = max(M, N)
    total_out = M * N
    if D <= 512:
        bm, bn, bk = 32, 64, 128
    elif D <= 1408:
        bm, bn, bk = 64, 128, 64
    elif D <= 2560:
        bm, bn, bk = 128, 256, 32
    else:
        bm, bn, bk = 256, 256, 32
    if total_out <= 512 * 512:
        nw, ns = 4, 5
    elif total_out <= 1536 * 1536:
        nw, ns = 4, 6
    else:
        nw, ns = 8, 4
    bm = _clamp_pot(bm, M, lo=16)
    bn = _clamp_pot(bn, N, lo=16)
    bk = _clamp_pot(bk, K, lo=32)
    if D <= 512:
        l2g = 1
    elif D <= 1024:
        l2g = 4
    elif D <= 2048:
        l2g = 8
    else:
        l2g = 16
    return {
        'block_sizes': [bm, bn, bk],
        'loop_orders': [[0, 1]],
        'l2_groupings': [l2g],
        'range_unroll_factors': [0, 0],
        'range_warp_specializes': [None, None],
        'range_num_stages': [0, 0],
        'range_multi_buffers': [None, None],
        'range_flattens': [None, None],
        'load_eviction_policies': ['', ''],
        'num_warps': nw,
        'num_stages': ns,
        'indexing': ['pointer', 'pointer', 'pointer'],
        'atomic_indexing': [],
        'pid_type': 'flat',
    }


# --- Merging: template (partial) into full Helion Config ---------

def _default_full_config() -> dict:
    return {
        'block_sizes': [64, 64, 64],
        'loop_orders': [[0, 1]],
        'l2_groupings': [1],
        'range_unroll_factors': [0, 0],
        'range_warp_specializes': [None, None],
        'range_num_stages': [0, 0],
        'range_multi_buffers': [None, None],
        'range_flattens': [None, None],
        'load_eviction_policies': ['', ''],
        'num_warps': 4,
        'num_stages': 3,
        'indexing': ['pointer', 'pointer', 'pointer'],
        'atomic_indexing': [],
        'pid_type': 'flat',
    }


def _merge_template_with_defaults(template: dict, M: int, K: int, N: int) -> dict:
    """The template fields listed in derived JSON are a partial config;
    merge with sensible Helion defaults, then clamp block_sizes to the
    actual axis dims so a template built from a smaller archive shape
    doesn't over-tile the query."""
    cfg = _default_full_config()
    cfg.update(template)
    bs = cfg.get('block_sizes')
    if isinstance(bs, list) and len(bs) == 3:
        cfg['block_sizes'] = [
            _clamp_pot(bs[0], M, lo=16),
            _clamp_pot(bs[1], N, lo=16),
            _clamp_pot(bs[2], K, lo=32),
        ]
    return cfg


# --- Entry points ---------------------------------------------------

def _shrink_to_fit(cfg: dict, M: int, K: int, N: int, dtype_size: int) -> dict:
    """Last-resort shmem fit: halve num_stages (down to 1) until the
    config is under the shmem budget. If even num_stages=1 doesn't fit,
    halve block_k (down to 32). Preserves all other knobs.
    """
    cfg = dict(cfg)
    ns = int(cfg.get("num_stages", 1)) or 1
    while ns > 1 and not _fits_shmem_budget({**cfg, "num_stages": ns}, M, K, N, dtype_size):
        ns //= 2
    cfg["num_stages"] = max(ns, 1)
    if not _fits_shmem_budget(cfg, M, K, N, dtype_size):
        bs = list(cfg.get("block_sizes") or [0, 0, 0])
        while bs[2] > 32 and not _fits_shmem_budget({**cfg, "block_sizes": bs}, M, K, N, dtype_size):
            bs[2] //= 2
        cfg["block_sizes"] = bs
    return cfg


def autotune_matmul(*args) -> dict:
    x = args[0]; y = args[1]
    M = int(x.shape[0])
    K = int(x.shape[1])
    N = int(y.shape[1])
    dtype_size = x.element_size()
    dtype = "fp16_bf16" if x.dtype in (torch.float16, torch.bfloat16) else "fp32"
    bucket = _matmul_bucket(dtype, M, K, N)

    # Ordered best->worst templates for this bucket (None if no rule).
    templates = _lookup_template("matmul", bucket)
    if templates:
        for t in templates:
            tmpl = t.get("template") if isinstance(t, dict) and "template" in t else t
            cfg = _merge_template_with_defaults(tmpl, M, K, N)
            if _fits_shmem_budget(cfg, M, K, N, dtype_size):
                return cfg
        # All templates in the bucket blow the shmem budget: take the
        # best one and shrink-to-fit rather than falling through to the
        # adaptive path (which may be worse than a shrunk-oracle config).
        tmpl = templates[0].get("template") if isinstance(templates[0], dict) and "template" in templates[0] else templates[0]
        return _shrink_to_fit(_merge_template_with_defaults(tmpl, M, K, N), M, K, N, dtype_size)

    # Fallback: archive tree + adaptive. Shrink-to-fit if needed.
    cfg = _fallback_config(M, K, N)
    if not _fits_shmem_budget(cfg, M, K, N, dtype_size):
        cfg = _shrink_to_fit(cfg, M, K, N, dtype_size)
    return cfg


def key_matmul(*args) -> int:
    x = args[0]; y = args[1]
    M = int(x.shape[0]); K = int(x.shape[1]); N = int(y.shape[1])
    dtype = "fp16_bf16" if x.dtype in (torch.float16, torch.bfloat16) else "fp32"
    bucket = _matmul_bucket(dtype, M, K, N)
    if _lookup_template("matmul", bucket) is not None:
        return hash(frozenset(bucket.items())) & 0xFFFFFF
    if _is_in_archive_range(M, K, N):
        return _archive_dispatch_index(M)
    cfg = _fallback_config(M, K, N)
    bs = cfg['block_sizes']
    return 1000 + (bs[0].bit_length() << 16) | (bs[1].bit_length() << 8) | bs[2].bit_length()
