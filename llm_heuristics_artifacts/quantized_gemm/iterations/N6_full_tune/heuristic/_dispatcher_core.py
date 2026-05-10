"""Shared dispatcher engine for the quantized-GEMM heuristics.

The three quantized kernels (int4, int16, fp4) share a matmul-shaped
tuning surface — same shape bucket, same shared-memory footprint
formula, same template merge + shrink-to-fit logic. Differences are:

- kernel_class label used for rule lookup
- per-kernel fallback config family (since each weight dtype
  favors different tile shapes)
- per-kernel in-range predicate for using an archive table

This module centralizes everything shared; the three kernel
dispatcher files stay small and just plug in their kernel-specific
pieces.

No kernel-name strings leak into any feature. No config hashes used
as keys.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

_RULES_PATH_ENV = "HELION_QUANTIZED_GEMM_OBSERVED_HEURISTICS_PATH"
_DEFAULT_RULES_PATH = (
    Path(__file__).resolve().parent.parent / "derived_general_heuristics.json"
)


def _load_rules() -> dict:
    path = Path(os.environ.get(_RULES_PATH_ENV) or _DEFAULT_RULES_PATH)
    return json.loads(path.read_text())


_RULES = _load_rules()


# --- Shape bucketing (must mirror scripts/llm_heuristics_research.py) -----
# Any drift between these bins and the generator's bucketing silently
# breaks rule lookup — the dispatcher computes a key the JSON has no
# entry for.

_MATMUL_MN_BINS = (64, 128, 256, 512, 1024, 4096)
_MATMUL_K_BINS = (64, 128, 256, 512, 1024, 4096, 32768)


def _bin_le(value: int, bins: tuple[int, ...]) -> str:
    for b in bins:
        if value <= b:
            return f"<={b}"
    return f">{bins[-1]}"


def _aspect_bucket(m: int, n: int, k: int) -> str:
    values = [v for v in (m, n, k) if v > 0]
    if len(values) != 3:
        return "unknown"
    ratio = max(values) / max(1, min(values))
    if ratio < 4:
        return "balanced"
    smallest = min(values)
    if m == smallest:
        return "skinny_m"
    if n == smallest:
        return "skinny_n"
    return "skinny_k"


def matmul_bucket(dtype: str, M: int, K: int, N: int) -> dict:
    return {
        "aspect": _aspect_bucket(M, N, K),
        "dtype": dtype,
        "k_bin": _bin_le(K, _MATMUL_K_BINS),
        "m_bin": _bin_le(M, _MATMUL_MN_BINS),
        "n_bin": _bin_le(N, _MATMUL_MN_BINS),
    }


def lookup_templates(kernel_class: str, bucket: dict) -> list | None:
    """Return all templates for the first matching rule, best-first.

    Supports both schemas: runtime-filtered rules use ``templates``,
    derived-general rules use ``selected_templates``. Caller applies
    the shmem budget filter to the query shape.
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
        return sorted(
            templates,
            key=lambda t: (
                -t.get("win_count", 0),
                t.get("geomean_slowdown", 2.0),
            ),
        )
    return None


# --- B200 shared-memory budget ------------------------------------

_B200_SHMEM_LIMIT_BYTES = 232448
_SHMEM_SAFETY_MARGIN = 0.90


def _estimate_matmul_shmem_bytes(
    block_m: int, block_n: int, block_k: int,
    num_stages: int, a_dtype_size: int, b_dtype_size: int,
) -> int:
    """Shared-memory bytes for A[block_m,block_k] + B[block_k,block_n]
    double-buffered for ``num_stages`` pipeline stages.

    The fp32 accumulator lives in registers, not shmem.

    Packed weight kernels (int4, fp4) store a B tile of
    [block_k, block_n] nibbles = block_k*block_n*0.5 bytes, because
    the loader pulls the packed int8 representation from HBM. This
    function takes per-operand dtype sizes so callers can pass 1 for
    packed-int8/int4/fp4 and 2 for int16/bf16.
    """
    a_tile = block_m * block_k * a_dtype_size
    b_tile = block_k * block_n * b_dtype_size
    return (a_tile + b_tile) * num_stages


def fits_shmem_budget(
    cfg: dict, M: int, K: int, N: int,
    a_dtype_size: int, b_dtype_size: int,
) -> bool:
    """True if ``cfg``'s tile sizes fit within the B200 shmem budget."""
    bs = cfg.get("block_sizes") or [0, 0, 0]
    if len(bs) < 3:
        return True
    bm, bn, bk = int(bs[0]), int(bs[1]), int(bs[2])
    # Template built from a larger shape can over-tile a small query;
    # clamp to the actual axis so the estimate is realistic.
    bm = min(bm, M); bn = min(bn, N); bk = min(bk, K)
    ns = int(cfg.get("num_stages", 1)) or 1
    est = _estimate_matmul_shmem_bytes(bm, bn, bk, ns, a_dtype_size, b_dtype_size)
    return est <= int(_B200_SHMEM_LIMIT_BYTES * _SHMEM_SAFETY_MARGIN)


def _clamp_pot(value: int, hi: int, lo: int = 16) -> int:
    v = min(int(value), int(hi))
    if v < lo:
        return lo
    out = 1
    while (out << 1) <= v:
        out <<= 1
    return max(out, lo)


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


def merge_template_with_defaults(template: dict, M: int, K: int, N: int) -> dict:
    cfg = _default_full_config()
    cfg.update(template)
    bs = cfg.get('block_sizes')
    if isinstance(bs, list) and len(bs) == 3:
        cfg['block_sizes'] = [
            _clamp_pot(bs[0], M, lo=16),
            _clamp_pot(bs[1], N, lo=16),
            _clamp_pot(bs[2], K, lo=16),
        ]
    return cfg


def shrink_to_fit(
    cfg: dict, M: int, K: int, N: int,
    a_dtype_size: int, b_dtype_size: int,
) -> dict:
    cfg = dict(cfg)
    ns = int(cfg.get("num_stages", 1)) or 1
    while ns > 1 and not fits_shmem_budget(
        {**cfg, "num_stages": ns}, M, K, N, a_dtype_size, b_dtype_size
    ):
        ns //= 2
    cfg["num_stages"] = max(ns, 1)
    if not fits_shmem_budget(cfg, M, K, N, a_dtype_size, b_dtype_size):
        bs = list(cfg.get("block_sizes") or [0, 0, 0])
        while bs[2] > 16 and not fits_shmem_budget(
            {**cfg, "block_sizes": bs}, M, K, N, a_dtype_size, b_dtype_size
        ):
            bs[2] //= 2
        cfg["block_sizes"] = bs
    return cfg


# --- Fallback selection -------------------------------------------

def choose_fallback(
    fallback_table: dict, M: int, K: int, N: int,
) -> dict:
    """Select a fallback config from a per-group table.

    Groups (priority order, first match wins):
      small_m  — M <= 256
      small_n  — N <= 256
      small_k  — K <= 256
      balanced — max/min ratio < 2
      rect     — everything else

    The table maps these group names to ``(block_sizes_base, config)``
    tuples. ``config`` is used verbatim except ``block_sizes``, which
    is clamped to fit the live axes.
    """
    if M <= 256:
        g = 'small_m'
    elif N <= 256:
        g = 'small_n'
    elif K <= 256:
        g = 'small_k'
    elif max(M, N, K) / max(1, min(M, N, K)) < 2:
        g = 'balanced'
    else:
        g = 'rect'
    cfg = dict(fallback_table[g])
    bs = cfg.get('block_sizes') or [64, 64, 64]
    cfg['block_sizes'] = [
        _clamp_pot(bs[0], M, lo=16),
        _clamp_pot(bs[1], N, lo=16),
        _clamp_pot(bs[2], K, lo=16),
    ]
    return cfg


# --- Dispatch entry point -----------------------------------------

def dispatch(
    kernel_class: str,
    M: int, K: int, N: int,
    a_dtype_size: int, b_dtype_size: int,
    fallback_table: dict,
    dtype_family: str = "fp16_bf16",
) -> dict:
    """Single-config dispatch.

    1. Bucket the query shape.
    2. If a rule matches the bucket, walk its templates best-first and
       return the first that fits the shmem budget.
    3. If every template blows the budget, shrink-to-fit the best one.
    4. Otherwise return the fallback config for the live shape group.
    """
    bucket = matmul_bucket(dtype_family, M, K, N)
    templates = lookup_templates(kernel_class, bucket)
    if templates:
        for t in templates:
            tmpl = t.get("template") if isinstance(t, dict) and "template" in t else t
            cfg = merge_template_with_defaults(tmpl, M, K, N)
            if fits_shmem_budget(cfg, M, K, N, a_dtype_size, b_dtype_size):
                return cfg
        tmpl0 = templates[0].get("template") if isinstance(templates[0], dict) and "template" in templates[0] else templates[0]
        return shrink_to_fit(
            merge_template_with_defaults(tmpl0, M, K, N),
            M, K, N, a_dtype_size, b_dtype_size,
        )
    cfg = choose_fallback(fallback_table, M, K, N)
    if not fits_shmem_budget(cfg, M, K, N, a_dtype_size, b_dtype_size):
        cfg = shrink_to_fit(cfg, M, K, N, a_dtype_size, b_dtype_size)
    return cfg


def dispatch_key(kernel_class: str, M: int, K: int, N: int,
                 dtype_family: str = "fp16_bf16") -> int:
    """Stable hash for Helion's shape cache."""
    bucket = matmul_bucket(dtype_family, M, K, N)
    if lookup_templates(kernel_class, bucket) is not None:
        return hash(frozenset(bucket.items())) & 0xFFFFFF
    if M <= 256:
        g = 0
    elif N <= 256:
        g = 1
    elif K <= 256:
        g = 2
    elif max(M, N, K) / max(1, min(M, N, K)) < 2:
        g = 3
    else:
        g = 4
    return 1_000_000 + g
