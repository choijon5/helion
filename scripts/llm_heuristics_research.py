# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Summarize AOT measurement CSVs into compact heuristics for the LLM autotuner.

The script is intentionally offline-first:

1. Load measured AOT CSVs from ``aot_pretune_data/b200``.
2. Aggregate repeated measurements by median timing.
3. Find per-shape winners, robust config families, and shape-regime patterns.
4. Emit compact JSON/Markdown artifacts that can be reviewed or sent to an LLM.
5. Optionally call Claude Opus 4.7 through Helion's LLM transport or the local
   Claude CLI / AI Gateway wrapper to critique and refine the heuristic guidance.

The generated artifacts are meant to guide prompt and seed-config changes.  They
are not runtime heuristics by themselves.
"""

from __future__ import annotations

import argparse
import collections
import csv
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Sequence

DEFAULT_DATA_ROOT = Path(
    "/home/jongsokchoi/helion_2_aot_pretune_data_all/aot_pretune_data/b200"
)
DEFAULT_OUTPUT_DIR = Path("/tmp/helion_llm_heuristics_research")
DEFAULT_MODEL = "claude-opus-4-7"
REPO_ROOT = Path(__file__).resolve().parents[1]
KEY_CONFIG_FIELDS = (
    "block_sizes",
    "num_warps",
    "num_stages",
    "pid_type",
    "indexing",
    "l2_groupings",
    "range_unroll_factors",
    "range_num_stages",
    "range_warp_specializes",
    "range_multi_buffers",
    "range_flattens",
    "load_eviction_policies",
    "maxnreg",
    "num_sm_multiplier",
)
GENERAL_TEMPLATE_FIELDS = (
    "block_sizes",
    "num_warps",
    "num_stages",
    "pid_type",
    "l2_groupings",
    "reduction_loops",
    "split_k",
    "num_sm_multiplier",
)
RULE_TEMPLATE_LIMIT = 3
RUNTIME_HEURISTIC_FILTERS = {
    "min_rule_shapes": 2,
    "min_holdout_coverage": 0.75,
    "max_rule_holdout_geomean_slowdown": 1.05,
    "max_rule_holdout_p90_slowdown": 1.10,
    "min_template_shapes": 2,
    "max_template_geomean_slowdown": 1.01,
    "max_template_p90_slowdown": 1.10,
}


@dataclass(frozen=True)
class Measurement:
    """One raw timing row from an AOT measurement CSV."""

    kernel: str
    run_id: str
    shape_hash: str
    config_hash: str
    config: dict[str, object]
    shape_features: dict[str, object]
    timing_ms: float


@dataclass(frozen=True)
class AggregatedMeasurement:
    """Median timing for one kernel/shape/config tuple."""

    kernel: str
    shape_hash: str
    config_hash: str
    config: dict[str, object]
    shape_features: dict[str, object]
    median_ms: float
    sample_count: int
    run_ids: tuple[str, ...]


def _json_loads_dict(text: str) -> dict[str, object]:
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object, got {type(value).__name__}")
    return value


def _finite_float(value: str) -> float | None:
    try:
        result = float(value)
    except ValueError:
        return None
    if not math.isfinite(result):
        return None
    return result


def _run_id_from_csv(path: Path) -> str:
    # .../<kernel>/runs/<run_id>/measurements_...
    try:
        return path.parent.name
    except IndexError:
        return "unknown"


def load_measurements(data_root: Path) -> list[Measurement]:
    """Load every measured run CSV below ``data_root``."""
    measurements: list[Measurement] = []
    for csv_path in sorted(data_root.glob("*/runs/*/measurements_*.csv")):
        run_id = _run_id_from_csv(csv_path)
        with csv_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                timing = _finite_float(row.get("timing_ms", ""))
                if timing is None:
                    continue
                measurements.append(
                    Measurement(
                        kernel=row["kernel_name"],
                        run_id=run_id,
                        shape_hash=row["shape_hash"],
                        config_hash=row["config_hash"],
                        config=_json_loads_dict(row["config"]),
                        shape_features=_json_loads_dict(row["shape_features"]),
                        timing_ms=timing,
                    )
                )
    return measurements


def _stable_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def aggregate_measurements(
    measurements: Sequence[Measurement],
) -> list[AggregatedMeasurement]:
    """Median-aggregate repeated observations for identical shape/config pairs."""
    grouped: dict[tuple[str, str, str, str], list[Measurement]] = (
        collections.defaultdict(list)
    )
    for measurement in measurements:
        key = (
            measurement.kernel,
            measurement.shape_hash,
            measurement.config_hash,
            _stable_json(measurement.config),
        )
        grouped[key].append(measurement)

    result: list[AggregatedMeasurement] = []
    for group in grouped.values():
        first = group[0]
        result.append(
            AggregatedMeasurement(
                kernel=first.kernel,
                shape_hash=first.shape_hash,
                config_hash=first.config_hash,
                config=first.config,
                shape_features=first.shape_features,
                median_ms=statistics.median(item.timing_ms for item in group),
                sample_count=len(group),
                run_ids=tuple(sorted({item.run_id for item in group})),
            )
        )
    return result


def _number(value: object) -> int | float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return value
    return None


def _int_feature(features: dict[str, object], name: str) -> int | None:
    value = _number(features.get(name))
    if value is None:
        return None
    return int(value)


def _bin_le(value: int | None, bins: Sequence[int]) -> str:
    if value is None:
        return "unknown"
    for bound in bins:
        if value <= bound:
            return f"<={bound}"
    return f">{bins[-1]}"


def _geomean(values: Iterable[float]) -> float:
    data = [value for value in values if value > 0 and math.isfinite(value)]
    if not data:
        return math.inf
    return math.exp(sum(math.log(value) for value in data) / len(data))


def _p90(values: Iterable[float]) -> float:
    data = sorted(value for value in values if math.isfinite(value))
    if not data:
        return math.inf
    if len(data) < 10:
        return max(data)
    return statistics.quantiles(data, n=10)[8]


def _dtype_family(features: dict[str, object]) -> str:
    dtype = str(features.get("arg0_dtype", "unknown"))
    if "float8" in dtype:
        return "fp8"
    if "float16" in dtype or "bfloat16" in dtype:
        return "fp16_bf16"
    if "float32" in dtype:
        return "fp32"
    if "int" in dtype:
        return "int"
    return "other"


def _numel_bin(features: dict[str, object]) -> str:
    return _bin_le(
        _int_feature(features, "arg0_numel"),
        [4096, 65536, 1048576, 16777216, 134217728],
    )


def _row_shape(features: dict[str, object]) -> tuple[int | None, int | None]:
    ndim = _int_feature(features, "arg0_ndim")
    if ndim is None or ndim < 2:
        return None, None
    rows = 1
    for dim_idx in range(ndim - 1):
        dim = _int_feature(features, f"arg0_dim{dim_idx}")
        if dim is None:
            return None, _int_feature(features, f"arg0_dim{ndim - 1}")
        rows *= dim
    return rows, _int_feature(features, f"arg0_dim{ndim - 1}")


def _matmul_shape_features(
    features: dict[str, object],
) -> tuple[int | None, int | None, int | None]:
    arg0_ndim = _int_feature(features, "arg0_ndim")
    arg1_ndim = _int_feature(features, "arg1_ndim")
    if arg0_ndim is None or arg1_ndim is None or arg0_ndim < 2 or arg1_ndim < 2:
        return None, None, None
    m = _int_feature(features, f"arg0_dim{arg0_ndim - 2}")
    k = _int_feature(features, f"arg0_dim{arg0_ndim - 1}")
    n = _int_feature(features, f"arg1_dim{arg1_ndim - 1}")
    return m, n, k


def _aspect_bucket(m: int | None, n: int | None, k: int | None) -> str:
    values = [value for value in (m, n, k) if value is not None and value > 0]
    if len(values) != 3:
        return "unknown"
    min_dim = min(values)
    max_dim = max(values)
    if max_dim / min_dim < 4:
        return "balanced"
    if m == min_dim:
        return "skinny_m"
    if n == min_dim:
        return "skinny_n"
    return "skinny_k"


def infer_kernel_class(kernel: str, features: dict[str, object]) -> str:
    """Map measured kernels to reusable workload classes.

    This is intentionally coarser than example names. Runtime classification can
    target the same class labels from source/FX traits and tensor shapes.
    """
    del features
    if kernel == "attention":
        return "attention"
    if kernel == "grouped_gemm":
        return "grouped_matmul"
    if kernel == "fp8_gemm":
        return "matmul_fp8"
    if kernel == "matmul":
        return "matmul"
    if kernel == "matmul_bf16_int4":
        return "matmul_int4"
    if kernel == "_bf16xint16_gemm":
        return "matmul_int16"
    if kernel == "nvfp4_matmul":
        return "matmul_fp4"
    if kernel == "vector_add":
        return "elementwise"
    if kernel == "softmax":
        return "row_softmax"
    if kernel == "cross_entropy":
        return "row_cross_entropy"
    if kernel == "rms_norm":
        return "row_norm_rms"
    if kernel == "layer_norm":
        return "row_norm_layer"
    return "unknown"


def shape_bucket_for_class(
    kernel_class: str, features: dict[str, object]
) -> dict[str, object]:
    """Return a compact shape regime used for cross-shape template selection."""
    dtype_family = _dtype_family(features)
    if kernel_class == "attention":
        seq = _int_feature(features, "arg0_dim2")
        head_dim = _int_feature(features, "arg0_dim3")
        batch = _int_feature(features, "arg0_dim0")
        heads = _int_feature(features, "arg0_dim1")
        batch_heads = None if batch is None or heads is None else batch * heads
        return {
            "seq_bin": _bin_le(seq, [1024, 2048, 4096, 8192, 16384]),
            "head_dim_bin": _bin_le(head_dim, [64, 128, 256]),
            "batch_heads_bin": _bin_le(batch_heads, [32, 64, 128, 256]),
            "dtype": dtype_family,
        }
    if kernel_class in {
        "matmul",
        "matmul_fp8",
        "grouped_matmul",
        "matmul_int4",
        "matmul_int16",
        "matmul_fp4",
    }:
        m, n, k = _matmul_shape_features(features)
        return {
            "m_bin": _bin_le(m, [64, 128, 256, 512, 1024, 4096]),
            "n_bin": _bin_le(n, [64, 128, 256, 512, 1024, 4096]),
            "k_bin": _bin_le(k, [64, 128, 256, 512, 1024, 4096, 32768]),
            "aspect": _aspect_bucket(m, n, k),
            "dtype": dtype_family,
        }
    if kernel_class.startswith("row_"):
        rows, cols = _row_shape(features)
        return {
            "rows_bin": _bin_le(rows, [512, 2048, 4096, 16384, 65536, 262144]),
            "cols_bin": _bin_le(cols, [512, 1024, 2048, 4096, 8192, 16384, 32768]),
            "dtype": dtype_family,
        }
    if kernel_class == "elementwise":
        return {"numel_bin": _numel_bin(features), "dtype": dtype_family}
    return {"dtype": dtype_family}


def _bucket_key(kernel_class: str, bucket: dict[str, object]) -> str:
    return f"{kernel_class}:{_stable_json(bucket)}"


def _shape_label(kernel: str, features: dict[str, object]) -> str:
    if kernel in {
        "matmul",
        "fp8_gemm",
        "matmul_bf16_int4",
        "_bf16xint16_gemm",
        "nvfp4_matmul",
    }:
        m = _int_feature(features, "arg0_dim0")
        k = _int_feature(features, "arg0_dim1")
        n = _int_feature(features, "arg1_dim1")
        return f"M={m},N={n},K={k},size={_bin_le(max(v for v in (m, n, k) if v is not None), [1024, 4096, 8192])}"

    if kernel == "attention":
        batch = _int_feature(features, "arg0_dim0")
        heads = _int_feature(features, "arg0_dim1")
        seq = _int_feature(features, "arg0_dim2")
        head_dim = _int_feature(features, "arg0_dim3")
        return (
            f"B={batch},H={heads},seq={seq},head_dim={head_dim},"
            f"seq_bin={_bin_le(seq, [1024, 2048, 4096, 8192, 16384])}"
        )

    if kernel == "vector_add":
        n = _int_feature(features, "arg0_dim0")
        return f"N={n},N_bin={_bin_le(n, [262144, 1048576, 16777216, 134217728])}"

    if kernel in {"softmax", "layer_norm", "rms_norm", "cross_entropy"}:
        rows = _int_feature(features, "arg0_dim0")
        cols = _int_feature(features, "arg0_dim1")
        return (
            f"rows={rows},cols={cols},"
            f"cols_bin={_bin_le(cols, [512, 1024, 2048, 4096, 8192, 16384, 32768])}"
        )

    dims = {
        key: value
        for key, value in sorted(features.items())
        if "_dim" in key and isinstance(value, int)
    }
    return ",".join(f"{key}={value}" for key, value in list(dims.items())[:6])


def _config_compact(config: dict[str, object]) -> dict[str, object]:
    return {key: config[key] for key in KEY_CONFIG_FIELDS if key in config}


def _template_compact(config: dict[str, object]) -> dict[str, object]:
    return {key: config[key] for key in GENERAL_TEMPLATE_FIELDS if key in config}


def _config_family(config: dict[str, object]) -> str:
    compact = _config_compact(config)
    family_keys = ("block_sizes", "num_warps", "num_stages", "pid_type")
    return _stable_json({key: compact[key] for key in family_keys if key in compact})


def _best_by_shape(
    rows: Sequence[AggregatedMeasurement],
) -> dict[tuple[str, str], AggregatedMeasurement]:
    best: dict[tuple[str, str], AggregatedMeasurement] = {}
    for row in rows:
        key = (row.kernel, row.shape_hash)
        old = best.get(key)
        if old is None or row.median_ms < old.median_ms:
            best[key] = row
    return best


def _slowdowns_by_row(
    rows: Sequence[AggregatedMeasurement],
    best: dict[tuple[str, str], AggregatedMeasurement],
) -> dict[tuple[str, str, str], float]:
    slowdowns: dict[tuple[str, str, str], float] = {}
    for row in rows:
        best_ms = best[(row.kernel, row.shape_hash)].median_ms
        if best_ms <= 0:
            continue
        slowdowns[(row.kernel, row.shape_hash, row.config_hash)] = (
            row.median_ms / best_ms
        )
    return slowdowns


def _median(values: Iterable[float]) -> float:
    data = list(values)
    if not data:
        return math.inf
    return statistics.median(data)


def _top_config_summaries(
    kernel: str,
    rows: Sequence[AggregatedMeasurement],
    best: dict[tuple[str, str], AggregatedMeasurement],
    slowdowns: dict[tuple[str, str, str], float],
    *,
    limit: int,
) -> list[dict[str, object]]:
    rows_by_config: dict[str, list[AggregatedMeasurement]] = collections.defaultdict(
        list
    )
    for row in rows:
        if row.kernel == kernel:
            rows_by_config[row.config_hash].append(row)

    winners = collections.Counter(
        row.config_hash
        for (row_kernel, _shape), row in best.items()
        if row_kernel == kernel
    )

    summaries: list[dict[str, object]] = []
    for config_hash, config_rows in rows_by_config.items():
        config = config_rows[0].config
        config_slowdowns = [
            slowdowns[(row.kernel, row.shape_hash, row.config_hash)]
            for row in config_rows
            if (row.kernel, row.shape_hash, row.config_hash) in slowdowns
        ]
        winner_rows = [
            row
            for row in config_rows
            if best[(row.kernel, row.shape_hash)].config_hash == config_hash
        ]
        summaries.append(
            {
                "config_hash": config_hash,
                "shape_coverage": len({row.shape_hash for row in config_rows}),
                "win_count": winners[config_hash],
                "median_slowdown": round(_median(config_slowdowns), 4),
                "p90_slowdown": round(
                    statistics.quantiles(config_slowdowns, n=10)[8]
                    if len(config_slowdowns) >= 10
                    else max(config_slowdowns, default=math.inf),
                    4,
                ),
                "compact_config": _config_compact(config),
                "winning_shape_examples": [
                    _shape_label(kernel, row.shape_features) for row in winner_rows[:3]
                ],
            }
        )
    summaries.sort(
        key=lambda item: (
            -int(item["win_count"]),
            float(item["median_slowdown"]),
            -int(item["shape_coverage"]),
        )
    )
    return summaries[:limit]


def _regime_summaries(
    kernel: str,
    rows: Sequence[AggregatedMeasurement],
    best: dict[tuple[str, str], AggregatedMeasurement],
    *,
    limit: int,
) -> list[dict[str, object]]:
    shape_to_regime: dict[str, str] = {}
    for row in rows:
        if row.kernel == kernel and row.shape_hash not in shape_to_regime:
            shape_to_regime[row.shape_hash] = _shape_label(kernel, row.shape_features)

    configs_by_regime: dict[str, collections.Counter[str]] = collections.defaultdict(
        collections.Counter
    )
    config_by_hash: dict[str, dict[str, object]] = {}
    for (row_kernel, shape_hash), row in best.items():
        if row_kernel != kernel:
            continue
        regime = shape_to_regime.get(shape_hash, "unknown")
        configs_by_regime[regime][row.config_hash] += 1
        config_by_hash[row.config_hash] = row.config

    summaries: list[dict[str, object]] = []
    for regime, counter in configs_by_regime.items():
        top = counter.most_common(3)
        summaries.append(
            {
                "regime": regime,
                "shape_count": sum(counter.values()),
                "top_winners": [
                    {
                        "config_hash": config_hash,
                        "wins": wins,
                        "family": _config_family(config_by_hash[config_hash]),
                        "compact_config": _config_compact(config_by_hash[config_hash]),
                    }
                    for config_hash, wins in top
                ],
            }
        )
    summaries.sort(key=lambda item: (-int(item["shape_count"]), str(item["regime"])))
    return summaries[:limit]


def summarize_measurements(
    rows: Sequence[AggregatedMeasurement],
    *,
    top_configs_per_kernel: int,
    regimes_per_kernel: int,
) -> dict[str, object]:
    """Build a compact, JSON-serializable heuristic summary."""
    best = _best_by_shape(rows)
    slowdowns = _slowdowns_by_row(rows, best)
    kernels = sorted({row.kernel for row in rows})
    summary: dict[str, object] = {
        "schema_version": 1,
        "aggregation": "median timing per (kernel, shape_hash, config_hash, config)",
        "kernels": {},
    }
    kernels_out = summary["kernels"]
    assert isinstance(kernels_out, dict)
    for kernel in kernels:
        kernel_rows = [row for row in rows if row.kernel == kernel]
        shape_hashes = {row.shape_hash for row in kernel_rows}
        config_hashes = {row.config_hash for row in kernel_rows}
        kernels_out[kernel] = {
            "aggregated_rows": len(kernel_rows),
            "raw_samples": sum(row.sample_count for row in kernel_rows),
            "shapes": len(shape_hashes),
            "configs": len(config_hashes),
            "winning_configs": _top_config_summaries(
                kernel,
                kernel_rows,
                best,
                slowdowns,
                limit=top_configs_per_kernel,
            ),
            "shape_regimes": _regime_summaries(
                kernel,
                kernel_rows,
                best,
                limit=regimes_per_kernel,
            ),
        }
    return summary


def _template_key_from_config(config: dict[str, object]) -> str:
    return _stable_json(_template_compact(config))


def _shape_best_ms(
    rows: Sequence[AggregatedMeasurement],
) -> dict[tuple[str, str], float]:
    return {
        key: row.median_ms
        for key, row in _best_by_shape(rows).items()
        if row.median_ms > 0
    }


def _template_slowdowns_by_shape(
    bucket_rows: Sequence[AggregatedMeasurement],
    best_ms_by_shape: dict[tuple[str, str], float],
) -> dict[str, dict[str, float]]:
    timings: dict[str, dict[str, float]] = collections.defaultdict(dict)
    for row in bucket_rows:
        best_ms = best_ms_by_shape.get((row.kernel, row.shape_hash))
        if best_ms is None or best_ms <= 0:
            continue
        shape_id = f"{row.kernel}:{row.shape_hash}"
        template_key = _template_key_from_config(row.config)
        old = timings[template_key].get(shape_id)
        if old is None or row.median_ms < old:
            timings[template_key][shape_id] = row.median_ms / best_ms
    return dict(timings)


def _template_summary(
    template_key: str,
    shape_slowdowns: dict[str, float],
    *,
    winning_template_shapes: set[str],
) -> dict[str, object]:
    slowdowns = list(shape_slowdowns.values())
    return {
        "template": json.loads(template_key),
        "shape_coverage": len(shape_slowdowns),
        "win_count": len(winning_template_shapes & set(shape_slowdowns)),
        "geomean_slowdown": round(_geomean(slowdowns), 4),
        "median_slowdown": round(_median(slowdowns), 4),
        "p90_slowdown": round(_p90(slowdowns), 4),
    }


def _select_templates_greedy(
    template_slowdowns: dict[str, dict[str, float]],
    *,
    all_shapes: set[str],
    limit: int,
) -> tuple[list[str], dict[str, object]]:
    selected: list[str] = []
    current = dict.fromkeys(all_shapes, math.inf)
    remaining = set(template_slowdowns)

    def objective(shape_costs: dict[str, float]) -> tuple[float, float, int]:
        covered = [value for value in shape_costs.values() if math.isfinite(value)]
        if not covered:
            return math.inf, math.inf, 0
        coverage_ratio = len(covered) / max(1, len(shape_costs))
        coverage_penalty = coverage_ratio**-0.25
        return _geomean(covered) * coverage_penalty, _p90(covered), -len(covered)

    current_score = objective(current)
    for _ in range(limit):
        best_key: str | None = None
        best_candidate_costs: dict[str, float] | None = None
        best_score: tuple[float, float, int] | None = None
        for template_key in sorted(remaining):
            candidate = dict(current)
            for shape_hash, slowdown in template_slowdowns[template_key].items():
                candidate[shape_hash] = min(candidate[shape_hash], slowdown)
            score = objective(candidate)
            if best_score is None or score < best_score:
                best_score = score
                best_key = template_key
                best_candidate_costs = candidate
        if best_key is None or best_candidate_costs is None:
            break
        assert best_score is not None
        if best_score >= current_score:
            break
        selected.append(best_key)
        remaining.remove(best_key)
        current = best_candidate_costs
        current_score = best_score

    covered = [value for value in current.values() if math.isfinite(value)]
    oracle = {
        "shape_coverage": len(covered),
        "shape_total": len(all_shapes),
        "coverage_ratio": round(len(covered) / max(1, len(all_shapes)), 4),
        "geomean_slowdown": round(_geomean(covered), 4),
        "median_slowdown": round(_median(covered), 4),
        "p90_slowdown": round(_p90(covered), 4),
    }
    return selected, oracle


def _cross_validate_templates(
    bucket_measurements: Sequence[AggregatedMeasurement],
    best_ms_by_shape: dict[tuple[str, str], float],
    *,
    limit: int,
) -> dict[str, object]:
    shapes = sorted({f"{row.kernel}:{row.shape_hash}" for row in bucket_measurements})
    if len(shapes) < 2:
        return {
            "shape_coverage": 0,
            "shape_total": len(shapes),
            "coverage_ratio": 0.0,
            "geomean_slowdown": math.inf,
            "median_slowdown": math.inf,
            "p90_slowdown": math.inf,
        }

    full_template_slowdowns = _template_slowdowns_by_shape(
        bucket_measurements, best_ms_by_shape
    )
    holdout_slowdowns: list[float] = []
    for holdout_shape in shapes:
        train_rows = [
            row
            for row in bucket_measurements
            if f"{row.kernel}:{row.shape_hash}" != holdout_shape
        ]
        train_shapes = {f"{row.kernel}:{row.shape_hash}" for row in train_rows}
        train_template_slowdowns = _template_slowdowns_by_shape(
            train_rows, best_ms_by_shape
        )
        selected, _oracle = _select_templates_greedy(
            train_template_slowdowns,
            all_shapes=train_shapes,
            limit=limit,
        )
        available = [
            full_template_slowdowns[template_key][holdout_shape]
            for template_key in selected
            if holdout_shape in full_template_slowdowns.get(template_key, {})
        ]
        if available:
            holdout_slowdowns.append(min(available))

    return {
        "shape_coverage": len(holdout_slowdowns),
        "shape_total": len(shapes),
        "coverage_ratio": round(len(holdout_slowdowns) / max(1, len(shapes)), 4),
        "geomean_slowdown": round(_geomean(holdout_slowdowns), 4),
        "median_slowdown": round(_median(holdout_slowdowns), 4),
        "p90_slowdown": round(_p90(holdout_slowdowns), 4),
    }


def derive_general_rules(
    rows: Sequence[AggregatedMeasurement],
    *,
    templates_per_rule: int = RULE_TEMPLATE_LIMIT,
) -> dict[str, object]:
    """Derive general shape-bucket rules from measured config families."""
    best = _best_by_shape(rows)
    best_ms_by_shape = _shape_best_ms(rows)
    bucket_rows: dict[str, list[AggregatedMeasurement]] = collections.defaultdict(list)
    bucket_meta: dict[str, dict[str, object]] = {}

    for row in rows:
        kernel_class = infer_kernel_class(row.kernel, row.shape_features)
        bucket = shape_bucket_for_class(kernel_class, row.shape_features)
        key = _bucket_key(kernel_class, bucket)
        bucket_rows[key].append(row)
        bucket_meta[key] = {
            "kernel_class": kernel_class,
            "shape_bucket": bucket,
            "source_kernels": sorted(
                {
                    *(
                        set(bucket_meta.get(key, {}).get("source_kernels", []))
                        if key in bucket_meta
                        else set()
                    ),
                    row.kernel,
                }
            ),
        }

    rules: list[dict[str, object]] = []
    class_totals: dict[str, dict[str, object]] = collections.defaultdict(
        lambda: {
            "shape_count": 0,
            "rule_count": 0,
            "raw_rows": 0,
            "oracle_geomean_slowdowns": [],
        }
    )
    for key, bucket_measurements in sorted(bucket_rows.items()):
        all_shapes = {f"{row.kernel}:{row.shape_hash}" for row in bucket_measurements}
        if not all_shapes:
            continue
        template_slowdowns = _template_slowdowns_by_shape(
            bucket_measurements, best_ms_by_shape
        )
        if not template_slowdowns:
            continue
        winning_template_shapes: dict[str, set[str]] = collections.defaultdict(set)
        for row in bucket_measurements:
            best_row = best.get((row.kernel, row.shape_hash))
            if best_row is None:
                continue
            winning_template_shapes[_template_key_from_config(best_row.config)].add(
                f"{row.kernel}:{row.shape_hash}"
            )
        selected, oracle = _select_templates_greedy(
            template_slowdowns,
            all_shapes=all_shapes,
            limit=templates_per_rule,
        )
        holdout = _cross_validate_templates(
            bucket_measurements,
            best_ms_by_shape,
            limit=templates_per_rule,
        )
        selected_summaries = [
            _template_summary(
                template_key,
                template_slowdowns[template_key],
                winning_template_shapes=winning_template_shapes.get(
                    template_key, set()
                ),
            )
            for template_key in selected
        ]
        meta = bucket_meta[key]
        kernel_class = str(meta["kernel_class"])
        class_total = class_totals[kernel_class]
        class_total["shape_count"] = int(class_total["shape_count"]) + len(all_shapes)
        class_total["rule_count"] = int(class_total["rule_count"]) + 1
        class_total["raw_rows"] = int(class_total["raw_rows"]) + len(
            bucket_measurements
        )
        cast_list = class_total["oracle_geomean_slowdowns"]
        assert isinstance(cast_list, list)
        if math.isfinite(float(oracle["geomean_slowdown"])):
            cast_list.append(float(oracle["geomean_slowdown"]))
        holdout_list = class_total.setdefault("holdout_geomean_slowdowns", [])
        assert isinstance(holdout_list, list)
        if math.isfinite(float(holdout["geomean_slowdown"])):
            holdout_list.append(float(holdout["geomean_slowdown"]))

        rules.append(
            {
                "kernel_class": kernel_class,
                "shape_bucket": meta["shape_bucket"],
                "source_kernels": meta["source_kernels"],
                "shape_count": len(all_shapes),
                "aggregated_rows": len(bucket_measurements),
                "selected_templates": selected_summaries,
                "selected_oracle": oracle,
                "leave_one_shape_out": holdout,
            }
        )

    rules.sort(
        key=lambda item: (
            str(item["kernel_class"]),
            -int(item["shape_count"]),
            _stable_json(item["shape_bucket"]),
        )
    )
    class_summary: dict[str, object] = {}
    for kernel_class, total in sorted(class_totals.items()):
        slowdowns = total.pop("oracle_geomean_slowdowns")
        assert isinstance(slowdowns, list)
        holdout_slowdowns = total.pop("holdout_geomean_slowdowns", [])
        assert isinstance(holdout_slowdowns, list)
        class_summary[kernel_class] = {
            **total,
            "rule_geomean_slowdown_geomean": round(_geomean(slowdowns), 4),
            "rule_geomean_slowdown_p90": round(_p90(slowdowns), 4),
            "holdout_geomean_slowdown_geomean": round(_geomean(holdout_slowdowns), 4),
            "holdout_geomean_slowdown_p90": round(_p90(holdout_slowdowns), 4),
        }
    return {
        "schema_version": 1,
        "selection": (
            "Greedy structural template families per kernel_class+shape_bucket. "
            "Templates intentionally omit noisy indexing/eviction fields."
        ),
        "template_fields": list(GENERAL_TEMPLATE_FIELDS),
        "class_summary": class_summary,
        "rules": rules,
    }


def render_general_rules_report(general_rules: dict[str, object]) -> str:
    """Render the derived general rule table in Markdown."""
    class_summary = general_rules["class_summary"]
    assert isinstance(class_summary, dict)
    overview_rows: list[list[object]] = []
    for kernel_class, raw_summary in sorted(class_summary.items()):
        summary = raw_summary
        assert isinstance(summary, dict)
        overview_rows.append(
            [
                f"`{kernel_class}`",
                summary["rule_count"],
                summary["shape_count"],
                summary["raw_rows"],
                summary["rule_geomean_slowdown_geomean"],
                summary["rule_geomean_slowdown_p90"],
                summary["holdout_geomean_slowdown_geomean"],
                summary["holdout_geomean_slowdown_p90"],
            ]
        )

    parts = [
        "# Data-Derived General Heuristics",
        "",
        (
            "These rules are selected from structural config families, not exact "
            "AOT config hashes. Lower slowdown is better; 1.0 means the selected "
            "template family contains the measured winner for that shape bucket."
        ),
        "",
        _markdown_table(
            [
                "Kernel class",
                "Rules",
                "Shapes",
                "Rows",
                "Geo slowdown",
                "P90 slowdown",
                "Holdout geo",
                "Holdout p90",
            ],
            overview_rows,
        ),
    ]

    rules = general_rules["rules"]
    assert isinstance(rules, list)
    for rule in rules:
        assert isinstance(rule, dict)
        oracle = rule["selected_oracle"]
        assert isinstance(oracle, dict)
        holdout = rule["leave_one_shape_out"]
        assert isinstance(holdout, dict)
        templates = rule["selected_templates"]
        assert isinstance(templates, list)
        parts.extend(
            [
                "",
                f"## `{rule['kernel_class']}` `{_stable_json(rule['shape_bucket'])}`",
                "",
                (
                    f"Shapes: {rule['shape_count']}; rows: {rule['aggregated_rows']}; "
                    f"oracle geo slowdown: {oracle['geomean_slowdown']}; "
                    f"p90: {oracle['p90_slowdown']}; "
                    f"coverage: {oracle['shape_coverage']}/{oracle['shape_total']}; "
                    f"holdout geo slowdown: {holdout['geomean_slowdown']}; "
                    f"holdout p90: {holdout['p90_slowdown']}; "
                    f"holdout coverage: {holdout['shape_coverage']}/{holdout['shape_total']}"
                ),
            ]
        )
        template_rows: list[list[object]] = []
        for index, template in enumerate(templates, start=1):
            assert isinstance(template, dict)
            template_rows.append(
                [
                    index,
                    template["shape_coverage"],
                    template["win_count"],
                    template["geomean_slowdown"],
                    template["p90_slowdown"],
                    "`" + _stable_json(template["template"]) + "`",
                ]
            )
        parts.append(
            _markdown_table(
                ["#", "Covered", "Wins", "Geo", "P90", "Template"],
                template_rows,
            )
        )
    return "\n".join(parts) + "\n"


def _passes_runtime_rule_filter(rule: dict[str, object]) -> bool:
    holdout = rule["leave_one_shape_out"]
    assert isinstance(holdout, dict)
    return (
        int(rule["shape_count"]) >= RUNTIME_HEURISTIC_FILTERS["min_rule_shapes"]
        and float(holdout["coverage_ratio"])
        >= RUNTIME_HEURISTIC_FILTERS["min_holdout_coverage"]
        and float(holdout["geomean_slowdown"])
        <= RUNTIME_HEURISTIC_FILTERS["max_rule_holdout_geomean_slowdown"]
        and float(holdout["p90_slowdown"])
        <= RUNTIME_HEURISTIC_FILTERS["max_rule_holdout_p90_slowdown"]
    )


def _passes_runtime_template_filter(template: dict[str, object]) -> bool:
    return (
        int(template["shape_coverage"])
        >= RUNTIME_HEURISTIC_FILTERS["min_template_shapes"]
        and float(template["geomean_slowdown"])
        <= RUNTIME_HEURISTIC_FILTERS["max_template_geomean_slowdown"]
        and float(template["p90_slowdown"])
        <= RUNTIME_HEURISTIC_FILTERS["max_template_p90_slowdown"]
    )


def derive_runtime_heuristics(general_rules: dict[str, object]) -> dict[str, object]:
    """Filter data-derived structural rules down to runtime seed heuristics."""
    source_rules = general_rules["rules"]
    assert isinstance(source_rules, list)
    rules: list[dict[str, object]] = []
    class_totals: dict[str, dict[str, object]] = collections.defaultdict(
        lambda: {
            "rule_count": 0,
            "shape_count": 0,
            "template_count": 0,
            "holdout_geomean_slowdowns": [],
            "holdout_p90_slowdowns": [],
        }
    )

    for raw_rule in source_rules:
        assert isinstance(raw_rule, dict)
        if not _passes_runtime_rule_filter(raw_rule):
            continue
        selected_templates = raw_rule["selected_templates"]
        assert isinstance(selected_templates, list)
        templates: list[dict[str, object]] = []
        for raw_template in selected_templates:
            assert isinstance(raw_template, dict)
            if not _passes_runtime_template_filter(raw_template):
                continue
            templates.append(
                {
                    "template": raw_template["template"],
                    "shape_coverage": raw_template["shape_coverage"],
                    "win_count": raw_template["win_count"],
                    "geomean_slowdown": raw_template["geomean_slowdown"],
                    "p90_slowdown": raw_template["p90_slowdown"],
                }
            )
        if not templates:
            continue

        holdout = raw_rule["leave_one_shape_out"]
        assert isinstance(holdout, dict)
        kernel_class = str(raw_rule["kernel_class"])
        rules.append(
            {
                "kernel_class": kernel_class,
                "shape_bucket": raw_rule["shape_bucket"],
                "source_kernels": raw_rule["source_kernels"],
                "shape_count": raw_rule["shape_count"],
                "selected_oracle": raw_rule["selected_oracle"],
                "leave_one_shape_out": holdout,
                "templates": templates,
            }
        )

        total = class_totals[kernel_class]
        total["rule_count"] = int(total["rule_count"]) + 1
        total["shape_count"] = int(total["shape_count"]) + int(raw_rule["shape_count"])
        total["template_count"] = int(total["template_count"]) + len(templates)
        holdout_geos = total["holdout_geomean_slowdowns"]
        holdout_p90s = total["holdout_p90_slowdowns"]
        assert isinstance(holdout_geos, list)
        assert isinstance(holdout_p90s, list)
        holdout_geos.append(float(holdout["geomean_slowdown"]))
        holdout_p90s.append(float(holdout["p90_slowdown"]))

    class_summary: dict[str, object] = {}
    for kernel_class, total in sorted(class_totals.items()):
        holdout_geos = total["holdout_geomean_slowdowns"]
        holdout_p90s = total["holdout_p90_slowdowns"]
        assert isinstance(holdout_geos, list)
        assert isinstance(holdout_p90s, list)
        class_summary[kernel_class] = {
            "rule_count": total["rule_count"],
            "shape_count": total["shape_count"],
            "template_count": total["template_count"],
            "holdout_geomean_slowdown_geomean": round(_geomean(holdout_geos), 4),
            "holdout_p90_slowdown_max": round(max(holdout_p90s, default=math.inf), 4),
        }

    return {
        "schema_version": 1,
        "source": "B200 AOT measurement CSVs",
        "selection": (
            "Runtime seed rules filtered from data-derived structural templates. "
            "Rules are keyed by inferred kernel class and compact shape bucket; "
            "templates intentionally omit noisy indexing and eviction-policy fields."
        ),
        "filters": RUNTIME_HEURISTIC_FILTERS,
        "template_fields": list(GENERAL_TEMPLATE_FIELDS),
        "class_summary": class_summary,
        "rules": rules,
    }


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def render_report(summary: dict[str, object]) -> str:
    """Render a reviewable Markdown report from the compact summary."""
    kernels = summary["kernels"]
    assert isinstance(kernels, dict)
    overview_rows: list[list[object]] = []
    for kernel, raw_data in sorted(kernels.items()):
        data = raw_data
        assert isinstance(data, dict)
        overview_rows.append(
            [
                f"`{kernel}`",
                data["raw_samples"],
                data["aggregated_rows"],
                data["shapes"],
                data["configs"],
            ]
        )

    parts = [
        "# LLM Autotune Heuristics Research",
        "",
        "## Dataset",
        "",
        _markdown_table(
            ["Kernel", "Raw samples", "Aggregated rows", "Shapes", "Configs"],
            overview_rows,
        ),
    ]

    for kernel, raw_data in sorted(kernels.items()):
        data = raw_data
        assert isinstance(data, dict)
        parts.extend(["", f"## `{kernel}`", ""])
        winners = data["winning_configs"]
        assert isinstance(winners, list)
        winner_rows: list[list[object]] = []
        for index, item in enumerate(winners[:5], start=1):
            assert isinstance(item, dict)
            winner_rows.append(
                [
                    index,
                    item["win_count"],
                    item["shape_coverage"],
                    item["median_slowdown"],
                    "`" + str(item["config_hash"]) + "`",
                    "`" + _stable_json(item["compact_config"]) + "`",
                ]
            )
        parts.append(
            _markdown_table(
                [
                    "#",
                    "Wins",
                    "Covered shapes",
                    "Median slowdown",
                    "Config hash",
                    "Compact config",
                ],
                winner_rows,
            )
        )

        regimes = data["shape_regimes"]
        assert isinstance(regimes, list)
        if regimes:
            parts.extend(["", "Top shape regimes:"])
            for regime in regimes[:5]:
                assert isinstance(regime, dict)
                top = regime["top_winners"]
                assert isinstance(top, list)
                top_text = "; ".join(
                    f"{winner['config_hash']} wins={winner['wins']}"
                    for winner in top
                    if isinstance(winner, dict)
                )
                parts.append(
                    f"- `{regime['regime']}`: {regime['shape_count']} shapes; {top_text}"
                )

    return "\n".join(parts) + "\n"


def render_opus_prompt(
    summary: dict[str, object],
    report: str,
    general_rules: dict[str, object],
    general_rules_report: str,
) -> str:
    """Build a compact research prompt for Opus 4.7."""
    compact_json = json.dumps(summary, sort_keys=True, separators=(",", ":"))
    if len(compact_json) > 55000:
        compact_json = compact_json[:55000] + "...<truncated>"
    compact_rules = json.dumps(general_rules, sort_keys=True, separators=(",", ":"))
    if len(compact_rules) > 55000:
        compact_rules = compact_rules[:55000] + "...<truncated>"
    return (
        "You are helping improve Helion's LLM-guided GPU-kernel autotuner. "
        "Use the measured B200 AOT data below to propose concrete prompt and "
        "seed-config heuristics that improve best-found kernel performance "
        "with fewer compiled configs. Focus on reusable workload traits, not "
        "hard-coded tutorial names. Prefer the data-derived structural-template "
        "rules over exact config hashes when they disagree.\n\n"
        "Return concise Markdown with these sections:\n"
        "1. Prompt rules to add or replace.\n"
        "2. Seed config templates by workload trait/shape regime, using only "
        "fields that appear stable in the structural-template analysis.\n"
        "3. Bad patterns to steer away from.\n"
        "4. Evaluation plan and success metrics.\n\n"
        "Data-derived structural rule report:\n"
        f"{general_rules_report[:24000]}\n\n"
        "Data-derived structural rules JSON:\n"
        f"{compact_rules}\n\n"
        "Markdown report:\n"
        f"{report[:24000]}\n\n"
        "Compact JSON summary:\n"
        f"{compact_json}\n"
    )


def write_artifacts(
    output_dir: Path,
    summary: dict[str, object],
    report: str,
    general_rules: dict[str, object],
    general_rules_report: str,
    runtime_heuristics: dict[str, object],
    opus_prompt: str,
    runtime_heuristics_path: Path | None,
) -> None:
    """Write JSON and Markdown artifacts for review."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "llm_heuristics_configs.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "llm_heuristics_report.md").write_text(report)
    (output_dir / "derived_general_heuristics.json").write_text(
        json.dumps(general_rules, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "derived_general_heuristics.md").write_text(general_rules_report)
    runtime_text = json.dumps(runtime_heuristics, indent=2, sort_keys=True) + "\n"
    (output_dir / "runtime_observed_heuristics_b200.json").write_text(runtime_text)
    if runtime_heuristics_path is not None:
        runtime_heuristics_path.parent.mkdir(parents=True, exist_ok=True)
        runtime_heuristics_path.write_text(runtime_text)
    (output_dir / "opus_prompt.md").write_text(opus_prompt)


def call_opus(
    *,
    model: str,
    prompt: str,
    output_dir: Path,
    request_timeout_s: float,
) -> None:
    """Ask Opus to critique the generated heuristic summary."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from helion.autotuner.llm.transport import call_provider
    from helion.autotuner.llm.transport import infer_provider

    provider = infer_provider(model)
    response = call_provider(
        provider,
        model=model,
        api_base=None,
        api_key=None,
        messages=[{"role": "user", "content": prompt}],
        max_output_tokens=4096,
        request_timeout_s=request_timeout_s,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "opus_response.md").write_text(response)


def call_opus_cli(
    *,
    model: str,
    prompt: str,
    output_dir: Path,
    request_timeout_s: float,
    max_budget_usd: float,
) -> None:
    """Ask Opus through the local Claude Code CLI / AI Gateway wrapper."""
    if shutil.which("claude") is None:
        raise RuntimeError("claude CLI was not found on PATH")

    completed = subprocess.run(
        [
            "claude",
            "-p",
            "--model",
            model,
            "--tools",
            "",
            "--max-budget-usd",
            str(max_budget_usd),
        ],
        input=prompt,
        text=True,
        capture_output=True,
        timeout=request_timeout_s,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "opus_response.md").write_text(completed.stdout)


def has_opus_credentials() -> bool:
    """Return whether the Anthropic transport has credentials available."""
    return any(
        os.environ.get(name) for name in ("HELION_LLM_API_KEY", "ANTHROPIC_API_KEY")
    )


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-configs-per-kernel", type=int, default=8)
    parser.add_argument("--regimes-per-kernel", type=int, default=12)
    parser.add_argument("--call-llm", action="store_true")
    parser.add_argument("--call-claude-cli", action="store_true")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--request-timeout-s", type=float, default=120.0)
    parser.add_argument("--claude-budget-usd", type=float, default=5.0)
    parser.add_argument(
        "--runtime-heuristics-path",
        type=Path,
        default=None,
        help="Optional checked-in JSON path for filtered runtime heuristics.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    measurements = load_measurements(args.data_root)
    if not measurements:
        print(f"No measurement rows found under {args.data_root}", file=sys.stderr)
        return 1

    aggregated = aggregate_measurements(measurements)
    summary = summarize_measurements(
        aggregated,
        top_configs_per_kernel=args.top_configs_per_kernel,
        regimes_per_kernel=args.regimes_per_kernel,
    )
    report = render_report(summary)
    general_rules = derive_general_rules(aggregated)
    general_rules_report = render_general_rules_report(general_rules)
    runtime_heuristics = derive_runtime_heuristics(general_rules)
    opus_prompt = render_opus_prompt(
        summary, report, general_rules, general_rules_report
    )
    write_artifacts(
        args.output_dir,
        summary,
        report,
        general_rules,
        general_rules_report,
        runtime_heuristics,
        opus_prompt,
        args.runtime_heuristics_path,
    )

    print(f"Loaded {len(measurements)} raw timing rows")
    print(f"Aggregated to {len(aggregated)} kernel/shape/config rows")
    print(f"Wrote artifacts to {args.output_dir}")

    if args.call_llm:
        if not has_opus_credentials():
            print(
                "Cannot call Opus: set HELION_LLM_API_KEY or ANTHROPIC_API_KEY.",
                file=sys.stderr,
            )
            return 2
        call_opus(
            model=args.model,
            prompt=opus_prompt,
            output_dir=args.output_dir,
            request_timeout_s=args.request_timeout_s,
        )
        print(f"Wrote Opus response to {args.output_dir / 'opus_response.md'}")
    if args.call_claude_cli:
        call_opus_cli(
            model=args.model,
            prompt=opus_prompt,
            output_dir=args.output_dir,
            request_timeout_s=args.request_timeout_s,
            max_budget_usd=args.claude_budget_usd,
        )
        print(f"Wrote Opus response to {args.output_dir / 'opus_response.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
