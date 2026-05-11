"""Generate a ``fallbacks`` section for observed_heuristics_b200.json.

Fallbacks are per-kernel-class, per-shape-group template configs used
when exact-bucket rule lookup misses. For each group defined by
``helion.autotuner.observed_heuristics._fallback_group_for_class``,
this script picks the archive shape with the median best-perf in that
group and emits its winning config as that group's fallback.

Usage::

    python scripts/observed_heuristics_fallbacks.py \
        --archive-root aot_pretune_data/b200 \
        --kernel-class matmul_int4 --kernel matmul_bf16_int4 \
        --output /tmp/fallbacks_int4.json

The output JSON is a map ``{group: {template: {...}}}`` matching the
on-disk schema. Merge multiple invocations (one per kernel_class) into
the top-level ``fallbacks`` block of ``observed_heuristics_b200.json``.

Uses only the measurement CSVs — no GPU required.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from collections import defaultdict


def _best_per_shape(csv_path: Path) -> dict[tuple[int, int, int], tuple[float, dict]]:
    """Walk a measurements CSV; for each (M, K, N), return (min_perf, config)."""
    best: dict[tuple[int, int, int], tuple[float, dict]] = {}
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            features = json.loads(row["shape_features"])
            m = features.get("arg0_dim0")
            k = features.get("arg0_dim1")
            n = features.get("arg1_dim1")
            if m is None or k is None or n is None:
                continue
            shape = (int(m), int(k), int(n))
            perf = float(row["timing_ms"])
            cur = best.get(shape)
            if cur is None or perf < cur[0]:
                best[shape] = (perf, json.loads(row["config"]))
    return best


def _matmul_group(m: int, n: int, k: int) -> str:
    """Same grouping as observed_heuristics._fallback_group_for_class."""
    if m <= 256:
        return "small_m"
    if n <= 256:
        return "small_n"
    if k <= 256:
        return "small_k"
    dims = [m, n, k]
    if max(dims) / max(1, min(dims)) < 2:
        return "balanced"
    return "rect"


def _row_group(rows: int, cols: int) -> str:
    if rows <= 512:
        return "short"
    if cols <= 1024:
        return "narrow"
    if cols >= 8192:
        return "wide"
    return "square"


def _elementwise_group(numel: int) -> str:
    if numel <= 65536:
        return "tiny"
    if numel <= 1048576:
        return "mid"
    return "huge"


def _attention_group(batch_heads: int, q_seq: int, head_dim: int) -> str:
    if q_seq <= 1024:
        return "short_seq"
    if q_seq >= 8192:
        return "long_seq"
    if head_dim <= 64:
        return "small_head"
    return "mid_seq"


_GROUPERS: dict[str, callable] = {
    "matmul": _matmul_group,
    "matmul_fp8": _matmul_group,
    "matmul_int4": _matmul_group,
    "matmul_int16": _matmul_group,
    "matmul_fp4": _matmul_group,
    "grouped_matmul": _matmul_group,
    "row_softmax": _row_group,
    "row_norm_rms": _row_group,
    "row_norm_layer": _row_group,
    "row_cross_entropy": _row_group,
    "elementwise": _elementwise_group,
    "attention": _attention_group,
}


def _shape_to_group(kernel_class: str, shape: tuple[int, ...]) -> str | None:
    grouper = _GROUPERS.get(kernel_class)
    if grouper is None:
        return None
    if kernel_class.startswith("matmul") or kernel_class == "grouped_matmul":
        m, k, n = shape
        return grouper(m, n, k)
    # Other kernel classes will need their own shape unpacking here; the
    # quantized-GEMM loop only exercises the matmul grouper today.
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-root", type=Path, required=True,
                        help="e.g. aot_pretune_data/b200")
    parser.add_argument("--kernel-class", required=True,
                        help="Semantic class, e.g. matmul_int4")
    parser.add_argument("--kernel", required=True,
                        help="Literal kernel name in the archive dir, e.g. matmul_bf16_int4")
    parser.add_argument("--output", type=Path, required=True,
                        help="JSON output path")
    args = parser.parse_args()

    run_dirs = sorted((args.archive_root / args.kernel / "runs").glob("*"))
    if not run_dirs:
        print(f"No runs found under {args.archive_root}/{args.kernel}", file=sys.stderr)
        return 1

    # Use the most recent run by dir-name.
    run_dir = run_dirs[-1]
    csvs = sorted(run_dir.glob("measurements_*.csv"))
    if not csvs:
        print(f"No measurement CSV in {run_dir}", file=sys.stderr)
        return 1

    best = _best_per_shape(csvs[0])
    print(f"Loaded {len(best)} unique shapes from {csvs[0]}", file=sys.stderr)

    # Bucket by group.
    by_group: dict[str, list] = defaultdict(list)
    for shape, (perf, cfg) in best.items():
        g = _shape_to_group(args.kernel_class, shape)
        if g is None:
            continue
        by_group[g].append((perf, shape, cfg))

    fallbacks: dict[str, dict] = {}
    for group, entries in by_group.items():
        entries.sort(key=lambda x: x[0])
        median_idx = len(entries) // 2
        perf, shape, cfg = entries[median_idx]
        fallbacks[group] = {
            "template": cfg,
            "source": {
                "archive_shape": {"M": shape[0], "K": shape[1], "N": shape[2]},
                "archive_perf_ms": perf,
                "n_archive_shapes": len(entries),
            },
        }
        print(
            f"  {group:>10}: median shape M={shape[0]} K={shape[1]} "
            f"N={shape[2]} perf={perf:.5f}ms "
            f"(of {len(entries)} in-group archive shapes)",
            file=sys.stderr,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fallbacks, indent=2) + "\n")
    print(f"Wrote {len(fallbacks)} fallback groups to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
