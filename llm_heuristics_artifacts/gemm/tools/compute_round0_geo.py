"""Compute round0_best_geo for the norm hill-climbing loop.

Inputs:
- baseline and heuristics CSVs written by ``run_live.py``. CSVs may live in
  separate directories (one per arm).
- optional companion ``*.meta.json`` files (for reporting; metadata is NOT
  used to pick CSVs — the caller provides paths explicitly).

Per-workload definition:

    round0_best_geo[workload, repeat] =
      min(perf_ms where generation==0 and status==ok in heuristics arm)
    / min(perf_ms where generation==0 and status==ok in baseline arm)

We geomean across (workload, repeat), then also across train / heldout
splits and across kernels for family-level metrics.

Usage::

    python compute_round0_geo.py \
        --baseline path/to/N0_live/baseline \
        --heuristics path/to/Nx/heuristics \
        --output path/to/Nx/scores.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def _load_csv(csv_path: Path) -> list[dict[str, Any]]:
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def _read_arm_dir(arm_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """Return {kernel -> list of rows} collected from all CSVs in ``arm_dir``."""
    by_kernel: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for csv_path in sorted(arm_dir.glob("*.csv")):
        for row in _load_csv(csv_path):
            by_kernel[row["kernel"]].append(row)
    return by_kernel


def _best_round0(rows: list[dict[str, Any]]) -> dict[tuple[str, int], float]:
    """{(shape_id, repeat) -> min(perf_ms where status==ok)}"""
    out: dict[tuple[str, int], float] = {}
    for r in rows:
        if r.get("status") != "ok":
            continue
        if int(r.get("generation", 0)) != 0:
            continue
        key = (r["shape_id"], int(r["repeat"]))
        try:
            perf = float(r["perf_ms"])
        except (TypeError, ValueError):
            continue
        if not math.isfinite(perf):
            continue
        existing = out.get(key)
        if existing is None or perf < existing:
            out[key] = perf
    return out


def _compute_ratios(
    baseline_rows: list[dict[str, Any]],
    heuristics_rows: list[dict[str, Any]],
) -> dict[tuple[str, int], dict[str, Any]]:
    b = _best_round0(baseline_rows)
    h = _best_round0(heuristics_rows)
    # split lookup per shape
    split_by_shape: dict[str, str] = {}
    for r in baseline_rows + heuristics_rows:
        split_by_shape.setdefault(r["shape_id"], r.get("split", "unknown"))

    result: dict[tuple[str, int], dict[str, Any]] = {}
    for key in sorted(set(b.keys()) & set(h.keys())):
        shape_id, repeat = key
        ratio = h[key] / b[key]
        result[key] = {
            "shape_id": shape_id,
            "repeat": repeat,
            "split": split_by_shape.get(shape_id, "unknown"),
            "baseline_best_ms": b[key],
            "heuristics_best_ms": h[key],
            "round0_best_ratio": ratio,
        }
    return result


def _geomean(values: list[float]) -> float:
    values = [v for v in values if v > 0 and math.isfinite(v)]
    if not values:
        return float("nan")
    return math.exp(sum(math.log(v) for v in values) / len(values))


def _summarize(per_repeat: dict[tuple[str, int], dict[str, Any]]) -> dict[str, Any]:
    train = [e["round0_best_ratio"] for e in per_repeat.values() if e["split"] == "train"]
    heldout = [e["round0_best_ratio"] for e in per_repeat.values() if e["split"] == "heldout"]
    allv = [e["round0_best_ratio"] for e in per_repeat.values()]
    return {
        "n_train_pairs": len(train),
        "n_heldout_pairs": len(heldout),
        "round0_best_geo_train": _geomean(train),
        "round0_best_geo_heldout": _geomean(heldout),
        "round0_best_geo_overall": _geomean(allv),
        "heldout_minus_train": (
            _geomean(heldout) - _geomean(train)
            if train and heldout
            else float("nan")
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline",
        required=True,
        type=Path,
        help="Directory (or CSV) with baseline arm results.",
    )
    parser.add_argument(
        "--heuristics",
        required=False,
        type=Path,
        help="Directory (or CSV) with heuristics arm results. Omit to report baseline-only stats.",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    def _load(path: Path) -> dict[str, list[dict[str, Any]]]:
        if path.is_dir():
            return _read_arm_dir(path)
        by: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in _load_csv(path):
            by[row["kernel"]].append(row)
        return by

    baseline = _load(args.baseline)
    if args.heuristics is None:
        # Baseline-only: just report per-kernel noise and per-shape best.
        by_kernel_summary: dict[str, Any] = {}
        for kernel, rows in baseline.items():
            best = _best_round0(rows)
            by_shape: dict[str, list[float]] = defaultdict(list)
            for (shape_id, repeat), perf in best.items():
                by_shape[shape_id].append(perf)
            shapes_summary = []
            for shape_id in sorted(by_shape):
                perfs = sorted(by_shape[shape_id])
                shapes_summary.append(
                    {
                        "shape_id": shape_id,
                        "n_repeats": len(perfs),
                        "best_ms_median": perfs[len(perfs) // 2] if perfs else None,
                        "best_ms_min": min(perfs) if perfs else None,
                        "best_ms_max": max(perfs) if perfs else None,
                        "relative_spread": (
                            (max(perfs) - min(perfs)) / min(perfs)
                            if perfs and min(perfs) > 0
                            else None
                        ),
                    }
                )
            by_kernel_summary[kernel] = {
                "n_rows": len(rows),
                "n_shape_repeat_pairs": len(best),
                "per_shape": shapes_summary,
            }
        args.output.write_text(json.dumps({"baseline_only": by_kernel_summary}, indent=2))
        print(f"Wrote {args.output}")
        for k, v in by_kernel_summary.items():
            print(f"[{k}] shapes={len(v['per_shape'])} pairs={v['n_shape_repeat_pairs']}")
        return 0

    heuristics = _load(args.heuristics)

    per_kernel: dict[str, Any] = {}
    family_all_train: list[float] = []
    family_all_heldout: list[float] = []
    family_all: list[float] = []

    for kernel in sorted(set(baseline) | set(heuristics)):
        if kernel not in baseline or kernel not in heuristics:
            per_kernel[kernel] = {"error": "missing arm"}
            continue
        pairs = _compute_ratios(baseline[kernel], heuristics[kernel])
        summary = _summarize(pairs)
        per_kernel[kernel] = {
            "summary": summary,
            "per_repeat": [pairs[k] for k in sorted(pairs)],
        }
        for entry in pairs.values():
            family_all.append(entry["round0_best_ratio"])
            (family_all_train if entry["split"] == "train" else family_all_heldout).append(
                entry["round0_best_ratio"]
            )

    report = {
        "family": {
            "n_train_pairs": len(family_all_train),
            "n_heldout_pairs": len(family_all_heldout),
            "round0_best_geo_train": _geomean(family_all_train),
            "round0_best_geo_heldout": _geomean(family_all_heldout),
            "round0_best_geo_overall": _geomean(family_all),
            "heldout_minus_train": (
                _geomean(family_all_heldout) - _geomean(family_all_train)
                if family_all_train and family_all_heldout
                else float("nan")
            ),
        },
        "per_kernel": per_kernel,
    }
    args.output.write_text(json.dumps(report, indent=2))
    print(f"Wrote {args.output}")
    print(
        f"Family train={report['family']['round0_best_geo_train']:.4f} "
        f"heldout={report['family']['round0_best_geo_heldout']:.4f} "
        f"delta={report['family']['heldout_minus_train']:.4f}"
    )
    for kernel, data in per_kernel.items():
        s = data.get("summary")
        if s is None:
            print(f"[{kernel}] {data}")
            continue
        print(
            f"[{kernel}] train={s['round0_best_geo_train']:.4f} "
            f"heldout={s['round0_best_geo_heldout']:.4f} "
            f"delta={s['heldout_minus_train']:.4f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
