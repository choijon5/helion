"""
Merge best configs across multiple AOT autotune runs.

Each `pretune_runner.py` invocation creates a new timestamped subdir under
`.helion_aot/job_<kernel>/<timestamp>/`.  Running the orchestrator N times
accumulates N subdirs with their own JSONs and CSVs.  This script scans all
of them and, per (kernel, shape), picks the config with the lowest measured
timing across all runs.

Output:
    .helion_aot/best_configs_<hardware_id>.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys


def load_measurements(csv_path: Path) -> dict[tuple[str, str], tuple[float, dict, dict]]:
    """Load (kernel, shape_hash) -> (timing_ms, config, shape_features) map.

    Returns the LOWEST timing for each (kernel, shape, config) seen.
    """
    by_key: dict[tuple[str, str], tuple[float, dict, dict]] = {}
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                t = float(row["timing_ms"])
            except (KeyError, ValueError, TypeError):
                continue
            key = (row["kernel_name"], row["shape_hash"])
            cfg = json.loads(row["config"])
            sf = json.loads(row["shape_features"])
            existing = by_key.get(key)
            if existing is None or t < existing[0]:
                by_key[key] = (t, cfg, sf)
    return by_key


def merge(roots: list[Path]) -> dict[str, dict]:
    """Scan all `.helion_aot/job_*/<run>/measurements_*.csv` under each root and merge.

    Returns: {kernel_name: {shape_hash: {timing_ms, config, shape_features}}}
    """
    best: dict[str, dict[str, dict]] = {}
    n_csvs = 0
    n_rows = 0

    csv_paths = []
    for root in roots:
        if root.exists():
            csv_paths.extend(root.glob("job_*/*/measurements_*.csv"))
    for csv_path in csv_paths:
        n_csvs += 1
        run_best = load_measurements(csv_path)
        n_rows += len(run_best)
        for (kernel, shape_hash), (t, cfg, sf) in run_best.items():
            kbest = best.setdefault(kernel, {})
            entry = kbest.get(shape_hash)
            if entry is None or t < entry["timing_ms"]:
                kbest[shape_hash] = {
                    "timing_ms": t,
                    "config": cfg,
                    "shape_features": sf,
                }
    print(
        f"[merge] scanned {n_csvs} measurements CSVs, {n_rows} (kernel, shape) "
        f"keys merged",
        file=sys.stderr,
    )
    return best


def to_tuned_configs_json(best: dict[str, dict[str, dict]]) -> dict[str, list[dict]]:
    """Format like helion's tuned_configs JSON, one entry per (kernel, shape)."""
    out: dict[str, list[dict]] = {}
    for kernel, shapes in best.items():
        out[kernel] = []
        for shape_hash, entry in sorted(shapes.items()):
            out[kernel].append(
                {
                    "config": entry["config"],
                    "shape_hash": shape_hash,
                    "timing_ms": entry["timing_ms"],
                    "shape_features": entry["shape_features"],
                }
            )
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        action="append",
        help="Root output dir (repeat to merge multiple). Default: helion_2/.helion_aot",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON path (default: <first-root>/best_configs.json)",
    )
    args = parser.parse_args()

    roots = args.root or [
        Path("/home/jongsokchoi/helion_2/.helion_aot"),
        # Old wrong-cwd location from earlier scheduler runs.
        Path("/home/jongsokchoi/helion_2/examples/aot_pretune/.helion_aot"),
    ]
    roots = [r for r in roots if r.exists()]
    if not roots:
        print("No roots exist", file=sys.stderr)
        return 1
    print(f"Merging from: {[str(r) for r in roots]}", file=sys.stderr)

    best = merge(roots)
    out = args.out or (roots[0] / "best_configs.json")
    out.write_text(json.dumps(to_tuned_configs_json(best), indent=2))
    print(f"Wrote {out}")

    # Print summary
    print("\nKernel summary:")
    for k, shapes in sorted(best.items()):
        timings = [e["timing_ms"] for e in shapes.values()]
        print(
            f"  {k:>15s}: {len(shapes)} shapes, "
            f"timing min={min(timings)*1000:.1f}us  "
            f"max={max(timings)*1000:.1f}us"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
