"""Produce fresh archive measurement CSVs for the norm family.

Per the archive-expansion rule in ``plan.md``, this runner always uses
Helion's default autotuner, ``LFBOTreeSearch``, at full effort. Captures
every benchmarked (config, timing) pair. Writes the output in the exact
schema the archived AOT CSVs use, so
``heuristic_generator.generate_heuristic`` consumes it unchanged.

CSV schema:
  kernel_name, shape_hash, config_hash, config, shape_features, timing_ms

One CSV per kernel per run. One run per kernel here.

Usage::

    python fresh_archive.py \
        --kernel layer_norm \
        --shape-grid .../shape_grid.json \
        --output-dir iterations/N4a_fresh_archive/<kernel>/ \
        [--only-split train]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from llm_heuristics_artifacts.norms.tools.workloads import build_kernel_and_args

from helion.autotuner.surrogate_pattern_search import LFBOTreeSearch
from helion.autotuner.effort_profile import PATTERN_SEARCH_DEFAULTS
from helion.runtime.config import Config
from helion.autotuner.base_search import BenchmarkResult


class _CapturingLFBOTreeSearch(LFBOTreeSearch):
    """LFBOTreeSearch wrapper that captures every (config, perf, status)
    tuple ever benchmarked, not just the final surviving population.

    The base class replaces ``self.population`` each generation, so
    ``_all_benchmark_results`` is the source of truth for archive
    expansion.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._all_benchmark_results: list[BenchmarkResult] = []

    def benchmark_batch(self, configs, *, desc=""):  # type: ignore[override]
        results = super().benchmark_batch(configs, desc=desc)
        self._all_benchmark_results.extend(results)
        return results


_CANONICAL_KERNEL_NAME = {
    "layer_norm": "layer_norm",
    "rms_norm": "rms_norm",
    "softmax": "softmax",
}


def _shape_hash(features: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(features, sort_keys=True).encode()
    ).hexdigest()[:16]


def _config_hash(cfg: Config) -> str:
    return hashlib.sha256(
        json.dumps(dict(cfg), sort_keys=True).encode()
    ).hexdigest()[:16]


def _extract_shape_features(args: tuple[Any, ...]) -> dict[str, Any]:
    """Match AOTAutotuneCache/extract_shape_features output format."""
    out: dict[str, Any] = {}
    for i, arg in enumerate(args):
        if isinstance(arg, torch.Tensor):
            out[f"arg{i}_ndim"] = arg.ndim
            for d in range(arg.ndim):
                out[f"arg{i}_dim{d}"] = int(arg.shape[d])
            out[f"arg{i}_numel"] = int(arg.numel())
            out[f"arg{i}_dtype"] = str(arg.dtype)
            out[f"arg{i}_dtype_size"] = arg.element_size()
            # dtype_cat matches heuristic_generator's scheme
            if arg.dtype == torch.bool:
                out[f"arg{i}_dtype_cat"] = 0
            elif arg.dtype in (torch.int8, torch.int16, torch.int32, torch.int64,
                                torch.uint8):
                out[f"arg{i}_dtype_cat"] = 1
            elif arg.dtype in (torch.float16, torch.bfloat16, torch.float32,
                                torch.float64):
                out[f"arg{i}_dtype_cat"] = 2
            else:
                out[f"arg{i}_dtype_cat"] = 4
        elif isinstance(arg, (int, float)):
            out[f"arg{i}_scalar"] = arg
    return out


def _run_one_shape(
    kernel_name: str,
    shape_entry: dict[str, Any],
    dtype_name: str,
    *,
    max_generations: int,
    copies: int,
    initial_population: int,
) -> list[tuple[dict[str, Any], Config, float]]:
    kernel_fn, args = build_kernel_and_args(kernel_name, shape_entry, dtype_name)
    bound = kernel_fn.bind(args)
    # Per plan.md "Archive Expansion Rule", use the default LFBOTreeSearch
    # autotuner at full effort. The capturing wrapper accumulates every
    # (config, perf, status) from every generation, not just the final
    # surviving population.
    search = _CapturingLFBOTreeSearch(
        bound,
        args,
        initial_population=initial_population,
        copies=copies,
        max_generations=max_generations,
    )
    search.autotune()
    shape_features = _extract_shape_features(args)
    # Dedupe by config hash; keep the fastest observed perf for each config.
    best_by_config: dict[str, tuple[Config, float]] = {}
    for res in search._all_benchmark_results:
        if res.status != "ok":
            continue
        if res.perf == float("inf") or res.perf <= 0:
            continue
        ch = _config_hash(res.config)
        if ch not in best_by_config or res.perf < best_by_config[ch][1]:
            best_by_config[ch] = (res.config, res.perf)
    return [
        (shape_features, cfg, perf * 1000.0)
        for (cfg, perf) in best_by_config.values()
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", required=True,
                        choices=list(_CANONICAL_KERNEL_NAME))
    parser.add_argument("--shape-grid", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--only-split", choices=["train", "heldout", "all"],
                        default="train")
    parser.add_argument("--only-shape", default=None)
    parser.add_argument(
        "--max-generations",
        type=int,
        default=PATTERN_SEARCH_DEFAULTS.max_generations,
        help="LFBOTreeSearch max_generations. Default is the full-effort profile value.",
    )
    parser.add_argument(
        "--copies",
        type=int,
        default=PATTERN_SEARCH_DEFAULTS.copies,
        help="LFBOTreeSearch copies (number of parallel search paths).",
    )
    parser.add_argument(
        "--initial-population",
        type=int,
        default=PATTERN_SEARCH_DEFAULTS.initial_population,
        help="Initial random-population size for LFBOTreeSearch.",
    )
    args = parser.parse_args(argv)

    grid = json.loads(args.shape_grid.read_text())
    shapes = grid["kernels"][args.kernel]["shapes"]
    if args.only_split != "all":
        shapes = [s for s in shapes if s["split"] == args.only_split]
    if args.only_shape:
        wanted = {s.strip() for s in args.only_shape.split(",")}
        shapes = [s for s in shapes if s["id"] in wanted]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / f"measurements_cuda_NVIDIA_B200_13.0.csv"
    meta_path = args.output_dir / f"run_metadata.json"

    fieldnames = [
        "kernel_name", "shape_hash", "config_hash", "config",
        "shape_features", "timing_ms",
    ]
    wall_t0 = time.perf_counter()
    written = 0
    per_shape_counts: dict[str, int] = {}
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for shape_entry in shapes:
            print(f"\n=== {args.kernel} {shape_entry['id']} "
                  f"({shape_entry.get('label','')}) split={shape_entry['split']} ===",
                  file=sys.stderr)
            try:
                results = _run_one_shape(
                    args.kernel, shape_entry, grid.get("dtype", "bfloat16"),
                    max_generations=args.max_generations,
                    copies=args.copies,
                    initial_population=args.initial_population,
                )
            except Exception as e:  # noqa: BLE001
                tb = traceback.format_exc()
                print(f"ERROR on {shape_entry['id']}: {type(e).__name__}: {e}\n{tb}",
                      file=sys.stderr)
                continue
            seen_hashes: set[str] = set()
            for features, cfg, perf_ms in results:
                ch = _config_hash(cfg)
                if ch in seen_hashes:
                    continue  # dedupe within a shape
                seen_hashes.add(ch)
                w.writerow({
                    "kernel_name": _CANONICAL_KERNEL_NAME[args.kernel],
                    "shape_hash": _shape_hash(features),
                    "config_hash": ch,
                    "config": json.dumps(dict(cfg), sort_keys=True),
                    "shape_features": json.dumps(features, sort_keys=True),
                    "timing_ms": perf_ms,
                })
                written += 1
            per_shape_counts[shape_entry["id"]] = len(seen_hashes)
            f.flush()

    wall_elapsed = time.perf_counter() - wall_t0
    meta = {
        "version": 1,
        "kernel": args.kernel,
        "shape_grid_path": str(args.shape_grid.resolve()),
        "only_split": args.only_split,
        "only_shape": args.only_shape,
        "n_shapes_run": len(shapes),
        "n_rows_written": written,
        "per_shape_unique_configs": per_shape_counts,
        "max_generations": args.max_generations,
        "copies": args.copies,
        "initial_population": args.initial_population,
        "autotuner": "LFBOTreeSearch",
        "csv_path": str(csv_path.resolve()),
        "wall_elapsed_s": wall_elapsed,
        "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Wrote {csv_path} ({written} rows, {wall_elapsed:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
