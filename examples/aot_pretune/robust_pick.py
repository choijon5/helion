"""
Robust post-merge: pick the best config per (kernel, shape) by re-benchmarking
the top-K candidates with K_BENCH independent do_bench calls and taking the
median.

This filters lowest-of-N bias from the raw merge — `pick_best_configs.py`
chose configs by lowest single timing in CSV, which can favour lucky
measurements.  This script:

  1. Reads all measurement CSVs across both data dirs.
  2. For each (kernel, shape_hash), keeps the top-K configs by raw CSV timing.
  3. For each candidate, calls `do_bench` K_BENCH times → K_BENCH medians.
  4. Picks the candidate with the lowest median across re-benchmarks.

Output:
    .helion_aot/robust_best_configs.json
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path

import torch
import triton.testing as triton_testing


SCRIPT_DIR = Path(__file__).resolve().parent
TUTORIAL_KERNELS_PATH = SCRIPT_DIR / "tutorial_kernels.py"


def load_kernels_module():
    spec = importlib.util.spec_from_file_location(
        "tutorial_kernels", TUTORIAL_KERNELS_PATH
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so helion's compiler can resolve __module__
    # lookups (e.g. when retrieving the kernel source for fp8_gemm).
    sys.modules["tutorial_kernels"] = mod
    spec.loader.exec_module(mod)
    return mod


def gather_candidates(
    roots: list[Path], top_k: int
) -> dict[str, dict[str, list[dict]]]:
    """Return {kernel: {shape_hash: [top-K candidate dicts]}}."""
    rows: dict[tuple[str, str], list[dict]] = {}
    for root in roots:
        if not root.exists():
            continue
        for csv_path in root.glob("job_*/*/measurements_*.csv"):
            with csv_path.open() as f:
                for row in csv.DictReader(f):
                    try:
                        timing = float(row["timing_ms"])
                    except (KeyError, ValueError):
                        continue
                    key = (row["kernel_name"], row["shape_hash"])
                    rows.setdefault(key, []).append(
                        {
                            "timing_ms": timing,
                            "config": json.loads(row["config"]),
                            "shape_features": json.loads(row["shape_features"]),
                            "config_hash": row.get("config_hash"),
                        }
                    )

    out: dict[str, dict[str, list[dict]]] = {}
    for (kernel, shape_hash), entries in rows.items():
        # Dedup by config_hash, keeping the lowest existing timing per config.
        by_cfg: dict[str, dict] = {}
        for e in entries:
            ch = e["config_hash"] or json.dumps(e["config"], sort_keys=True)
            existing = by_cfg.get(ch)
            if existing is None or e["timing_ms"] < existing["timing_ms"]:
                by_cfg[ch] = e
        # Sort by raw csv timing, keep top K.
        deduped = sorted(by_cfg.values(), key=lambda e: e["timing_ms"])[:top_k]
        out.setdefault(kernel, {})[shape_hash] = deduped
    return out


def shape_features_for(args: tuple) -> dict:
    """Compute the shape_features dict the way helion's CSV does."""
    out: dict[str, object] = {}
    for i, a in enumerate(args):
        if isinstance(a, torch.Tensor):
            out[f"arg{i}_ndim"] = a.ndim
            for d in range(a.ndim):
                out[f"arg{i}_dim{d}"] = a.shape[d]
            out[f"arg{i}_numel"] = a.numel()
            out[f"arg{i}_dtype"] = str(a.dtype)
            out[f"arg{i}_dtype_size"] = a.element_size()
            # Best-effort; matches helion's _flatten_dtype_category roughly
            out[f"arg{i}_dtype_cat"] = 2 if a.dtype.is_floating_point else 0
    return out


def features_match(target: dict, candidate: dict) -> bool:
    """True if numeric tensor dims/dtypes match.

    Compares only the fields that uniquely identify a tensor input:
    ndim, dim_i, dtype.  shape_features in CSVs may have extra fields like
    'arg0_dtype_cat' that helion encodes differently; match on the obvious
    subset.
    """
    keys = [
        k for k in target
        if k.endswith("_ndim") or "_dim" in k or k.endswith("_dtype") or k.endswith("_dtype_size")
    ]
    return all(target.get(k) == candidate.get(k) for k in keys)


def rebenchmark(
    kernel,
    args: tuple,
    config_dict: dict,
    n_calls: int,
    warmup_ms: int,
    rep_ms: int,
) -> list[float]:
    """Return list of n_calls do_bench medians."""
    import helion

    kernel.reset()
    bound = kernel.bind(tuple(args))

    # Drop helion-AOT-only key if present (e.g. _triton_*)
    cleaned = dict(config_dict)
    cleaned.pop("atomic_indexing", None)
    config = helion.Config(**cleaned)
    bound.set_config(config)

    # Warmup compile
    bound(*args)

    timings = []
    for _ in range(n_calls):
        t = triton_testing.do_bench(
            lambda: bound(*args),
            warmup=warmup_ms,
            rep=rep_ms,
            return_mode="median",
        )
        timings.append(t)
    return timings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--roots",
        type=Path,
        nargs="+",
        default=[
            Path("/home/jongsokchoi/helion_2/.helion_aot"),
            Path("/home/jongsokchoi/helion_2/examples/aot_pretune/.helion_aot"),
        ],
        help="AOT data directories to scan for candidates.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Top-K candidates per shape to re-benchmark (default: 3).",
    )
    parser.add_argument(
        "--n-bench",
        type=int,
        default=5,
        help="Number of do_bench calls per candidate (default: 5).",
    )
    parser.add_argument(
        "--warmup-ms",
        type=int,
        default=50,
        help="do_bench warmup ms (default: 50).",
    )
    parser.add_argument(
        "--rep-ms",
        type=int,
        default=200,
        help="do_bench rep ms (default: 200).",
    )
    parser.add_argument(
        "--kernels",
        nargs="+",
        default=None,
        help="Kernels to process (default: all).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/home/jongsokchoi/helion_2/.helion_aot/robust_best_configs.json"),
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="CUDA device index (sets CUDA_VISIBLE_DEVICES if set).",
    )
    args = parser.parse_args()

    if args.gpu is not None:
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    print(f"Loading kernel module from {TUTORIAL_KERNELS_PATH}", file=sys.stderr)
    tk = load_kernels_module()

    KERNEL_INFO: dict[str, tuple[Callable, Callable]] = {
        "vector_add": (tk.vector_add, tk._vector_add_inputs),
        "matmul": (tk.matmul, tk._matmul_inputs),
        "softmax": (tk.softmax, tk._softmax_inputs),
        "layer_norm": (tk.layer_norm, tk._layer_norm_inputs),
        "attention": (tk.attention, tk._attention_inputs),
        "grouped_gemm": (tk.grouped_gemm, tk._grouped_gemm_inputs),
        "fp8_gemm": (tk.fp8_gemm, tk._fp8_gemm_inputs),
    }

    target_kernels = args.kernels or list(KERNEL_INFO.keys())
    print(f"Gathering candidates from {[str(r) for r in args.roots]}", file=sys.stderr)
    cands = gather_candidates(args.roots, args.top_k)

    final: dict[str, list[dict]] = {}
    for kernel_name in target_kernels:
        if kernel_name not in KERNEL_INFO:
            print(f"Unknown kernel: {kernel_name}", file=sys.stderr)
            continue
        kernel, inputs_fn = KERNEL_INFO[kernel_name]
        kernel_cands = cands.get(kernel_name, {})
        if not kernel_cands:
            print(f"No candidates for {kernel_name}", file=sys.stderr)
            continue

        print(f"\n=== {kernel_name}: {len(kernel_cands)} shapes ===", file=sys.stderr)
        all_inputs = inputs_fn()
        # Pre-compute shape_features for each input set so we can match by csv hash key
        feature_lookup: list[tuple[dict, tuple]] = [
            (shape_features_for(args), args) for args in all_inputs
        ]

        kernel_results: list[dict] = []
        for shape_hash, candidates in kernel_cands.items():
            # Match candidate's shape_features to one of our inputs
            target = candidates[0]["shape_features"]
            matched_args = None
            for feats, inp_args in feature_lookup:
                if features_match(target, feats):
                    matched_args = inp_args
                    break
            if matched_args is None:
                print(
                    f"  [{shape_hash[:8]}]: NO MATCHING INPUT for "
                    f"{target} — skipping",
                    file=sys.stderr,
                )
                continue

            t0 = time.time()
            best_median = float("inf")
            best_cand = None
            best_timings: list[float] = []
            for ci, cand in enumerate(candidates):
                try:
                    timings = rebenchmark(
                        kernel,
                        matched_args,
                        cand["config"],
                        n_calls=args.n_bench,
                        warmup_ms=args.warmup_ms,
                        rep_ms=args.rep_ms,
                    )
                    median = statistics.median(timings)
                    if median < best_median:
                        best_median = median
                        best_cand = cand
                        best_timings = timings
                except Exception as e:
                    print(
                        f"  [{shape_hash[:8]}] cand{ci}: FAILED {type(e).__name__}: {e}",
                        file=sys.stderr,
                    )

            elapsed = time.time() - t0
            if best_cand is not None:
                kernel_results.append(
                    {
                        "shape_hash": shape_hash,
                        "shape_features": best_cand["shape_features"],
                        "config": best_cand["config"],
                        "csv_timing_ms": best_cand["timing_ms"],
                        "rebench_median_ms": best_median,
                        "rebench_timings_ms": best_timings,
                    }
                )
                print(
                    f"  [{shape_hash[:8]}] best={best_median:.4f}ms "
                    f"(csv was {best_cand['timing_ms']:.4f}ms)  "
                    f"in {elapsed:.1f}s",
                    file=sys.stderr,
                )
        final[kernel_name] = kernel_results

    args.out.write_text(json.dumps(final, indent=2))
    print(f"\nWrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
