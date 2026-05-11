"""Exp-1 runner that uses the native runtime's observed-heuristics path.

Arm semantics:
- baseline:   HELION_AUTOTUNE_OBSERVED_HEURISTICS=0 + effort=none  -> Helion default config
- heuristics: HELION_AUTOTUNE_OBSERVED_HEURISTICS=1 + effort=none  -> observed-heuristic seed config

No dispatcher .py files involved — the seed lookup is the runtime's own
`observed_heuristic_default_config` reading the merged JSON.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from llm_heuristics_artifacts.quantized_gemm.tools.workloads import build_kernel_and_args
from helion.autotuner.benchmarking import do_bench


def _config_hash(cfg) -> str:
    payload = json.dumps(dict(cfg), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _bench_single_config(kernel_fn, args, cfg) -> tuple[float, str]:
    bound = kernel_fn.bind(args)
    compiled = bound.compile_config(cfg)
    try:
        fn = lambda: compiled(*args)
        fn()
        torch.cuda.synchronize()
        perf = do_bench(fn, return_mode="median", warmup=1, rep=50)
        return float(perf), "ok"
    except Exception as e:
        print(f"ERROR benchmarking: {type(e).__name__}: {e}", file=sys.stderr)
        return float("inf"), "error"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", required=True,
                        choices=["matmul_bf16_int4", "_bf16xint16_gemm", "nvfp4_matmul"])
    parser.add_argument("--arm", required=True, choices=["baseline", "heuristics"])
    parser.add_argument("--shape-grid", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--only-split", choices=["train", "heldout", "all"], default="all")
    parser.add_argument("--only-shape", default=None)
    args = parser.parse_args()

    grid = json.loads(args.shape_grid.read_text())
    kernel_cfg = grid["kernels"][args.kernel]
    shapes = kernel_cfg["shapes"]
    if args.only_split != "all":
        shapes = [s for s in shapes if s["split"] == args.only_split]
    if args.only_shape:
        wanted = {s.strip() for s in args.only_shape.split(",")}
        shapes = [s for s in shapes if s["id"] in wanted]

    # Settings are read at kernel-bind time, not at each call, so the env
    # vars must be set **before** the helion module imports the kernel.
    # We rely on the shell script setting these before invoking Python; this
    # check surfaces the mistake fast instead of silently yielding garbage.
    expected_effort = "none"
    expected_heur = "1" if args.arm == "heuristics" else "0"
    actual_effort = os.environ.get("HELION_AUTOTUNE_EFFORT")
    actual_heur = os.environ.get("HELION_AUTOTUNE_OBSERVED_HEURISTICS")
    if actual_effort != expected_effort:
        raise RuntimeError(
            f"Env HELION_AUTOTUNE_EFFORT must be {expected_effort!r} before "
            f"Python starts; found {actual_effort!r}. "
            f"Set via shell: HELION_AUTOTUNE_EFFORT=none python ..."
        )
    if actual_heur != expected_heur:
        raise RuntimeError(
            f"Env HELION_AUTOTUNE_OBSERVED_HEURISTICS must be {expected_heur!r} "
            f"for arm={args.arm!r} before Python starts; found {actual_heur!r}."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / f"{args.kernel}_{args.arm}.csv"
    meta_path = args.output_dir / f"{args.kernel}_{args.arm}.meta.json"

    fieldnames = [
        "kernel", "shape_id", "shape_label", "split", "repeat", "arm",
        "generation", "config_hash", "config", "status", "perf_ms",
        "compile_time_s", "seeded_by_heuristic",
    ]
    wall_t0 = time.perf_counter()
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for s in shapes:
            dtype = kernel_cfg.get("dtype")
            kernel_fn, kern_args = build_kernel_and_args(args.kernel, s, dtype)
            bound = kernel_fn.bind(kern_args)
            # Pass the real args explicitly — without them, _implicit_config
            # falls through to fake_args whose shape is SymInt, and shape
            # bucketing returns 'unknown'/empty rule lookup. That caches a
            # wrong config for the rest of the run, and every shape picks
            # the same bad fallback.
            cfg = bound._implicit_config(kern_args)
            if cfg is None:
                cfg = bound.config_spec.default_config()
            for rep in range(args.repeats):
                torch.manual_seed(20260509 + rep * 1000)
                try:
                    perf, status = _bench_single_config(kernel_fn, kern_args, cfg)
                except Exception as e:
                    print(f"ERROR {args.kernel} {s['id']} rep={rep}: {e}\n{traceback.format_exc()}",
                          file=sys.stderr)
                    perf, status = float("inf"), "error"
                w.writerow({
                    "kernel": args.kernel, "shape_id": s["id"],
                    "shape_label": s.get("label", ""), "split": s["split"],
                    "repeat": rep, "arm": args.arm, "generation": 0,
                    "config_hash": _config_hash(cfg),
                    "config": json.dumps(dict(cfg), sort_keys=True),
                    "status": status, "perf_ms": perf, "compile_time_s": "",
                    "seeded_by_heuristic": "1" if args.arm == "heuristics" else "0",
                })
                f.flush()

    wall = time.perf_counter() - wall_t0
    meta = {
        "version": 1, "kernel": args.kernel, "arm": args.arm,
        "mode": "no_autotune_effort_none_native_observed_heuristics",
        "shape_grid_path": str(args.shape_grid.resolve()),
        "shapes_run": [s["id"] for s in shapes],
        "repeats": args.repeats,
        "observed_heuristics_enabled": args.arm == "heuristics",
        "csv_path": str(csv_path.resolve()),
        "wall_elapsed_s": wall,
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Wrote {csv_path}")
    print(f"Elapsed: {wall:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
