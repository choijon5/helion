"""No-autotune benchmark: time a single config per shape.

For the baseline arm: use ``HELION_AUTOTUNE_EFFORT=none`` so Helion
returns the ``config_spec.default_config()``, then benchmark that one
config directly.

For the heuristics arm: load the heuristic file, call ``autotune_<kname>``
with the live args, and benchmark the returned config.

Output CSV has the same schema as ``run_live.py`` so
``compute_round0_geo.py`` can score it against any baseline.

This is the honest "no autotune" baseline for Experiment 1 — it
answers: "if the user does not autotune at all, does the heuristic's
per-shape config beat Helion's generic default config?"
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
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

from llm_heuristics_artifacts.gemm.tools.workloads import build_kernel_and_args
from helion.autotuner.benchmarking import do_bench
from helion.runtime.config import Config


def _config_hash(cfg: Config) -> str:
    return hashlib.sha256(
        json.dumps(dict(cfg), sort_keys=True).encode()
    ).hexdigest()[:16]


def _load_heuristic_config(
    heuristic_path: Path, kernel_name: str, args: tuple[Any, ...]
) -> Config:
    spec = importlib.util.spec_from_file_location("_ext_heuristic", heuristic_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load heuristic module at {heuristic_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    autotune_fn = getattr(module, f"autotune_{kernel_name}", None)
    if autotune_fn is None:
        raise RuntimeError(f"Heuristic has no autotune_{kernel_name} function")
    cfg_dict = autotune_fn(*args)
    return Config(**cfg_dict)


def _bench_single_config(
    kernel_fn, args: tuple[Any, ...], cfg: Config
) -> tuple[float, str]:
    """Compile + benchmark a single config. Returns (perf_ms, status)."""
    import contextlib

    bound = kernel_fn.bind(args)
    # Override the kernel's default config with the chosen one for this run.
    compiled = bound.compile_config(cfg)
    try:
        # Warm-up + median timing, same harness as the autotuner uses.
        fn = lambda: compiled(*args)
        fn()  # warmup
        torch.cuda.synchronize()
        perf = do_bench(fn, return_mode="median", warmup=1, rep=50)
        return float(perf), "ok"
    except Exception as e:  # noqa: BLE001
        print(f"ERROR benchmarking: {type(e).__name__}: {e}", file=sys.stderr)
        return float("inf"), "error"


def _default_config(kernel_fn, args: tuple[Any, ...]) -> Config:
    """Return the Helion default config (what effort=none produces)."""
    bound = kernel_fn.bind(args)
    return bound.config_spec.default_config()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", required=True, choices=["matmul", "fp8_gemm"])
    parser.add_argument("--arm", required=True, choices=["baseline", "heuristics"])
    parser.add_argument("--shape-grid", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--only-split",
        choices=["train", "heldout", "all"],
        default="all",
    )
    parser.add_argument("--only-shape", default=None)
    args = parser.parse_args(argv)

    grid = json.loads(args.shape_grid.read_text())
    kernel_cfg = grid["kernels"][args.kernel]
    shapes = kernel_cfg["shapes"]
    if args.only_split != "all":
        shapes = [s for s in shapes if s["split"] == args.only_split]
    if args.only_shape:
        wanted = {s.strip() for s in args.only_shape.split(",")}
        shapes = [s for s in shapes if s["id"] in wanted]

    heuristic_path_env = os.environ.get("HELION_LLM_ROUND0_HEURISTIC_PATH")
    heuristic_path: Path | None = None
    if args.arm == "heuristics":
        if not heuristic_path_env:
            raise RuntimeError(
                "--arm heuristics requires HELION_LLM_ROUND0_HEURISTIC_PATH env var"
            )
        heuristic_path = Path(heuristic_path_env)
        if not heuristic_path.exists():
            raise RuntimeError(f"heuristic file does not exist: {heuristic_path}")

    # Force effort=none so the default-config path is taken for baseline
    # (does not affect heuristic arm since we bench the heuristic config
    # directly).
    os.environ["HELION_AUTOTUNE_EFFORT"] = "none"

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
        for shape_entry in shapes:
            dtype_name = grid.get("dtype")
            kernel_fn, kern_args = build_kernel_and_args(
                args.kernel, shape_entry, dtype_name
            )
            # Choose the config once per shape (it's shape-dependent but
            # deterministic, so same across repeats).
            if args.arm == "baseline":
                cfg = _default_config(kernel_fn, kern_args)
            else:
                cfg = _load_heuristic_config(
                    heuristic_path, kernel_fn.name, kern_args
                )
            for repeat in range(args.repeats):
                torch.manual_seed(20260509 + repeat * 1000)
                try:
                    perf, status = _bench_single_config(
                        kernel_fn, kern_args, cfg
                    )
                except Exception as e:  # noqa: BLE001
                    print(
                        f"ERROR on {args.kernel} {shape_entry['id']} "
                        f"rep={repeat}: {type(e).__name__}: {e}\n"
                        f"{traceback.format_exc()}",
                        file=sys.stderr,
                    )
                    perf, status = float("inf"), "error"
                w.writerow({
                    "kernel": args.kernel,
                    "shape_id": shape_entry["id"],
                    "shape_label": shape_entry.get("label", ""),
                    "split": shape_entry["split"],
                    "repeat": repeat,
                    "arm": args.arm,
                    "generation": 0,
                    "config_hash": _config_hash(cfg),
                    "config": json.dumps(dict(cfg), sort_keys=True),
                    "status": status,
                    "perf_ms": perf,
                    "compile_time_s": "",
                    "seeded_by_heuristic": "1" if args.arm == "heuristics" else "0",
                })
                f.flush()

    wall = time.perf_counter() - wall_t0
    meta = {
        "version": 1,
        "kernel": args.kernel,
        "arm": args.arm,
        "mode": "no_autotune_effort_none",
        "shape_grid_path": str(args.shape_grid.resolve()),
        "dtype": grid.get("dtype"),
        "shapes_run": [s["id"] for s in shapes],
        "repeats": args.repeats,
        "heuristic_path": str(heuristic_path) if heuristic_path else None,
        "csv_path": str(csv_path.resolve()),
        "wall_elapsed_s": wall,
        "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Wrote {csv_path}")
    print(f"Wrote {meta_path}")
    print(f"Elapsed: {wall:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
