"""Exp-2 runner using the native observed-heuristics integration.

Uses the stock ``LLMGuidedSearch`` from Helion. Arm semantics:
- baseline:   HELION_AUTOTUNE_OBSERVED_HEURISTICS=0 -> LLM round-0 only
- heuristics: HELION_AUTOTUNE_OBSERVED_HEURISTICS=1 -> merged JSON seeds
              are prepended by LLMGuidedSearch automatically.

No dispatcher .py files. No custom subclass.
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
from helion.autotuner.llm_search import LLMGuidedSearch


def _config_hash(cfg) -> str:
    return hashlib.sha256(
        json.dumps(dict(cfg), sort_keys=True).encode()
    ).hexdigest()[:16]


def _run_one_shape(kernel_name, shape_entry, dtype_name, *,
                   llm_model, llm_provider, configs_per_round,
                   initial_random_configs, request_timeout_s):
    kernel_fn, args = build_kernel_and_args(kernel_name, shape_entry, dtype_name)
    bound = kernel_fn.bind(args)
    search = LLMGuidedSearch(
        bound, args,
        provider=llm_provider, model=llm_model,
        configs_per_round=configs_per_round,
        initial_random_configs=initial_random_configs,
        max_rounds=1, finishing_rounds=0,
        request_timeout_s=request_timeout_s,
    )
    search.autotune()
    rows = []
    for res in search._all_benchmark_results:
        perf_ms = res.perf * 1000.0 if res.status == "ok" else float("inf")
        rows.append({
            "generation": 0,
            "config_hash": _config_hash(res.config),
            "config": json.dumps(dict(res.config), sort_keys=True),
            "status": res.status,
            "perf_ms": perf_ms,
            "compile_time_s": res.compile_time if res.compile_time is not None else "",
            "seeded_by_heuristic": "",  # native path — runtime doesn't mark seed origins per-config
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", required=True,
                        choices=["matmul_bf16_int4", "_bf16xint16_gemm", "nvfp4_matmul"])
    parser.add_argument("--arm", required=True, choices=["baseline", "heuristics"])
    parser.add_argument("--shape-grid", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--configs-per-round", type=int, default=5)
    parser.add_argument("--initial-random-configs", type=int, default=3)
    parser.add_argument("--request-timeout-s", type=float, default=600.0)
    parser.add_argument("--only-split", choices=["train", "heldout", "all"], default="all")
    parser.add_argument("--only-shape", default=None)
    args = parser.parse_args()

    grid = json.loads(args.shape_grid.read_text())
    kernel_cfg = grid["kernels"][args.kernel]
    shapes = kernel_cfg["shapes"]
    dtype_name = kernel_cfg.get("dtype")
    if args.only_split != "all":
        shapes = [s for s in shapes if s["split"] == args.only_split]
    if args.only_shape:
        wanted = {s.strip() for s in args.only_shape.split(",")}
        shapes = [s for s in shapes if s["id"] in wanted]

    expected_heur = "1" if args.arm == "heuristics" else "0"
    actual_heur = os.environ.get("HELION_AUTOTUNE_OBSERVED_HEURISTICS")
    if actual_heur != expected_heur:
        raise RuntimeError(
            f"Env HELION_AUTOTUNE_OBSERVED_HEURISTICS must be {expected_heur!r} "
            f"for arm={args.arm!r} before Python starts; found {actual_heur!r}."
        )
    llm_model = os.environ.get("HELION_LLM_MODEL", "us.anthropic.claude-opus-4-7")
    llm_provider = os.environ.get("HELION_LLM_PROVIDER") or None

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
            for rep in range(args.repeats):
                torch.manual_seed(20260509 + rep * 1000)
                try:
                    rows = _run_one_shape(
                        args.kernel, s, dtype_name,
                        llm_model=llm_model, llm_provider=llm_provider,
                        configs_per_round=args.configs_per_round,
                        initial_random_configs=args.initial_random_configs,
                        request_timeout_s=args.request_timeout_s,
                    )
                except Exception as e:
                    print(f"ERROR {args.kernel} {s['id']} rep={rep}: {e}\n{traceback.format_exc()}",
                          file=sys.stderr)
                    w.writerow({
                        "kernel": args.kernel, "shape_id": s["id"],
                        "shape_label": s.get("label", ""), "split": s["split"],
                        "repeat": rep, "arm": args.arm, "generation": 0,
                        "config_hash": "", "config": "", "status": "error",
                        "perf_ms": float("inf"), "compile_time_s": "",
                        "seeded_by_heuristic": "",
                    })
                    continue
                for r in rows:
                    w.writerow({
                        "kernel": args.kernel, "shape_id": s["id"],
                        "shape_label": s.get("label", ""), "split": s["split"],
                        "repeat": rep, "arm": args.arm, **r,
                    })
                f.flush()

    wall = time.perf_counter() - wall_t0
    meta = {
        "version": 1, "kernel": args.kernel, "arm": args.arm,
        "mode": "llm_guided_search_max_rounds_1_native_observed_heuristics",
        "shape_grid_path": str(args.shape_grid.resolve()),
        "shapes_run": [s["id"] for s in shapes],
        "repeats": args.repeats, "configs_per_round": args.configs_per_round,
        "initial_random_configs": args.initial_random_configs,
        "observed_heuristics_enabled": args.arm == "heuristics",
        "llm_model": llm_model, "llm_provider": llm_provider,
        "csv_path": str(csv_path.resolve()),
        "wall_elapsed_s": wall,
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Wrote {csv_path}  elapsed {wall:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
