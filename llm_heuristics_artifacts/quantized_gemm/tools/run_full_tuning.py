"""Per-shape full-autotune harness.

Drives Helion's default autotuner (``LFBOTreeSearch`` under
``HELION_AUTOTUNE_EFFORT=full``) for each shape in the expansion grid,
so the full pipeline runs — pattern_search + lfbo_pattern_search +
differential_evolution + random_search + llm_search + finishing_rounds
— instead of a single optimizer in isolation.

Every benchmarked config across every stage is captured by wrapping
``LocalBenchmarkProvider.benchmark`` and emitting one archive-schema
CSV row per (shape, config, timing). Duplicates are deduped to the
best observed perf per config.

Invariants:
- CSVs go under ``aot_pretune_data/b200/<kernel>/runs/<run_id>/``.
- Schema matches the archive exactly:
  ``kernel_name, shape_hash, config_hash, config, shape_features, timing_ms``.
- No existing archive data is touched.

Usage:
    python run_full_tuning.py --kernel matmul \
        --shapes-json ...expansion_shapes.json \
        --archive-root aot_pretune_data/b200 \
        --run-id 20260508_full_tuned

Budget: expect 5–15 minutes per shape; 40 shapes * 2 kernels * ~10 min
≈ 13 GPU-hours. Overnight-safe.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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

from llm_heuristics_artifacts.quantized_gemm.tools.workloads import build_kernel_and_args
from helion.autotuner import benchmark_provider as _bp_module


_DTYPE_CAT = {
    torch.bool: 0,
    torch.int8: 1, torch.int16: 1, torch.int32: 1, torch.int64: 1,
    torch.uint8: 1, torch.uint16: 1, torch.uint32: 1, torch.uint64: 1,
    torch.float16: 2, torch.bfloat16: 2, torch.float32: 2, torch.float64: 2,
    torch.complex64: 3, torch.complex128: 3,
}


def _dtype_cat(dt: torch.dtype) -> int:
    return _DTYPE_CAT.get(dt, 4)


def _tensor_features(tensor: torch.Tensor, prefix: str) -> dict[str, Any]:
    return {
        f"{prefix}_ndim": tensor.ndim,
        f"{prefix}_dim0": int(tensor.shape[0]) if tensor.ndim > 0 else 0,
        f"{prefix}_dim1": int(tensor.shape[1]) if tensor.ndim > 1 else 0,
        f"{prefix}_numel": int(tensor.numel()),
        f"{prefix}_dtype": str(tensor.dtype),
        f"{prefix}_dtype_size": tensor.element_size(),
        f"{prefix}_dtype_cat": _dtype_cat(tensor.dtype),
    }


def _shape_features(args: tuple[torch.Tensor, ...]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for i, a in enumerate(args):
        if isinstance(a, torch.Tensor):
            out.update(_tensor_features(a, f"arg{i}"))
    return out


def _stable_hash(payload: str) -> str:
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _shape_hash(shape_features: dict[str, Any]) -> str:
    return _stable_hash(json.dumps(shape_features, sort_keys=True))


def _config_hash(config_dict: dict[str, Any]) -> str:
    return _stable_hash(json.dumps(config_dict, sort_keys=True))


def _install_benchmark_hook() -> tuple[dict[str, tuple[dict, float]], callable]:
    """Monkey-patch LocalBenchmarkProvider.benchmark to tee results.

    Returns a (row_by_config, restore_fn) pair. Every call to the patched
    benchmark records each config's best observed perf into
    ``row_by_config`` (keyed by config_hash).
    """
    row_by_config: dict[str, tuple[dict, float]] = {}
    original = _bp_module.LocalBenchmarkProvider.benchmark

    def wrapped(self, configs, *, desc="Benchmarking"):
        results = original(self, configs, desc=desc)
        for r in results:
            if r.status != "ok":
                continue
            if not math.isfinite(r.perf):
                continue
            cfg = dict(r.config)
            ch = _config_hash(cfg)
            existing = row_by_config.get(ch)
            if existing is None or r.perf < existing[1]:
                row_by_config[ch] = (cfg, r.perf)
        return results

    _bp_module.LocalBenchmarkProvider.benchmark = wrapped

    def restore():
        _bp_module.LocalBenchmarkProvider.benchmark = original

    return row_by_config, restore


def _tune_one_shape(
    kernel_name: str, shape_entry: dict[str, Any], dtype_name: str,
) -> list[dict[str, Any]]:
    kernel_fn, args = build_kernel_and_args(
        kernel_name, {"args": shape_entry}, dtype_name
    )
    bound = kernel_fn.bind(args)

    # Make sure we run the full pipeline, ignoring any inherited preset.
    bound.settings.autotune_effort = "full"

    row_by_config, restore = _install_benchmark_hook()
    try:
        bound.autotune(args, force=True)
    finally:
        restore()

    features = _shape_features(args)
    shape_h = _shape_hash(features)
    rows: list[dict[str, Any]] = []
    for ch, (cfg, perf) in row_by_config.items():
        rows.append(
            {
                "kernel_name": kernel_fn.name,
                "shape_hash": shape_h,
                "config_hash": ch,
                "config": json.dumps(cfg, sort_keys=True),
                "shape_features": json.dumps(features, sort_keys=True),
                "timing_ms": perf,
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", required=True, choices=["matmul_bf16_int4", "_bf16xint16_gemm", "nvfp4_matmul"])
    parser.add_argument("--shapes-json", required=True, type=Path)
    parser.add_argument("--archive-root", required=True, type=Path)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--hardware-id", default="cuda_NVIDIA_B200_13.0")
    parser.add_argument("--only-bucket", default=None)
    parser.add_argument("--only-tag", default=None)
    args = parser.parse_args(argv)

    # Belt-and-suspenders: set the env var too so any sub-invocation
    # (e.g. LLM stage subprocess) picks up "full" as well.
    os.environ.setdefault("HELION_AUTOTUNE_EFFORT", "full")

    grid = json.loads(args.shapes_json.read_text())
    kernel_cfg = grid["shapes"][args.kernel]
    dtype_name = kernel_cfg["dtype"]
    bucket_keys = [k for k in kernel_cfg.keys() if k != "dtype"]
    if args.only_bucket:
        wanted = {b.strip() for b in args.only_bucket.split(",")}
        bucket_keys = [k for k in bucket_keys if k in wanted]
    tag_filter = None
    if args.only_tag:
        tag_filter = {t.strip() for t in args.only_tag.split(",")}

    run_id = args.run_id or f"{time.strftime('%Y%m%d_%H%M%S')}_full_tuned"
    run_dir = args.archive_root / args.kernel / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / f"measurements_{args.hardware_id}.csv"
    meta_path = run_dir / "run_metadata.json"

    fieldnames = ["kernel_name", "shape_hash", "config_hash", "config",
                  "shape_features", "timing_ms"]
    wall_t0 = time.perf_counter()
    shapes_done = 0
    total_rows = 0
    file_mode = "a" if csv_path.exists() else "w"
    with open(csv_path, file_mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if file_mode == "w":
            writer.writeheader()
        for bucket in bucket_keys:
            for shape in kernel_cfg[bucket]:
                if tag_filter is not None and shape.get("tag") not in tag_filter:
                    continue
                torch.manual_seed(20260509 + hash(shape.get("tag", "")) % 10000)
                t_shape = time.perf_counter()
                try:
                    rows = _tune_one_shape(args.kernel, shape, dtype_name)
                except Exception as e:  # noqa: BLE001
                    print(
                        f"ERROR on {args.kernel} bucket={bucket} tag={shape.get('tag')}: "
                        f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
                        file=sys.stderr,
                    )
                    continue
                for r in rows:
                    writer.writerow(r)
                f.flush()
                shapes_done += 1
                total_rows += len(rows)
                elapsed = time.perf_counter() - t_shape
                print(
                    f"[{elapsed:.1f}s] {args.kernel} {bucket}/{shape.get('tag')} "
                    f"M={shape['M']} K={shape['K']} N={shape['N']}: "
                    f"{len(rows)} unique configs recorded"
                )

    wall = time.perf_counter() - wall_t0
    meta = {
        "run_id": run_id,
        "hardware_id": args.hardware_id,
        "kernel": args.kernel,
        "source": "full autotune (LFBOTreeSearch + effort=full) for expansion shapes",
        "shapes_json": str(args.shapes_json.resolve()),
        "autotune_effort": "full",
        "buckets_run": bucket_keys,
        "shapes_done": shapes_done,
        "total_rows": total_rows,
        "wall_elapsed_s": wall,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "csv_path": str(csv_path.resolve()),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"\nWrote {csv_path}")
    print(f"Wrote {meta_path}")
    print(f"Elapsed: {wall:.1f}s  shapes={shapes_done}  rows={total_rows}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
