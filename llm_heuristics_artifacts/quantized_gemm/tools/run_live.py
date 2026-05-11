"""Live benchmark runner for the quantized-GEMM hill-climbing loop.

Ported from llm_heuristics_artifacts/gemm/tools/run_live.py. Same
mechanics (baseline vs heuristics arm, LLMGuidedSearch max_rounds=1,
optional heuristic seed configured via env var) — differences:

- SUPPORTED_KERNELS = the 3 quantized kernel names
- workloads builder is the quantized one
- dtype per-kernel lookup (grid["kernels"][k]["dtype"])
- heuristic entry-point resolution also tries stripping a leading
  underscore (for _bf16xint16_gemm whose dispatcher exports
  autotune_bf16xint16_gemm).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
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
from helion.runtime.config import Config

SUPPORTED_KERNELS = ("matmul_bf16_int4", "_bf16xint16_gemm", "nvfp4_matmul")


def _config_hash(cfg: Config) -> str:
    payload = json.dumps(dict(cfg), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _load_round0_heuristic_config(
    heuristic_path: Path, kernel_name: str, args: tuple[Any, ...]
) -> Config | None:
    spec = importlib.util.spec_from_file_location("_ext_heuristic", heuristic_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load heuristic module at {heuristic_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    autotune_fn = getattr(module, f"autotune_{kernel_name}", None)
    if autotune_fn is None:
        autotune_fn = getattr(module, f"autotune_{kernel_name.lstrip('_')}", None)
    if autotune_fn is None:
        return None
    cfg_dict = autotune_fn(*args)
    if cfg_dict is None:
        return None
    return Config(**cfg_dict)


class _SeededLLMGuidedSearch(LLMGuidedSearch):
    def __init__(
        self,
        *args,
        seed_config: Config | None = None,
        skip_llm_stage: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._extra_seed_config = seed_config
        self._skip_llm_stage = skip_llm_stage

    def _build_seed_configs(self):
        seeds = super()._build_seed_configs()
        if self._extra_seed_config is None:
            return seeds
        seen = {self._config_key(c) for c in seeds}
        extra_key = self._config_key(self._extra_seed_config)
        if extra_key in seen:
            return seeds
        return [self._extra_seed_config, *seeds]

    def _call_llm_async(self, messages):
        if self._skip_llm_stage:
            import concurrent.futures
            f: concurrent.futures.Future = concurrent.futures.Future()
            f.set_result('{"configs": []}')
            return f
        return super()._call_llm_async(messages)


def _run_one_shape(
    kernel_name: str,
    shape_entry: dict[str, Any],
    dtype_name: str,
    *,
    heuristic_path: Path | None,
    llm_model: str,
    llm_provider: str | None,
    configs_per_round: int,
    initial_random_configs: int,
    request_timeout_s: float,
    skip_llm_stage: bool = False,
) -> list[dict[str, Any]]:
    kernel_fn, args = build_kernel_and_args(kernel_name, shape_entry, dtype_name)
    bound = kernel_fn.bind(args)
    seed_config = None
    if heuristic_path is not None:
        seed_config = _load_round0_heuristic_config(
            heuristic_path, kernel_fn.name, args
        )
    search = _SeededLLMGuidedSearch(
        bound,
        args,
        provider=llm_provider,
        model=llm_model,
        configs_per_round=configs_per_round,
        initial_random_configs=initial_random_configs,
        max_rounds=1,
        finishing_rounds=0,
        request_timeout_s=request_timeout_s,
        seed_config=seed_config,
        skip_llm_stage=skip_llm_stage,
    )
    search.autotune()
    rows: list[dict[str, Any]] = []
    for res in search._all_benchmark_results:
        perf_ms = res.perf * 1000.0 if res.status == "ok" else float("inf")
        rows.append(
            {
                "generation": 0,
                "config_hash": _config_hash(res.config),
                "config": json.dumps(dict(res.config), sort_keys=True),
                "status": res.status,
                "perf_ms": perf_ms,
                "compile_time_s": res.compile_time if res.compile_time is not None else "",
                "seeded_by_heuristic": "1"
                if seed_config is not None
                and _config_hash(res.config) == _config_hash(seed_config)
                else "0",
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", required=True, choices=SUPPORTED_KERNELS)
    parser.add_argument("--arm", required=True, choices=["baseline", "heuristics"])
    parser.add_argument("--shape-grid", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--configs-per-round", type=int, default=5)
    parser.add_argument("--initial-random-configs", type=int, default=3)
    parser.add_argument("--request-timeout-s", type=float, default=600.0)
    parser.add_argument(
        "--only-split",
        choices=["train", "heldout", "all"],
        default="all",
    )
    parser.add_argument("--only-shape", default=None)
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip the LLM round-0 call.",
    )
    args = parser.parse_args(argv)

    grid = json.loads(args.shape_grid.read_text())
    kernel_cfg = grid["kernels"][args.kernel]
    shapes = kernel_cfg["shapes"]
    dtype_name = kernel_cfg.get("dtype")
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
    else:
        if heuristic_path_env:
            print(
                "WARN: arm=baseline but HELION_LLM_ROUND0_HEURISTIC_PATH is set; "
                "ignoring heuristic path for baseline arm.",
                file=sys.stderr,
            )

    llm_model = os.environ.get("HELION_LLM_MODEL", "us.anthropic.claude-opus-4-7")
    llm_provider = os.environ.get("HELION_LLM_PROVIDER")
    if llm_provider == "":
        llm_provider = None

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
            for repeat in range(args.repeats):
                torch.manual_seed(20260509 + repeat * 1000)
                try:
                    rows = _run_one_shape(
                        args.kernel,
                        shape_entry,
                        dtype_name,
                        heuristic_path=heuristic_path,
                        llm_model=llm_model,
                        llm_provider=llm_provider,
                        configs_per_round=args.configs_per_round,
                        initial_random_configs=args.initial_random_configs,
                        request_timeout_s=args.request_timeout_s,
                        skip_llm_stage=args.no_llm,
                    )
                except Exception as e:  # noqa: BLE001
                    tb = traceback.format_exc()
                    print(
                        f"ERROR on {args.kernel} {shape_entry['id']} "
                        f"rep={repeat}: {type(e).__name__}: {e}\n{tb}",
                        file=sys.stderr,
                    )
                    w.writerow(
                        {
                            "kernel": args.kernel,
                            "shape_id": shape_entry["id"],
                            "shape_label": shape_entry.get("label", ""),
                            "split": shape_entry["split"],
                            "repeat": repeat,
                            "arm": args.arm,
                            "generation": 0,
                            "config_hash": "",
                            "config": "",
                            "status": "error",
                            "perf_ms": float("inf"),
                            "compile_time_s": "",
                            "seeded_by_heuristic": "0",
                        }
                    )
                    continue
                for r in rows:
                    w.writerow(
                        {
                            "kernel": args.kernel,
                            "shape_id": shape_entry["id"],
                            "shape_label": shape_entry.get("label", ""),
                            "split": shape_entry["split"],
                            "repeat": repeat,
                            "arm": args.arm,
                            **r,
                        }
                    )
                f.flush()

    wall_elapsed = time.perf_counter() - wall_t0
    meta = {
        "version": 1,
        "kernel": args.kernel,
        "arm": args.arm,
        "shape_grid_path": str(args.shape_grid.resolve()),
        "dtype": dtype_name,
        "shapes_run": [s["id"] for s in shapes],
        "repeats": args.repeats,
        "configs_per_round": args.configs_per_round,
        "initial_random_configs": args.initial_random_configs,
        "request_timeout_s": args.request_timeout_s,
        "llm_model": llm_model,
        "llm_provider": llm_provider,
        "skip_llm_stage": bool(args.no_llm),
        "heuristic_path": str(heuristic_path) if heuristic_path else None,
        "helion_anthropic_thinking_budget": os.environ.get(
            "HELION_LLM_ANTHROPIC_THINKING_BUDGET"
        ),
        "helion_anthropic_effort": os.environ.get(
            "HELION_LLM_ANTHROPIC_REASONING_EFFORT"
        ),
        "csv_path": str(csv_path.resolve()),
        "wall_elapsed_s": wall_elapsed,
        "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Wrote {csv_path}")
    print(f"Wrote {meta_path}")
    print(f"Elapsed: {wall_elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
