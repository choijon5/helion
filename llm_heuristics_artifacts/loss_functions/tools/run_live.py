"""Live benchmark runner for the loss function hill-climbing loop.

Runs one arm (baseline or heuristics) on one kernel across the committed
shape grid, writing per-shape CSVs + metadata that the scorer consumes.

The heuristics mechanism is selected by env vars, *not* CLI flags, so the
baseline and heuristics arms use identical code paths. Supported envs:

- ``HELION_LLM_ROUND0_HEURISTIC_PATH``: path to a generated AOT
  ``heuristic_*.py`` file. When set, the runner calls ``autotune_<kname>``
  on the live args and passes the returned config as a round-0 seed to
  ``LLMGuidedSearch``.

Example::

    # baseline
    python run_live.py --kernel cross_entropy --arm baseline --repeats 3 \
        --shape-grid .../shape_grid.json --output-dir .../N0_live/baseline

    # heuristics with AOT seed
    HELION_LLM_ROUND0_HEURISTIC_PATH=.../heuristic_cross_entropy.py \
    python run_live.py --kernel cross_entropy --arm heuristics --repeats 3 \
        --shape-grid .../shape_grid.json --output-dir .../Nx/heuristics
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

from llm_heuristics_artifacts.loss_functions.tools.workloads import build_kernel_and_args
import helion
from helion.autotuner.base_search import BenchmarkResult
from helion.autotuner.llm_search import LLMGuidedSearch
from helion.runtime.config import Config


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
        return None
    cfg_dict = autotune_fn(*args)
    if cfg_dict is None:
        return None
    return Config(**cfg_dict)


class _SeededLLMGuidedSearch(LLMGuidedSearch):
    """LLMGuidedSearch variant that adds a heuristic-picked seed config."""

    def __init__(self, *args, seed_config: Config | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._extra_seed_config = seed_config

    def _build_seed_configs(self):
        seeds = super()._build_seed_configs()
        if self._extra_seed_config is None:
            return seeds
        # Put the heuristic seed at the front; dedupe by flatten key.
        seen = {self._config_key(c) for c in seeds}
        extra_key = self._config_key(self._extra_seed_config)
        if extra_key in seen:
            return seeds
        return [self._extra_seed_config, *seeds]


def _run_one_shape(
    kernel_name: str,
    shape_entry: dict[str, Any],
    dtype_name: str,
    *,
    heuristic_path: Path | None,
    llm_model: str,
    llm_provider: str,
    configs_per_round: int,
    initial_random_configs: int,
    max_rounds: int,
    request_timeout_s: float,
    track_round_progress: bool = False,
    round_progress_data: list[dict[str, Any]] | None = None,
    repeat_num: int = 0,
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
        max_rounds=max_rounds,
        finishing_rounds=0,
        request_timeout_s=request_timeout_s,
        seed_config=seed_config,
    )

    # Track per-round progress if requested
    if track_round_progress and round_progress_data is not None:
        # Wrap the _finalize_round method to capture best config after each round
        original_finalize_round = search._finalize_round

        def _wrapped_finalize_round(round_num: int) -> None:
            original_finalize_round(round_num)
            # After finalize, best config is updated - capture it
            if search.population:
                best_member = min(search.population, key=lambda m: m.perf)
                # PopulationMember.perf is in milliseconds (from BenchmarkResult.perf)
                best_perf_ms = best_member.perf if best_member.status == "ok" else float("inf")

                # Calculate improvement from round 0
                if round_num == 0:
                    improvement_pct = 0.0
                else:
                    # Find round 0 best
                    round_0_entry = next(
                        (e for e in round_progress_data
                         if e["kernel"] == kernel_name
                         and e["shape_id"] == shape_entry["id"]
                         and e["repeat"] == repeat_num
                         and e["round"] == 0),
                        None
                    )
                    if round_0_entry and round_0_entry["best_so_far_ms"] > 0:
                        improvement_pct = (
                            (round_0_entry["best_so_far_ms"] - best_perf_ms)
                            / round_0_entry["best_so_far_ms"] * 100.0
                        )
                    else:
                        improvement_pct = 0.0

                round_progress_data.append({
                    "kernel": kernel_name,
                    "shape_id": shape_entry["id"],
                    "repeat": repeat_num,
                    "round": round_num,
                    "best_so_far_ms": best_perf_ms,
                    "new_configs_tested": configs_per_round if round_num > 0 else initial_random_configs + 1,
                    "improvement_pct": improvement_pct,
                })

        search._finalize_round = _wrapped_finalize_round

    search.autotune()
    rows: list[dict[str, Any]] = []
    for res in search._all_benchmark_results:
        # res.perf is ALREADY in milliseconds per Helion's BenchmarkResult contract.
        # (Previously this multiplied by 1000, inflating all CSV numbers by 1000×.)
        perf_ms = res.perf if res.status == "ok" else float("inf")
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
    parser.add_argument(
        "--kernel",
        required=True,
        choices=["cross_entropy", "jsd", "kl_div", "grpo_loss", "fused_linear_jsd", "softmax", "matmul", "attention", "layernorm"],
    )
    parser.add_argument("--arm", required=True, choices=["baseline", "heuristics"])
    parser.add_argument("--shape-grid", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--configs-per-round", type=int, default=5)
    parser.add_argument("--max-rounds", type=int, default=1, help="Number of LLM refinement rounds")
    parser.add_argument("--initial-random-configs", type=int, default=3)
    parser.add_argument("--request-timeout-s", type=float, default=600.0)
    parser.add_argument(
        "--only-split",
        choices=["train", "heldout", "all"],
        default="all",
        help="Run only shapes with this split tag.",
    )
    parser.add_argument(
        "--only-shape",
        default=None,
        help="Run only shapes with matching id (comma-separated for multiple).",
    )
    parser.add_argument(
        "--track-round-progress",
        action="store_true",
        help="Track and save best config after each round to a separate CSV.",
    )
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
    else:
        if heuristic_path_env:
            print(
                "WARN: arm=baseline but HELION_LLM_ROUND0_HEURISTIC_PATH is set; "
                "ignoring heuristic path for baseline arm.",
                file=sys.stderr,
            )

    llm_model = os.environ.get("HELION_LLM_MODEL", "us.anthropic.claude-opus-4-7")
    llm_provider = os.environ.get("HELION_LLM_PROVIDER", "bedrock")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / f"{args.kernel}_{args.arm}.csv"
    meta_path = args.output_dir / f"{args.kernel}_{args.arm}.meta.json"
    round_progress_path = args.output_dir / f"{args.kernel}_{args.arm}_round_progress.csv"

    fieldnames = [
        "kernel",
        "shape_id",
        "shape_label",
        "split",
        "repeat",
        "arm",
        "generation",
        "config_hash",
        "config",
        "status",
        "perf_ms",
        "compile_time_s",
        "seeded_by_heuristic",
    ]

    # Prepare round progress tracking
    round_progress_data: list[dict[str, Any]] = []

    wall_t0 = time.perf_counter()
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for shape_entry in shapes:
            for repeat in range(args.repeats):
                torch.manual_seed(20260508 + repeat * 1000)
                try:
                    rows = _run_one_shape(
                        args.kernel,
                        shape_entry,
                        grid.get("dtype", "bfloat16"),
                        heuristic_path=heuristic_path,
                        llm_model=llm_model,
                        llm_provider=llm_provider,
                        configs_per_round=args.configs_per_round,
                        initial_random_configs=args.initial_random_configs,
                        max_rounds=args.max_rounds,
                        request_timeout_s=args.request_timeout_s,
                        track_round_progress=args.track_round_progress,
                        round_progress_data=round_progress_data,
                        repeat_num=repeat,
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

    if args.track_round_progress and round_progress_data:
        rp_fields = [
            "kernel",
            "shape_id",
            "repeat",
            "round",
            "best_so_far_ms",
            "new_configs_tested",
            "improvement_pct",
        ]
        with open(round_progress_path, "w", newline="") as rf:
            rw = csv.DictWriter(rf, fieldnames=rp_fields)
            rw.writeheader()
            for entry in round_progress_data:
                rw.writerow(entry)

    meta = {
        "version": 1,
        "kernel": args.kernel,
        "arm": args.arm,
        "shape_grid_path": str(args.shape_grid.resolve()),
        "dtype": grid.get("dtype", "bfloat16"),
        "shapes_run": [s["id"] for s in shapes],
        "repeats": args.repeats,
        "configs_per_round": args.configs_per_round,
        "max_rounds": args.max_rounds,
        "initial_random_configs": args.initial_random_configs,
        "request_timeout_s": args.request_timeout_s,
        "llm_model": llm_model,
        "llm_provider": llm_provider,
        "heuristic_path": str(heuristic_path) if heuristic_path else None,
        "helion_anthropic_thinking_budget": os.environ.get(
            "HELION_LLM_ANTHROPIC_THINKING_BUDGET"
        ),
        "helion_anthropic_effort": os.environ.get(
            "HELION_LLM_ANTHROPIC_REASONING_EFFORT"
        ),
        "csv_path": str(csv_path.resolve()),
        "round_progress_path": str(round_progress_path.resolve()) if args.track_round_progress else None,
        "wall_elapsed_s": wall_elapsed,
        "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Wrote {csv_path}")
    print(f"Wrote {meta_path}")
    if args.track_round_progress and round_progress_data:
        print(f"Wrote {round_progress_path} ({len(round_progress_data)} rows)")
    print(f"Elapsed: {wall_elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
