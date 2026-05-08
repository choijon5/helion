"""Live benchmark runner for the norm hill-climbing loop.

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
    python run_live.py --kernel layer_norm --arm baseline --repeats 3 \
        --shape-grid .../shape_grid.json --output-dir .../N0_live/baseline

    # heuristics with AOT seed
    HELION_LLM_ROUND0_HEURISTIC_PATH=.../heuristic_layer_norm.py \
    python run_live.py --kernel layer_norm --arm heuristics --repeats 3 \
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

from llm_heuristics_artifacts.norms.tools.workloads import build_kernel_and_args
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
        # Fall back to the first autotune_* function in the module. The
        # archive-side kernel name (e.g. "rms_norm") can differ from the
        # Python function name (e.g. "rms_norm_fwd"), so looking up by the
        # exact live-kernel name may miss.
        for attr in dir(module):
            if attr.startswith("autotune_") and callable(getattr(module, attr)):
                autotune_fn = getattr(module, attr)
                break
    if autotune_fn is None:
        raise RuntimeError(
            f"No autotune_* function found in heuristic module {heuristic_path}"
        )
    cfg_dict = autotune_fn(*args)
    if cfg_dict is None:
        return None
    return Config(**cfg_dict)


def _load_round0_heuristic_library(heuristic_path: Path) -> list[Config]:
    """Extract the full list of candidate configs (``_C = [...]``) from a
    generated AOT heuristic file.

    The tree-chosen config from ``_load_round0_heuristic_config`` is
    always one of these; using the whole list as seeds gives round 0
    many good starting points instead of one.
    """
    import ast

    tree = ast.parse(heuristic_path.read_text())
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "_C"
            and isinstance(node.value, ast.List)
        ):
            configs: list[Config] = []
            for elt in node.value.elts:
                if not isinstance(elt, ast.Dict):
                    continue
                try:
                    cfg_dict = ast.literal_eval(elt)
                except (ValueError, SyntaxError):
                    continue
                configs.append(Config(**cfg_dict))
            if configs:
                return configs
    raise RuntimeError(
        f"Could not extract _C config list from heuristic module {heuristic_path}"
    )


class _SeededLLMGuidedSearch(LLMGuidedSearch):
    """LLMGuidedSearch variant that adds heuristic-picked seed configs.

    Two modes:
      - ``seed_config`` (single config): inserted at the front of the seed
        list; default + random seeds are preserved.
      - ``seed_library`` (list of configs): ALL library configs are
        inserted at the front of the seed list. This mode is intended
        for "heuristic dominates round 0" experiments (N2b onwards)
        where we also drop ``initial_random_configs`` to 0 at the caller.
    """

    def __init__(
        self,
        *args,
        seed_config: Config | None = None,
        seed_library: list[Config] | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._extra_seed_config = seed_config
        self._extra_seed_library: list[Config] = list(seed_library or [])
        self._seed_config_compile_failed = False

    def _build_seed_configs(self):
        seeds = super()._build_seed_configs()
        extras: list[Config] = []
        if self._extra_seed_library:
            extras.extend(self._extra_seed_library)
        elif self._extra_seed_config is not None:
            extras.append(self._extra_seed_config)
        if not extras:
            return seeds
        seen = {self._config_key(c) for c in seeds}
        out: list[Config] = []
        for cfg in extras:
            key = self._config_key(cfg)
            if key in seen:
                continue
            seen.add(key)
            out.append(cfg)
        return [*out, *seeds]

    def _ingest_results(self, results):
        """Detect whether the heuristic-seeded config compiled and ran ok."""
        super()._ingest_results(results)
        if self._extra_seed_config is None:
            return
        seed_key = self._config_key(self._extra_seed_config)
        for r in results:
            if self._config_key(r.config) == seed_key and r.status != "ok":
                self._seed_config_compile_failed = True
                break


def _run_one_shape(
    kernel_name: str,
    shape_entry: dict[str, Any],
    dtype_name: str,
    *,
    heuristic_path: Path | None,
    heuristic_mode: str,
    llm_model: str,
    llm_provider: str,
    configs_per_round: int,
    initial_random_configs: int,
    request_timeout_s: float,
) -> list[dict[str, Any]]:
    kernel_fn, args = build_kernel_and_args(kernel_name, shape_entry, dtype_name)
    bound = kernel_fn.bind(args)
    seed_config = None
    seed_library: list[Config] = []
    effective_randoms = initial_random_configs
    if heuristic_path is not None:
        if heuristic_mode == "library":
            seed_library = _load_round0_heuristic_library(heuristic_path)
            effective_randoms = 0
            # Tree-picked config: prepend so it is still first after dedupe.
            picked = _load_round0_heuristic_config(
                heuristic_path, kernel_fn.name, args
            )
            if picked is not None:
                lib_keys = {json.dumps(dict(c), sort_keys=True) for c in seed_library}
                pk = json.dumps(dict(picked), sort_keys=True)
                if pk in lib_keys:
                    # reorder: picked first, then others
                    seed_library = [picked] + [
                        c
                        for c in seed_library
                        if json.dumps(dict(c), sort_keys=True) != pk
                    ]
                else:
                    seed_library = [picked, *seed_library]
                seed_config = picked
        else:  # "tree" (single tree-picked config)
            seed_config = _load_round0_heuristic_config(
                heuristic_path, kernel_fn.name, args
            )
    search = _SeededLLMGuidedSearch(
        bound,
        args,
        provider=llm_provider,
        model=llm_model,
        configs_per_round=configs_per_round,
        initial_random_configs=effective_randoms,
        max_rounds=1,
        finishing_rounds=0,
        request_timeout_s=request_timeout_s,
        seed_config=seed_config if not seed_library else None,
        seed_library=seed_library,
    )
    search.autotune()
    rows: list[dict[str, Any]] = []
    library_hashes = {_config_hash(c) for c in seed_library}
    for res in search._all_benchmark_results:
        perf_ms = res.perf * 1000.0 if res.status == "ok" else float("inf")
        ch = _config_hash(res.config)
        seeded = "0"
        if seed_library and ch in library_hashes:
            seeded = "1"
        elif seed_config is not None and ch == _config_hash(seed_config):
            seeded = "1"
        rows.append(
            {
                "generation": 0,
                "config_hash": ch,
                "config": json.dumps(dict(res.config), sort_keys=True),
                "status": res.status,
                "perf_ms": perf_ms,
                "compile_time_s": res.compile_time if res.compile_time is not None else "",
                "seeded_by_heuristic": seeded,
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", required=True, choices=["layer_norm", "rms_norm", "softmax"])
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
        help="Run only shapes with this split tag.",
    )
    parser.add_argument(
        "--only-shape",
        default=None,
        help="Run only shapes with matching id (comma-separated for multiple).",
    )
    parser.add_argument(
        "--heuristic-mode",
        choices=["tree", "library"],
        default="tree",
        help="How to consume the heuristic. 'tree' injects only the "
             "decision-tree-chosen config; 'library' injects all configs "
             "from the heuristic's _C list and drops random seeds.",
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
                        heuristic_mode=args.heuristic_mode,
                        llm_model=llm_model,
                        llm_provider=llm_provider,
                        configs_per_round=args.configs_per_round,
                        initial_random_configs=args.initial_random_configs,
                        request_timeout_s=args.request_timeout_s,
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
        "dtype": grid.get("dtype", "bfloat16"),
        "shapes_run": [s["id"] for s in shapes],
        "repeats": args.repeats,
        "configs_per_round": args.configs_per_round,
        "initial_random_configs": args.initial_random_configs,
        "request_timeout_s": args.request_timeout_s,
        "llm_model": llm_model,
        "llm_provider": llm_provider,
        "heuristic_path": str(heuristic_path) if heuristic_path else None,
        "heuristic_mode": args.heuristic_mode,
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
