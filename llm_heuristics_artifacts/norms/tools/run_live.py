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
from helion.autotuner.llm_seeded_lfbo import LLMSeededLFBOTreeSearch
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


def _load_archive_evidence(
    archive_csvs: list[Path],
    args: tuple[Any, ...],
    *,
    k_nearest_shapes: int = 4,
    top_k_configs: int = 3,
) -> str | None:
    """Build a compact evidence block for the round-0 prompt.

    Finds the ``k_nearest_shapes`` archived shapes closest to the live
    args (by log(numel) + log(row/col ratio)) and lists the
    ``top_k_configs`` fastest archived configs per match. Returns a
    human-readable block suitable to prepend to the LLM prompt, or
    ``None`` if the archive has no compatible shapes.

    The match uses only shape features, never config hashes or
    archive-only identifiers, so the output is safe to include in a
    prompt without feature-audit hits.
    """
    import csv as _csv
    import math as _math

    if not archive_csvs:
        return None

    # Extract live-shape features
    live_tensor = args[0] if args and isinstance(args[0], torch.Tensor) else None
    if live_tensor is None or live_tensor.ndim < 2:
        return None
    live_rows = int(live_tensor.shape[0])
    live_cols = int(live_tensor.shape[1])
    if live_rows <= 0 or live_cols <= 0:
        return None

    live_log_numel = _math.log2(live_rows * live_cols)
    live_log_ratio = _math.log2(max(live_cols, 1) / max(live_rows, 1))

    # Group archive rows by (shape_hash, rows, cols); pick fastest config per hash
    shape_info: dict[str, dict[str, Any]] = {}
    for csv_path in archive_csvs:
        try:
            with open(csv_path) as f:
                reader = _csv.DictReader(f)
                for row in reader:
                    try:
                        features = json.loads(row["shape_features"])
                        cfg = json.loads(row["config"])
                        perf = float(row["timing_ms"])
                    except (KeyError, ValueError, json.JSONDecodeError):
                        continue
                    rows_a = features.get("arg0_dim0")
                    cols_a = features.get("arg0_dim1")
                    if rows_a is None or cols_a is None:
                        continue
                    h = row.get("shape_hash", "")
                    if h not in shape_info:
                        shape_info[h] = {
                            "rows": int(rows_a),
                            "cols": int(cols_a),
                            "configs": [],
                        }
                    shape_info[h]["configs"].append((perf, cfg))
        except (FileNotFoundError, OSError):
            continue

    if not shape_info:
        return None

    # Rank shapes by distance to live shape
    def _dist(entry: dict[str, Any]) -> float:
        r, c = entry["rows"], entry["cols"]
        if r <= 0 or c <= 0:
            return float("inf")
        return abs(_math.log2(r * c) - live_log_numel) + abs(
            _math.log2(c / r) - live_log_ratio
        )

    ranked = sorted(shape_info.items(), key=lambda kv: _dist(kv[1]))[
        :k_nearest_shapes
    ]
    if not ranked:
        return None

    lines: list[str] = [
        "",
        "## Archived measurements for similar shapes",
        "",
        "The following configs were tuned offline on this kernel for "
        "shapes near the one you are about to propose configs for, on "
        "NVIDIA B200. Lower timing is better. Use these as evidence, "
        "not a constraint. Prefer 3-5 configs that look like these or "
        "like plausible refinements, plus 1-2 that try a different "
        "family.",
        "",
        f"Live shape: rows={live_rows}, cols={live_cols}, "
        f"numel={live_rows * live_cols}, dtype={live_tensor.dtype}.",
        "",
    ]
    for _, entry in ranked:
        top_cfgs = sorted(entry["configs"], key=lambda t: t[0])[:top_k_configs]
        if not top_cfgs:
            continue
        lines.append(
            f"Archived shape rows={entry['rows']}, cols={entry['cols']} "
            f"(numel={entry['rows'] * entry['cols']}):"
        )
        for perf, cfg in top_cfgs:
            brief = {
                k: cfg.get(k)
                for k in (
                    "block_sizes",
                    "num_warps",
                    "num_stages",
                    "pid_type",
                    "indexing",
                    "reduction_loops",
                    "range_warp_specializes",
                )
                if k in cfg
            }
            lines.append(f"  - {brief}  {perf:.3f} ms")
        lines.append("")
    return "\n".join(lines)


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
        evidence_block: str | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._extra_seed_config = seed_config
        self._extra_seed_library: list[Config] = list(seed_library or [])
        self._evidence_block = evidence_block
        self._seed_config_compile_failed = False

    def _build_initial_prompt(self) -> str:
        base = super()._build_initial_prompt()
        if not self._evidence_block:
            return base
        return f"{base}\n{self._evidence_block}"

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
    evidence_archives: list[Path] | None = None,
) -> list[dict[str, Any]]:
    kernel_fn, args = build_kernel_and_args(kernel_name, shape_entry, dtype_name)
    bound = kernel_fn.bind(args)
    seed_config = None
    seed_library: list[Config] = []
    effective_randoms = initial_random_configs
    evidence_block: str | None = None
    if evidence_archives:
        evidence_block = _load_archive_evidence(evidence_archives, args)
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
        evidence_block=evidence_block,
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
    parser.add_argument(
        "--evidence-archive",
        default=None,
        help="Comma-separated list of archive measurement CSVs. When set, "
             "the round-0 prompt includes top configs from the nearest "
             "archived shapes. Env var HELION_LLM_PROMPT_EVIDENCE_CSVS has "
             "the same effect.",
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

    # Evidence archive CSVs: CLI flag OR HELION_LLM_PROMPT_EVIDENCE_CSVS env.
    evidence_arg = args.evidence_archive or os.environ.get(
        "HELION_LLM_PROMPT_EVIDENCE_CSVS"
    )
    evidence_archives: list[Path] | None = None
    if evidence_arg:
        evidence_archives = [
            Path(p.strip()) for p in evidence_arg.split(",") if p.strip()
        ]
        for p in evidence_archives:
            if not p.exists():
                raise RuntimeError(f"evidence archive CSV does not exist: {p}")

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
                        evidence_archives=evidence_archives,
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
        "evidence_archives": [str(p) for p in (evidence_archives or [])],
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
