# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Run overnight autoresearch loops for observed LLM autotuner heuristics.

This driver intentionally keeps the measured product question narrow:

    Does the full heuristic mechanism beat baseline on final kernel runtime and
    end-to-end autotune wall time?

It runs repeated benchmark suites through ``llm_heuristics_experiment.py``,
aggregates geomean cost ratios, and optionally asks Claude Opus 4.7 to critique
the measured results and propose the next iteration.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
from typing import TYPE_CHECKING
from typing import cast

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_SCRIPT = REPO_ROOT / "scripts" / "llm_heuristics_experiment.py"
LOOP_DIR = Path("/tmp/helion_heuristics_loop")
DEFAULT_MODEL = "claude-opus-4-7"
DEFAULT_CLAUDE_MODEL = "claude-opus-4-7"

SUITES: dict[str, dict[str, object]] = {
    "core_rows": {
        "workloads": [
            "softmax_4k_1k",
            "softmax_4k_2k",
            "softmax_1k_1k",
            "softmax_1k_2k",
            "softmax_2k_1k",
            "softmax_2k_2k",
            "softmax_1k_4k",
            "softmax_2k_4k",
            "rms_norm_4k",
            "rms_norm_4k_1k",
            "rms_norm_4k_2k",
            "rms_norm_2048x4096",
            "rms_norm_8192x2048",
            "rms_norm_1024x8192",
            "rms_norm_1024x16384",
            "softmax_4k",
            "softmax_2k_8k",
            "softmax_1k_16k",
            "layer_norm_4k",
            "layer_norm_4k_1k",
            "layer_norm_4k_2k",
            "layer_norm_2k_8k",
            "layer_norm_1k_16k",
            "cross_entropy_32k",
            "cross_entropy_4k_16k",
            "cross_entropy_1k_64k",
            "sum_5120x2560",
            "sum_4096x1024",
            "sum_2048x8192",
        ],
        "arms": ["baseline", "heuristics_product", "heuristics_product_top1"],
    },
    "broad": {
        "workloads": [
            "add_1m",
            "add_16m",
            "exp_1m",
            "exp_16m",
            "geglu_4096x4096",
            "geglu_2048x8192",
            "swiglu_4096x4096",
            "swiglu_2048x8192",
            "batch_softmax_16x512x1024",
            "batch_softmax_8x1024x2048",
            "bmm_8x256x384x512",
            "bmm_16x128x512x256",
            "matmul_1k",
            "matmul_skinny_m",
            "matmul_skinny_n",
            "matmul_k_heavy",
            "matmul_split_k_64x32768x64",
            "attention_1k_d64",
        ],
        "arms": ["baseline", "heuristics_product"],
    },
    "attention": {
        "workloads": [
            "attention_512_d64",
            "attention_1k_d64",
            "attention_2k_d64",
            "attention_2k_d128",
            "attention_4k_d64",
            "attention_4k_d128",
        ],
        "arms": ["baseline", "heuristics_product", "heuristics_product_top1"],
    },
    "matmul_control": {
        "workloads": [
            "matmul_256",
            "matmul_512",
            "matmul_1k",
            "matmul_2k",
            "matmul_skinny_m",
            "matmul_skinny_n",
            "matmul_k_heavy",
            "bmm_8x256x384x512",
            "bmm_16x128x512x256",
        ],
        "arms": ["baseline", "heuristics", "heuristics_product"],
    },
}
SUITES["overnight"] = {
    "workloads": list(
        dict.fromkeys(
            [
                *cast("list[str]", SUITES["core_rows"]["workloads"]),
                *cast("list[str]", SUITES["broad"]["workloads"]),
                *cast("list[str]", SUITES["attention"]["workloads"]),
            ]
        )
    ),
    "arms": ["baseline", "heuristics_product"],
}


def _geomean(values: Iterable[float]) -> float:
    collected = list(values)
    return math.exp(sum(math.log(value) for value in collected) / len(collected))


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _stats(values: Sequence[float]) -> dict[str, float]:
    return {
        "geomean": _geomean(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p25": _percentile(values, 0.25),
        "p75": _percentile(values, 0.75),
        "min": min(values),
        "max": max(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


_COST_METRIC_FIELDS = (
    "perf_cost",
    "time_cost",
    "cfg_cost",
    "compile_time_total_cost",
    "compile_time_mean_cost",
    "compile_time_p75_cost",
    "compile_time_p90_cost",
    "compile_time_max_cost",
    "benchmark_time_total_cost",
    "benchmark_time_mean_cost",
    "benchmark_time_p75_cost",
    "benchmark_time_p90_cost",
    "benchmark_time_max_cost",
)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _stats_ratio(
    arm_stats: dict[str, object],
    base_stats: dict[str, object],
    key: str,
) -> float | None:
    arm_value = _optional_float(arm_stats.get(key))
    base_value = _optional_float(base_stats.get(key))
    if arm_value is None or base_value is None or base_value <= 0:
        return None
    return arm_value / base_value


def _summarize_metric(
    rows: Sequence[dict[str, object]], name: str
) -> dict[str, float] | None:
    values = [
        parsed
        for row in rows
        if (parsed := _optional_float(row.get(name))) is not None and parsed > 0
    ]
    return _stats(values) if values else None


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _suite_defaults(name: str) -> tuple[list[str], list[str]]:
    suite = SUITES[name]
    return list(cast("list[str]", suite["workloads"])), list(
        cast("list[str]", suite["arms"])
    )


def _run_repeat(args: argparse.Namespace, repeat: int) -> None:
    output_dir = Path(args.output_root) / f"repeat_{repeat:02d}"
    summary = output_dir / "summary.json"
    if summary.exists() and not args.force:
        print(f"repeat {repeat}: using existing {summary}", flush=True)
        return

    cmd = [
        sys.executable,
        str(EXPERIMENT_SCRIPT),
        "suite",
        "--workloads",
        ",".join(args.workloads),
        "--arms",
        ",".join(args.arms),
        "--autotuner",
        args.autotuner,
        "--model",
        args.model,
        "--effort",
        args.effort,
        "--verify-runs",
        str(args.verify_runs),
        "--timeout-s",
        str(args.timeout_s),
        "--output-dir",
        str(output_dir),
    ]
    if args.range_heuristics_path is not None:
        cmd.extend(["--range-heuristics-path", str(args.range_heuristics_path)])
    if args.llm_max_rounds is not None:
        cmd.extend(["--llm-max-rounds", str(args.llm_max_rounds)])
    if args.llm_configs_per_round is not None:
        cmd.extend(["--llm-configs-per-round", str(args.llm_configs_per_round)])
    if args.llm_initial_random_configs is not None:
        cmd.extend(
            [
                "--llm-initial-random-configs",
                str(args.llm_initial_random_configs),
            ]
        )
    if args.llm_round0_mode != "off":
        cmd.extend(["--llm-round0-mode", args.llm_round0_mode])
        if args.llm_round0_dir is not None:
            cmd.extend(["--llm-round0-dir", str(args.llm_round0_dir)])
    env = dict(os.environ)
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    log_path = Path(args.output_root) / f"repeat_{repeat:02d}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"repeat {repeat}: running {' '.join(cmd)}", flush=True)
    print(f"repeat {repeat}: log {log_path}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        print(
            f"repeat {repeat}: experiment failed with exit {completed.returncode}; "
            f"see {log_path}",
            flush=True,
        )


def _load_cost_rows(
    output_root: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    cost_rows: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    for summary_path in sorted(output_root.glob("repeat_*/summary.json")):
        repeat = summary_path.parent.name
        raw_results = json.loads(summary_path.read_text())
        by_workload: dict[str, dict[str, dict[str, object]]] = {}
        for raw in raw_results:
            if "error" in raw:
                errors.append({"repeat": repeat, **raw})
                continue
            by_workload.setdefault(str(raw["workload"]), {})[str(raw["arm"])] = raw
        for workload, by_arm in sorted(by_workload.items()):
            baseline = by_arm.get("baseline")
            if baseline is None:
                errors.append(
                    {
                        "repeat": repeat,
                        "workload": workload,
                        "arm": "baseline",
                        "error": "missing baseline",
                    }
                )
                continue
            base_verified = cast("dict[str, object]", baseline["verified"])
            base_ms = float(base_verified["median_ms"])
            base_time = float(baseline["wall_time_s"])
            base_configs = int(baseline["configs_tested"])
            base_compile_stats = cast(
                "dict[str, object]",
                baseline.get("compile_time_per_config_stats", {}),
            )
            base_benchmark_stats = cast(
                "dict[str, object]",
                baseline.get("benchmark_time_per_batch_stats", {}),
            )
            for arm, result in sorted(by_arm.items()):
                if arm == "baseline":
                    continue
                verified = cast("dict[str, object]", result["verified"])
                arm_ms = float(verified["median_ms"])
                arm_time = float(result["wall_time_s"])
                arm_configs = int(result["configs_tested"])
                arm_compile_stats = cast(
                    "dict[str, object]",
                    result.get("compile_time_per_config_stats", {}),
                )
                arm_benchmark_stats = cast(
                    "dict[str, object]",
                    result.get("benchmark_time_per_batch_stats", {}),
                )
                row: dict[str, object] = {
                    "repeat": repeat,
                    "workload": workload,
                    "arm": arm,
                    "kernel_class": result.get("kernel_class"),
                    "matched_observed_rule": result.get("matched_observed_rule"),
                    "shape_bucket": result.get("shape_bucket"),
                    "base_ms": base_ms,
                    "arm_ms": arm_ms,
                    "perf_cost": arm_ms / base_ms,
                    "base_time_s": base_time,
                    "arm_time_s": arm_time,
                    "time_cost": arm_time / base_time,
                    "base_configs": base_configs,
                    "arm_configs": arm_configs,
                    "cfg_cost": arm_configs / base_configs,
                    "base_compile_time_per_config_stats": base_compile_stats,
                    "arm_compile_time_per_config_stats": arm_compile_stats,
                    "base_benchmark_time_per_batch_stats": base_benchmark_stats,
                    "arm_benchmark_time_per_batch_stats": arm_benchmark_stats,
                }
                for stat_name in ("total", "mean", "p75", "p90", "max"):
                    row[f"compile_time_{stat_name}_cost"] = _stats_ratio(
                        arm_compile_stats,
                        base_compile_stats,
                        stat_name,
                    )
                    row[f"benchmark_time_{stat_name}_cost"] = _stats_ratio(
                        arm_benchmark_stats,
                        base_benchmark_stats,
                        stat_name,
                    )
                cost_rows.append(row)
    return cost_rows, errors


def _summarize(cost_rows: Sequence[dict[str, object]]) -> dict[str, object]:
    by_arm: dict[str, dict[str, dict[str, float]]] = {}
    by_workload_arm: dict[str, dict[str, dict[str, float]]] = {}
    for arm in sorted({str(row["arm"]) for row in cost_rows}):
        rows = [row for row in cost_rows if row["arm"] == arm]
        by_arm[arm] = {}
        for metric in _COST_METRIC_FIELDS:
            metric_summary = _summarize_metric(rows, metric)
            if metric_summary is not None:
                by_arm[arm][metric] = metric_summary
    for workload in sorted({str(row["workload"]) for row in cost_rows}):
        for arm in sorted({str(row["arm"]) for row in cost_rows}):
            rows = [
                row
                for row in cost_rows
                if row["workload"] == workload and row["arm"] == arm
            ]
            if not rows:
                continue
            by_workload_arm[f"{workload}:{arm}"] = {}
            for metric in _COST_METRIC_FIELDS:
                metric_summary = _summarize_metric(rows, metric)
                if metric_summary is not None:
                    by_workload_arm[f"{workload}:{arm}"][metric] = metric_summary
    return {"summary_by_arm": by_arm, "summary_by_workload_arm": by_workload_arm}


def _write_artifacts(
    *,
    args: argparse.Namespace,
    cost_rows: Sequence[dict[str, object]],
    errors: Sequence[dict[str, object]],
) -> tuple[Path, Path]:
    output_root = Path(args.output_root)
    summary = _summarize(cost_rows)
    artifact = {
        "suite": args.suite,
        "workloads": args.workloads,
        "arms": args.arms,
        "repeats": args.repeats,
        "autotuner": args.autotuner,
        "model": args.model,
        "rows": list(cost_rows),
        "errors": list(errors),
        **summary,
    }
    json_path = output_root / "aggregate_results.json"
    json_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    lines: list[str] = [
        f"# LLM heuristic autoresearch: {args.suite}",
        "",
        f"- Output root: `{output_root}`",
        f"- Autotuner: `{args.autotuner}`",
        f"- Model: `{args.model}`",
        f"- Repeats requested: `{args.repeats}`",
        f"- Workloads: `{', '.join(args.workloads)}`",
        f"- Arms: `{', '.join(args.arms)}`",
        "",
        "Lower is better.",
        "",
        "| arm | perf geo | time geo | cfg geo | perf range | time range | cfg range |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, by_metric in sorted(summary["summary_by_arm"].items()):
        perf = by_metric["perf_cost"]
        time_cost = by_metric["time_cost"]
        cfg = by_metric["cfg_cost"]
        lines.append(
            f"| {arm} | {perf['geomean']:.3f} | {time_cost['geomean']:.3f} | "
            f"{cfg['geomean']:.3f} | {perf['min']:.3f}-{perf['max']:.3f} | "
            f"{time_cost['min']:.3f}-{time_cost['max']:.3f} | "
            f"{cfg['min']:.3f}-{cfg['max']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Per-workload geomeans:",
            "",
            "| workload | arm | perf geo | time geo | cfg geo |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for key, by_metric in sorted(summary["summary_by_workload_arm"].items()):
        workload, arm = key.split(":", 1)
        lines.append(
            f"| {workload} | {arm} | "
            f"{by_metric['perf_cost']['geomean']:.3f} | "
            f"{by_metric['time_cost']['geomean']:.3f} | "
            f"{by_metric['cfg_cost']['geomean']:.3f} |"
        )
    attention_keys = [
        key
        for key in sorted(summary["summary_by_workload_arm"])
        if key.startswith("attention_")
    ]
    if attention_keys:
        lines.extend(
            [
                "",
                "Attention time distribution:",
                "",
                "| workload | arm | perf geo | time geo | time p25 | time median | time p75 | time max | compile total | compile max | bench total |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for key in attention_keys:
            workload, arm = key.split(":", 1)
            by_metric = summary["summary_by_workload_arm"][key]
            perf = by_metric["perf_cost"]
            time_cost = by_metric["time_cost"]
            compile_total = by_metric.get("compile_time_total_cost", {})
            compile_max = by_metric.get("compile_time_max_cost", {})
            benchmark_total = by_metric.get("benchmark_time_total_cost", {})
            lines.append(
                f"| {workload} | {arm} | "
                f"{perf['geomean']:.3f} | "
                f"{time_cost['geomean']:.3f} | "
                f"{time_cost['p25']:.3f} | "
                f"{time_cost['median']:.3f} | "
                f"{time_cost['p75']:.3f} | "
                f"{time_cost['max']:.3f} | "
                f"{compile_total.get('geomean', float('nan')):.3f} | "
                f"{compile_max.get('geomean', float('nan')):.3f} | "
                f"{benchmark_total.get('geomean', float('nan')):.3f} |"
            )
    if errors:
        lines.extend(["", "Errors:"])
        for error in errors:
            lines.append(f"- `{error}`")
    md_path = output_root / "aggregate_summary.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(md_path.read_text(), flush=True)
    return json_path, md_path


def _call_claude(args: argparse.Namespace, json_path: Path, md_path: Path) -> None:
    response_path = LOOP_DIR / "claude" / f"{args.suite}_autoresearch_response.md"
    response_path.parent.mkdir(parents=True, exist_ok=True)
    prompt = (
        "You are Claude Opus 4.7 in an autoresearch loop with Codex. "
        f"Read {md_path} and {json_path}. "
        "The only product question is whether the full heuristics mechanism "
        "beats baseline on final kernel perf and autotune wall time across many kernels. "
        "Do not recommend prompt/seeds as product policies. "
        "Argue with Codex, identify which heuristic classes are actually helping, "
        "which should be disabled, what data is missing, and what exact next "
        f"experiment should run. Write your response to {response_path}. "
        "Update /tmp/helion_heuristics_loop/claude/proposed_policy.json only if "
        "the policy should change. Do not run benchmarks."
    )
    cmd = [
        "claude",
        "-p",
        "--model",
        args.claude_model,
        "--effort",
        "max",
        "--permission-mode",
        "bypassPermissions",
        "--add-dir",
        str(LOOP_DIR),
        "--add-dir",
        str(REPO_ROOT),
        "--max-budget-usd",
        str(args.claude_budget_usd),
        "--output-format",
        "text",
        prompt,
    ]
    print(f"Calling Claude: {' '.join(cmd[:12])} ...", flush=True)
    try:
        completed = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            text=True,
            check=False,
            timeout=args.claude_timeout_s,
        )
        if completed.returncode != 0:
            print(f"Claude exited {completed.returncode}", flush=True)
    except subprocess.TimeoutExpired:
        print(
            f"Claude timed out after {args.claude_timeout_s}s; "
            f"checking whether it wrote {response_path}",
            flush=True,
        )
    if response_path.exists():
        print(f"Claude response: {response_path}", flush=True)
    else:
        print("Claude response was not written", flush=True)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=sorted(SUITES), default="core_rows")
    parser.add_argument("--workloads", default="")
    parser.add_argument("--arms", default="")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--output-root", type=Path, default=Path("/tmp/helion_llm_autoresearch")
    )
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--autotuner", default="LLMSeededLFBOTreeSearch")
    parser.add_argument("--effort", default="full", choices=["quick", "full"])
    parser.add_argument("--verify-runs", type=int, default=10)
    parser.add_argument("--timeout-s", type=float, default=1800.0)
    parser.add_argument("--range-heuristics-path", type=Path, default=None)
    parser.add_argument("--llm-max-rounds", type=int, default=None)
    parser.add_argument("--llm-configs-per-round", type=int, default=None)
    parser.add_argument("--llm-initial-random-configs", type=int, default=None)
    parser.add_argument(
        "--llm-round0-mode",
        choices=("off", "record", "replay", "paired-no-match"),
        default="off",
    )
    parser.add_argument("--llm-round0-dir", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-run", action="store_true")
    parser.add_argument("--call-claude", action="store_true")
    parser.add_argument("--claude-model", default=DEFAULT_CLAUDE_MODEL)
    parser.add_argument("--claude-budget-usd", type=float, default=8.0)
    parser.add_argument("--claude-timeout-s", type=float, default=900.0)
    args = parser.parse_args(argv)
    workloads, arms = _suite_defaults(args.suite)
    if args.workloads:
        workloads = _split_csv(args.workloads)
    if args.arms:
        arms = _split_csv(args.arms)
    args.workloads = workloads
    args.arms = arms
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    Path(args.output_root).mkdir(parents=True, exist_ok=True)
    if not args.skip_run:
        for repeat in range(1, args.repeats + 1):
            _run_repeat(args, repeat)
    cost_rows, errors = _load_cost_rows(Path(args.output_root))
    if not cost_rows:
        raise RuntimeError(f"No successful cost rows found in {args.output_root}")
    json_path, md_path = _write_artifacts(args=args, cost_rows=cost_rows, errors=errors)
    if args.call_claude:
        _call_claude(args, json_path, md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
