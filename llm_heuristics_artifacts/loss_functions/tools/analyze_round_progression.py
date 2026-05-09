"""Per-round analysis of LLM autotuner baseline data.

Reads the ``{kernel}_baseline_round_progress.csv`` files produced by
``run_live.py --track-round-progress`` and emits a structured report:

- Per-kernel improvement curves (round 0 → N)
- Plateau detection (first round where improvement < 2%)
- Regression detection (rounds where best_so_far got worse)
- Cross-kernel comparison (geometric mean of best_ms per round, relative improvement)
- Actionable recommendations for Phase 2 prompt optimization

Usage::

    python analyze_round_progression.py \\
        --baseline-dir /home/dev/helion_choijon5/llm_heuristics_artifacts/baseline \\
        --output /home/dev/helion_choijon5/llm_heuristics_artifacts/PHASE_1_ANALYSIS.md
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

PLATEAU_THRESHOLD_PCT = 2.0
REGRESSION_THRESHOLD_PCT = -1.0


def _read_round_progress(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                row["repeat"] = int(row["repeat"])
                row["round"] = int(row["round"])
                row["best_so_far_ms"] = float(row["best_so_far_ms"])
                row["new_configs_tested"] = int(row["new_configs_tested"])
                row["improvement_pct"] = float(row["improvement_pct"])
            except (ValueError, KeyError):
                continue
            if math.isinf(row["best_so_far_ms"]) or math.isnan(row["best_so_far_ms"]):
                continue
            rows.append(row)
    return rows


def _geomean(values: list[float]) -> float:
    if not values:
        return float("nan")
    values = [v for v in values if v > 0]
    if not values:
        return float("nan")
    return math.exp(sum(math.log(v) for v in values) / len(values))


def _per_kernel_summary(kernel: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute per-round aggregate best_ms for one kernel.

    For each round, take the best_so_far_ms across repeats per shape, then
    geomean across shapes. Also compute % improvement from previous round.
    """
    max_round = max((r["round"] for r in rows), default=0)
    shape_ids = sorted({r["shape_id"] for r in rows})

    per_round_geomean: list[float] = []
    per_round_shape_bests: list[dict[str, float]] = []
    for rnd in range(max_round + 1):
        shape_bests: dict[str, float] = {}
        for sid in shape_ids:
            candidates = [
                r["best_so_far_ms"]
                for r in rows
                if r["shape_id"] == sid and r["round"] == rnd and r["best_so_far_ms"] > 0
            ]
            if candidates:
                shape_bests[sid] = min(candidates)
        per_round_shape_bests.append(shape_bests)
        per_round_geomean.append(_geomean(list(shape_bests.values())))

    # Per-round delta (vs previous round)
    round_improvements: list[float] = []
    for i in range(len(per_round_geomean)):
        if i == 0 or math.isnan(per_round_geomean[i]) or math.isnan(per_round_geomean[i - 1]):
            round_improvements.append(0.0)
        else:
            prev = per_round_geomean[i - 1]
            curr = per_round_geomean[i]
            round_improvements.append((prev - curr) / prev * 100.0 if prev > 0 else 0.0)

    # Plateau: first round after round 0 where improvement < 2%
    plateau_round: int | None = None
    for i, imp in enumerate(round_improvements):
        if i > 0 and imp < PLATEAU_THRESHOLD_PCT:
            plateau_round = i
            break

    # Regression: any round that made things worse
    regression_rounds = [i for i, imp in enumerate(round_improvements) if i > 0 and imp < REGRESSION_THRESHOLD_PCT]

    # Total improvement (round 0 -> last)
    if len(per_round_geomean) >= 2 and per_round_geomean[0] > 0:
        total_improvement = (per_round_geomean[0] - per_round_geomean[-1]) / per_round_geomean[0] * 100.0
    else:
        total_improvement = 0.0

    return {
        "kernel": kernel,
        "num_shapes": len(shape_ids),
        "max_round": max_round,
        "per_round_geomean_ms": per_round_geomean,
        "per_round_improvement_pct": round_improvements,
        "plateau_round": plateau_round,
        "regression_rounds": regression_rounds,
        "total_improvement_pct": total_improvement,
    }


def _recommendations(summaries: list[dict[str, Any]]) -> list[str]:
    """Generate actionable recommendations from per-kernel summaries."""
    recs: list[str] = []

    # Round with biggest average gain
    num_rounds = max((s["max_round"] for s in summaries), default=0) + 1
    if num_rounds >= 2:
        round_gains: list[list[float]] = [[] for _ in range(num_rounds)]
        for s in summaries:
            for i, imp in enumerate(s["per_round_improvement_pct"]):
                if i < num_rounds:
                    round_gains[i].append(imp)
        avg_per_round = [statistics.mean(rg) if rg else 0.0 for rg in round_gains]
        best_round = max(range(1, num_rounds), key=lambda i: avg_per_round[i]) if num_rounds > 1 else 1
        recs.append(
            f"Biggest average gain is at round {best_round} (+{avg_per_round[best_round]:.1f}%). "
            f"→ Focus prompt optimization on round {best_round}."
        )

    # Kernels with plateau
    early_plateau = [s for s in summaries if s["plateau_round"] is not None and s["plateau_round"] <= 1]
    if early_plateau:
        names = ", ".join(s["kernel"] for s in early_plateau)
        recs.append(
            f"Early plateau (round ≤1) on: {names}. "
            f"→ Try **Approach 6 (Success Pattern Learning)** to help LLM exploit winners faster."
        )

    # Kernels with regression
    regressing = [s for s in summaries if s["regression_rounds"]]
    if regressing:
        details = ", ".join(f"{s['kernel']} (round {s['regression_rounds']})" for s in regressing)
        recs.append(
            f"Regression detected: {details}. "
            f"→ Try **Approach 1 (Multi-Round Tuning, cap rounds)** or simplify late-round prompts."
        )

    # Kernel-type pattern
    loss_kernels = {"cross_entropy", "softmax", "layernorm"}  # memory-bound
    compute_kernels = {"matmul", "attention"}
    loss_total = [s["total_improvement_pct"] for s in summaries if s["kernel"] in loss_kernels]
    compute_total = [s["total_improvement_pct"] for s in summaries if s["kernel"] in compute_kernels]
    if loss_total and compute_total:
        loss_mean = statistics.mean(loss_total)
        compute_mean = statistics.mean(compute_total)
        gap = abs(loss_mean - compute_mean)
        if gap > 5.0:
            better = "memory-bound" if loss_mean > compute_mean else "compute-bound"
            recs.append(
                f"Kernel-type gap detected: memory-bound avg {loss_mean:.1f}%, compute-bound avg {compute_mean:.1f}% "
                f"(Δ {gap:.1f}%, {better} improves more). "
                f"→ Try **Approach 4 (Theoretical Guidance)** or **Approach 7 (Compiler-Detected Patterns)**."
            )

    return recs


def _render_report(summaries: list[dict[str, Any]], recs: list[str]) -> str:
    lines: list[str] = []
    lines.append("# Phase 1: Per-Round Analysis Report\n")
    lines.append("*Auto-generated from baseline/{kernel}/{kernel}_baseline_round_progress.csv*\n")

    lines.append("\n## Per-Kernel Summary\n")
    lines.append("| Kernel | Shapes | Rounds | Total Δ | Plateau@ | Regressions |")
    lines.append("|--------|--------|--------|---------|----------|-------------|")
    for s in summaries:
        plateau = str(s["plateau_round"]) if s["plateau_round"] is not None else "—"
        regressions = ",".join(str(r) for r in s["regression_rounds"]) or "—"
        lines.append(
            f"| {s['kernel']} | {s['num_shapes']} | {s['max_round'] + 1} | "
            f"{s['total_improvement_pct']:+.1f}% | {plateau} | {regressions} |"
        )

    lines.append("\n## Per-Round Improvement (% vs previous round)\n")
    num_rounds = max((s["max_round"] for s in summaries), default=0) + 1
    header = "| Kernel |" + "".join(f" R{i} |" for i in range(num_rounds))
    sep = "|--------|" + "--------|" * num_rounds
    lines.append(header)
    lines.append(sep)
    for s in summaries:
        row = f"| {s['kernel']} |"
        for i in range(num_rounds):
            if i < len(s["per_round_improvement_pct"]):
                imp = s["per_round_improvement_pct"][i]
                row += f" {imp:+.1f}% |" if i > 0 else " —    |"
            else:
                row += " —    |"
        lines.append(row)

    lines.append("\n## Per-Round Geometric Mean (ms)\n")
    lines.append(header.replace("R0", "R0 (init)"))
    lines.append(sep)
    for s in summaries:
        row = f"| {s['kernel']} |"
        for i in range(num_rounds):
            if i < len(s["per_round_geomean_ms"]) and not math.isnan(s["per_round_geomean_ms"][i]):
                row += f" {s['per_round_geomean_ms'][i]:.3f} |"
            else:
                row += " —    |"
        lines.append(row)

    lines.append("\n## Recommendations for Phase 2\n")
    if recs:
        for r in recs:
            lines.append(f"- {r}")
    else:
        lines.append("- (No specific recommendations — patterns are uniform.)")

    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, required=True, help="Directory containing per-kernel baseline subdirs")
    parser.add_argument("--output", type=Path, required=True, help="Output markdown report path")
    parser.add_argument("--json-output", type=Path, default=None, help="Optional structured JSON output")
    args = parser.parse_args(argv)

    summaries: list[dict[str, Any]] = []
    for kernel_dir in sorted(args.baseline_dir.iterdir()):
        if not kernel_dir.is_dir() or kernel_dir.name.startswith("_"):
            continue
        kernel = kernel_dir.name
        rp_path = kernel_dir / f"{kernel}_baseline_round_progress.csv"
        if not rp_path.exists():
            print(f"SKIP {kernel}: no round_progress CSV at {rp_path}")
            continue
        rows = _read_round_progress(rp_path)
        if not rows:
            print(f"SKIP {kernel}: round_progress CSV empty")
            continue
        summaries.append(_per_kernel_summary(kernel, rows))

    if not summaries:
        print("ERROR: no kernels had usable round_progress data")
        return 1

    recs = _recommendations(summaries)
    report = _render_report(summaries, recs)
    args.output.write_text(report)
    print(f"Wrote {args.output}")

    if args.json_output:
        args.json_output.write_text(json.dumps({"summaries": summaries, "recommendations": recs}, indent=2, default=str))
        print(f"Wrote {args.json_output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
