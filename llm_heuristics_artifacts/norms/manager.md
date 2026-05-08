# Row-Reduction Heuristic Hill-Climbing Manager

Workflow for running the `plan.md` gates using Claude subagents via the
`Agent` tool. No Codex dependency; all proposal, review, and analysis roles
are Claude subagents.

## Role

The manager coordinates subagents and validates artifacts. The manager does
not directly propose policies, edit the autotuner, or interpret raw LLM
output.

The manager may:

- read CSVs, JSON, and generated heuristic code;
- invoke the offline scoring harness to reproduce AOT metrics;
- run lightweight verification (`git status --short`, `nvidia-smi`);
- write per-iteration reports under `iterations/N<n>_<slug>/`.

The manager must not:

- run live GPU benchmarks (delegate to a harness subagent);
- commit, push, or stage unrelated files;
- propose autotuner-source changes without explicit user approval.

## LLM Autotuner Configuration

Every live-benchmark arm in this loop must use:

```bash
export HELION_LLM_PROVIDER=anthropic
export HELION_LLM_MODEL=claude-opus-4-7
# Once N1 lands the extended-thinking patch:
export HELION_LLM_ANTHROPIC_THINKING_BUDGET=32000   # or the approved value
# API key goes through the standard Anthropic env var:
export ANTHROPIC_API_KEY=<user-provided>
```

These apply to both arms. Differences between arms come from the heuristic
mechanism alone (seed injection, observed-heuristics JSON, or nothing for
the baseline arm).

## Offline Scoring Harness (CPU only)

Used to reproduce AOT `max_slowdown` / `geomean_slowdown` before attempting
a live run:

```bash
/home/dev/.conda/envs/pytorch_env/bin/python -c "
import sys; sys.path.insert(0, '/home/dev/helion_choijon5')
from pathlib import Path
from helion.autotuner.heuristic_generator import generate_heuristic, PerformanceTarget

target = PerformanceTarget(
    goal_type='max_slowdown', threshold=1.10, min_configs=1, max_configs=10,
    backend='decision_tree', feature_selection=True, print_score_matrix=False,
    verbose=False, skip_write=True,
)
csv = Path('<PATH_TO_MEASUREMENTS_CSV>')
out = Path('/tmp/norm_iter/<RUN_TAG>'); out.mkdir(parents=True, exist_ok=True)
results = generate_heuristic(csv, out, target=target)
for k, r in results.items():
    s = r.performance_stats
    print(k, s['max_slowdown'], s['geomean_slowdown'], r.model_accuracy, len(r.selected_configs))
"
```

Swap `backend='decision_tree'` for `'nearest_neighbor'` to compare backends.

Canonical CSV paths:

| source          | path                                                                                                |
|-----------------|-----------------------------------------------------------------------------------------------------|
| layer_norm      | `aot_pretune_data/b200/layer_norm/runs/20260503_131231_.../measurements_cuda_NVIDIA_B200_13.0.csv`  |
| layer_norm pool | `/tmp/norm_baseline_single/layer_norm_pooled.csv` (rebuild from 5 run CSVs if missing)               |
| rms_norm        | `aot_pretune_data/b200/rms_norm/runs/20260503_001250_.../measurements_cuda_NVIDIA_B200_13.0.csv`    |
| softmax         | `aot_pretune_data/b200/softmax/runs/20260503_114301_.../measurements_cuda_NVIDIA_B200_13.0.csv`     |
| softmax pool    | `/tmp/norm_baseline_single/softmax_pooled.csv` (rebuild from 4 run CSVs if missing)                 |

## Train / Held-Out Discipline

Generality is the primary acceptance criterion for this loop. Every
subagent that builds or scores a heuristic must honor:

1. **Train shapes only** go into heuristic construction — including the
   measurement CSV rows fed to `heuristic_generator.py`, any
   observed-heuristics JSON filter, and any hand-authored rule. The shape
   split lives in `iterations/N0_live/shape_grid.json`; do not redraw it.
2. **Reports always show three numbers per kernel:** `train`, `heldout`,
   and `heldout − train`. Missing either number is an incomplete report.
3. **Cross-kernel rotation** at N4b builds from two of the three norms
   and scores on the third. Tag which kernel was held out.
4. **Feature audit.** Every candidate heuristic file must be grepped for
   kernel-name strings (`layer_norm`, `rms_norm`, `softmax`) outside the
   top-level dispatch function, and for any run or config hash leaking
   from the CSVs into the features. Any hit is an automatic FAIL.

The review subagent role's explicit checklist includes these four items.

## Live Scoring Harness (B200)

This loop does not yet have a `llm_heuristics_autoresearch.py` analogue. The
first harness subagent builds a minimal runner in
`llm_heuristics_artifacts/norms/tools/run_live.py` that:

1. Takes `--kernel`, `--shapes-json`, `--arm {baseline,heuristics}`,
   `--repeats`, and `--output-csv`.
2. Constructs the kernel + args from `examples/<kernel>.py` for each shape.
3. Runs `LLMSeededSearch` with `llm_max_rounds=1` and the configured model.
4. Writes a row per (workload, repeat, generation, config, perf_ms, status)
   and a companion JSON metadata file for scoring.

The score computation sits in `tools/compute_round0_geo.py` and uses
metadata paths, not filename parsing.

Until the runner exists, live-benchmark gates stay blocked; the only
runnable gates are offline (N0 offline, N3 offline-only steps, N4 JSON
synthesis).

## Subagent Roles

Spawn via the `Agent` tool. Never reuse the same subagent for both
proposal and review; create a fresh instance for review.

- **diagnose** (Explore): read-only analysis of CSVs and generated
  heuristic code; identifies worst shapes and candidate features.
- **propose** (general-purpose): proposes exactly one change. No runs, no
  edits. Must state hypothesis, expected delta, and rollback plan.
- **review** (general-purpose): checks for overfitting, feature leakage,
  unsafe source edits, and missing guardrails. Read-only.
- **implement** (general-purpose): applies the reviewed change. Narrow
  file allowlist. No unrelated edits.
- **harness** (general-purpose): runs the accepted command on GPU 0 and
  writes results under `iterations/N<n>_<slug>/`. Does not change source.
- **analysis** (general-purpose): computes `round0_best_geo` from CSV
  metadata and writes the per-gate report.

## Assignment Format

```text
Task: <one sentence>
Gate: N<n>
Role: diagnose | propose | review | implement | harness | analysis
Inputs:
- current champion: <iteration path>
- plan: llm_heuristics_artifacts/norms/plan.md
- CSVs: <absolute paths>
- autotuner env: Opus-4.7 + max reasoning (see manager.md)
Allowed actions:
- <explicitly allowed>
Forbidden actions:
- live benchmarks unless role is harness
- edits outside <allowed file list>
- commits, pushes, stages
- running pip or system package managers
Required output:
- <iteration report path>
Acceptance criteria:
- <numeric gate from plan.md>
```

## Subagent Report Format

```text
## Result
status: PASS | FAIL | BLOCKED
gate: N<n>
role: <one of the roles above>

## Summary
- <one sentence>
- <one sentence>

## Offline Scores (if applicable)
| source            | configs | max_slowdown | geomean | acc |
|-------------------|--------:|-------------:|--------:|----:|
| layer_norm single | ...     | ...          | ...     | ... |
| layer_norm pooled | ...     | ...          | ...     | ... |
| rms_norm          | ...     | ...          | ...     | ... |
| softmax single    | ...     | ...          | ...     | ... |
| softmax pooled    | ...     | ...          | ...     | ... |

## Live Scores (if applicable)
| kernel     | scope   | n | round0_best_geo | verified_geo | wall_geo |
|------------|---------|--:|----------------:|-------------:|---------:|
| layer_norm | train   | . | ...             | ...          | ...      |
| layer_norm | heldout | . | ...             | ...          | ...      |
| rms_norm   | train   | . | ...             | ...          | ...      |
| rms_norm   | heldout | . | ...             | ...          | ...      |
| softmax    | train   | . | ...             | ...          | ...      |
| softmax    | heldout | . | ...             | ...          | ...      |
| family     | train   | . | ...             | ...          | ...      |
| family     | heldout | . | ...             | ...          | ...      |

## Generality Deltas (if applicable)
| kernel     | train | heldout | heldout − train | feature audit |
|------------|------:|--------:|----------------:|---------------|
| layer_norm | ...   | ...     | ...             | clean / flag  |
| rms_norm   | ...   | ...     | ...             | clean / flag  |
| softmax    | ...   | ...     | ...             | clean / flag  |

## Proposed / Applied Change
- mechanism:
- file(s) touched:
- reversibility:

## Notes / Risks
- <bullets>

## Next
- <one concrete next step>
```

## Lightweight Verification

After every subagent return:

```bash
git status --short
git diff --stat
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv
```

If anything is staged, or files outside the allowlist changed, stop and
ask the user.

## When to Ask for Help

- A proposal needs to edit `helion/autotuner/heuristic_generator.py`,
  `helion/autotuner/llm_search.py`, or `helion/autotuner/llm/transport.py`.
- A gate wants to change the primary metric, the LLM model, or the
  acceptance threshold.
- A live run takes longer than 30 minutes per kernel.
- rms_norm `round0_best_geo` regresses > 3% under any candidate.
- Two consecutive gates FAIL with no clear diagnostic path.
