# GEMM Heuristic Hill-Climbing Manager

Workflow for running the `plan.md` gates using Claude subagents via the
`Agent` tool. Mirrors `../norms/manager.md`; differences called out
inline.

## Role

The manager coordinates subagents and validates artifacts. The manager
does not directly propose policies, edit the autotuner, or interpret
raw LLM output.

The manager may:

- read CSVs, JSON, and generated heuristic code;
- invoke the offline scoring harness to reproduce AOT metrics;
- run lightweight verification (`git status --short`, `nvidia-smi`);
- write per-iteration reports under `iterations/N<n>_<slug>/`.

The manager must not:

- run live GPU benchmarks (delegate to a harness subagent) — except
  short smoke tests (1 shape, 1 repeat) to verify the harness starts;
- commit, push, or stage unrelated files;
- propose autotuner-source changes without explicit user approval.

## LLM Autotuner Configuration

Every live-benchmark arm in this loop must use:

```bash
export HELION_LLM_PROVIDER=bedrock
export HELION_LLM_MODEL=us.anthropic.claude-opus-4-7
# Opus 4.7 adaptive-thinking; _bedrock.py auto-selects on model name.
export HELION_LLM_ANTHROPIC_THINKING_BUDGET=8000
# IMDSv2 creds are picked up automatically on EC2.
# Elsewhere: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION.
```

If Bedrock is unreachable, fall back to:

```bash
export HELION_LLM_PROVIDER=anthropic
export ANTHROPIC_API_KEY=<user-provided>
```

These apply to both arms. Differences between arms come from the
heuristic mechanism alone (seed injection, observed-heuristics JSON, or
nothing for the baseline arm).

## Offline Scoring Harness (CPU only)

```bash
source /home/dev/miniconda3/etc/profile.d/conda.sh && conda activate helion_choijon5
cd /home/dev/helion_choijon5

python - <<'PY'
from pathlib import Path
from helion.autotuner.heuristic_generator import generate_heuristic, PerformanceTarget

target = PerformanceTarget(
    goal_type='max_slowdown', threshold=1.10, min_configs=1, max_configs=10,
    backend='decision_tree', feature_selection=True, print_score_matrix=False,
    verbose=False, skip_write=True,
)
csv = Path('<PATH_TO_MEASUREMENTS_CSV>')
out = Path('/tmp/gemm_iter/<RUN_TAG>'); out.mkdir(parents=True, exist_ok=True)
results = generate_heuristic(csv, out, target=target)
for k, r in results.items():
    s = r.performance_stats
    print(k, s['max_slowdown'], s['geomean_slowdown'], r.model_accuracy, len(r.selected_configs))
PY
```

Swap `backend='decision_tree'` for `'nearest_neighbor'` to compare
backends.

Canonical CSV paths (relative to repo root):

| source             | path                                                                                        |
|--------------------|---------------------------------------------------------------------------------------------|
| matmul single      | `aot_pretune_data/b200/matmul/runs/20260430_192807_2dc612/measurements_cuda_NVIDIA_B200_13.0.csv` |
| matmul expanded    | `aot_pretune_data/b200/matmul/runs/20260502_110931_expanded_matmul/measurements_cuda_NVIDIA_B200_13.0.csv` |
| fp8_gemm a         | `aot_pretune_data/b200/fp8_gemm/runs/20260430_192807_699fa9/measurements_cuda_NVIDIA_B200_13.0.csv` |
| fp8_gemm b         | `aot_pretune_data/b200/fp8_gemm/runs/20260501_151413_cb87f1/measurements_cuda_NVIDIA_B200_13.0.csv` |
| fp8_gemm c         | `aot_pretune_data/b200/fp8_gemm/runs/20260501_180243_009304/measurements_cuda_NVIDIA_B200_13.0.csv` |
| fp8_gemm d         | `aot_pretune_data/b200/fp8_gemm/runs/20260501_220932_cb0d7e/measurements_cuda_NVIDIA_B200_13.0.csv` |
| fp8_gemm expanded  | `aot_pretune_data/b200/fp8_gemm/runs/20260502_110931_expanded_fp8_gemm/measurements_cuda_NVIDIA_B200_13.0.csv` |

## Train / Held-Out Discipline

Generality is the primary acceptance criterion for this loop. Every
subagent that builds or scores a heuristic must honor:

1. **Train shapes only** go into heuristic construction — including the
   measurement CSV rows fed to `heuristic_generator.py`, any
   observed-heuristics JSON filter, and any hand-authored rule. The
   shape split lives in `iterations/N0_live/shape_grid.json`; do not
   redraw it.
2. **Reports always show three numbers per kernel:** `train`,
   `heldout`, and `heldout − train`. Missing either number is an
   incomplete report.
3. **Cross-kernel rotation** at N4b builds from one GEMM and scores on
   the other. Tag which kernel was held out.
4. **Feature audit.** Every candidate heuristic file must be grepped
   for kernel-name strings (`matmul`, `fp8_gemm`) outside the
   top-level dispatch function, and for any run or config hash leaking
   from the CSVs into the features. Any hit is an automatic FAIL.

## Live Scoring Harness (B200)

The live runner `tools/run_live.py` reuses the same pattern as the
norms loop but is GEMM-specific in its arg builders (see
`tools/workloads.py`). It:

1. Takes `--kernel`, `--shapes-json`, `--arm {baseline,heuristics}`,
   `--repeats`, and `--output-dir`.
2. Constructs the kernel + args from `examples/<kernel>.py` for each
   shape.
3. Runs `LLMGuidedSearch` with `max_rounds=1` and the configured model.
4. Writes a row per (workload, repeat, generation, config, perf_ms,
   status) and a companion JSON metadata file for scoring.

The score computation sits in `tools/compute_round0_geo.py` and uses
metadata paths, not filename parsing.

## Subagent Roles

Spawn via the `Agent` tool. Never reuse the same subagent for both
proposal and review.

- **diagnose** (Explore): read-only analysis of CSVs and heuristic
  code; identifies worst shapes and candidate features.
- **propose** (general-purpose): proposes exactly one change. No runs,
  no edits.
- **review** (general-purpose): checks for overfitting, feature
  leakage, unsafe source edits, missing guardrails. Read-only.
- **implement** (general-purpose): applies the reviewed change. Narrow
  file allowlist. No unrelated edits.
- **harness** (general-purpose): runs the accepted command on GPU 0
  and writes results under `iterations/N<n>_<slug>/`. Does not change
  source.
- **analysis** (general-purpose): computes `round0_best_geo` from CSV
  metadata and writes the per-gate report.

## Assignment Format

```text
Task: <one sentence>
Gate: N<n>
Role: diagnose | propose | review | implement | harness | analysis
Inputs:
- current champion: <iteration path>
- plan: llm_heuristics_artifacts/gemm/plan.md
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

## Lightweight Verification

After every subagent return:

```bash
git status --short
git diff --stat
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv
```

If anything is staged, or files outside the allowlist changed, stop
and ask the user.

## When to Ask for Help

- A proposal needs to edit `helion/autotuner/heuristic_generator.py`,
  `helion/autotuner/llm_search.py`, or
  `helion/autotuner/llm/transport.py`.
- A gate wants to change the primary metric, the LLM model, or the
  acceptance threshold.
- A live run takes longer than 45 minutes per kernel (GEMM with
  `static_shapes=True` needs one compile per shape; budget
  accordingly).
- fp8_gemm `round0_best_geo` regresses > 3% under any candidate.
- Two consecutive gates FAIL with no clear diagnostic path.
