# GEMM Hill-Climbing Setup (portable)

These notes mirror `../norms/SETUP.md`. GEMM uses the same env, wheel,
and Bedrock transport; only the archive checkout and the shape grid
differ.

## Getting the AOT measurement data

```bash
cd /home/dev/helion_choijon5
git fetch origin choijon5/aot-pretune-data
git checkout origin/choijon5/aot-pretune-data -- \
  aot_pretune_data/b200/matmul/ \
  aot_pretune_data/b200/fp8_gemm/ \
  aot_pretune_data/b200/run_index.json \
  aot_pretune_data/README.md
```

This drops ~11 MB of CSVs into your working tree but does not switch
branches or stage them.

## Prerequisites

- NVIDIA B200 host (`sm_100`).
- AWS IAM role (instance profile or `AWS_ACCESS_KEY_ID` +
  `AWS_SECRET_ACCESS_KEY`) with `bedrock:InvokeModel` on
  `us.anthropic.claude-opus-4-7` in some region (tested:
  `us-east-2`). If you don't have Bedrock, set `ANTHROPIC_API_KEY` and
  pass `HELION_LLM_PROVIDER=anthropic`.
- The `helion_choijon5` conda env created per `../norms/SETUP.md`. The
  same env serves both loops.

## Running the hill-climb gates

```bash
source /home/dev/miniconda3/etc/profile.d/conda.sh && conda activate helion_choijon5
cd /home/dev/helion_choijon5

export HELION_LLM_PROVIDER=bedrock
export HELION_LLM_MODEL=us.anthropic.claude-opus-4-7
export HELION_LLM_ANTHROPIC_THINKING_BUDGET=8000

# N0 baseline: 2 kernels x 12 shapes x 3 repeats via LLMGuidedSearch(max_rounds=1)
for K in matmul fp8_gemm; do
  python llm_heuristics_artifacts/gemm/tools/run_live.py \
    --kernel $K \
    --arm baseline \
    --shape-grid llm_heuristics_artifacts/gemm/iterations/N0_live/shape_grid.json \
    --output-dir llm_heuristics_artifacts/gemm/iterations/N0_live/baseline \
    --repeats 3
done

# Score baseline alone (per-shape noise + best-so-far)
python llm_heuristics_artifacts/gemm/tools/compute_round0_geo.py \
  --baseline llm_heuristics_artifacts/gemm/iterations/N0_live/baseline \
  --output   llm_heuristics_artifacts/gemm/iterations/N0_live/baseline_summary.json
```

Heuristics arm for gate N2 (once an AOT heuristic is picked):

```bash
export HELION_LLM_ROUND0_HEURISTIC_PATH=/absolute/path/to/heuristic_matmul.py
python llm_heuristics_artifacts/gemm/tools/run_live.py \
  --kernel matmul \
  --arm heuristics \
  --shape-grid llm_heuristics_artifacts/gemm/iterations/N0_live/shape_grid.json \
  --output-dir llm_heuristics_artifacts/gemm/iterations/N2_seed/heuristics \
  --repeats 3

python llm_heuristics_artifacts/gemm/tools/compute_round0_geo.py \
  --baseline   llm_heuristics_artifacts/gemm/iterations/N0_live/baseline \
  --heuristics llm_heuristics_artifacts/gemm/iterations/N2_seed/heuristics \
  --output     llm_heuristics_artifacts/gemm/iterations/N2_seed/scores.json
```

## Files in this tree

- `plan.md` — living plan + gate definitions + terminal goal.
- `manager.md` — subagent workflow, scoring harness, report formats.
- `N0_baseline.json` — offline AOT heuristic reproduction.
- `iterations/N0_live/shape_grid.json` — 12-shape grid per kernel with
  7/5 train/heldout split.
- `tools/workloads.py` — builds kernel + args for a shape entry.
- `tools/run_live.py` — runs one arm × one kernel across the grid.
- `tools/compute_round0_geo.py` — computes `round0_best_geo` from arm
  CSVs.

## Gotchas

- `matmul` example uses `static_shapes=False`; `fp8_gemm` uses
  `static_shapes=True`. Per-shape compile cost differs; budget longer
  for fp8_gemm live runs.
- `fp8_gemm` requires `torch.float8_e4m3fn` input tensors. The
  workload builder constructs them via `torch.randn(...).to(...)` with
  clamping into fp8 range.
- `tl.dot` requires `M ≥ 16, N ≥ 16, K ≥ 32`. The shape grid keeps all
  dims ≥ 32 to avoid this; the stretch set in N4b may include `(1,
  N, K)` decode shapes which will need a special-case block size.
