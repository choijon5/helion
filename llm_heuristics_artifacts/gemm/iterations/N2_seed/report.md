# N2 — AOT heuristic as a round-0 seed

**Status: FAIL (heldout gate)**

## Result

| kernel   | scope   | round0_best_geo | delta (hld − trn) |
|----------|---------|----------------:|------------------:|
| fp8_gemm | train   | **0.7620**      |                    |
| fp8_gemm | heldout | 0.8386          | +0.0766            |
| matmul   | train   | 0.8248          |                    |
| matmul   | heldout | 0.8857          | +0.0610            |
| family   | train   | **0.7928**      |                    |
| family   | heldout | 0.8619          | +0.0691            |

Train goal `≤ 0.80`: met on family and fp8_gemm. Matmul train at 0.8248
(8.2% speedup). Held-out goal `≤ 0.80`: missed on both kernels by ~4–9
points. Overfit delta: both kernels exceed the 0.05 threshold. Both
signal the same thing: the heuristic built from square archive shapes
does not generalize to skewed shapes.

## Worst held-out shapes (regressions and near-miss)

| kernel   | shape       | label             | median ratio | notes |
|----------|-------------|-------------------|-------------:|-------|
| matmul   | MM_012      | 4096×4096×128     | 1.13 (reg)   | N-skinny, heuristic picks big-M config; tiles wrong for N=128 |
| matmul   | MM_010      | 1024×2048×4096    | 1.00         | K and N both > M; heuristic keys only on M=1024 |
| fp8_gemm | FP8_011     | 128×2048×2048     | ~0.95        | M-skinny; heuristic maps M=128 to the smallest config, underused N/K |
| fp8_gemm | FP8_012     | 4096×4096×128     | 1.00         | N-skinny, same pattern as MM_012 |

## What's actually happening

- Feature selector kept only `arg0_dim0` (M) for both heuristics (train
  shapes are square so M uniquely identifies each shape). The decision
  tree learned a pure-M dispatch.
- On held-out non-square shapes, the dispatch picks configs based on M
  alone. For MM_012 (M=4096) it chooses the "big" config tuned for
  3840³, which has block_n/N tiles sized for huge N. Running that on
  N=128 is inefficient.
- No regression on train (by construction — train is in-distribution)
  but heldout regression on MM_012 is the dealbreaker.

## Overfit audit

The feature selector removed K (`arg0_dim1`) and N (`arg1_dim1`) because
on the train set they were collinear with M. The removal is an
overfitting artifact, not a signal that K and N don't matter.

## Decision

PROMOTE N2 as baseline-champion on **train**; FAIL held-out gate. Go to
N3 with the specific hypothesis that forcing the tree to use K and N is
the highest-leverage change.

## Artifacts

- `heuristic/heuristic_matmul.py`, `heuristic/heuristic_fp8_gemm.py`
- `heuristic/heuristic_matmul_summary.json`,
  `heuristic/heuristic_fp8_gemm_summary.json`
- `heuristics/matmul_heuristics.csv`,
  `heuristics/fp8_gemm_heuristics.csv`
- `heuristics/*.meta.json`
- `scores.json`
