# GEMM Heuristic Hill-Climbing

This directory tracks hill-climbing experiments that try to improve LLM-seeded
autotuning (`LLMSeededSearch` / `LLMGuidedSearch`) on the GEMM kernel family:
`matmul` and `fp8_gemm`. Both share the shape triple `(M, K, N)` with one
matmul-style K-reduction per output tile; they differ in accumulator dtype
(fp16→fp32 vs fp8→fp32) and in the Triton intrinsic (`torch.addmm` vs
`hl.dot`), which changes which configs are fast.

The loop mirrors `../norms/` in structure but differs in three ways:

- **Target kernels** are dense GEMMs, not row-reductions.
- **Family size** is 2 (matmul, fp8_gemm). The archive also contains
  `grouped_gemm`, but its AOT run used a different API (3D inputs instead of
  today's jagged 2D + offsets); seed injection across that API gap is not
  sound, so grouped_gemm is out of scope for this loop.
- **Cross-kernel rotation** is 2-way (build from matmul → score on fp8_gemm
  and vice versa) instead of 3-way.

Contents:

- `plan.md`: gate-by-gate plan with the `round0_best_geo ≤ 0.80` terminal
  goal.
- `manager.md`: workflow for managing Claude subagents in this loop.
- `SETUP.md`: portable setup notes (env, archive, Bedrock creds).
- `N0_baseline.json`: offline AOT heuristic reproduction (input data to the
  heuristic, not the live score).
- `iterations/N<n>_<slug>/`: per-iteration candidate policy, harness command,
  CSVs, and score report.
