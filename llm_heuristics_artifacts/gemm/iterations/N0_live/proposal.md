# N0-Live: Shape Grid Proposal (GEMM)

## Summary

A deterministic 12-shape grid per kernel (7 train, 5 held-out) for
`matmul` (fp16) and `fp8_gemm` (fp8_e4m3fn) on NVIDIA B200, seed 20260509.

## Design Rationale

**Training set (archive-aligned):**
- All 7 train shapes are square `(M, K, N) = (D, D, D)` with D ∈
  `{256, 512, 1024, 1536, 2048, 3072, 3840}`. This matches the archive
  distribution (M=N=K, step 128, 256→3840) so the AOT heuristic built
  from archived CSVs has strong signal to draw on.
- Covers both small (256³ → 65K out) and large (3840³ → 14.7M out)
  corners; covers POT (1024, 2048, 4096 would — but 4096 is heldout)
  and non-POT (1536, 3072, 3840).
- `numel_out` ranges ~65K to ~14.7M, ~225× dynamic range.

**Held-out set (outside archive signature):**
- `4096³` — one step beyond the archive cap (3968); the nearest
  archive shape is `(3840,3840,3840)` so seed extrapolation is tested.
- `2048×1024×2048` — K half of M=N; rank-reduction-like signature that
  the archive never sees.
- `1024×2048×4096` — rectangular, K < N > M; covers FFN expansion.
- `128×2048×2048` — M-skinny, K=N; resembles decode prefill with very
  small batch.
- `4096×4096×128` — N-skinny; resembles output projection with very
  small expansion.

Together, held-out covers one square-at-new-scale shape and four
distinct non-square signatures (K-skinny, M-skinny, N-skinny,
rectangular).

**Feature diversity sanity:**
- train M: {256, 512, 1024, 1536, 2048, 3072, 3840}
- heldout M: {128, 1024, 2048, 4096, 4096}
- train N: same as train M (square)
- heldout N: {128, 2048, 2048, 4096, 4096}
- train K: same as train M (square)
- heldout K: {1024, 2048, 2048, 4096, 4096}
- Both halves cover multiple orders of magnitude of numel_out.

## Overlap with Archived CSVs

- **matmul archive:** 30–39 distinct shapes, all square fp16, D ∈
  {256..3968, step 128}. Our train shapes exactly match the archive
  grid at 7 distinct Ds. Held-out `4096³` is step 128 above the cap;
  non-square held-out shapes have no archive counterpart.
- **fp8_gemm archive:** 30–39 distinct shapes, all square fp8, D ∈
  {256..3968, step 128}. Same alignment.

## Kernel Entry Points

- **matmul:** `examples.matmul.matmul(x, y)` → `[M, N]`.
- **fp8_gemm:** `examples.fp8_gemm.fp8_gemm(x, y)` → `[M, N]`
  (`static_shapes=True`, which forces a per-shape compile).

## Risks & Mitigation

- **Overfit to archive:** held-out set intentionally excludes any
  archive shape signature. Generality gate tracks `heldout − train`
  delta; if > 0.05, policy fails.
- **128-dim skinny shapes may hit `tl.dot` size minima:** for
  non-`static_shapes` kernels the autotuner will still pick valid block
  sizes. We keep the min dim at 128 to avoid the `M ≥ 16, N ≥ 16, K ≥
  32` floor tl.dot enforces.
- **Fp8 numeric saturation:** our workload builder clamps to
  `[-448, 448]`. Results should compute; we don't verify against a
  reference in round 0.
- **fp8_gemm `static_shapes=True` per-shape compile:** live run wall
  time is dominated by compile, not benchmark. Budget ≤ 45 minutes
  per kernel for 12 shapes × 3 repeats × (1 seed + 3 random + ≤ 5
  LLM configs).

## Assumptions

1. **Per-kernel dtype is fixed:** matmul is fp16, fp8_gemm is fp8. No
   dtype mixing in N0/N2/N3.
2. **Device:** NVIDIA B200 GPU 0.
3. **Seed is fixed (20260509).** Derived from `norms` seed +1 to avoid
   coincidental overlap.
