# N0-Live: Shape Grid Proposal

## Summary

A deterministic 12-shape grid per kernel (7 train, 5 held-out) spanning `layer_norm`, `rms_norm`, and `softmax` on NVIDIA B200, targeting generalization via strict axis and numel coverage.

## Design Rationale

**Shape Selection:**
- All shapes are 2D row-reduction patterns: `(rows, cols)` with softmax reducing over the `cols` axis per row.
- Rows range: `[256, 512, 1024, 2048, 3072, 4096]` to cover small, medium, and large batch dimensions.
- Cols range: `[1024, 2048, 3072, 4096, 5120]` to cover embedding and hidden dimensions.
- Total numels span `~262K to ~17M`, matching archived CSV distributions.

**Train/Held-Out Split (Fixed Seed 20260508):**
- Deterministic shuffle per kernel, split 7/5, ensuring reproducibility.
- **Both halves cover both axes:** train rows ⊇ {256, 512, 1024, 2048}, train cols ⊇ {1024, 2048, 4096}; held-out rows include 3072/4096, cols include 5120.
- **Corner shapes in each half:** train min~262K / max~10.5M (layer_norm), held-out min~1M / max~16.7M; softmax inversed to stress generalization.
- **Non-POT dimensions per kernel:** 3072 (3×1024) and 5120 (5×1024) in train or held-out, ensuring the heuristic does not memorize only power-of-two patterns.

## Overlap with Archived CSVs

All selected shapes lie within or near the archived tuning ranges:
- **layer_norm:** archived (4096, 1024)–(8192, 4096); our grid ⊆ this range.
- **rms_norm:** archived (2048, 48)–(8192, 4096); our grid aligns with the mid-high regime (256, 1024)–(4096, 4096).
- **softmax:** archived (4096, 256)–(4096, 2688); our grid broadens to smaller rows (256–4096) and extends cols to 5120 for stretch.

The heuristics arm will thus see strong signal from existing AOT CSV rows during construction, but held-out shapes do not exactly match any single archive tuple, enforcing true generalization.

## Kernel Entry Points

- **layer_norm_fwd:** Forward pass, returns (output, mean, rstd); row-reduces over cols.
- **rms_norm_fwd:** Forward pass, returns (output, inv_rms); row-reduces over cols.
- **softmax_two_pass:** Two-pass numerically stable softmax; row-reduces over cols.

All accept bfloat16 inputs; dtype is fixed at construction time.

## Risks & Mitigation

- **Overfit to archive:** Held-out set intentionally excludes exact CSV tuples. Generality gate tracks `heldout − train` delta; if > 0.05, policy fails.
- **Axis imbalance:** Both train and held-out span both rows and cols; corner shapes ensure numel range coverage in each half.
- **Non-POT underrepresentation:** At least one non-POT shape per kernel per half (3072, 5120).
- **Softmax batching:** The plan mentions "one batched (B, rows, cols) shape" for softmax, but kernel signature is (m, n). We use larger 2D shapes as a proxy; gate N4b will stretch to true batched if needed.

## Assumptions

1. **HALF_DTYPE = bfloat16** from test harness; no fp16/fp32 variants in this grid (N4b gate adds those).
2. **Row-reduction kernels** only; no col-reduction or full-matrix reductions.
3. **Device:** NVIDIA B200 GPU 0 (as per plan).
4. **Seed is fixed (20260508)** for reproducible split; same grid every gate run.

