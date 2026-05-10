# Quantized-GEMM observed-heuristics: policy doc

**Status: ready for review. No source commits yet. No push to PR.**

This doc explains what the quantized-GEMM hill-climbing loop produced,
what it's good for, and where each file lives relative to a future
upstream PR.

## Summary

Three kernel-specific observed-heuristic dispatchers for:

- `matmul_bf16_int4` (A:bf16 * B:int4-packed)
- `_bf16xint16_gemm` (x:bf16 * w:int16)
- `nvfp4_matmul` (A:bf16 * B:fp4_e2m1-packed)

Each dispatcher reads one JSON file of observed heuristics (derived
from 120 B200 archive shapes × full autotune), buckets the live
query shape, and returns a single `helion.Config` dict that can
be used as the round-0 seed for `LLMGuidedSearch` (Exp-2) or as a
no-autotune default (Exp-1).

## Measured impact (all on B200, Opus 4.7, 12-shape live grid)

| kernel             | Exp-1 (no-autotune) heldout | Exp-2 (LLM-on) heldout |
|--------------------|----------------------------:|-----------------------:|
| `matmul_bf16_int4` |                      0.132  |                 0.581  |
| `_bf16xint16_gemm` |                      0.107  |                 0.827  |
| `nvfp4_matmul`     |                      0.198  |                 0.609  |
| **family geomean** |                      0.141  |                 0.664  |

- Exp-1: single-config heuristic vs Helion's generic default — family
  heldout 0.141 = ~7.1× faster no-autotune.
- Exp-2: heuristic seed prepended to LLM round-0 vs LLM alone —
  family heldout 0.664 = ~34% faster round-0 best.

## How the dispatchers work

1. Load the derived-heuristics JSON at import time
   (`derived_general_heuristics.json`). This JSON has 57 data-derived
   rules (19 per quantized kernel) keyed on
   `(kernel_class, shape_bucket)` tuples where shape_bucket is
   `{aspect, dtype, m_bin, n_bin, k_bin}` with generator-aligned bins.
2. At query time: compute the bucket of `(M, K, N, dtype)`, then look
   for a matching rule for the kernel's semantic class
   (`matmul_int4`, `matmul_int16`, `matmul_fp4`).
3. If a rule matches, walk its templates in win-count-descending
   order. Return the first template whose estimated shmem footprint
   fits B200 (232 KB × 0.9 safety margin). If every template blows
   the budget, halve `num_stages` then `block_k` until it fits
   (shrink-to-fit).
4. If no rule matches, use a per-kernel fallback table keyed on
   shape group: `small_m` (M≤256), `small_n` (N≤256), `small_k`
   (K≤256), `balanced` (max/min < 2), `rect` (else). Each entry is
   a median-performing winner from the Q2 archive for that group.
5. Every config's `block_sizes` are clamped to the query's actual
   axes (so a template trained on 4096³ doesn't over-tile a 256³
   query).

No kernel-name strings are used as rule keys or features. No
config hashes are used in dispatch. No GPU is touched in the
dispatcher — it's pure CPU code.

## Files in this loop's output

### Input data (generated during Q2, not part of the PR)

- `aot_pretune_data/b200/matmul_bf16_int4/runs/20260509_q2_llmseeded/measurements_cuda_NVIDIA_B200_13.0.csv`
  — 8348 rows
- `aot_pretune_data/b200/_bf16xint16_gemm/runs/20260509_q2_llmseeded/measurements_cuda_NVIDIA_B200_13.0.csv`
  — 7486 rows
- `aot_pretune_data/b200/nvfp4_matmul/runs/20260509_q2_llmseeded/measurements_cuda_NVIDIA_B200_13.0.csv`
  — 10917 rows

CSVs live on the `choijon5/aot-pretune-data` branch per plan.md's
hard constraints, not on the code branch.

### Heuristics JSON (generated during Q3)

- `llm_heuristics_artifacts/quantized_gemm/iterations/Q3_heuristic/derived_general_heuristics.json`
  (57 rules, the dispatcher's primary source)
- `llm_heuristics_artifacts/quantized_gemm/iterations/Q3_heuristic/runtime_observed_heuristics_b200.json`
  (3 rules, strict LOSO-filtered — usable as a more conservative
  subset if we want a schema-v2 curated variant)

### Dispatcher code (Q4)

- `llm_heuristics_artifacts/quantized_gemm/iterations/N6_full_tune/heuristic/_dispatcher_core.py`
  — shared engine (~250 lines)
- `llm_heuristics_artifacts/quantized_gemm/iterations/N6_full_tune/heuristic/heuristic_matmul_bf16_int4.py`
- `llm_heuristics_artifacts/quantized_gemm/iterations/N6_full_tune/heuristic/heuristic_bf16xint16_gemm.py`
- `llm_heuristics_artifacts/quantized_gemm/iterations/N6_full_tune/heuristic/heuristic_nvfp4_matmul.py`

### Tooling

- `llm_heuristics_artifacts/quantized_gemm/tools/workloads.py` — arg builders
- `llm_heuristics_artifacts/quantized_gemm/tools/run_full_tuning.py` — archive tuning driver
- `llm_heuristics_artifacts/quantized_gemm/tools/resume_q2.sh` — per-shape subprocess driver with BENCHMARK_SUBPROCESS=1 crash isolation
- `llm_heuristics_artifacts/quantized_gemm/tools/run_no_autotune.py` — Exp-1 runner
- `llm_heuristics_artifacts/quantized_gemm/tools/run_live.py` — Exp-2 runner
- `llm_heuristics_artifacts/quantized_gemm/tools/compute_round0_geo.py` — scorer
- `llm_heuristics_artifacts/quantized_gemm/tools/run_q5.sh`, `run_q6.sh` — experiment drivers

### Script edit (Q3, required for infer_kernel_class)

- `scripts/llm_heuristics_research.py` — additive: 3 new classes
  (`matmul_int4`, `matmul_int16`, `matmul_fp4`) added to
  `infer_kernel_class`, `shape_bucket_for_class`, `_shape_label`. No
  behavior change for existing kernels.

## Deployment options (not decided here)

Three ways to ship this:

1. **As a research artifact only** — keep everything under
   `llm_heuristics_artifacts/quantized_gemm/`, don't touch
   `helion/autotuner/`. Users opt in by setting
   `HELION_LLM_ROUND0_HEURISTIC_PATH=<path>` as env var and the
   runtime picks it up. Lowest risk, no public API.
2. **Upstream into `helion/autotuner/data/`** following PR 2378's
   pattern — promote `runtime_observed_heuristics_b200.json` (strict,
   3-rule) or `derived_general_heuristics.json` (57-rule) into
   `helion/autotuner/data/observed_heuristics_b200.json` with
   schema_version bump to 2 and per-rule `validation` blocks filled
   in from Q5/Q6 scores. Requires the autotuner to know how to
   consume kernel-class-keyed rules for these 3 kernels.
3. **Hybrid** — ship the JSON upstream (option 2) but leave the
   three `heuristic_*.py` dispatchers as example code in
   `examples/` or as documentation, not as autotuner-source.

Option 1 is what's actually implemented today and is zero-risk.
Options 2/3 need user sign-off per manager.md's rule against
autotuner-source changes.

## Known caveats

- **nvfp4 Δheldout-train = +0.034** (Exp-2) is the tightest of the
  three, ~70% of the overfit gate. If adding new nvfp4 shapes
  (mobile-scale or batch-decode signatures) to the live grid, watch
  for this margin shrinking. Could be tightened by collecting more
  nvfp4 archive shapes at skinny_m (the bucket with the fewest
  covered shapes in Q3).
- **int16 has zero promoted runtime-JSON rules** but still wins
  0.107 on Exp-1 (all fallback table). The Exp-2 0.827 shows the
  fallback is strictly helpful but less dramatic than int4/fp4's
  rule-driven picks. Revisit if we get more int16 measurements.
- **Absolute perf_ms in Q6 CSVs is scaled 1000×** due to a
  `res.perf * 1000` in `run_live.py` ported from the gemm loop.
  Scorer ratios are unaffected. Clean up whenever touching these
  files again.

## PR preparation (not executed without approval)

Suggested PR title:
> `[Pretune] Quantized-GEMM observed heuristics (int4 / int16 / fp4)`

Suggested body:
> Follows the dense-GEMM pretune loop on `choijon5/gemm-hill-climb`.
>
> - New `_dispatcher_core.py` engine + 3 kernel dispatchers under
>   `llm_heuristics_artifacts/quantized_gemm/`.
> - Additive `scripts/llm_heuristics_research.py` edit: new kernel
>   classes `matmul_int4`, `matmul_int16`, `matmul_fp4`.
> - Measured family-level Exp-1 heldout = 0.141 (~7× over Helion
>   default), Exp-2 heldout = 0.664 (~34% over LLM-alone round-0).
> - No autotuner-source changes.
>
> Archive CSVs on `choijon5/aot-pretune-data`, run id
> `20260509_q2_llmseeded`.
