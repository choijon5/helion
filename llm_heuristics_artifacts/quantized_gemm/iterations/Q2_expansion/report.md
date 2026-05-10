# Q2 — Quantized-GEMM Archive Expansion

**Status: DONE (2026-05-10 06:28 UTC). 40/40 shapes × 3 kernels = 120 shapes, no missing.**

## Outcome

Archive tuning for the quantized-GEMM family is complete. Run id
`20260509_q2_llmseeded` under
`aot_pretune_data/b200/<kernel>/runs/`:

| kernel             | shapes | unique configs (rows) |
|--------------------|-------:|----------------------:|
| `matmul_bf16_int4` | 40/40  |                 8348  |
| `_bf16xint16_gemm` | 40/40  |                 7486  |
| `nvfp4_matmul`     | 40/40  |                10917  |

## Autotuner + model

- `HELION_AUTOTUNER=LLMSeededLFBOTreeSearch` (per Q0 decision).
- `HELION_AUTOTUNE_EFFORT=full`.
- `HELION_LLM_PROVIDER=bedrock`, `HELION_LLM_MODEL=us.anthropic.claude-opus-4-7`,
  `HELION_LLM_ANTHROPIC_THINKING_BUDGET=8000` (Opus 4.7 adaptive thinking).
- `HELION_AUTOTUNE_BENCHMARK_SUBPROCESS=1` (see Crash isolation).
- `HELION_AUTOTUNE_IGNORE_ERRORS=1`.

## Wall time

- Overnight run (2026-05-09): `_bf16xint16_gemm` 40/40, `matmul_bf16_int4`
  33/40, `nvfp4_matmul` 7/40. Crashed on a `matmul_bf16_int4` config
  that triggered `cudaErrorMisalignedAddress`.
- Resume driver (2026-05-10 01:34 → 06:28 UTC): 36 shapes, **16 827 s
  wall (~4 h 40 m)**, mean 467 s/shape, max 955 s (`nvfp4 rect_4k_2k_1k`),
  min 231 s (`nvfp4 m128_k2048_n2048`).

## Crash isolation (took two tries)

- **First retry**: rewrote the driver into a per-shape bash loop so a
  Python-level crash couldn't leak across shapes. Still hit
  `TritonUnrecoverableRuntimeError: CUDA error: misaligned address`
  on `matmul_bf16_int4/k512` with `block_sizes=[16,64,512]` during
  benchmark. I had set `HELION_AUTOTUNE_PRECOMPILE=spawn`, which
  only isolates the precompile step — the crash was in the benchmark.
- **Second retry (what worked)**: swapped to
  `HELION_AUTOTUNE_BENCHMARK_SUBPROCESS=1`. Benchmark runs in a child
  process; a CUDA misalignment there does not corrupt the parent's
  context. Completed 36 shapes with zero propagated crashes.

Per-shape subprocess isolation (bash loop) is still in place as
belt-and-suspenders, but BENCHMARK_SUBPROCESS is the primary defense.

## Notable shapes

Longest tunes were all large rectangular nvfp4 shapes:
`rect_4k_2k_1k` (955 s), `rect_1k_2k_4k` (884 s), `rect_2k_4k_1k`
(665 s), `rect_8k_2k_512` (662 s), `rect_1536_3072_1536` (633 s),
`skinny_k k128_m4k_n4k` (615 s). This is the LFBO surrogate doing
more work where the config landscape is wider.

## Artifacts

- `aot_pretune_data/b200/matmul_bf16_int4/runs/20260509_q2_llmseeded/`
- `aot_pretune_data/b200/_bf16xint16_gemm/runs/20260509_q2_llmseeded/`
- `aot_pretune_data/b200/nvfp4_matmul/runs/20260509_q2_llmseeded/`
- `tools/resume_q2.sh` — per-shape driver with BENCHMARK_SUBPROCESS=1.
- `iterations/Q2_expansion/resume_q2_driver.log` — per-shape timings.
- `iterations/Q2_expansion/<kernel>_<tag>.log` — individual shape logs.

## Handoff to Q3

Next gate is generating the observed-heuristics JSON from the archive.
Matches the dense-GEMM recipe:

```bash
python scripts/llm_heuristics_research.py \
    --data-root aot_pretune_data/b200 \
    --kernels matmul_bf16_int4 _bf16xint16_gemm nvfp4_matmul \
    --out llm_heuristics_artifacts/quantized_gemm/iterations/Q3_heuristic
```

Then inspect per-kernel rule counts and how many buckets pass the
strict LOSO filter.
