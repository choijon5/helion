# Q5b — Exp-1 re-benchmark with 21-rule schema-v2 JSON

**Status: DONE (2026-05-11). The 21-rule
`helion/autotuner/data/observed_heuristics_b200_quantized.json` delivers
the same Exp-1 speedup as the 57-rule internal derived JSON. Family
heldout drift: +0.7%.**

## Setup

Identical to Q5 except the dispatcher reads the 21-rule schema-v2 file
instead of the 57-rule `derived_general_heuristics.json`. Redirected
via `HELION_QUANTIZED_GEMM_OBSERVED_HEURISTICS_PATH`.

## Results

Round-0 best-ratio geomeans (heuristic / baseline). Lower is better.

| kernel             | Q5 (57 rules) heldout | Q5b (21 rules) heldout | Δ |
|--------------------|----------------------:|-----------------------:|--:|
| `_bf16xint16_gemm` |                 0.107 |                 **0.110** | +0.003 |
| `matmul_bf16_int4` |                 0.132 |                 **0.132** |  0.000 |
| `nvfp4_matmul`     |                 0.198 |                 **0.198** |  0.000 |
| **family**         |                 0.141 |                 **0.142** | +0.001 |

- All three kernels still well under the target (heldout ≤ 0.20).
- Family heldout drift: 0.7%, well inside noise.
- No overfitting (Δheldout − train negative across all kernels).

## Combined v2 validation summary

Schema-v2 file passes both experiments:

| experiment | family heldout | target | pass |
|------------|---------------:|-------:|:----:|
| Q5b (vs Helion default, no autotune) | 0.142 | ≤ 0.20 | ✅ |
| Q6b (vs Opus round-0 LLM-on)         | 0.663 | ≤ 0.95 | ✅ |

## On the one marginal rule kept

`matmul_int4 balanced <=512` has Q6 = 1.001 (tied with Opus round-0).
Decision: keep. Justification from per-repeat drill-in on its only
live shape (I4_002, 512³):

| experiment | baseline | heuristic | ratio | outcome |
|------------|---------:|----------:|------:|---------|
| Q5 (no-autotune) | 27.7 μs | 17.4 μs | **0.628** | heuristic wins ~37% |
| Q6 (LLM-on)      | 15.39 ms | 15.36/17.41/15.39 ms | 1.001 | tie (one noisy repeat) |

The rule still clearly helps users running `HELION_AUTOTUNE_EFFORT=none`;
it breaks even against Opus-assisted autotuning. Net-positive overall.

## Artifacts

- `baseline/*_baseline.csv` + `.meta.json`
- `heuristics/*_heuristics.csv` + `.meta.json`
- `scores.json`
- `logs/*.log`
