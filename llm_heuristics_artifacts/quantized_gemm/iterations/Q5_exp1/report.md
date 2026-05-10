# Q5 — Exp-1 live score (no-autotune, single config per shape)

**Status: DONE (2026-05-10). All three kernels PASS the Exp-1 target
(heldout ≤ 0.20). No overfitting — heldout < train for every kernel.**

## Setup

- `HELION_AUTOTUNE_EFFORT=none` forces Helion to use
  `config_spec.default_config()` on the baseline arm.
- Heuristic arm: per-kernel dispatcher from
  `iterations/N6_full_tune/heuristic/heuristic_<kernel>.py` picks a
  single config per live shape via bucket → rule → template (or
  fallback table if the bucket has no promoted rule).
- 3 kernels × 2 arms × 12 shapes × 3 repeats = 216 benchmarks.
- Wall time: ~60 s total (no compile-search needed per config).

## Results

Round-0 best-ratio geomeans (heuristic / baseline). Lower is better;
< 1.0 means the heuristic beats the default config.

| kernel             | train  | heldout | Δ (heldout − train) | target | pass |
|--------------------|-------:|--------:|--------------------:|-------:|:----:|
| `_bf16xint16_gemm` | 0.142  | **0.107** | -0.034           | ≤ 0.20 |  ✅  |
| `matmul_bf16_int4` | 0.157  | **0.132** | -0.025           | ≤ 0.20 |  ✅  |
| `nvfp4_matmul`     | 0.213  | **0.198** | -0.015           | ≤ 0.20 |  ✅  |
| **family**         | 0.168  | **0.141** | -0.027           |  —     |      |

Interpreting family heldout 0.141: the heuristic's single-config
dispatch is ~7.1× faster than Helion's default config on held-out
shapes. Better than the dense-GEMM Exp-1 number (0.081 for dense
was ~12×, but dense had a richer JSON with more promoted rules).

Δheldout-train is negative for all three kernels → the heuristic
generalizes to the 5 held-out shapes better than to train. No
evidence of overfitting.

## Why nvfp4 is the worst of the three

Expected from Q3: nvfp4 had only 2 promoted rules (both skinny_n),
so most of the 12 live shapes fall to the fallback table — less
precision than int4's or int16's rule+fallback mix.
`nvfp4_matmul` heldout 0.198 is still under target, but the
headroom is slim. Worth revisiting the nvfp4 fallback table after
Q6 if LLM-on shows a gap.

## Artifacts

- `baseline/<kernel>_baseline.csv` + `.meta.json` — default-config
  benchmarks, 3 repeats each.
- `heuristics/<kernel>_heuristics.csv` + `.meta.json` — dispatcher
  picks, 3 repeats each.
- `scores.json` — full per-repeat table + per-kernel/family geos.
- `logs/*.log` — per-run stdout/stderr.
- `tools/run_q5.sh` — driver script.

## Go/no-go for Q6

Strong PASS. Proceeding to Q6 (LLM-on, target heldout ≤ 0.95). Q6 is
a harder test: the baseline arm now includes the autotuner's LLM
round-0 proposals, so the heuristic has to add value on top of
Opus's picks rather than just beating the generic default.
