# Q6 — Exp-2 live score (LLM-on, Opus 4.7 via Bedrock)

**Status: DONE (2026-05-10 17:12 UTC). All three kernels PASS
heldout ≤ 0.95. Overfit gate (Δheldout−train ≤ 0.05) respected.**

## Setup

- `LLMGuidedSearch` (`max_rounds=1`, `configs_per_round=5`,
  `initial_random_configs=3`). Opus 4.7 via Bedrock,
  `HELION_LLM_ANTHROPIC_THINKING_BUDGET=8000`.
- Baseline arm: Helion default + random seeds + LLM round-0 proposals
  (no heuristic seed).
- Heuristic arm: same + the per-kernel dispatcher's pick prepended to
  the seed list.
- `HELION_AUTOTUNE_BENCHMARK_SUBPROCESS=1` for crash isolation.
- 3 kernels × 2 arms × 12 shapes × 3 repeats = 216 LLM-guided searches.
- Total wall: **25 min 26 s** (17:47 → 17:12 UTC).

## Results

Round-0 best-ratio geomeans (heuristic / baseline). < 1.0 means the
heuristic arm's round-0 best beats the LLM-only arm's round-0 best.

| kernel             | train   | heldout  | Δ (heldout − train) | target | pass |
|--------------------|--------:|---------:|--------------------:|-------:|:----:|
| `_bf16xint16_gemm` |  0.829  | **0.827**| -0.002              | ≤ 0.95 |  ✅  |
| `matmul_bf16_int4` |  0.576  | **0.581**| +0.005              | ≤ 0.95 |  ✅  |
| `nvfp4_matmul`     |  0.575  | **0.609**| +0.034              | ≤ 0.95 |  ✅  |
| **family**         |  0.650  | **0.664**| +0.014              |  —     |      |

Overfit gate from plan.md: `heldout − train ≤ 0.05` on Exp-2. All
kernels under the gate. `nvfp4_matmul` at +0.034 is the tightest —
noted for Q7 follow-up.

## Interpretation

- **Family heldout 0.664** means the heuristic + LLM combo finds a
  config in round 0 that is ~34% faster than the LLM alone picks
  in round 0, averaged over 45 held-out (shape, repeat) pairs.
- **int4 and fp4 each save ~40% on top of Opus**. These are the
  two kernels with packed weight operands — their tuning surface
  has sharper edges that the learned heuristic captures but the
  LLM's one-shot proposals don't always reach.
- **int16 saves only 17%** on top of Opus. Explanation (from Q3):
  int16 had no rules pass the strict LOSO filter; the dispatcher
  relies heavily on the fallback table there. The LLM round-0 calls
  on int16 already produce strong picks from mainstream matmul-style
  configs, so the fallback's marginal contribution is smaller.
- **No regression** (all three < 1.0). Even on int16 the heuristic
  seed never hurts — the seed is prepended rather than replacing the
  LLM's proposals, so the worst case is "same as baseline".

## Comparison vs dense-GEMM loop

For reference:

| loop          | Exp-1 family heldout | Exp-2 family heldout |
|---------------|----------------------:|----------------------:|
| dense GEMM    |                 0.081 |                 0.849 |
| quantized GEMM|                 0.141 |                 0.664 |

Exp-1 numbers are worse for quantized (dense had more buckets with
rules that promoted through the strict filter, and more archive
measurements). But Exp-2 numbers are **better for quantized** —
because Opus's round-0 picks for quantized GEMM start farther from
optimal than for dense GEMM, so the heuristic seed has more room
to help.

## Artifacts

- `baseline/<kernel>_baseline.csv` + `.meta.json`
- `heuristics/<kernel>_heuristics.csv` + `.meta.json`
- `scores.json` — per-kernel summary + per-(shape, repeat) table
- `logs/*.log` — per-run LLM calls, tracebacks
- `tools/run_q6.sh` — driver

## Known caveat: ms-vs-seconds oddity in absolute CSV values

`run_live.py` multiplies `res.perf` by 1000 before writing to CSV,
inheriting a convention from the gemm loop's port. `BenchmarkResult.perf`
is actually already in ms, so absolute CSV values in this folder are
scaled 1000× vs the Q5 CSVs. The ratio-based scorer is unaffected
(both arms use the same scaling), and all the `round0_best_geo`
numbers above are correct. Flagged for a follow-up cleanup; does not
block Q7.

## Go/no-go for Q7

**PASS on both experiments, all three kernels.** Plan.md's Q7 rule:
"if Exp-1 and Exp-2 pass on at least 2/3 kernels, write a policy doc
and prepare a PR." We pass on 3/3. Proceeding to Q7.
