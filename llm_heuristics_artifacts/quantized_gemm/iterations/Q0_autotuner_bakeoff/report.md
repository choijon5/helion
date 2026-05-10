# Q0 — Autotuner bake-off: LFBOTreeSearch vs LLMSeededLFBOTreeSearch

**Status: DONE. Decision: adopt LLMSeededLFBOTreeSearch for archive
tuning from here on (including the Q2 quantized-GEMM expansion).**

## Setup

- Environment: HELION_AUTOTUNE_EFFORT=full, B200 GPU 0,
  HELION_LLM_PROVIDER=bedrock, HELION_LLM_MODEL=us.anthropic.claude-opus-4-7,
  HELION_LLM_ANTHROPIC_THINKING_BUDGET=8000.
- Kernel: examples.matmul.matmul (float16).
- Per-run subprocess via `tools/run_bakeoff.py` so CUDA state is
  clean for each run.
- Shapes (chose 3 from the dense-GEMM grid with different
  characteristics):
  - 256³ — small balanced, heavily represented in the archive.
  - 2048³ — mid balanced, well-understood.
  - 128×2048×2048 — skinny_m held-out shape where dense-GEMM N6
    heuristic was weakest vs LLM picks.

## Setup bug found + fix

First run of LLMSeededLFBOTreeSearch failed instantly on all 3 shapes:
`ValueError: Unsupported LLM provider 'bedrock'`. Root cause: the
choijon5/gemm-hill-climb branch was forked from main and therefore
did not carry the bedrock transport (`helion/autotuner/llm/_bedrock.py`
and the bedrock additions to `helion/autotuner/llm/transport.py`
only live on choijon5/norms-hill-climb). All prior
choijon5/gemm-hill-climb live runs worked because they were executed
from the norms-branch working tree; the gemm branch's own code was
incomplete.

Fix: ported `_bedrock.py` (304 lines, unmodified) and the
bedrock-aware transport.py (513 lines, replaces the 399-line main-branch
version) from origin/choijon5/norms-hill-climb. The branch is now
self-contained.

## Results

| shape          | autotuner                | wall (s) | configs | best (μs) | wall ratio | perf ratio |
|----------------|--------------------------|---------:|--------:|----------:|-----------:|-----------:|
| 256³           | LFBOTreeSearch           |     76.7 |     389 |      7.14 |  —         | —          |
| 256³           | LLMSeededLFBOTreeSearch  |     69.9 |     371 |      7.10 |  0.91×     | 0.996× (slightly better) |
| 2048³          | LFBOTreeSearch           |    477.1 |     518 |     23.55 |  —         | —          |
| 2048³          | LLMSeededLFBOTreeSearch  |    159.2 |     210 |     23.55 |  **0.33×** | 1.000× (tied) |
| 128×2048×2048  | LFBOTreeSearch           |    142.5 |     629 |     11.30 |  —         | —          |
| 128×2048×2048  | LLMSeededLFBOTreeSearch  |     86.4 |     451 |     11.30 |  **0.61×** | 1.000× (tied) |

Geomeans across the 3 shapes:

- Wall-clock: LLMSeeded / LFBO = **0.569**  (1.76× faster)
- Best perf: LLMSeeded / LFBO = **0.9985**  (identical within noise)
- Total wall: LFBO 696s, LLMSeeded **315s**

## Decision per plan.md

Decision rule from the plan was:
- If LLMSeeded is ≥1.5× faster AND within 3% perf: adopt LLMSeeded.
- Otherwise: stick with LFBO.

LLMSeeded is 1.76× faster and **not** within 3% worse — it is tied or
slightly better. **Adopt LLMSeededLFBOTreeSearch** for all archive
tuning from Q2 onward.

## Why it's faster

LLMSeeded seeds the LFBOTreeSearch's starting population from a round
of LLMGuidedSearch proposals rather than random draws. Opus 4.7 picks
a small set of configs that are already strong (≥95% as good as the
final oracle in our dense-GEMM experience). LFBO's surrogate converges
faster from those starting points — it needs fewer generations to find
the oracle because the initial best-so-far is already near-optimal.

The saving scales with tuning depth: 256³ (small search space, easy
to converge) saved 9%; 2048³ (wider search space) saved 67%.
Expect most of our real tuning shapes to look like the 2048³ case.

## Budget implication for Q2 expansion

Original plan (LFBO-only): ~3-10 min/shape × 120 shapes (3 kernels ×
40) ≈ 12 GPU-hours.

With LLMSeeded: ~7 GPU-hours for 120 shapes. Still overnight-safe;
finishes in under half a work day.

## Artifacts

- `LFBOTreeSearch_<shape>.json` × 3  
- `LLMSeededLFBOTreeSearch_<shape>.json` × 3  
- `tools/run_bakeoff.py` — per-run harness.  
- `/tmp/q0_bakeoff.log`, `/tmp/q0_llmseed.log` — run logs.

## Next

Q1: smoke-test the 3 quantized kernels' arg builders (int4, int16,
fp4 packing) end-to-end, then commit the N0_live/shape_grid.json and
iterations/N1_expand/expansion_shapes.json. Q2: launch the expansion
tuning with `HELION_AUTOTUNER=LLMSeededLFBOTreeSearch`.
