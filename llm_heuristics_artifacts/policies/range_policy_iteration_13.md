# Iteration 13: Freeze Prompt-Only RMS/Softmax, Split Attention Into Seed-Mechanism Research

Author: Claude Opus 4.7 via no-tool CLI route; Codex will review before benchmark.

## Decision Summary

Iteration 13 should take a hybrid step:

1. Freeze the prompt-only range policy at the three active classes from iteration 12:
   `row_norm_rms`, `row_softmax_narrow`, and `row_softmax_mid`.
2. Stop running further loose prompt-only attention experiments.
3. Open a separate attention d128 seed-mechanism research arm using existing
   data-derived seed support. This is research, not a product policy.

The key reason is the iteration 11/12 contrast:

- Iteration 11 active d128 prompt guidance: `perf_geo=0.806`,
  `time_geo=1.080`, `compile_total=1.523`.
- Iteration 12 no d128 guidance: `perf_geo=1.025`, `time_geo=1.111`.

Prompt guidance likely found better d128 kernels, but prompt-only ranges did
not control compile/wall time. Removing guidance lost the perf win. Another
prompt-only narrowing pass is therefore not the right next move.

## Frozen Prompt-Only Policy

The frozen useful set is:

```text
group               perf geo   time geo   verdict
main_rms               0.979      0.758   material wall-time win, neutral perf
main_softmax_narrow    0.985      0.626   material wall-time win, neutral perf
main_softmax_mid       1.000      0.540   material wall-time win, neutral perf
```

These should only be revalidated if the first-round prompt mechanism, prompt
template, scoring, or seed pool changes.

## Attention d128 Mechanism Research

Run a separate research comparison on GPU 2:

- `baseline`
- `seeds`
- `range_prompt`
- `seeds_plus_range_prompt` if the harness already supports it, otherwise use
  the closest existing composition arm and note the mismatch.

Suggested workloads:

- `attention_2k_d128`
- `attention_4k_d128`

Minimum gate for the `seeds` arm vs baseline, geomean across both d128 shapes:

- `perf_geo <= 0.95`
- `time_geo <= 1.20`
- `compile_total <= 1.20`
- each shape has `perf <= 1.00`

If seed configs clear the gate, iteration 14 should design a real constrained
candidate or seed-injection mechanism and validate it on held-out attention and
non-attention shapes. If seed configs fail, close the attention d128 direction
for this loop and keep the active surface to frozen RMS/softmax.

## Risks

- Seed configs are data-derived from prior AOT/autotune data, but this still
  tests a stronger mechanism than product prompt guidance.
- A win is research evidence only. It is not a green light to ship hardcoded
  shape-specific configs.
- If seed configs improve perf but still regress wall time, the likely next
  mechanism is constrained candidate generation rather than seed-only injection.
