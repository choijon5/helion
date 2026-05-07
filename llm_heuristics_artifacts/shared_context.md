# Helion LLM Heuristics Shared Context

This file is the memory anchor for Codex/Claude heuristic iterations. Read it
before proposing a policy, critique, or benchmark plan.

## Latest Canonical State

Latest completed overnight run:
`/tmp/helion_round0_objective_20260505_230436`

Launcher PID:
`1774067` completed at 2026-05-06 03:34 PDT. GPU 2 was idle after completion.

Active corrected objective:
`round0_best_geo` is primary. It is the geometric mean of
`min(perf_ms)` over autotune CSV rows where `generation == 0` and
`status == ok`, compared against the baseline arm. Lower is better.

Analyzer correction:
Use JSON metadata for `workload`, `arm`, and `autotune_log_csv`; do not split
filenames to infer arms, because workload names can contain arm-like suffixes
such as `softmax_4k_2k`. For hybrid/LFBO, score the `_stage1_llm.csv` file
when present.

Latest completed run:
`/tmp/helion_round0_objective_20260505_230436`

Latest accepted policy:
`/tmp/helion_heuristics_loop/claude/range_policy_iteration_12.json`

GPU for new benchmarks:
Use `CUDA_VISIBLE_DEVICES=2` from iteration 11 onward. GPU 2 was checked
idle before launching iteration 11 (`4 MiB`, `0%` utilization).

Latest status:

- Overnight corrected guided iteration 11 (`LLMGuidedSearch`,
  `llm_max_rounds=1`, 19 workloads x 3 repeats):

```text
arm             round0_best_geo   verified_geo
heuristics                0.953          0.958
range_prompt              0.959          0.975
seeds                     0.969          0.966
```

- Overnight corrected guided iteration 12 (`LLMGuidedSearch`,
  `llm_max_rounds=1`, range prompt only, 19 workloads x 3 repeats):

```text
arm             round0_best_geo   verified_geo
range_prompt              0.974          0.989
```

- Overnight corrected hybrid/LFBO handoff (`LLMSeededLFBOTreeSearch`,
  `llm_max_rounds=1`, 8 workloads x 3 repeats):

```text
arm             round0_best_geo   verified_geo
heuristics                0.949          0.973
range_prompt              0.999          0.970
seeds                     0.988          0.978
```

- Current interpretation:
  Guided round-0 is best with iteration-11 `heuristics` globally, with
  `range_prompt` close. Hybrid final verified perf is slightly best with
  `range_prompt`, but differences are small. The data strongly argues for a
  per-kernel/per-shape router instead of one global arm: attention d64 likes
  seeds/heuristics, attention 2k d128 likes heuristics/range, attention 4k d128
  should avoid the current range/heuristics policy, RMS prefers range prompt
  for round-0, and softmax has modest range/heuristic wins. BMM guided round-0
  looks useful, but hybrid stage-1/handoff regressed and needs targeted
  investigation.
- Objective correction:
  For hybrid/LFBO, the primary metric is now `round0_best`: the best measured
  config after seed batch + LLM round-0 proposals, relative to baseline
  `round0_best`. Compile time and wall time are secondary diagnostics only.
  Seed-batch best is useful only to diagnose whether the heuristic seed pool is
  better before the LLM response; it is not the final objective if
  `llm_max_rounds=1`.
- RMS: validated/preserved. Iteration 12 GPU-2 group result:
  `perf_geo=0.979`, `time_geo=0.758`, `cfg_geo=1.099`,
  `max_shape_perf=1.000`. Material wall-time win reproduces with neutral perf,
  but config count remains somewhat elevated.
- Softmax narrow: validated/preserved. Iteration 12 GPU-2 group result:
  `perf_geo=0.985`, `time_geo=0.626`, `cfg_geo=1.025`,
  `max_shape_perf=1.003`. Breadth result remains a strong wall-time win.
- Softmax mid: validated/preserved. Iteration 12 GPU-2 group result:
  `perf_geo=1.000`, `time_geo=0.540`, `cfg_geo=1.018`,
  `max_shape_perf=1.001`. Mid wall-time result remains strong with neutral perf.
- Attention d128 short: blocked/sanity only in iteration 12. With active d128
  guidance removed, `attention_2k_d128` result was `perf_geo=1.025`,
  `time_geo=1.111`, `cfg_geo=0.974`, max time `1.370`,
  `compile_total=0.888`. This loses the earlier prompted perf win and still
  fails wall-time. The earlier d128 prompt likely found useful kernels, but
  prompt-only ranges are not enough to exploit them safely; use a stronger
  seed or constrained-candidate mechanism before testing attention again.
- Attention d64 long: blocked/sanity only. Iteration 12 no-guidance sanity:
  `perf_geo=0.993`, `time_geo=0.972`, `cfg_geo=1.001`, max time `1.046`.
  Sanity is in band; keep blocked/no-guidance.

Latest aggregate, all 16 workloads, lower is better:

```text
arm             perf geo   time geo   cfg geo   perf range    time range    cfg range
range_prompt       0.989      0.688     1.042   0.903-1.089  0.339-1.370  0.760-1.923
```

Main useful heuristic set as of iteration 12:
RMS + softmax narrow + softmax mid. Attention should remain blocked/diagnostic
until a seed or constrained-candidate mechanism can preserve the d128 perf
signal without wall-time regression.

### Range Policy Iteration 13 Request

Status: Claude proposal completed via no-tool Opus 4.7 route; Codex reviewed.

- Inputs Claude must read:
  - This shared context file, especially `Latest Canonical State`.
  - `/tmp/helion_heuristics_loop/claude/range_policy_iteration_12.json`
  - `/tmp/helion_heuristics_loop/claude/range_policy_iteration_12.md`
  - `/tmp/helion_llm_autoresearch_range_prompt_iter12_20260505_1412/aggregate_results.json`
  - `/tmp/helion_llm_autoresearch_range_prompt_iter12_20260505_1412/aggregate_summary.md`
  - `/tmp/helion_heuristics_loop/codex/range_policy_iteration_12_review.md`
- Required output files:
  - `/tmp/helion_heuristics_loop/claude/range_policy_iteration_13.md`
  - `/tmp/helion_heuristics_loop/claude/range_policy_iteration_13.json`
- Claude task:
  - Propose the next policy or harness iteration; do not run benchmarks.
  - Preserve the validated useful set unless there is a strong data-backed
    reason to change it: RMS, softmax narrow, softmax mid.
  - Treat attention d64 long as blocked/no-guidance.
  - For attention d128, interpret iteration 12 correctly: removing guidance
    lost the earlier prompted perf win (`perf_geo=1.025`, `time_geo=1.111`),
    while iteration 11 guidance gave strong perf (`perf_geo=0.806`) but failed
    wall-time/compile (`time_geo=1.080`, `compile_total=1.523`). The next
    useful attention experiment should change the mechanism, e.g. seed configs
    or constrained candidate generation, rather than another prompt-only range.
  - Keep GPU 2 as canonical benchmark GPU for future runs.
  - Keep the objective first-round only unless the proposed mechanism requires
    a clearly justified comparison arm.
  - Treat small deltas as noise; require about `>=5%` perf or `>=20%` wall-time
    improvement before calling something real.

### Range Policy Iteration 13 Review

Status: direction accepted, benchmark plan narrowed before running.

- Claude wrote:
  - `/tmp/helion_heuristics_loop/claude/range_policy_iteration_13.md`
  - `/tmp/helion_heuristics_loop/claude/range_policy_iteration_13.json`
- Codex review:
  `/tmp/helion_heuristics_loop/codex/range_policy_iteration_13_review.md`
- JSON validated:
  `/tmp/helion_heuristics_loop/codex/range_policy_iteration_13.validated.json`
- Accepted policy direction:
  - Freeze prompt-only RMS, narrow softmax, and mid softmax.
  - Do not run another loose prompt-only attention d128 narrowing pass.
  - Test attention d128 seed mechanism separately.
- Codex correction:
  - `seeds_plus_range_prompt` is not an existing harness arm.
  - The current `range_prompt` policy blocks d128, so it is not a useful d128
    prompt comparison unless an older d128-active policy is used.
  - The first clean experiment is `baseline` vs `seeds` on `attention_2k_d128`
    and `attention_4k_d128`.

### Attention d128 Seed Mechanism Iteration 13 Benchmark

Status: completed on GPU 2.

- Output root:
  `/tmp/helion_llm_autoresearch_attention_seed_iter13_20260505_1535`
- Autotuner/model:
  `LLMGuidedSearch`, `gpt-5-2`
- Mode:
  first-round only (`llm_max_rounds=1`)
- Workloads:
  `attention_2k_d128`, `attention_4k_d128`
- Arms:
  `baseline`, `seeds`
- Final aggregate, lower is better:

```text
arm      perf geo   time geo   cfg geo   perf range    time range    cfg range
seeds       1.030      1.057     0.980   0.995-1.184  0.913-1.334  0.923-1.000
```

- Per-workload geomeans:

```text
workload              perf geo   time geo   cfg geo
attention_2k_d128        1.002      1.046     0.974
attention_4k_d128        1.058      1.069     0.987
```

- Attention time attribution:

```text
workload              perf geo   time geo   time p25   median   p75    max   compile total   compile max   bench total
attention_2k_d128        1.002      1.046      0.992    1.000  1.081  1.162          0.923         0.763         0.950
attention_4k_d128        1.058      1.069      0.957    1.002  1.168  1.334          1.017         1.032         1.001
```

- Gate result:
  Seed-only fails. It needed `perf_geo <= 0.95`, `time_geo <= 1.20`,
  `compile_total <= 1.20`, and per-shape perf `<= 1.00`. The aggregate
  `perf_geo=1.030` fails the perf gate, and `attention_4k_d128` fails the
  per-shape perf gate at `1.058`. This does not preserve the iteration-11
  d128 prompt perf signal.
- Interpretation:
  Existing observed seed configs are not enough for attention d128. Close the
  seed-only direction for d128 in this loop unless a real constrained-candidate
  mechanism is implemented and tested. Frozen RMS/softmax remains the useful
  policy surface.
- Seed-quality reinterpretation:
  Compile/wall-time is not the right metric when evaluating whether the seed
  set contains a good starting config. Parsing the round-0 autotune CSVs shows:

```text
view                                             perf ratio, lower better
observed seed-batch best / baseline seed best                      0.535
all round-0 configs best / baseline all round-0 best                1.023
selected verified best / baseline selected verified best            1.030
```

  By workload:

```text
workload              seed-batch best ratio   all round-0 best ratio
attention_2k_d128                     0.287                    1.000
attention_4k_d128                     1.000                    1.046
```

  This means the observed seed list is much better than baseline random seeds
  for `attention_2k_d128`, but neutral for `attention_4k_d128`. Once the LLM's
  first-round configs are included, the best config is not improved; the LLM
  already finds the same/better config. If the product target is a better seed
  for downstream LFBO, this signal is promising only for the 2k d128 bucket and
  should be tested in a pure seed-to-LFBO handoff, not judged by compile time.

### Round-0 Best Re-Scoring

Status: completed from existing autotune CSVs.

Definition:
`round0_best = min(perf_ms)` among all `generation == 0` configs in the
autotune CSV. For `LLMGuidedSearch` with `llm_max_rounds=1`, this is the best
config available for the LLM-stage handoff after seed configs and first LLM
proposal batch.

- Iteration 12 `range_prompt` vs baseline, lower is better:

```text
group                     round0_best_geo   max workload
main_rms                            0.953          1.000
main_softmax_narrow                 0.972          1.001
main_softmax_mid                    1.004          1.015
attention_sanity                    1.021          1.043
all                                 0.978          1.043
```

- Iteration 11 `range_prompt` vs baseline:

```text
all round0_best_geo = 0.967
attention_2k_d128 round0_best_geo = 0.838
rms_norm_2048x4096 round0_best_geo = 0.883
```

- Iteration 13 `seeds` vs baseline for d128 attention:

```text
all round0_best_geo = 1.023
attention_2k_d128 round0_best_geo = 1.000
attention_4k_d128 round0_best_geo = 1.046
```

Interpretation:
Seed-only is not useful for the final round-0 handoff objective. Range/prompt
guidance has modest round-0 best wins overall, stronger for RMS and some
attention policies, but the current active prompt-only policy does not improve
attention. Future experiments should optimize/report `round0_best_geo` first.

### Range Policy Iteration 12 Request

Status: Claude proposal completed; Codex reviewed and accepted for benchmark
on GPU 2.

- Inputs Claude must read:
  - This shared context file, especially `Latest Canonical State`.
  - `/tmp/helion_heuristics_loop/claude/range_policy_iteration_11.json`
  - `/tmp/helion_heuristics_loop/claude/range_policy_iteration_11.md`
  - `/tmp/helion_llm_autoresearch_range_prompt_iter11_20260505_1313/aggregate_results.json`
  - `/tmp/helion_llm_autoresearch_range_prompt_iter11_20260505_1313/aggregate_summary.md`
  - `/tmp/helion_heuristics_loop/codex/range_policy_iteration_11_review.md`
- Required output files:
  - `/tmp/helion_heuristics_loop/claude/range_policy_iteration_12.md`
  - `/tmp/helion_heuristics_loop/claude/range_policy_iteration_12.json`
- Claude task:
  - Propose the next policy or harness iteration; do not run benchmarks.
  - Preserve the validated useful set unless there is a strong data-backed
    reason to change it: RMS, softmax narrow, softmax mid.
  - Treat attention d64 long as blocked/no-guidance.
  - Attention d128 short has now failed the wall-time/compile objective after
    the cheaper prompt trial: iteration 11 `perf_geo=0.806`, `time_geo=1.080`,
    `compile_total=1.523`. The range prompt is guidance, not a hard constraint;
    the d128 CSV still includes non-4-warp configs under the range-prompt arm.
    Because the user cares about lower wall-time and compile time as well as
    kernel perf, do NOT keep d128 as a useful heuristic unless proposing a
    clearly different mechanism with minimal code change.
  - Keep GPU 2 as canonical benchmark GPU for future runs.
  - Keep the objective first-round prompt-only (`LLMGuidedSearch`,
    `llm_max_rounds=1`) unless there is a strong reason to change it.
  - Treat small deltas as noise; require about `>=5%` perf or `>=20%` wall-time
    improvement before calling something real.

### Range Policy Iteration 12 Review

Status: accepted for benchmark.

- Claude wrote:
  - `/tmp/helion_heuristics_loop/claude/range_policy_iteration_12.md`
  - `/tmp/helion_heuristics_loop/claude/range_policy_iteration_12.json`
- Codex review:
  `/tmp/helion_heuristics_loop/codex/range_policy_iteration_12_review.md`
- JSON validated:
  `/tmp/helion_heuristics_loop/codex/range_policy_iteration_12.validated.json`
- Accepted policy direction:
  - Remove `attention:d128_short` from active guidance.
  - Keep RMS, softmax narrow, and softmax mid unchanged as validated useful set.
  - Keep `attention_2k_d128` and `attention_4k_d64` as no-guidance sanity
    workloads.
  - Do not run another d128 prompt-tightening trial unless the mechanism changes
    from guidance to a real constrained candidate mechanism or new telemetry
    identifies a specific compile-cost cause.

### Range Prompt Iteration 12 Benchmark

Status: completed on GPU 2.

- Output root:
  `/tmp/helion_llm_autoresearch_range_prompt_iter12_20260505_1412`
- Policy JSON:
  `/tmp/helion_heuristics_loop/claude/range_policy_iteration_12.json`
- GPU:
  `CUDA_VISIBLE_DEVICES=2`
- Mode:
  first-round only (`LLMGuidedSearch`, `llm_max_rounds=1`),
  prompt-only range heuristic.
- Workloads:
  same 16-workload set as iteration 11; 14 main RMS/softmax workloads plus
  2 no-guidance attention sanity workloads.
- Final aggregate, all 16 workloads, lower is better:

```text
arm             perf geo   time geo   cfg geo   perf range    time range    cfg range
range_prompt       0.989      0.688     1.042   0.903-1.089  0.339-1.370  0.760-1.923
```

- Final group geomeans, lower is better:

```text
group                                      perf geo   time geo   cfg geo   max shape perf   max repeat perf   max time
all                                           0.989      0.688     1.042            1.025            1.089      1.370
main_rms                                      0.979      0.758     1.099            1.000            1.015      1.038
main_softmax_narrow                           0.985      0.626     1.025            1.003            1.005      1.019
main_softmax_mid                              1.000      0.540     1.018            1.001            1.010      0.909
sanity_attention_d128_short_unprompted        1.025      1.111     0.974            1.025            1.089      1.370
sanity_attention_d64_long_unprompted          0.993      0.972     1.001            0.993            1.000      1.046
```

- Final attention attribution:

```text
workload              perf geo   time geo   time p25   median   p75    max   compile total   compile max   bench total
attention_2k_d128        1.025      1.111      1.002    1.044  1.207  1.370          0.888         0.933         0.951
attention_4k_d64         0.993      0.972      0.937    0.961  1.004  1.046          0.897         0.881         1.018
```

- Final interpretation:
  RMS, narrow softmax, and mid softmax remain the useful prompt-only heuristic
  set on GPU 2. Attention d128 without active guidance is worse than baseline
  on kernel perf and wall-time, so the strong d128 result from iterations 10-11
  was likely prompt-attributable. However, prompt-only ranges did not control
  wall-time/compile when d128 was active. Next attention work should use a
  seed or constrained-candidate mechanism rather than another loose prompt
  range.
- Repeat 1 interim, lower is better:

```text
group                                      perf geo   time geo   cfg geo
all                                           0.990      0.769     1.004
main_rms                                      0.984      0.827     1.040
main_softmax_narrow                           0.987      0.744     0.958
main_softmax_mid                              1.000      0.593     1.056
sanity_attention_d128_short_unprompted        0.995      1.044     1.000
sanity_attention_d64_long_unprompted          0.999      1.046     0.962
```

- Repeat 1 attention sanity:
  Without d128 guidance, `attention_2k_d128` is neutral (`perf_cost=0.995`,
  `time_cost=1.044`, `compile_total=0.978`). This suggests the earlier d128
  kernel perf win was prompt/heuristic-attributable, but prompt-only ranges did
  not control compile cost. Continue attention research with a stronger
  mechanism (seeded configs or constrained candidate generation), not another
  loose prompt-only range.
- Repeats 1-2 interim, lower is better:

```text
group                                      perf geo   time geo   cfg geo   max time
all                                           0.990      0.749     1.022      1.370
main_rms                                      0.983      0.790     1.102      0.954
main_softmax_narrow                           0.985      0.726     0.965      1.019
main_softmax_mid                              0.998      0.564     1.021      0.909
sanity_attention_d128_short_unprompted        1.041      1.196     1.000      1.370
sanity_attention_d64_long_unprompted          0.989      1.003     1.021      1.046
```

- Repeats 1-2 attention sanity:
  `attention_2k_d128` without guidance is bad/noisy (`perf_geo=1.041`,
  `time_geo=1.196`, max time `1.370`). This strengthens the conclusion that
  the earlier guided d128 kernel perf win was real, but prompt-only ranges are
  not enough to exploit it safely. Need a controlled seed/constrained-candidate
  experiment for attention.

### Range Policy Iteration 11 Request

Status: Claude Opus 4.7 max-effort proposal completed; Codex reviewed and
accepted for benchmark on GPU 2.

- Inputs Claude must read:
  - This shared context file, especially `Latest Canonical State`.
  - `/tmp/helion_heuristics_loop/claude/range_policy_iteration_10.json`
  - `/tmp/helion_heuristics_loop/claude/range_policy_iteration_10.md`
  - `/tmp/helion_llm_autoresearch_range_prompt_iter10_20260505_1211/aggregate_results.json`
  - `/tmp/helion_llm_autoresearch_range_prompt_iter10_20260505_1211/aggregate_summary.md`
  - `/tmp/helion_heuristics_loop/codex/range_policy_iteration_10_review.md`
- Required output files:
  - `/tmp/helion_heuristics_loop/claude/range_policy_iteration_11.md`
  - `/tmp/helion_heuristics_loop/claude/range_policy_iteration_11.json`
- Claude task:
  - Propose the next policy or harness iteration; do not run benchmarks.
  - Preserve the validated useful set unless there is a strong data-backed
    reason to change it: RMS, softmax narrow, softmax mid.
  - Treat attention d64 long as blocked/no-guidance.
  - Attention d128 short failed the iteration-10 replication gate on wall-time:
    `perf_geo=0.822`, `time_geo=1.106`, `compile_total=1.250`,
    `bench_total=1.071`. Because the user cares about lower wall-time and
    compile time as well as kernel perf, do NOT broaden d128 now.
  - For attention d128 short, decide between demotion, a narrower/cheaper
    prompt policy (for example reduced `candidate_count`, tighter ranges, or
    anti-ranges), or removing attention from the benchmarked useful set. Any
    proposed change must be general, not a hardcoded config, and must be
    expressible with current matcher fields or call out the smallest code
    change needed.
  - Keep the objective first-round prompt-only (`LLMGuidedSearch`,
    `llm_max_rounds=1`) unless there is a strong reason to change it.
  - Treat small deltas as noise; require about `>=5%` perf or `>=20%` wall-time
    improvement before calling something real.

### Range Policy Iteration 11 Review

Status: accepted for benchmark.

- Claude wrote:
  - `/tmp/helion_heuristics_loop/claude/range_policy_iteration_11.md`
  - `/tmp/helion_heuristics_loop/claude/range_policy_iteration_11.json`
- Codex review:
  `/tmp/helion_heuristics_loop/codex/range_policy_iteration_11_review.md`
- JSON validated:
  `/tmp/helion_heuristics_loop/codex/range_policy_iteration_11.validated.json`
- Accepted policy direction:
  - RMS, narrow softmax, and mid softmax are unchanged from iteration 10 and
    remain the only validated useful set for success gating.
  - Attention d128 short is a diagnostic tightening trial, not validated:
    `candidate_count` changes `4 -> 2`, and `num_warps` changes `[4, 8] -> [4]`.
  - Attention d64 long remains blocked/no-guidance.
  - Caveat: `candidate_count=2` is prompt guidance, not a hard execution cap.
    The benchmark must verify the d128 prompt/log and compile telemetry.
- Next benchmark should use:
  `/tmp/helion_heuristics_loop/claude/range_policy_iteration_11.json`
  on GPU 2.

### Range Prompt Iteration 11 Benchmark

Status: completed on GPU 2.

- Output root:
  `/tmp/helion_llm_autoresearch_range_prompt_iter11_20260505_1313`
- Policy JSON:
  `/tmp/helion_heuristics_loop/claude/range_policy_iteration_11.json`
- GPU:
  `CUDA_VISIBLE_DEVICES=2`
- GPU consistency note:
  Do not switch GPUs within an iteration. Iteration 11 is already bound to
  GPU 2; if cross-iteration comparability to GPU-7 iteration 10 is needed,
  rerun the relevant policy on GPU 2 instead of mixing GPU 2 and GPU 7 repeats.
- Mode:
  first-round only (`LLMGuidedSearch`, `llm_max_rounds=1`),
  prompt-only range heuristic.
- Workloads:
  same 16-workload iteration-10 set.
- Final aggregate, all 16 workloads, lower is better:

```text
arm             perf geo   time geo   cfg geo   perf range    time range    cfg range
range_prompt       0.973      0.676     1.026   0.798-1.026  0.370-1.198  0.760-1.368
```

- Final group geomeans, lower is better:

```text
group                                 perf geo   time geo   cfg geo   max shape perf   max repeat perf   max time
all                                      0.973      0.676     1.026            1.012            1.026      1.198
main_rms                                 0.972      0.735     1.096            1.012            1.026      1.152
main_softmax_narrow                      0.989      0.635     0.980            1.003            1.010      0.935
main_softmax_mid                         0.999      0.501     1.004            1.000            1.006      0.580
main_attention_d128_short                0.806      1.080     1.055            0.806            0.822      1.198
sanity_attention_d64_long_unprompted     0.991      0.991     1.014            0.991            1.014      1.050
```

- Final attention attribution:

```text
workload              perf geo   time geo   time p25   median   p75    max   compile total   compile max   bench total
attention_2k_d128        0.806      1.080      1.030    1.125  1.161  1.198          1.523         1.065         1.050
attention_4k_d64         0.991      0.991      0.965    1.030  1.040  1.050          1.024         0.982         0.985
```

- Final interpretation:
  RMS, narrow softmax, and mid softmax continue to look useful on GPU 2.
  Attention d128 remains unacceptable for the user-facing objective: the
  tightened policy preserves strong perf (`perf_geo=0.806`) but still regresses
  wall-time (`time_geo=1.080`) and compile total (`compile_total=1.523`).
  This is worse than the iteration-10 compile issue, not fixed by
  `candidate_count 4 -> 2` plus `num_warps [4,8] -> [4]`.
- Mechanism note:
  The range heuristic is prompt guidance, not a hard constraint. The d128
  autotune CSVs still include some non-4-warp configs under the range-prompt
  arm, likely from default/random seed configs and/or LLM noncompliance. This
  limits how much a prompt-only range can reduce compile cost.
- Repeat 1 interim, lower is better:

```text
group                                 perf geo   time geo   cfg geo
all                                      0.970      0.696     1.004
main_rms                                 0.974      0.768     1.079
main_softmax_narrow                      0.986      0.663     0.967
main_softmax_mid                         0.997      0.497     0.946
main_attention_d128_short                0.800      1.198     1.083
sanity_attention_d64_long_unprompted     0.954      0.900     0.962
```

- Repeat 1 attention attribution:
  `attention_2k_d128` still has strong perf (`perf_cost=0.800`) but fails
  wall-time/compile (`time_cost=1.198`, `compile_total=1.465`). The d128
  tightening did not fix wall-time in repeat 1. `attention_4k_d64` sanity was
  in band (`perf_cost=0.954`, `time_cost=0.900`).
- Repeats 1-2 interim, lower is better:

```text
group                                 perf geo   time geo   cfg geo   max perf   max time
all                                      0.971      0.660     1.019      1.026      1.198
main_rms                                 0.971      0.699     1.107      1.026      1.001
main_softmax_narrow                      0.984      0.623     0.991      1.010      0.888
main_softmax_mid                         0.998      0.506     0.931      1.001      0.580
main_attention_d128_short                0.811      1.058     1.083      0.822      1.198
sanity_attention_d64_long_unprompted     0.983      0.963     0.981      1.014      1.030
```

- Repeats 1-2 attention attribution:
  `attention_2k_d128` still has strong perf (`perf_geo=0.811`) but fails
  wall-time (`time_geo=1.058`) and compile remains elevated
  (`compile_total=1.400`). Unless repeat 3 reverses this, the tightened d128
  policy should not be kept as useful. `attention_4k_d64` sanity remains in
  band (`perf_geo=0.983`, `time_geo=0.963`).

### Range Policy Iteration 10 Request

Status: Claude proposal reviewed by Codex; accepted for benchmark.

- Inputs Claude must read:
  - This shared context file, especially `Latest Canonical State`.
  - `/tmp/helion_heuristics_loop/claude/range_policy_iteration_9.json`
  - `/tmp/helion_heuristics_loop/claude/range_policy_iteration_9.md`
  - `/tmp/helion_llm_autoresearch_range_prompt_iter9_20260505_1112/aggregate_results.json`
  - `/tmp/helion_llm_autoresearch_range_prompt_iter9_20260505_1112/aggregate_summary.md`
  - `/tmp/helion_heuristics_loop/codex/range_policy_iteration_9_review.md`
- Required output files:
  - `/tmp/helion_heuristics_loop/claude/range_policy_iteration_10.md`
  - `/tmp/helion_heuristics_loop/claude/range_policy_iteration_10.json`
- Claude task:
  - Propose the next policy or harness iteration; do not run benchmarks.
  - Preserve the validated useful set unless there is a strong data-backed
    reason to change it: RMS, softmax narrow, softmax mid.
  - Treat attention d64 long as blocked/no-guidance after iteration 9 sanity
    passed (`time_geo=1.028`, max time `1.086`).
  - For attention d128 short, decide whether to keep the one-shape
    validate-active policy, add realistic breadth workloads, or demote it. If
    adding workloads, identify exact workload shapes and whether the harness
    already contains them or needs the smallest code change.
  - Do not broaden to weak classes unless the AOT evidence and prior benchmark
    results justify it materially.
  - Keep the objective first-round prompt-only (`LLMGuidedSearch`,
    `llm_max_rounds=1`) unless there is a strong reason to change it.
  - Treat small deltas as noise; require about `>=5%` perf or `>=20%` wall-time
    improvement before calling something real.

### Range Policy Iteration 10 Review

Status: accepted for benchmark.

- Claude wrote:
  - `/tmp/helion_heuristics_loop/claude/range_policy_iteration_10.md`
  - `/tmp/helion_heuristics_loop/claude/range_policy_iteration_10.json`
- Codex review:
  `/tmp/helion_heuristics_loop/codex/range_policy_iteration_10_review.md`
- JSON validated:
  `/tmp/helion_heuristics_loop/codex/range_policy_iteration_10.validated.json`
- Accepted policy direction:
  - Iteration 10 is a stability replication, not a new heuristic change.
  - Policy keys and matcher-relevant fields are identical to iteration 9 for
    all four policies.
  - The benchmark should answer whether `attention_2k_d128` returns to the
    stronger historical band or continues weakening.
- Next benchmark should use:
  `/tmp/helion_heuristics_loop/claude/range_policy_iteration_10.json`

### Range Prompt Iteration 10 Benchmark

Status: completed.

- Output root:
  `/tmp/helion_llm_autoresearch_range_prompt_iter10_20260505_1211`
- Policy JSON:
  `/tmp/helion_heuristics_loop/claude/range_policy_iteration_10.json`
- Autotuner/model:
  `LLMGuidedSearch`, `gpt-5-2`
- Mode:
  first-round only (`llm_max_rounds=1`), prompt-only range heuristic.
- GPU:
  `CUDA_VISIBLE_DEVICES=7`
- Repeats:
  `3`
- Arms:
  `baseline`, `range_prompt`
- Workloads:
  RMS 5, softmax narrow 6, softmax mid 3, attention d128 validate 1,
  attention d64 sanity 1.
- Final aggregate, all 16 workloads, lower is better:

```text
arm             perf geo   time geo   cfg geo   perf range    time range    cfg range
range_prompt       0.974      0.717     1.044   0.801-1.051  0.437-1.287  0.833-2.364
```

- Group geomeans, lower is better:

```text
group                                 perf geo   time geo   cfg geo   max shape perf   max repeat perf   max time
all                                      0.974      0.717     1.044            1.005            1.051      1.287
main_rms                                 0.976      0.719     1.161            1.005            1.051      1.260
main_softmax_narrow                      0.981      0.702     0.950            0.999            1.001      0.950
main_softmax_mid                         1.002      0.572     1.081            1.005            1.020      0.786
main_attention_d128_short                0.822      1.106     1.027            0.822            0.836      1.287
sanity_attention_d64_long_unprompted     0.999      1.032     0.987            0.999            1.022      1.161
```

- Attention attribution:

```text
workload              perf geo   time geo   time p25   median   p75    max   compile total   compile max   bench total
attention_2k_d128        0.822      1.106      1.026    1.031  1.159  1.287          1.250         1.055         1.071
attention_4k_d64         0.999      1.032      0.972    0.981  1.071  1.161          1.095         1.111         0.966
```

- Interpretation:
  Iteration 10 confirms the d128 prompt still finds faster kernels
  (`perf_geo=0.822`) but fails the user-facing objective because wall-time and
  compile/bench overhead are unstable (`time_geo=1.106`, max `1.287`). Do not
  promote or broaden d128 until that is addressed. RMS, softmax narrow, and
  softmax mid remain the only validated useful set.
- Repeat 1 aligned summary, lower is better:

```text
arm             perf geo   time geo   cfg geo
range_prompt       0.98       0.71      1.05
```

- Repeat 2 aligned summary, lower is better:

```text
arm             perf geo   time geo   cfg geo
range_prompt       0.98       0.75      1.08
```

- Repeat 1 attention replication check:
  `attention_2k_d128` perf recovered (`perf_cost=0.83`) but wall-time slightly
  missed the replication gate (`time_cost=1.03`). `attention_4k_d64`
  no-guidance sanity regressed on wall-time (`time_cost=1.16`). The final
  aggregate above supersedes this interim note.
- Repeat 2 attention replication check:
  `attention_2k_d128` perf stayed strong (`perf_cost=0.80`) but wall-time
  regressed badly (`time_cost=1.29`). The d128 perf signal remains, but the
  wall-time path is not stable. `attention_4k_d64` sanity was fine in this
  repeat (`time_cost=0.96`). The final aggregate above supersedes this interim
  note.
- Command:

```bash
HELION_AUTOTUNE_RANDOM_SEED=20260505 HELION_AUTOTUNE_BENCH_SUBPROCESS=1 \
/home/jongsokchoi/.conda/envs/helion_2/bin/python scripts/llm_heuristics_autoresearch.py \
  --suite core_rows \
  --workloads rms_norm_2048x4096,rms_norm_4k,rms_norm_8192x2048,rms_norm_1024x16384,rms_norm_1024x8192,softmax_4k_1k,softmax_4k_2k,softmax_1k_1k,softmax_1k_2k,softmax_2k_1k,softmax_2k_2k,softmax_4k,softmax_1k_4k,softmax_2k_4k,attention_2k_d128,attention_4k_d64 \
  --arms baseline,range_prompt \
  --repeats 3 \
  --output-root /tmp/helion_llm_autoresearch_range_prompt_iter10_20260505_1211 \
  --gpu 7 \
  --model gpt-5-2 \
  --autotuner LLMGuidedSearch \
  --effort full \
  --llm-max-rounds 1 \
  --range-heuristics-path /tmp/helion_heuristics_loop/claude/range_policy_iteration_10.json \
  --verify-runs 10 \
  --timeout-s 1800 \
  --force
```

### Range Policy Iteration 9 Request

Status: Claude proposal reviewed by Codex; accepted for benchmark.

- Inputs Claude must read:
  - This shared context file, especially `Latest Canonical State`.
  - `/tmp/helion_heuristics_loop/claude/range_policy_iteration_8.json`
  - `/tmp/helion_heuristics_loop/claude/range_policy_iteration_8.md`
  - `/tmp/helion_llm_autoresearch_range_prompt_iter8_20260505_1024/aggregate_results.json`
  - `/tmp/helion_llm_autoresearch_range_prompt_iter8_20260505_1024/aggregate_summary.md`
- Required output files:
  - `/tmp/helion_heuristics_loop/claude/range_policy_iteration_9.md`
  - `/tmp/helion_heuristics_loop/claude/range_policy_iteration_9.json`
- Claude task:
  - Propose the next policy or harness iteration; do not run benchmarks.
  - Preserve the validated useful set unless there is a strong data-backed
    reason to change it: RMS, softmax narrow, softmax mid.
  - For attention, do not promote the whole class. Either propose a guarded
    split that keeps the `attention_2k_d128` win while excluding or diagnosing
    `attention_4k_d64`, or explicitly mark attention as blocked.
  - If proposing a split, make the matcher expressible by current policy JSON
    fields or identify the smallest code change needed before benchmarking.
  - Keep the objective first-round prompt-only (`LLMGuidedSearch`,
    `llm_max_rounds=1`) unless there is a strong reason to change it.
  - Treat small deltas as noise; require about `>=5%` perf or `>=20%` wall-time
    improvement before calling something real.

### Range Policy Iteration 9 Review

Status: accepted for benchmark.

- Claude wrote:
  - `/tmp/helion_heuristics_loop/claude/range_policy_iteration_9.md`
  - `/tmp/helion_heuristics_loop/claude/range_policy_iteration_9.json`
- Codex review:
  `/tmp/helion_heuristics_loop/codex/range_policy_iteration_9_review.md`
- JSON validated:
  `/tmp/helion_heuristics_loop/codex/range_policy_iteration_9.validated.json`
- Accepted policy direction:
  - RMS, softmax narrow, and softmax mid are unchanged in matcher fields,
    ranges, candidate counts, status, lifecycle, and headline groups.
  - Attention is split. `attention_2k_d128` should match the new
    `d128_short` policy; `attention_4k_d64` should match no range policy.
  - The attention d128 rule remains `validate_active`, not broadly validated,
    because it still has only one shape despite a robust repeated signal.
- Manual routing check:

```text
attention_2k_d128 -> d128_short
attention_4k_d64  -> no match
```
- Next benchmark should use:
  `/tmp/helion_heuristics_loop/claude/range_policy_iteration_9.json`

### Range Prompt Iteration 9 Benchmark

Status: completed.

- Output root:
  `/tmp/helion_llm_autoresearch_range_prompt_iter9_20260505_1112`
- Policy JSON:
  `/tmp/helion_heuristics_loop/claude/range_policy_iteration_9.json`
- Autotuner/model:
  `LLMGuidedSearch`, `gpt-5-2`
- Mode:
  first-round only (`llm_max_rounds=1`), prompt-only range heuristic.
- GPU:
  `CUDA_VISIBLE_DEVICES=7`
- Repeats:
  `3`
- Arms:
  `baseline`, `range_prompt`
- Workloads:
  RMS 5, softmax narrow 6, softmax mid 3, attention d128 validate 1,
  attention d64 sanity 1.
- Final aggregate, all 16 workloads, lower is better:

```text
arm             perf geo   time geo   cfg geo   perf range    time range    cfg range
range_prompt       0.978      0.745     1.014   0.827-1.030  0.391-1.131  0.792-1.562
```

- Group geomeans, lower is better:

```text
group                                 perf geo   time geo   cfg geo   max shape perf   max repeat perf   max time
all                                      0.978      0.745     1.014            1.005            1.030      1.131
main_rms                                 0.975      0.779     1.120            1.000            1.004      1.131
main_softmax_narrow                      0.985      0.738     0.968            1.005            1.019      1.101
main_softmax_mid                         1.002      0.594     0.955            1.004            1.014      0.878
main_attention_d128_short                0.876      0.896     0.986            0.876            0.967      0.947
sanity_attention_d64_long_unprompted     0.990      1.028     1.013            0.990            1.030      1.086
```

- Attention attribution:

```text
workload              perf geo   time geo   time p25   median   p75    max   compile total   compile max   bench total
attention_2k_d128        0.876      0.896      0.872    0.899  0.923  0.947          0.962         0.925         1.009
attention_4k_d64         0.990      1.028      1.002    1.049  1.067  1.086          0.963         0.919         0.996
```

- Interpretation:
  Iteration 9 confirms the guarded attention split is safer than the broad
  attention rule: d64 no-guidance sanity is within band, and d128 still shows a
  useful perf signal. However, d128 remains one-shape evidence and one repeat
  was weak (`max_repeat_perf=0.967`), so do not promote attention broadly.
  RMS/softmax remain the validated useful set; their wall-time wins are weaker
  than iteration 8 but still pass the intended guardrails.
- Repeat 1 aligned summary, lower is better:

```text
arm             perf geo   time geo   cfg geo
range_prompt       0.97       0.76      1.00
```

- Repeat 2 aligned summary, lower is better:

```text
arm             perf geo   time geo   cfg geo
range_prompt       0.99       0.73      1.04
```

- Repeat 1 attention split check:
  `attention_2k_d128` stayed helpful (`perf_cost=0.83`, `time_cost=0.84`).
  `attention_4k_d64` no longer showed the prior wall-time regression
  (`perf_cost=0.96`, `time_cost=0.95`). The final aggregate above supersedes
  this interim note.
- Repeat 2 attention split check:
  `attention_2k_d128` was only a small perf/time win (`perf_cost=0.97`,
  `time_cost=0.95`), so the d128 promotion gate is not clearly passing yet.
  `attention_4k_d64` stayed inside the expected no-guidance sanity band
  (`perf_cost=1.03`, `time_cost=1.09`). The final aggregate above supersedes
  this interim note.
- Command:

```bash
HELION_AUTOTUNE_RANDOM_SEED=20260505 HELION_AUTOTUNE_BENCH_SUBPROCESS=1 \
/home/jongsokchoi/.conda/envs/helion_2/bin/python scripts/llm_heuristics_autoresearch.py \
  --suite core_rows \
  --workloads rms_norm_2048x4096,rms_norm_4k,rms_norm_8192x2048,rms_norm_1024x16384,rms_norm_1024x8192,softmax_4k_1k,softmax_4k_2k,softmax_1k_1k,softmax_1k_2k,softmax_2k_1k,softmax_2k_2k,softmax_4k,softmax_1k_4k,softmax_2k_4k,attention_2k_d128,attention_4k_d64 \
  --arms baseline,range_prompt \
  --repeats 3 \
  --output-root /tmp/helion_llm_autoresearch_range_prompt_iter9_20260505_1112 \
  --gpu 7 \
  --model gpt-5-2 \
  --autotuner LLMGuidedSearch \
  --effort full \
  --llm-max-rounds 1 \
  --range-heuristics-path /tmp/helion_heuristics_loop/claude/range_policy_iteration_9.json \
  --verify-runs 10 \
  --timeout-s 1800 \
  --force
```

### Range Policy Iteration 8 Request

Status: Claude proposal accepted; harness telemetry updated.

- Inputs Claude must read:
  - This shared context file, especially `Latest Canonical State`.
  - `/tmp/helion_heuristics_loop/claude/range_policy_iteration_7.json`
  - `/tmp/helion_heuristics_loop/claude/range_policy_iteration_7.md`
  - `/tmp/helion_llm_autoresearch_range_prompt_iter7_20260505_0907/aggregate_results.json`
- Required output files:
  - `/tmp/helion_heuristics_loop/claude/range_policy_iteration_8.md`
  - `/tmp/helion_heuristics_loop/claude/range_policy_iteration_8.json`
- Claude task:
  - Propose the next policy/harness iteration; do not run benchmarks.
  - Treat RMS, softmax narrow, and softmax mid as the current useful validated
    set unless there is a strong reason to change them.
  - For attention, decide between mechanism work and blocking. It has persistent
    wall-time regression on `attention_4k_d64` despite perf wins.
  - If proposing mechanism work, make it concrete and benchmarkable with the
    existing harness, or specify the smallest harness change needed.
  - Do not broaden to weak classes unless the AOT evidence and prior benchmark
    results justify it materially.
  - Keep exact-template prompt and exact seeds disabled unless explicitly
    justified.

### Range Policy Iteration 8 Review

Status: accepted for benchmark.

- Claude wrote:
  - `/tmp/helion_heuristics_loop/claude/range_policy_iteration_8.md`
  - `/tmp/helion_heuristics_loop/claude/range_policy_iteration_8.json`
- JSON syntax validated.
- Accepted policy direction:
  - RMS, softmax narrow, and softmax mid remain the validated useful set.
  - No range, bucket, candidate-count, or workload changes.
  - Attention remains diagnostic; iteration 8 adds attribution telemetry before
    deciding whether to block or add anti-ranges.
- Harness updated:
  - `scripts/llm_heuristics_experiment.py` now records
    `compile_time_per_config_stats` and `benchmark_time_per_batch_stats`.
  - `scripts/llm_heuristics_autoresearch.py` now aggregates compile-time and
    benchmark-time cost ratios, and includes attention attribution columns in
    the markdown summary.
- Validation passed:
  - `/home/jongsokchoi/.conda/envs/helion_2/bin/python -m ruff check scripts/llm_heuristics_experiment.py scripts/llm_heuristics_autoresearch.py`
  - `/home/jongsokchoi/.conda/envs/helion_2/bin/python -m ruff format --check scripts/llm_heuristics_experiment.py scripts/llm_heuristics_autoresearch.py`
- Iteration-8 benchmark should use:
  `/tmp/helion_heuristics_loop/claude/range_policy_iteration_8.json`

### Range Prompt Iteration 8 Benchmark

Status: completed.

- Output root:
  `/tmp/helion_llm_autoresearch_range_prompt_iter8_20260505_1024`
- Policy JSON:
  `/tmp/helion_heuristics_loop/claude/range_policy_iteration_8.json`
- Autotuner/model:
  `LLMGuidedSearch`, `gpt-5-2`
- Mode:
  first-round only (`llm_max_rounds=1`), prompt-only range heuristic.
- GPU:
  `CUDA_VISIBLE_DEVICES=7`
- Repeats:
  `3`
- Arms:
  `baseline`, `range_prompt`
- Workloads:
  RMS 5, softmax narrow 6, softmax mid 3, attention diagnostic 2.
- Extra telemetry:
  aggregate compile-time and benchmark-time cost ratios, plus attention time
  distribution.
- Final aggregate, all 16 workloads, lower is better:

```text
arm             perf geo   time geo   cfg geo   perf range    time range    cfg range
range_prompt       0.968      0.646     1.027   0.744-1.021  0.390-1.271  0.750-1.316
```

- Group geomeans, lower is better:

```text
group                     perf geo   time geo   cfg geo   max shape perf   max repeat perf   max time
all                          0.968      0.646     1.027            1.012            1.021      1.271
main_rms                     0.975      0.665     1.096            1.012            1.021      0.985
main_softmax_narrow          0.986      0.613     0.988            1.001            1.005      0.963
main_softmax_mid             0.999      0.520     1.033            1.003            1.015      0.582
diagnostic_attention         0.859      0.981     0.973            0.950            0.973      1.271
```

- Attention attribution:

```text
workload              perf geo   time geo   time p25   median   p75    max   compile total   compile max   bench total
attention_2k_d128        0.776      0.859      0.818    0.888  0.921  0.954          1.047         1.006         0.966
attention_4k_d64         0.950      1.120      1.051    1.065  1.168  1.271          0.974         0.922         0.925
```

- Interpretation:
  RMS, softmax narrow, and softmax mid stay validated/preserved. Attention is
  still not a single validated class: `2k_d128` is helpful, while `4k_d64`
  remains a wall-time regression even though compile and benchmark totals are
  lower. Do not promote attention without splitting or explaining that behavior.
- Repeat 1 aligned summary, lower is better:

```text
arm             perf geo   time geo   cfg geo
range_prompt       0.97       0.64      0.99
```

- Repeat 2 aligned summary, lower is better:

```text
arm             perf geo   time geo   cfg geo
range_prompt       0.97       0.63      1.07
```

- Repeat 1 note:
  `attention_2k_d128` improved materially on perf and slightly on time
  (`perf_cost=0.74`, `time_cost=0.95`), while `attention_4k_d64` was near-neutral
  on perf with a small wall-time regression (`perf_cost=0.96`, `time_cost=1.04`).
  The final aggregate above supersedes this interim note.
- Repeat 2 note:
  `attention_2k_d128` again improved materially on perf and somewhat on time
  (`perf_cost=0.78`, `time_cost=0.89`), while `attention_4k_d64` again stayed
  near-neutral on perf with a small wall-time regression
  (`perf_cost=0.97`, `time_cost=1.07`). The final aggregate above supersedes
  this interim note.
- Command:

```bash
HELION_AUTOTUNE_RANDOM_SEED=20260505 HELION_AUTOTUNE_BENCH_SUBPROCESS=1 \
/home/jongsokchoi/.conda/envs/helion_2/bin/python scripts/llm_heuristics_autoresearch.py \
  --suite core_rows \
  --workloads rms_norm_2048x4096,rms_norm_4k,rms_norm_8192x2048,rms_norm_1024x16384,rms_norm_1024x8192,softmax_4k_1k,softmax_4k_2k,softmax_1k_1k,softmax_1k_2k,softmax_2k_1k,softmax_2k_2k,softmax_4k,softmax_1k_4k,softmax_2k_4k,attention_2k_d128,attention_4k_d64 \
  --arms baseline,range_prompt \
  --repeats 3 \
  --output-root /tmp/helion_llm_autoresearch_range_prompt_iter8_20260505_1024 \
  --gpu 7 \
  --model gpt-5-2 \
  --autotuner LLMGuidedSearch \
  --effort full \
  --llm-max-rounds 1 \
  --range-heuristics-path /tmp/helion_heuristics_loop/claude/range_policy_iteration_8.json \
  --verify-runs 10 \
  --timeout-s 1800 \
  --force
```

## Current Goal

Improve Helion's LLM-guided autotuner by adding useful, general heuristics to
the initial LLM prompt. The desired policy should describe candidate families
and value ranges, not hard-coded exact configs. We care most about whether the
first LLM round materially improves best-found configs; if it does, it is more
likely to help `LLMSeededLFBOTreeSearch` reduce final runtime or compile time.

Claude proposes; Codex critiques against the data/code constraints; Claude
revises if needed. Benchmark only after Codex accepts the proposal. The latest
completed benchmark is iteration 8; the next step is an iteration-9 proposal
focused on preserving RMS/softmax wins and resolving or splitting attention.

## Materiality Threshold

Treat small deltas as noise. A few percent is not an improvement.

- Perf improvement should be roughly >=5% before calling it real.
- Wall-time improvement should be roughly >=20% before calling it real.
- One-off N=1 LFBO wall-time wins are weak evidence because policy=none arms
  have shown large trajectory divergence.

## Existing Data

AOT CSV data root:
`/home/jongsokchoi/helion_2_aot_pretune_data_all/aot_pretune_data/b200`

Runtime derived heuristic JSON:
`/home/jongsokchoi/helion_2_llm_priors/helion/autotuner/llm/data/observed_heuristics_b200.json`

Derivation script:
`/home/jongsokchoi/helion_2_llm_priors/scripts/llm_heuristics_research.py`

Current benchmark/experiment harness:
`/home/jongsokchoi/helion_2_llm_priors/scripts/llm_heuristics_experiment.py`

Main overnight hybrid result:
`/tmp/helion_llm_autoresearch_overnight_20260504_201938/aggregate_results.json`
`/tmp/helion_llm_autoresearch_overnight_20260504_201938/aggregate_summary.md`

Current Claude policy:
`/tmp/helion_heuristics_loop/claude/proposed_policy.json`

## What We Tried

Initial implementation derived exact sparse config templates from B200 AOT CSVs
and applied them in two ways:

- Initial prompt text via `build_observed_heuristic_guidance`.
- Round-0 seed configs via `observed_heuristic_seed_configs`.

The current mechanism classifies kernels at runtime from traced workload traits,
config-space block rank/reduction structure, tensor shapes, and dtype. It then
matches `kernel_class + shape_bucket` against the JSON rules.

Classes currently recognized include:
`attention`, `matmul`, `matmul_fp8`, `batched_matmul`, `grouped_matmul`,
`split_k_matmul`, `row_softmax`, `row_norm_layer`, `row_norm_rms`,
`row_cross_entropy`, and `elementwise`.

## Important Results So Far

Overnight broad hybrid run covered 46 workloads, arms baseline vs
`heuristics_product`, one repeat:

```text
arm                  perf geo   time geo   cfg geo
--------------------------------------------------
heuristics_product      1.009      0.981     1.046
```

This is not materially better. It is inside noise.

Clear or promising signals:

- `rms_norm_1024x16384`: perf 0.937, time 0.819.
- Some `add`, `bmm`, `attention`, `cross_entropy`, `rms_norm` rows showed
  large wall-time improvements, but several did not have matched rules or were
  N=1 only. Treat these as clues, not proof.

Regressions/problems:

- `matmul` exact-template heuristics were not useful; earlier runs showed
  material regressions.
- `row_softmax` did not generalize; broad run time geo was worse.
- `elementwise` vector-add-derived fp32 rules fired on `exp`, where `exp_1m`
  regressed badly.
- Attention buckets are inconsistent; the same broad rule can help one shape
  and hurt another.
- Some failures occurred in hybrid runs, including `matmul_skinny_m` shared
  memory OOR and `attention_2k_d64` illegal memory access.

## Current Direction

Move from exact templates to range-based prompt heuristics:

- For each kernel class/regime, tell the LLM to propose several candidates
  spanning ranges such as block size families, warp counts, stage counts,
  `pid_type`, reduction-loop choices, persistent-vs-flat scheduling, etc.
- Include anti-ranges/patterns to avoid when the data shows regressions.
- Do not seed exact configs unless later evidence says seeds help.
- Benchmark first-round-only LLM behavior first: `max_rounds=1`, enough
  candidates per round to see whether prompt guidance materially changes the
  first batch.

## Questions To Resolve Before Benchmarking

- Is the AOT CSV corpus sufficient for each class, or do we need more shapes?
- Which classes should receive range guidance now?
- Which classes should be explicitly excluded to avoid misleading the LLM?
- Should promising N=1 wall-time wins in add/bmm/cross_entropy be treated as
  real signals or as LFBO trajectory noise?
- What acceptance gate should we use before claiming a heuristic is useful?

## Iteration Log

### Range Policy Iteration 1

Status: Claude proposal reviewed by Codex; revision required before benchmarking.

- Codex created this shared context file so Claude CLI calls have persistent
  memory across stateless invocations.
- Codex regenerated the AOT-derived evidence snapshot from the CSV corpus:
  `83,985` raw timing rows aggregated to `47,677` kernel/shape/config rows.
- Snapshot directory:
  `/tmp/helion_heuristics_loop/codex/range_policy_data_snapshot`
- Claude has been asked to propose a first range-based initial-prompt policy
  and write:
  `/tmp/helion_heuristics_loop/claude/range_policy_iteration_1.md`
  `/tmp/helion_heuristics_loop/claude/range_policy_iteration_1.json`
- Claude eventually wrote both markdown and valid JSON. The JSON arrived after
  Codex had already started review, but Codex validated it afterward.
- Claude iteration-1 markdown proposed range policies for many classes. Codex
  rejected that as too broad for the first validation pass.
- Codex review:
  `/tmp/helion_heuristics_loop/codex/range_policy_iteration_1_review.md`
- Key Codex decisions from the review:
  - First validation must be prompt-only. Disable exact AOT template seeds to
    avoid confounding prompt quality with seed behavior.
  - Do not call anything "ship" yet. Use "candidate for first-round validation."
  - Keep the initial benchmark scope small: `row_norm_rms` as the main candidate;
    optional diagnostics for attention small-mid and narrow row_softmax only.
  - Hold matmul-family, cross_entropy, layer_norm, generic elementwise, and sum
    until more data or mechanism work exists.
  - Matmul aspect classification already exists; the problem is product evidence
    and rule selection, not basic aspect detection.
  - Elementwise fp32 vector_add data should not be projected onto exp/geglu/swiglu.
  - Cross_entropy and bmm overnight time wins did not have matched rules, so they
    are not heuristic evidence.
- No new benchmarking should run until Claude revises the policy and Codex
  reviews the revision.

### Range Policy Iteration 2

Status: mostly accepted by Codex; one more revision required before
benchmarking.

- Claude read `shared_context.md` and Codex's iteration-1 review, then wrote:
  `/tmp/helion_heuristics_loop/claude/range_policy_iteration_2.md`
  `/tmp/helion_heuristics_loop/claude/range_policy_iteration_2.json`
- Codex validated the JSON.
- Codex review:
  `/tmp/helion_heuristics_loop/codex/range_policy_iteration_2_review.md`
- What improved:
  - Scope is now first-round initial prompt only.
  - Exact AOT template seeds are disabled for validation.
  - Weak classes moved to `blocked_classes`.
  - Main candidate is `row_norm_rms`.
  - `attention small_mid` and `row_softmax narrow` are diagnostics only.
- Required fixes before benchmarking:
  - Replace nonexistent workload `rms_norm_4k_8k` with an existing workload such
    as `rms_norm_1024x8192`.
  - Add two-tier gates: directional signal vs material improvement. Only
    material improvement should be called real.
  - Make `range_prompt` suppress both exact-template seeds and exact-template
    prompt text.
  - Use section title `Range-Based Heuristics`.
  - Keep RMS as one class-level range entry for the first run.
  - Keep `num_stages=1` nuance out of the first prompt.
- No new benchmarking should run until Claude iteration 3 addresses these
  details and Codex confirms the JSON is ready.

### Range Policy Iteration 3

Status: Claude and Codex policy review converged; implementation/benchmark prep
can start.

- Claude wrote and Codex validated:
  `/tmp/helion_heuristics_loop/claude/range_policy_iteration_3.md`
  `/tmp/helion_heuristics_loop/claude/range_policy_iteration_3.json`
- Claude accepted all Codex iteration-2 fixes.
- Final pre-benchmark policy:
  - Objective: first-round, initial-prompt-only.
  - Arm `baseline`: no range prompt, no exact-template prompt, no exact seeds.
  - Arm `range_prompt`: range prompt on, exact-template prompt off, exact seeds
    off.
  - Section title: `Range-Based Heuristics`.
  - Main headline group: `row_norm_rms` only.
  - Diagnostics: curated `attention small_mid` and `row_softmax narrow`, reported
    separately and not included in the success claim.
  - Blocked/held: elementwise, matmul family, cross_entropy, layer_norm, generic
    sum/null class.
- Main RMS workloads:
  `rms_norm_2048x4096`, `rms_norm_4k`, `rms_norm_8192x2048`,
  `rms_norm_1024x16384`, `rms_norm_1024x8192`.
- Two-tier gates:
  - Directional gate: perf geo <= 0.99 and time geo <= 0.95. This is only a weak
    signal, not an improvement claim.
  - Material gate: perf geo <= 0.95 OR time geo <= 0.80 with perf geo <= 1.02.
    This is required to claim a real improvement.
- Codex started implementation support:
  - Added range prompt mode in `helion/autotuner/llm/heuristics.py`.
  - Added `Range-Based Heuristics` section wiring in
    `helion/autotuner/llm/prompting.py`.
  - Added `range_prompt` support to
    `scripts/llm_heuristics_experiment.py`.
  - Added range/first-round pass-through options to
    `scripts/llm_heuristics_autoresearch.py`.
  - Added targeted test coverage in `test/test_llm_autotuner.py`.
- Validation passed:
  - `ruff check helion/autotuner/llm/heuristics.py helion/autotuner/llm/prompting.py scripts/llm_heuristics_experiment.py scripts/llm_heuristics_autoresearch.py test/test_llm_autotuner.py`
  - `ruff format --check helion/autotuner/llm/heuristics.py helion/autotuner/llm/prompting.py scripts/llm_heuristics_experiment.py scripts/llm_heuristics_autoresearch.py test/test_llm_autotuner.py`
  - `pytest test/test_llm_autotuner.py::TestLLMGuidedSearch::test_observed_heuristics_are_opt_in test/test_llm_autotuner.py::TestLLMGuidedSearch::test_observed_heuristics_generate_valid_reduction_seeds test/test_llm_autotuner.py::TestLLMGuidedSearch::test_range_heuristics_are_prompt_only -q`
  - `pytest test/test_llm_autotuner.py::TestLLMGuidedSearch::test_range_heuristics_are_prompt_only -q`

### Range Prompt RMS Benchmark

Status: completed main validation.

- Output root:
  `/tmp/helion_llm_autoresearch_range_prompt_rms_20260505_0648`
- Autotuner/model:
  `LLMGuidedSearch`, `gpt-5-2`
- Mode:
  first-round only (`llm_max_rounds=1`), prompt-only range heuristic.
- GPU:
  `CUDA_VISIBLE_DEVICES=7`
- Repeats:
  `3`
- Arms:
  `baseline`, `range_prompt`
- Main RMS workloads:
  `rms_norm_2048x4096`, `rms_norm_4k`, `rms_norm_8192x2048`,
  `rms_norm_1024x16384`, `rms_norm_1024x8192`.
- Aggregate result, lower is better:

```text
arm             perf geo   time geo   cfg geo   perf range    time range    cfg range
range_prompt       0.975      0.755     1.117   0.896-1.001  0.597-1.136  1.000-1.923
```

- Per workload geomean, lower is better:

```text
workload              perf geo   time geo   cfg geo
rms_norm_1024x16384      0.983      0.739     1.260
rms_norm_1024x8192       0.999      0.810     1.069
rms_norm_2048x4096       0.901      0.632     1.191
rms_norm_4k              0.996      0.706     1.026
rms_norm_8192x2048       1.001      0.916     1.054
```

- Interpretation:
  - Perf improvement is directional only (`2.5%` geomean), not material by the
    user's `>=5%` threshold.
  - Wall-time improvement is material (`24.5%` geomean), clearing the material
    gate (`time_geo <= 0.80` with perf geomean <= `1.02`).
  - No per-workload perf geomean regressed materially; max perf cost was `1.001`.
  - Config count increased (`cfg_geo=1.117`), so the wall-time win likely comes
    from better first-round LLM trajectory/compile behavior, not fewer configs.

### Shared Context Update Protocol

Status: active requirement from the user.

- This file is the persistent memory shared between Codex and Claude CLI.
- Update this file on every research iteration before invoking Claude, after
  Codex critique, after benchmark results, and before changing the next policy.
- Claude CLI should be given this file every time because individual CLI calls
  should not be assumed to retain prior conversation state.
- Each update should include:
  - what policy/version is being evaluated,
  - what changed since the previous iteration,
  - exact benchmark command/output root when a run starts,
  - aligned geomean results when a run finishes,
  - Codex interpretation using the user's materiality rule: a few percent is
    noise; require roughly `>=5%` perf improvement or `>=20%` wall-time
    improvement with no material perf regression.

### Range Prompt Diagnostics Benchmark

Status: completed.

- Output root:
  `/tmp/helion_llm_autoresearch_range_prompt_diagnostics_20260505_0705`
- Autotuner/model:
  `LLMGuidedSearch`, `gpt-5-2`
- Mode:
  first-round only (`llm_max_rounds=1`), prompt-only range heuristic.
- GPU:
  `CUDA_VISIBLE_DEVICES=7`
- Repeats:
  `3`
- Arms:
  `baseline`, `range_prompt`
- Diagnostic workloads:
  `attention_2k_d128`, `attention_4k_d64`, `softmax_4k_1k`,
  `softmax_4k_2k`.
- Aggregate result, lower is better:

```text
arm             perf geo   time geo   cfg geo   perf range    time range    cfg range
range_prompt       0.958      0.869     1.025   0.862-1.080  0.507-1.216  0.920-1.533
```

- Per-workload geomean, lower is better:

```text
workload              perf geo   time geo   cfg geo
attention_2k_d128        0.976      1.078     1.013
attention_4k_d64         0.938      1.151     1.014
softmax_4k_1k            0.999      0.651     0.946
softmax_4k_2k            0.919      0.705     1.138
```

- Interpretation:
  - Overall diagnostic perf cost was `0.958`, close to but not clearly past the
    user's materiality threshold. Treat this as promising but not enough for a
    broad claim.
  - Overall time cost was `0.869`, below baseline but not a material `>=20%`
    improvement.
  - Attention remains mixed: both diagnostic attention workloads improved perf,
    but both regressed wall time. Do not promote attention yet.
  - Softmax narrow is the strongest next candidate: wall time improved
    materially on both tested workloads, while perf was neutral on `4k_1k` and
    materially better on `4k_2k`. The weakness is coverage: only two softmax
    shapes were tested in this diagnostic run.

### Range Policy Iteration 4 Request

Status: Claude proposal received; Codex review requires revision before
benchmarking.

- Inputs Claude must read:
  - This shared context file.
  - `/tmp/helion_heuristics_loop/claude/range_policy_iteration_3.json`
  - `/tmp/helion_heuristics_loop/codex/range_policy_iteration_2_review.md`
  - Fresh AOT evidence snapshot in
    `/tmp/helion_heuristics_loop/codex/range_policy_data_snapshot`
- Required output files:
  - `/tmp/helion_heuristics_loop/claude/range_policy_iteration_4.md`
  - `/tmp/helion_heuristics_loop/claude/range_policy_iteration_4.json`
- Claude task:
  - Preserve RMS as a material wall-time win unless it can improve perf without
    losing the compile-time result.
  - Decide whether softmax has enough AOT evidence to expand beyond the two
    diagnostic shapes, and propose a coverage-backed validation set if so.
  - Keep attention diagnostic-only unless the proposed policy directly addresses
    its wall-time regression.
  - Avoid hardcoded configs. Propose general value ranges by kernel class and
    shape bucket.
  - Keep the next benchmark first-round prompt-only unless Claude gives a strong
    reason to change the experimental design.
  - Apply the user's materiality rule: a few percent is noise.

### Range Policy Iteration 4 Review

Status: revision required before benchmarking.

- Claude wrote:
  - `/tmp/helion_heuristics_loop/claude/range_policy_iteration_4.md`
  - `/tmp/helion_heuristics_loop/claude/range_policy_iteration_4.json`
- JSON syntax validated.
- Codex review:
  `/tmp/helion_heuristics_loop/codex/range_policy_iteration_4_review.md`
- Accepted direction:
  - Preserve RMS range policy verbatim for this iteration.
  - Validate softmax narrow next, while reporting it separately from RMS.
  - Keep softmax mid and attention diagnostic-only.
  - Keep experiment first-round prompt-only with exact-template prompt/seeds
    disabled.
- Required fixes before benchmarking:
  - Attention matcher is currently broken:
    `head_dim_bin: "<=64,<=128"` should be
    `head_dim_bin_in: ["<=64", "<=128"]`; as written, neither diagnostic
    attention workload receives range guidance.
  - Softmax mid probe should use exact `cols_bin: "<=4096"` instead of
    `cols_bin_le: 4096`, because the current policy also matches narrow shapes
    and relies on policy order.
  - Softmax narrow can be tested as a main sub-headline, but success should
    require the material gate. Directional-only on two shapes is not enough.
  - RMS `time_geo <= 0.85` is a regression guard, not reproduction of the
    material time win. Material reproduction still requires `time_geo <= 0.80`
    with `perf_geo <= 1.02`.
- No benchmark should run on iteration-4 JSON until Claude revises these.

### Range Policy Iteration 5 Request

Status: Claude revision accepted for benchmarking.

- Inputs Claude must read:
  - This shared context file.
  - `/tmp/helion_heuristics_loop/claude/range_policy_iteration_4.json`
  - `/tmp/helion_heuristics_loop/claude/range_policy_iteration_4.md`
  - `/tmp/helion_heuristics_loop/codex/range_policy_iteration_4_review.md`
- Required output files:
  - `/tmp/helion_heuristics_loop/claude/range_policy_iteration_5.md`
  - `/tmp/helion_heuristics_loop/claude/range_policy_iteration_5.json`
- Claude task:
  - Apply the required Codex fixes exactly.
  - Keep the experiment first-round prompt-only.
  - Do not broaden the benchmark unless the revision explicitly explains why.
  - Keep exact-template prompt and exact-template seeds disabled.
  - Produce JSON that the runtime matcher can actually apply to the intended
    workloads.

### Range Policy Iteration 5 Review

Status: accepted for benchmark.

- Claude wrote:
  - `/tmp/helion_heuristics_loop/claude/range_policy_iteration_5.md`
  - `/tmp/helion_heuristics_loop/claude/range_policy_iteration_5.json`
- JSON syntax validated.
- Codex matcher sanity check passed:

```text
softmax_4k_1k      -> ['narrow']
softmax_4k_2k      -> ['narrow']
softmax_4k         -> ['mid_probe']
attention_2k_d128  -> ['small_mid']
attention_4k_d64   -> ['small_mid']
```

- Iteration-5 changes from rejected iteration 4:
  - Attention uses `head_dim_bin_in: ["<=64", "<=128"]`, so diagnostics now
    receive the intended guidance.
  - Softmax mid uses exact `cols_bin: "<=4096"`, so it no longer relies on
    policy order and no longer matches narrow shapes.
  - Softmax narrow success requires the material gate; directional-only on two
    shapes triggers harness expansion, not a success claim.
  - RMS gate split:
    `rms_regression_guard` is a no-regression check, and
    `rms_material_reproduction` is required to claim the previous material
    wall-time win reproduced.
- Next benchmark should use:
  `/tmp/helion_heuristics_loop/claude/range_policy_iteration_5.json`

### Range Prompt Iteration 5 Benchmark

Status: completed.

- Output root:
  `/tmp/helion_llm_autoresearch_range_prompt_iter5_20260505_0731`
- Policy JSON:
  `/tmp/helion_heuristics_loop/claude/range_policy_iteration_5.json`
- Autotuner/model:
  `LLMGuidedSearch`, `gpt-5-2`
- Mode:
  first-round only (`llm_max_rounds=1`), prompt-only range heuristic.
- GPU:
  `CUDA_VISIBLE_DEVICES=7`
- Repeats:
  `3`
- Arms:
  `baseline`, `range_prompt`
- Main RMS workloads:
  `rms_norm_2048x4096`, `rms_norm_4k`, `rms_norm_8192x2048`,
  `rms_norm_1024x16384`, `rms_norm_1024x8192`.
- Main softmax narrow workloads:
  `softmax_4k_1k`, `softmax_4k_2k`.
- Diagnostics:
  `attention_2k_d128`, `attention_4k_d64`, `softmax_4k`.
- Command:

```bash
HELION_AUTOTUNE_RANDOM_SEED=20260505 HELION_AUTOTUNE_BENCH_SUBPROCESS=1 \
/home/jongsokchoi/.conda/envs/helion_2/bin/python scripts/llm_heuristics_autoresearch.py \
  --suite core_rows \
  --workloads rms_norm_2048x4096,rms_norm_4k,rms_norm_8192x2048,rms_norm_1024x16384,rms_norm_1024x8192,softmax_4k_1k,softmax_4k_2k,attention_2k_d128,attention_4k_d64,softmax_4k \
  --arms baseline,range_prompt \
  --repeats 3 \
  --output-root /tmp/helion_llm_autoresearch_range_prompt_iter5_20260505_0731 \
  --gpu 7 \
  --model gpt-5-2 \
  --autotuner LLMGuidedSearch \
  --effort full \
  --llm-max-rounds 1 \
  --range-heuristics-path /tmp/helion_heuristics_loop/claude/range_policy_iteration_5.json \
  --verify-runs 10 \
  --timeout-s 1800 \
  --force
```

### Range Prompt Iteration 6 Final Results (Latest)

Status: canonical final result for iteration 6. This is the latest section and
supersedes any earlier partial repeat notes.

- Aggregate result, all 14 workloads, lower is better:

```text
arm             perf geo   time geo   cfg geo   perf range    time range    cfg range
range_prompt       0.973      0.734     1.008   0.788-1.036  0.420-1.229  0.760-1.300
```

- Group geomeans, lower is better. `max shape perf` is computed from
  per-workload geomeans, which is the right "per-shape" gate value after
  3 repeats. `max repeat perf` is shown separately as repeat-level noise.

```text
group                     perf geo   time geo   cfg geo   max shape perf   max repeat perf
main_rms                     0.984      0.705     1.101            1.007            1.022
main_softmax_narrow          0.991      0.705     0.955            1.011            1.036
diagnostic_attention         0.882      1.045     1.013            0.974            1.002
diagnostic_softmax_mid       0.999      0.568     0.889            0.999            1.000
```

- Per-workload geomeans, lower is better:

```text
workload                  perf geo   time geo   cfg geo   max repeat perf
attention_2k_d128            0.799      1.021     1.013            0.805
attention_4k_d64             0.974      1.071     1.013            1.002
rms_norm_1024x16384          0.962      0.634     1.091            0.995
rms_norm_1024x8192           0.996      0.688     1.040            1.000
rms_norm_2048x4096           0.957      0.704     1.187            1.002
rms_norm_4k                  0.999      0.743     1.069            1.003
rms_norm_8192x2048           1.007      0.762     1.120            1.022
softmax_1k_1k                0.997      0.798     0.915            1.008
softmax_1k_2k                1.001      0.835     1.013            1.003
softmax_2k_1k                1.011      0.645     0.929            1.036
softmax_2k_2k                0.994      0.560     1.058            0.999
softmax_4k                   0.999      0.568     0.889            1.000
softmax_4k_1k                0.999      0.736     0.919            1.000
softmax_4k_2k                0.947      0.696     0.905            0.990
```

- Gate interpretation:
  - Overall perf improvement is noise-band (`2.7%`), but overall wall-time
    improvement is material (`26.6%`).
  - RMS material reproduction passed using per-workload geomean as per-shape:
    `time_geo=0.705 <= 0.80`, `perf_geo=0.984 <= 1.02`,
    `max shape perf=1.007 <= 1.02`.
  - Softmax narrow breadth gate passed:
    `time_geo=0.705 <= 0.80`, `perf_geo=0.991 <= 1.02`,
    `max shape perf=1.011 <= 1.02`, `6` shapes.
  - The four new narrow softmax shapes alone are neutral on perf and materially
    faster on wall time:
    `perf_geo=1.001`, `time_geo=0.700`, `cfg_geo=0.977`,
    `max shape perf=1.011`.
  - Attention remains diagnostic-only. It has strong perf improvement
    (`perf_geo=0.882`) but still fails the wall-time promotion gate
    (`time_geo=1.045 > 1.02`).
  - Softmax mid remains promising but only one diagnostic shape.

### Range Policy Iteration 7 Request

Status: Claude proposal accepted; harness updated.

- Inputs Claude must read:
  - This shared context file.
  - `/tmp/helion_heuristics_loop/claude/range_policy_iteration_6.json`
  - `/tmp/helion_heuristics_loop/claude/range_policy_iteration_6.md`
  - `/tmp/helion_llm_autoresearch_range_prompt_iter6_20260505_0812/aggregate_results.json`
  - Fresh AOT evidence snapshot in
    `/tmp/helion_heuristics_loop/codex/range_policy_data_snapshot`
- Required output files:
  - `/tmp/helion_heuristics_loop/claude/range_policy_iteration_7.md`
  - `/tmp/helion_heuristics_loop/claude/range_policy_iteration_7.json`
- Claude task:
  - Propose the next policy/harness iteration; do not run benchmarks.
  - RMS material reproduction passed again; decide whether RMS should now be
    treated as validated and preserved.
  - Softmax narrow breadth gate passed on 6 shapes; decide whether it should be
    treated as validated and preserved, or whether another breadth expansion is
    needed before calling it useful.
  - Attention has now repeatedly improved perf but failed wall-time promotion.
    Decide whether to move it to blocked until compile/SMEM validation exists,
    or propose a concrete diagnostic that directly targets wall time.
  - Softmax mid is promising but only one shape; decide whether iteration 7
    should add mid harness shapes such as `softmax_1k_4k` or `softmax_2k_4k`
    and what gate should apply.
  - Keep exact-template prompt and exact seeds disabled unless the proposal
    explicitly justifies changing the experiment scope.
  - Keep using the materiality rule: a few percent is noise.

### Range Policy Iteration 7 Review

Status: accepted for benchmark.

- Claude wrote:
  - `/tmp/helion_heuristics_loop/claude/range_policy_iteration_7.md`
  - `/tmp/helion_heuristics_loop/claude/range_policy_iteration_7.json`
- JSON syntax validated.
- Accepted policy direction:
  - Treat RMS as `validated_preserved`; keep regression guard.
  - Treat narrow softmax as `validated_preserved`; keep regression guard.
  - Promote softmax mid to active validation by expanding from 1 to 3 shapes.
  - Keep attention diagnostic, but add p25/p75 per-repeat distribution reporting
    to distinguish variance from a persistent wall-time regression.
- Harness updated:
  - `scripts/llm_heuristics_experiment.py`
  - `scripts/llm_heuristics_autoresearch.py`
- Added workloads:
  - `softmax_1k_4k`: `M=1024,N=4096`
  - `softmax_2k_4k`: `M=2048,N=4096`
- Added `p25` and `p75` to aggregate stats and surfaced attention time
  distribution in the markdown summary.
- Validation passed:
  - `/home/jongsokchoi/.conda/envs/helion_2/bin/python -m ruff check scripts/llm_heuristics_experiment.py scripts/llm_heuristics_autoresearch.py`
  - `/home/jongsokchoi/.conda/envs/helion_2/bin/python -m ruff format --check scripts/llm_heuristics_experiment.py scripts/llm_heuristics_autoresearch.py`
- Matcher sanity check passed:

```text
softmax_1k_4k      -> ['mid_probe']
softmax_2k_4k      -> ['mid_probe']
softmax_4k         -> ['mid_probe']
softmax_1k_1k      -> ['narrow']
softmax_2k_2k      -> ['narrow']
attention_2k_d128  -> ['small_mid']
attention_4k_d64   -> ['small_mid']
```

- Iteration-7 benchmark should use:
  `/tmp/helion_heuristics_loop/claude/range_policy_iteration_7.json`

### Range Prompt Iteration 7 Benchmark

Status: completed.

- Output root:
  `/tmp/helion_llm_autoresearch_range_prompt_iter7_20260505_0907`
- Policy JSON:
  `/tmp/helion_heuristics_loop/claude/range_policy_iteration_7.json`
- Autotuner/model:
  `LLMGuidedSearch`, `gpt-5-2`
- Mode:
  first-round only (`llm_max_rounds=1`), prompt-only range heuristic.
- GPU:
  `CUDA_VISIBLE_DEVICES=7`
- Repeats:
  `3`
- Arms:
  `baseline`, `range_prompt`
- Main workloads:
  - RMS: `rms_norm_2048x4096`, `rms_norm_4k`,
    `rms_norm_8192x2048`, `rms_norm_1024x16384`,
    `rms_norm_1024x8192`.
  - Softmax narrow: `softmax_4k_1k`, `softmax_4k_2k`,
    `softmax_1k_1k`, `softmax_1k_2k`, `softmax_2k_1k`,
    `softmax_2k_2k`.
  - Softmax mid: `softmax_4k`, `softmax_1k_4k`, `softmax_2k_4k`.
- Diagnostics:
  `attention_2k_d128`, `attention_4k_d64`.
- Command:

```bash
HELION_AUTOTUNE_RANDOM_SEED=20260505 HELION_AUTOTUNE_BENCH_SUBPROCESS=1 \
/home/jongsokchoi/.conda/envs/helion_2/bin/python scripts/llm_heuristics_autoresearch.py \
  --suite core_rows \
  --workloads rms_norm_2048x4096,rms_norm_4k,rms_norm_8192x2048,rms_norm_1024x16384,rms_norm_1024x8192,softmax_4k_1k,softmax_4k_2k,softmax_1k_1k,softmax_1k_2k,softmax_2k_1k,softmax_2k_2k,softmax_4k,softmax_1k_4k,softmax_2k_4k,attention_2k_d128,attention_4k_d64 \
  --arms baseline,range_prompt \
  --repeats 3 \
  --output-root /tmp/helion_llm_autoresearch_range_prompt_iter7_20260505_0907 \
  --gpu 7 \
  --model gpt-5-2 \
  --autotuner LLMGuidedSearch \
  --effort full \
  --llm-max-rounds 1 \
  --range-heuristics-path /tmp/helion_heuristics_loop/claude/range_policy_iteration_7.json \
  --verify-runs 10 \
  --timeout-s 1800 \
  --force
```

### Range Prompt Iteration 7 Final Results

Status: canonical final result for iteration 7.

- Aggregate result, all 16 workloads, lower is better:

```text
arm             perf geo   time geo   cfg geo   perf range    time range    cfg range
range_prompt       0.969      0.707     1.008   0.770-1.026  0.427-1.345  0.800-1.857
```

- Group geomeans, lower is better. `max shape perf` is computed from
  per-workload geomeans. `max repeat perf` is shown separately as
  repeat-level noise.

```text
group                     perf geo   time geo   cfg geo   max shape perf   max repeat perf
all                          0.969      0.707     1.008            1.000            1.026
main_rms                     0.971      0.729     1.138            1.000            1.002
main_softmax_narrow          0.983      0.646     0.951            0.999            1.026
main_softmax_mid             0.997      0.602     0.914            1.000            1.001
diagnostic_attention         0.884      1.094     1.027            0.977            1.003
```

- Per-workload geomeans, lower is better:

```text
workload                  perf geo   time geo   cfg geo   max repeat perf
attention_2k_d128            0.800      1.006     1.000            0.834
attention_4k_d64             0.977      1.191     1.054            1.003
rms_norm_1024x16384          0.945      0.599     1.297            0.949
rms_norm_1024x8192           1.000      0.623     1.069            1.002
rms_norm_2048x4096           0.916      0.710     1.187            0.947
rms_norm_4k                  0.998      0.704     1.100            1.002
rms_norm_8192x2048           1.000      1.106     1.054            1.001
softmax_1k_1k                0.999      0.812     0.893            1.003
softmax_1k_2k                0.998      0.576     0.986            1.000
softmax_1k_4k                0.996      0.529     0.852            1.001
softmax_2k_1k                0.992      0.615     0.915            1.026
softmax_2k_2k                0.995      0.579     1.002            1.000
softmax_2k_4k                0.995      0.599     1.067            0.998
softmax_4k                   1.000      0.688     0.840            1.001
softmax_4k_1k                0.999      0.711     0.946            1.001
softmax_4k_2k                0.919      0.611     0.971            0.929
```

- Attention time distribution:

```text
workload              perf geo   time geo   p25    median   p75    max
attention_2k_d128        0.800      1.006  0.961   1.002  1.053  1.104
attention_4k_d64         0.977      1.191  1.126   1.240  1.293  1.345
```

- Gate interpretation:
  - Overall perf improvement is noise-band (`3.1%`), but overall wall-time
    improvement is material (`29.3%`).
  - RMS regression guard passed; RMS material reproduction also still passes:
    `time_geo=0.729`, `perf_geo=0.971`, `max shape perf=1.000`.
  - Narrow softmax regression guard passed; breadth still passes:
    `time_geo=0.646`, `perf_geo=0.983`, `max shape perf=0.999`.
  - Mid softmax breadth gate passed on 3 shapes:
    `time_geo=0.602`, `perf_geo=0.997`, `max shape perf=1.000`.
  - Main useful heuristic set is now RMS + narrow softmax + mid softmax:
    consistent material wall-time wins and neutral/noise-band perf.
  - Attention remains diagnostic-only. `attention_2k_d128` is fine, but
    `attention_4k_d64` has persistent wall-time regression:
    `time_geo=1.191`, `median=1.240`, `p25=1.126`.
    Next iteration should either add mechanism work for attention or block it.

- Partial result, repeat 1 lower is better:

```text
group / workload          perf cost   time cost   cfg cost
overall geomean              0.97        0.68        0.99
rms_norm_2048x4096           0.90        0.61        1.24
rms_norm_4k                  1.00        0.59        1.18
rms_norm_8192x2048           1.00        1.16        1.08
rms_norm_1024x16384          0.94        0.59        1.08
rms_norm_1024x8192           1.00        0.60        1.08
softmax_1k_1k                1.00        0.82        0.92
softmax_1k_2k                1.00        0.83        0.96
softmax_2k_1k                0.98        0.46        0.87
softmax_2k_2k                0.99        0.56        1.00
softmax_4k_1k                1.00        0.54        0.92
softmax_4k_2k                0.93        0.65        0.96
softmax_1k_4k                1.00        0.46        0.80
softmax_2k_4k                0.99        0.59        0.95
softmax_4k                   1.00        0.65        0.84
attention_2k_d128            0.80        1.00        0.96
attention_4k_d64             0.96        1.24        1.04
```

- Partial interpretation:
  - Repeat 1 is strong overall.
  - New mid-softmax shapes look good: neutral perf and material wall-time wins.
  - `rms_norm_8192x2048` is again the weak RMS wall-time point.
  - Attention still improves perf but regresses wall time on `attention_4k_d64`.

- Partial result, repeat 2 lower is better:

```text
group / workload          perf cost   time cost   cfg cost
overall geomean              0.97        0.74        1.02
rms_norm_2048x4096           0.95        0.98        1.04
rms_norm_4k                  1.00        0.60        1.08
rms_norm_8192x2048           1.00        1.04        1.04
rms_norm_1024x16384          0.95        0.60        1.86
rms_norm_1024x8192           1.00        0.71        1.04
softmax_1k_1k                1.00        0.86        0.84
softmax_1k_2k                1.00        0.43        1.00
softmax_2k_1k                1.03        1.00        0.88
softmax_2k_2k                1.00        0.55        1.05
softmax_4k_1k                1.00        0.82        0.96
softmax_4k_2k                0.91        0.63        1.04
softmax_1k_4k                0.99        0.73        0.92
softmax_2k_4k                0.99        0.66        0.96
softmax_4k                   1.00        0.60        0.84
attention_2k_d128            0.77        0.92        1.00
attention_4k_d64             0.97        1.01        1.08
```

- Partial interpretation after repeat 2:
  - New mid-softmax remains good.
  - Attention looked much better in repeat 2; final distribution will decide
    whether this is variance or an actual wall-time improvement.
  - `rms_norm_8192x2048` remains the weak RMS wall-time point.

### Range Prompt Iteration 6 Final Results

Status: canonical final result for iteration 6. Any earlier partial repeat
notes are superseded by this section.

- Aggregate result, all 14 workloads, lower is better:

```text
arm             perf geo   time geo   cfg geo   perf range    time range    cfg range
range_prompt       0.973      0.734     1.008   0.788-1.036  0.420-1.229  0.760-1.300
```

- Group geomeans, lower is better. `max shape perf` is computed from
  per-workload geomeans, which is the right "per-shape" gate value after
  3 repeats. `max repeat perf` is shown separately as repeat-level noise.

```text
group                     perf geo   time geo   cfg geo   max shape perf   max repeat perf
main_rms                     0.984      0.705     1.101            1.007            1.022
main_softmax_narrow          0.991      0.705     0.955            1.011            1.036
diagnostic_attention         0.882      1.045     1.013            0.974            1.002
diagnostic_softmax_mid       0.999      0.568     0.889            0.999            1.000
```

- Per-workload geomeans, lower is better:

```text
workload                  perf geo   time geo   cfg geo   max repeat perf
attention_2k_d128            0.799      1.021     1.013            0.805
attention_4k_d64             0.974      1.071     1.013            1.002
rms_norm_1024x16384          0.962      0.634     1.091            0.995
rms_norm_1024x8192           0.996      0.688     1.040            1.000
rms_norm_2048x4096           0.957      0.704     1.187            1.002
rms_norm_4k                  0.999      0.743     1.069            1.003
rms_norm_8192x2048           1.007      0.762     1.120            1.022
softmax_1k_1k                0.997      0.798     0.915            1.008
softmax_1k_2k                1.001      0.835     1.013            1.003
softmax_2k_1k                1.011      0.645     0.929            1.036
softmax_2k_2k                0.994      0.560     1.058            0.999
softmax_4k                   0.999      0.568     0.889            1.000
softmax_4k_1k                0.999      0.736     0.919            1.000
softmax_4k_2k                0.947      0.696     0.905            0.990
```

- Gate interpretation:
  - Overall perf improvement is noise-band (`2.7%`), but overall wall-time
    improvement is material (`26.6%`).
  - RMS material reproduction passed using per-workload geomean as per-shape:
    `time_geo=0.705 <= 0.80`, `perf_geo=0.984 <= 1.02`,
    `max shape perf=1.007 <= 1.02`.
  - Softmax narrow breadth gate passed:
    `time_geo=0.705 <= 0.80`, `perf_geo=0.991 <= 1.02`,
    `max shape perf=1.011 <= 1.02`, `6` shapes.
  - The four new narrow softmax shapes alone are neutral on perf and materially
    faster on wall time:
    `perf_geo=1.001`, `time_geo=0.700`, `cfg_geo=0.977`,
    `max shape perf=1.011`.
  - Attention remains diagnostic-only. It has strong perf improvement
    (`perf_geo=0.882`) but still fails the wall-time promotion gate
    (`time_geo=1.045 > 1.02`).
  - Softmax mid remains promising but only one diagnostic shape.

- Partial result, repeat 1 lower is better:

```text
group / workload          perf cost   time cost   cfg cost
overall geomean              0.98        0.73        0.98
rms_norm_2048x4096           1.00        0.61        1.30
rms_norm_4k                  1.00        0.63        1.04
rms_norm_8192x2048           1.00        0.88        1.04
rms_norm_1024x16384          0.99        0.69        1.00
rms_norm_1024x8192           0.99        0.66        1.04
softmax_1k_1k                0.99        0.77        0.80
softmax_1k_2k                1.00        0.84        0.96
softmax_2k_1k                1.04        0.74        0.80
softmax_2k_2k                1.00        0.55        1.04
softmax_4k_1k                1.00        0.82        0.88
softmax_4k_2k                0.99        0.54        0.88
attention_2k_d128            0.80        1.01        1.00
attention_4k_d64             0.96        1.10        1.04
softmax_4k                   1.00        0.58        1.06
```

- Partial interpretation:
  - Repeat 1 supports wall-time wins overall, but softmax breadth is not clean:
    `softmax_2k_1k` perf cost is `1.04`, above the strict `1.02` breadth gate.
  - RMS wall time is still strong but perf is mostly neutral in repeat 1.
  - Attention still improves perf but regresses wall time.

- Partial result, repeat 2 lower is better:

```text
group / workload          perf cost   time cost   cfg cost
overall geomean              0.97        0.79        1.01
rms_norm_2048x4096           0.98        0.93        1.04
rms_norm_4k                  1.00        0.64        1.13
rms_norm_8192x2048           1.00        1.03        1.04
rms_norm_1024x16384          0.95        0.68        1.00
rms_norm_1024x8192           1.00        0.69        1.04
softmax_1k_1k                1.01        0.75        1.04
softmax_1k_2k                1.00        0.93        1.04
softmax_2k_1k                1.00        0.87        0.96
softmax_2k_2k                0.99        0.56        1.14
softmax_4k_1k                1.00        0.50        0.96
softmax_4k_2k                0.92        0.98        0.92
attention_2k_d128            0.80        1.04        1.04
attention_4k_d64             0.96        1.23        1.00
softmax_4k                   1.00        0.56        0.88
```

- Partial interpretation after repeat 2:
  - Overall remains positive but less strong than iteration 5.
  - RMS has two weak wall-time points in repeat 2:
    `rms_norm_2048x4096` at `0.93` and `rms_norm_8192x2048` at `1.03`.
  - Expanded softmax perf is cleaner than repeat 1, but wall-time breadth
    depends on aggregate because `softmax_1k_2k` and `softmax_4k_2k` are weak.
  - Attention time regression persists.

- Partial result, repeat 1 lower is better:

```text
group / workload          perf cost   time cost   cfg cost
overall geomean              0.94        0.62        1.08
rms_norm_2048x4096           0.90        0.60        1.13
rms_norm_4k                  1.00        0.58        1.09
rms_norm_8192x2048           1.00        0.43        1.30
rms_norm_1024x16384          0.95        0.67        1.04
rms_norm_1024x8192           0.99        0.59        1.09
softmax_4k_1k                1.02        0.50        0.92
softmax_4k_2k                0.91        0.55        1.00
attention_2k_d128            0.74        0.92        1.00
attention_4k_d64             0.90        1.13        1.08
softmax_4k                   1.00        0.54        1.22
```

- Partial interpretation:
  - Repeat 1 looks strong overall, but no claims until all 3 repeats finish.
  - RMS wall-time win reproduced strongly in repeat 1.
  - Softmax narrow wall time is strong; one shape has perf cost `1.02`, right
    at the material-regression boundary.
  - Attention perf improved but one attention shape still regressed wall time.

- Partial result, repeat 2 lower is better:

```text
group / workload          perf cost   time cost   cfg cost
overall geomean              0.94        0.72        1.12
rms_norm_2048x4096           0.90        0.63        1.24
rms_norm_4k                  0.99        0.60        1.08
rms_norm_8192x2048           0.95        0.53        1.30
rms_norm_1024x16384          0.95        0.61        1.37
rms_norm_1024x8192           1.00        0.92        1.04
softmax_4k_1k                1.00        0.60        1.09
softmax_4k_2k                0.92        0.92        0.96
attention_2k_d128            0.80        0.92        1.00
attention_4k_d64             0.92        1.21        1.00
softmax_4k                   1.00        0.56        1.15
```

- Partial interpretation after repeat 2:
  - Overall geomean remains strong, but final claims still require repeat 3.
  - RMS remains a strong wall-time win, though `rms_norm_1024x8192` time was
    weaker in repeat 2.
  - Softmax narrow perf remains neutral/better; wall-time materiality depends
    on the final geomean because `softmax_4k_2k` only hit `0.92` in repeat 2.
  - Attention still improves perf but regresses wall time on `attention_4k_d64`.

- Aggregate result, all 10 workloads, lower is better:

```text
arm             perf geo   time geo   cfg geo   perf range    time range    cfg range
range_prompt       0.946      0.713     1.080   0.739-1.019  0.428-1.221  0.920-1.368
```

- Group geomeans, lower is better:

```text
group                     perf geo   time geo   cfg geo   perf max
all                          0.946      0.713     1.080      1.019
main_rms                     0.967      0.669     1.126      1.002
main_softmax_narrow          0.959      0.639     1.023      1.019
diagnostic_attention         0.857      1.060     1.013      1.000
diagnostic_softmax_mid       1.001      0.549     1.104      1.001
```

- Per-workload geomeans, lower is better:

```text
workload                  perf geo   time geo   cfg geo   perf max
attention_2k_d128            0.781      0.948     1.013      0.804
attention_4k_d64             0.941      1.185     1.014      1.000
rms_norm_1024x16384          0.963      0.670     1.140      0.991
rms_norm_1024x8192           0.997      0.797     1.042      1.001
rms_norm_2048x4096           0.900      0.614     1.183      0.903
rms_norm_4k                  0.996      0.699     1.070      1.001
rms_norm_8192x2048           0.983      0.585     1.207      1.002
softmax_4k                   1.001      0.549     1.104      1.001
softmax_4k_1k                1.006      0.610     0.988      1.019
softmax_4k_2k                0.915      0.669     1.059      0.921
```

- Gate interpretation:
  - RMS regression guard passed.
  - RMS material reproduction passed: `time_geo=0.669 <= 0.80`,
    `perf_geo=0.967 <= 1.02`, `perf max=1.002 <= 1.02`.
  - Softmax narrow material gate passed via wall-time branch:
    `time_geo=0.639 <= 0.80`, `perf_geo=0.959 <= 1.02`,
    `perf max=1.019 <= 1.02`.
  - Attention should not be promoted: `time_geo=1.060`, and
    `attention_4k_d64` time is `1.185`.
  - Softmax mid is promising on one diagnostic shape, but one shape is not
    enough to claim a general mid-softmax heuristic.

### Range Policy Iteration 6 Request

Status: Claude proposal accepted; harness updated.

- Inputs Claude must read:
  - This shared context file.
  - `/tmp/helion_heuristics_loop/claude/range_policy_iteration_5.json`
  - `/tmp/helion_heuristics_loop/claude/range_policy_iteration_5.md`
  - `/tmp/helion_llm_autoresearch_range_prompt_iter5_20260505_0731/aggregate_results.json`
  - Fresh AOT evidence snapshot in
    `/tmp/helion_heuristics_loop/codex/range_policy_data_snapshot`
- Required output files:
  - `/tmp/helion_heuristics_loop/claude/range_policy_iteration_6.md`
  - `/tmp/helion_heuristics_loop/claude/range_policy_iteration_6.json`
- Claude task:
  - Propose the next policy/harness iteration; do not run benchmarks.
  - Softmax narrow cleared material gate but only on two harness shapes. Decide
    whether iteration 6 should add harness workloads such as `softmax_2k_1k`,
    `softmax_8k_1k`, `softmax_2k_2k`, and `softmax_8k_2k` before making a
    broader claim.
  - RMS material reproduction passed. Decide whether to preserve RMS or split
    it by shape bucket to improve perf and/or reduce wall time without losing
    the material result.
  - Attention improved perf but still regressed wall time, especially
    `attention_4k_d64`. Keep it diagnostic or propose a narrower fix; do not
    promote attention unless the wall-time mechanism is addressed.
  - Keep exact-template prompt and exact seeds disabled unless the proposal
    explicitly argues for changing the experiment scope.
  - Keep using materiality rule: a few percent is noise.

### Range Policy Iteration 6 Review

Status: accepted for benchmark.

- Claude wrote:
  - `/tmp/helion_heuristics_loop/claude/range_policy_iteration_6.md`
  - `/tmp/helion_heuristics_loop/claude/range_policy_iteration_6.json`
- JSON syntax validated.
- Codex accepted Claude's substitution of `1k`/`2k` row softmax shapes for
  `8k` row shapes, because `8k` row shapes would not route to the current
  narrow policy (`rows_bin<=4096`).
- Harness updated:
  - `scripts/llm_heuristics_experiment.py`
  - `scripts/llm_heuristics_autoresearch.py`
- Added workloads:
  - `softmax_1k_1k`: `M=1024,N=1024`
  - `softmax_1k_2k`: `M=1024,N=2048`
  - `softmax_2k_1k`: `M=2048,N=1024`
  - `softmax_2k_2k`: `M=2048,N=2048`
- Validation passed:
  - `/home/jongsokchoi/.conda/envs/helion_2/bin/python -m ruff check scripts/llm_heuristics_experiment.py scripts/llm_heuristics_autoresearch.py`
  - `/home/jongsokchoi/.conda/envs/helion_2/bin/python -m ruff format --check scripts/llm_heuristics_experiment.py scripts/llm_heuristics_autoresearch.py`
- Matcher sanity check passed:

```text
softmax_1k_1k      -> ['narrow']
softmax_1k_2k      -> ['narrow']
softmax_2k_1k      -> ['narrow']
softmax_2k_2k      -> ['narrow']
softmax_4k_1k      -> ['narrow']
softmax_4k_2k      -> ['narrow']
softmax_4k         -> ['mid_probe']
attention_2k_d128  -> ['small_mid']
attention_4k_d64   -> ['small_mid']
```

- Iteration-6 benchmark should use:
  `/tmp/helion_heuristics_loop/claude/range_policy_iteration_6.json`

### Range Prompt Iteration 6 Benchmark

Status: completed.

- Output root:
  `/tmp/helion_llm_autoresearch_range_prompt_iter6_20260505_0812`
- Policy JSON:
  `/tmp/helion_heuristics_loop/claude/range_policy_iteration_6.json`
- Autotuner/model:
  `LLMGuidedSearch`, `gpt-5-2`
- Mode:
  first-round only (`llm_max_rounds=1`), prompt-only range heuristic.
- GPU:
  `CUDA_VISIBLE_DEVICES=7`
- Repeats:
  `3`
- Arms:
  `baseline`, `range_prompt`
- Main RMS workloads:
  `rms_norm_2048x4096`, `rms_norm_4k`, `rms_norm_8192x2048`,
  `rms_norm_1024x16384`, `rms_norm_1024x8192`.
- Main softmax narrow workloads:
  `softmax_4k_1k`, `softmax_4k_2k`, `softmax_1k_1k`,
  `softmax_1k_2k`, `softmax_2k_1k`, `softmax_2k_2k`.
- Diagnostics:
  `attention_2k_d128`, `attention_4k_d64`, `softmax_4k`.
- Command:

```bash
HELION_AUTOTUNE_RANDOM_SEED=20260505 HELION_AUTOTUNE_BENCH_SUBPROCESS=1 \
/home/jongsokchoi/.conda/envs/helion_2/bin/python scripts/llm_heuristics_autoresearch.py \
  --suite core_rows \
  --workloads rms_norm_2048x4096,rms_norm_4k,rms_norm_8192x2048,rms_norm_1024x16384,rms_norm_1024x8192,softmax_4k_1k,softmax_4k_2k,softmax_1k_1k,softmax_1k_2k,softmax_2k_1k,softmax_2k_2k,attention_2k_d128,attention_4k_d64,softmax_4k \
  --arms baseline,range_prompt \
  --repeats 3 \
  --output-root /tmp/helion_llm_autoresearch_range_prompt_iter6_20260505_0812 \
  --gpu 7 \
  --model gpt-5-2 \
  --autotuner LLMGuidedSearch \
  --effort full \
  --llm-max-rounds 1 \
  --range-heuristics-path /tmp/helion_heuristics_loop/claude/range_policy_iteration_6.json \
  --verify-runs 10 \
  --timeout-s 1800 \
  --force
```

### Range Prompt Iteration 6 Final Results (Canonical)

Status: canonical final result for iteration 6. This section supersedes earlier
partial repeat notes.

- Aggregate result, all 14 workloads, lower is better:

```text
arm             perf geo   time geo   cfg geo   perf range    time range    cfg range
range_prompt       0.973      0.734     1.008   0.788-1.036  0.420-1.229  0.760-1.300
```

- Group geomeans, lower is better. `max shape perf` is computed from
  per-workload geomeans, which is the correct "per-shape" gate value after
  3 repeats. `max repeat perf` is shown separately as repeat-level noise.

```text
group                     perf geo   time geo   cfg geo   max shape perf   max repeat perf
main_rms                     0.984      0.705     1.101            1.007            1.022
main_softmax_narrow          0.991      0.705     0.955            1.011            1.036
diagnostic_attention         0.882      1.045     1.013            0.974            1.002
diagnostic_softmax_mid       0.999      0.568     0.889            0.999            1.000
```

- Per-workload geomeans, lower is better:

```text
workload                  perf geo   time geo   cfg geo   max repeat perf
attention_2k_d128            0.799      1.021     1.013            0.805
attention_4k_d64             0.974      1.071     1.013            1.002
rms_norm_1024x16384          0.962      0.634     1.091            0.995
rms_norm_1024x8192           0.996      0.688     1.040            1.000
rms_norm_2048x4096           0.957      0.704     1.187            1.002
rms_norm_4k                  0.999      0.743     1.069            1.003
rms_norm_8192x2048           1.007      0.762     1.120            1.022
softmax_1k_1k                0.997      0.798     0.915            1.008
softmax_1k_2k                1.001      0.835     1.013            1.003
softmax_2k_1k                1.011      0.645     0.929            1.036
softmax_2k_2k                0.994      0.560     1.058            0.999
softmax_4k                   0.999      0.568     0.889            1.000
softmax_4k_1k                0.999      0.736     0.919            1.000
softmax_4k_2k                0.947      0.696     0.905            0.990
```

- Gate interpretation:
  - Overall perf improvement is noise-band (`2.7%`), but overall wall-time
    improvement is material (`26.6%`).
  - RMS material reproduction passed using per-workload geomean as per-shape:
    `time_geo=0.705 <= 0.80`, `perf_geo=0.984 <= 1.02`,
    `max shape perf=1.007 <= 1.02`.
  - Softmax narrow breadth gate passed:
    `time_geo=0.705 <= 0.80`, `perf_geo=0.991 <= 1.02`,
    `max shape perf=1.011 <= 1.02`, `6` shapes.
  - The four new narrow softmax shapes alone are neutral on perf and materially
    faster on wall time:
    `perf_geo=1.001`, `time_geo=0.700`, `cfg_geo=0.977`,
    `max shape perf=1.011`.
  - Attention remains diagnostic-only. It has strong perf improvement
    (`perf_geo=0.882`) but still fails the wall-time promotion gate
    (`time_geo=1.045 > 1.02`).
  - Softmax mid remains promising but only one diagnostic shape.
