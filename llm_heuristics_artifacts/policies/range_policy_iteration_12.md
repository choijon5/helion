# Range Policy Iteration 12 (Claude)

Author: Claude Opus 4.7. Iteration 12 builds on the iteration-11 final result
(`/tmp/helion_llm_autoresearch_range_prompt_iter11_20260505_1313/aggregate_results.json`)
and follows the user's iter-12 brief delivered in the prompt. The user's
brief states: do not keep `attention_d128_short` as a useful heuristic unless
a clearly different minimal mechanism is justified; prefer removing/blocking
d128 from the useful set if no such mechanism is justified.

**Iter-12 decision: REMOVE `attention_d128_short`.** Move the
`head_dim_bin_in: ["<=128"], seq_bin_in: ["<=2048"]` bucket from the active
policy list to `blocked_classes`. Demote `attention_2k_d128` from a main
benchmarked workload to a sanity workload alongside `attention_4k_d64`,
mirroring the d64_long handling since iter-9. iter-11 evidence shows the
tightening trial (option b) failed both the user-facing wall-time/compile
objective and Codex's iter-11 caveat condition; no clearly-different minimal
mechanism can be justified from the available telemetry.

RMS, narrow softmax, and mid softmax are preserved byte-identical to iter-11
(and to iter-3..iter-10 for matcher-relevant fields) because all three
iter-11 regression guards passed cleanly. `attention_d64_long` continues
blocked / no-guidance, joined now by `attention_d128_short`.

## TL;DR — what changed vs iteration 11

| # | Topic                              | Iter-11 form                                                                                                                                                                  | Iter-12 form                                                                                                                                                                                                                                                  |
|---|------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1 | RMS                                | validated_preserved, regression guard MANDATORY                                                                                                                                | unchanged (regression guard still mandatory; iter-11 result: time_geo=0.735, perf_geo=0.972, max_shape_perf=1.012)                                                                                                                                            |
| 2 | Narrow softmax                     | validated_preserved, regression guard MANDATORY                                                                                                                                | unchanged (regression guard still mandatory; iter-11 result: time_geo=0.635, perf_geo=0.989, max_shape_perf=1.003)                                                                                                                                            |
| 3 | Mid softmax                        | validated_preserved, regression guard MANDATORY                                                                                                                                | unchanged (regression guard still mandatory; iter-11 result: time_geo=0.501, perf_geo=0.999, max_shape_perf=1.000)                                                                                                                                            |
| 4 | Attention d128_short               | status=validate, lifecycle=diagnostic; tightening trial (candidate_count 4→2, num_warps [4,8]→[4]); replication gate REPORTED ALONGSIDE                                       | **REMOVED from active policy list**. Bucket added to `blocked_classes`. attention_2k_d128 moved from `main_workloads` to `sanity_workloads_list` with new `attention_d128_short_unprompted_sanity` reported-alongside gate (parallel structure to d64_long).   |
| 5 | Attention d64_long                 | excluded from policy list; sanity workload `sanity_attention_d64_long_unprompted`                                                                                              | unchanged (iter-11 sanity passed cleanly: time_geo=0.991, perf_geo=0.991, max=1.050; 3rd consecutive in-band pass — the d64_long block is now empirically confirmed)                                                                                          |
| 6 | Workload count                     | 16 (5 RMS + 6 narrow + 3 mid + 1 attention d128 main + 1 attention d64 sanity)                                                                                                 | 16 (5 RMS + 6 narrow + 3 mid + **1 attention d128 sanity** + 1 attention d64 sanity); count unchanged but d128 moved from main to sanity                                                                                                                       |
| 7 | Mandatory gates                    | 3: rms / narrow / mid regression guards                                                                                                                                        | 3 (unchanged)                                                                                                                                                                                                                                                 |
| 8 | blocked_classes                    | 13 entries                                                                                                                                                                     | **14 entries** (new: `attention_d128_short`)                                                                                                                                                                                                                  |
| 9 | Range / bucket text                | byte-identical to iter-10 for RMS / narrow / mid; tightening edits for d128_short                                                                                              | byte-identical to iter-11 for RMS / narrow / mid; **d128_short policy entry deleted**                                                                                                                                                                         |
| 10| Headline group rename              | `main_attention_d128_short`                                                                                                                                                    | **`sanity_attention_d128_short_unprompted`** (parallel to `sanity_attention_d64_long_unprompted`)                                                                                                                                                            |
| 11| Gate definitions                   | `attention_d128_short_replication_gate` (perf≤0.95, time≤1.02, ...)                                                                                                            | **DELETED** (no longer applies; bucket has no policy). Replaced with `attention_d128_short_unprompted_sanity` (expected_time ∈ [0.90, 1.10], expected_perf ∈ [0.95, 1.05]) — same shape as the d64 sanity gate.                                                |
| 12| Harness / matcher deltas           | none                                                                                                                                                                           | none — REMOVE is just a JSON edit (delete one policy entry, add one blocked_classes entry, move one workload from main to sanity, rename one headline group). No matcher fields change.                                                                       |
| 13| Canonical GPU                      | GPU 2                                                                                                                                                                          | GPU 2 (unchanged; iter-11 was first iter-12-canonical run)                                                                                                                                                                                                    |

The REMOVE is a pure JSON edit: zero matcher / harness / autotuner code
changes. The matcher already supports buckets with no policy entry as
no-guidance (the d64_long pattern since iter-9). attention_2k_d128 simply
joins attention_4k_d64 as a sanity workload — both attention shapes will
receive no Range-Based Heuristics section under the range_prompt arm,
producing baseline-equivalent results modulo LLM API nondeterminism.

## Iter-11 result recap

```text
arm             perf geo   time geo   cfg geo   perf range    time range    cfg range
range_prompt       0.973      0.676     1.026   0.798-1.026  0.370-1.198  0.760-1.368
```

Group geomeans, lower is better:

```text
group                                 perf geo   time geo   cfg geo   max shape perf   max repeat perf   max time
all                                      0.973      0.676     1.026            1.012            1.026      1.198
main_rms                                 0.972      0.735     1.096            1.012            1.026      1.152
main_softmax_narrow                      0.989      0.635     0.980            1.003            1.010      0.935
main_softmax_mid                         0.999      0.501     1.004            1.000            1.006      0.580
main_attention_d128_short                0.806      1.080     1.055            0.806            0.822      1.198
sanity_attention_d64_long_unprompted     0.991      0.991     1.014            0.991            1.014      1.050
```

Attention attribution per the iter-11 telemetry:

```text
workload              perf geo   time geo   time p25   median   p75    max   compile total   compile max   bench total
attention_2k_d128        0.806      1.080      1.030    1.125  1.161  1.198          1.523         1.065         1.050
attention_4k_d64         0.991      0.991      0.965    1.030  1.040  1.050          1.024         0.982         0.985
```

Per-repeat d128 detail under the iter-11 tightened policy:

| repeat   | perf  | time  | cfg   | compile_total | compile_mean | compile_max |
|----------|------:|------:|------:|--------------:|-------------:|------------:|
| repeat_01| 0.800 | 1.198 | 1.083 | 1.465         | 1.353        | 1.008       |
| repeat_02| 0.822 | 0.935 | 1.083 | 1.338         | 1.235        | 0.961       |
| repeat_03| 0.798 | 1.125 | 1.000 | **1.801**     | **1.801**    | 1.246       |

Per-repeat d64 sanity detail (no prompt, third consecutive in-band pass):

| repeat   | perf  | time  | compile_total |
|----------|------:|------:|--------------:|
| repeat_01| 0.954 | 0.900 | 0.764         |
| repeat_02| 1.014 | 1.030 | 1.016         |
| repeat_03| 1.005 | 1.050 | 1.385         |

## Reasoning per the user's iter-12 brief

The user's iter-12 brief asks five things:

1. Preserve validated RMS/softmax policies unless data strongly says otherwise.
2. Treat `attention_d64_long` as blocked / no-guidance.
3. Attention `d128_short` failed iter-11's wall-time/compile objective even
   under the cheaper prompt: `perf_geo=0.806`, `time_geo=1.080`,
   `compile_total=1.523`. Range prompt is guidance not constraint; d128 CSVs
   still include non-4-warp configs under range_prompt. Because the user
   cares about wall-time and compile equally with kernel perf, do NOT keep
   d128 as a useful heuristic unless proposing a clearly different minimal
   mechanism. **Prefer removing/blocking d128 from the useful set if no such
   mechanism is justified.**
4. Keep GPU 2 as canonical benchmark GPU.
5. Keep next benchmark first-round prompt-only unless strongly justified.

Materiality rule: small deltas are noise; ~≥5% perf or ~≥20% wall-time only
count as real.

### Decision 1 — RMS, narrow softmax, mid softmax: still preserved?

**Decision: yes, preserve all three as `validated_preserved` byte-identical
to iter-11.** All three iter-11 regression guards passed cleanly:

| class                | iter-3                | iter-5                | iter-6                | iter-7                | iter-8                | iter-9                                | iter-10                              | iter-11                              | Lifecycle (iter-12)  |
|----------------------|-----------------------|-----------------------|-----------------------|-----------------------|-----------------------|----------------------------------------|---------------------------------------|---------------------------------------|----------------------|
| row_norm_rms         | time=0.755 perf=0.975 | time=0.669 perf=0.967 | time=0.705 perf=0.984 | time=0.729 perf=0.971 | time=0.665 perf=0.975 | time=0.779 perf=0.975 max-shape=1.000 | time=0.719 perf=0.976 max-shape=1.005 | **time=0.735** perf=0.972 max-shape=1.012 | validated_preserved  |
| row_softmax narrow   | (diag)                | time=0.639 perf=0.959 | time=0.705 perf=0.991 | time=0.646 perf=0.983 | time=0.613 perf=0.986 | time=0.738 perf=0.985 max-shape=1.005 | time=0.702 perf=0.981 max-shape=0.999 | **time=0.635** perf=0.989 max-shape=1.003 | validated_preserved  |
| row_softmax mid      | —                     | (diag) time=0.549     | (diag) time=0.568     | time=0.602 perf=0.997 | time=0.520 perf=0.999 | time=0.594 perf=1.002 max-shape=1.004 | time=0.572 perf=1.002 max-shape=1.005 | **time=0.501** **perf=0.999** max-shape=1.000 | validated_preserved  |

iter-11 watch items, NOT acted on in iter-12:

- **Mid perf_geo=0.999 in iter-11** stepped down from iter-9/iter-10's
  near-ceiling 1.002 — an improvement; perf no longer hugs the 1.02 noise
  tolerance. Per-shape perf max=1.000 (vs 1.005 iter-10).
- **RMS time band continues stable**: 8-iter band {0.665, 0.669, 0.705,
  0.719, 0.729, 0.735, 0.755, 0.779}, mean 0.722, std 0.036. iter-11=0.735
  is right at the mean.
- **Narrow softmax time_geo=0.635** is the second-best result in the 7
  iterations narrow has been benchmarked. 7-iter band {0.613, 0.635, 0.646,
  0.702, 0.705, 0.738, 0.739} (iter-3 was diag-only). All comfortably under
  the 0.85 regression-guard ceiling and the 0.80 breadth-gate ceiling.

The RMS cfg_geo=1.096 in iter-11 (down from 1.161 in iter-10) is also a
healthy step — the prompt is steering toward fewer configs than iter-10,
which is good. None of these are at the "data strongly says otherwise"
threshold for changing the policy.

### Decision 2 — Attention d64 long: continue blocked / no-guidance?

**Decision: yes, continue exactly as in iter-11.** No policy entry for the
d64_long bucket. `attention_4k_d64` stays in the harness as the
`sanity_attention_d64_long_unprompted` workload.

iter-11 sanity result:

| metric         | iter-11 actual | expected band   | verdict                                         |
|----------------|----------------|-----------------|-------------------------------------------------|
| `time_cost`    | 0.991          | [0.90, 1.10]    | inside (geomean)                                |
| `perf_cost`    | 0.991          | [0.95, 1.05]    | inside                                          |
| max time_cost  | 1.050          | (≤ 1.10 desired)| **inside in all 3 repeats** (cleanest of 3 iters) |

Per-repeat detail:

| repeat   | perf  | time  | compile_total |
|----------|------:|------:|--------------:|
| repeat_01| 0.954 | 0.900 | 0.764         |
| repeat_02| 1.014 | 1.030 | 1.016         |
| repeat_03| 1.005 | 1.050 | 1.385         |

This is the **third consecutive** in-band sanity for d64_long:

| iter | time_geo | perf_geo | max_time | in band? |
|------|---------:|---------:|---------:|----------|
| 9    | 1.028    | 0.990    | 1.086    | yes      |
| 10   | 1.032    | 0.999    | 1.161    | geomean yes; max past 1.10 in repeat 1 |
| 11   | 0.991    | 0.991    | 1.050    | **yes (cleanest of three)**            |

The d64_long block is now empirically confirmed. iter-12 keeps the sanity
workload because dropping it now would lose comparison data exactly when
attention_2k_d128 joins as a new sanity workload (we want to compare d128
sanity behavior against d64 sanity behavior; both should be neutral). iter-13
or later may consider dropping d64 sanity once d128 sanity has accumulated
its own in-band history.

### Decision 3 — Attention d128 short: REMOVE

**Decision: REMOVE.** Move `head_dim_bin_in: ["<=128"], seq_bin_in:
["<=2048"]` to `blocked_classes`. Demote `attention_2k_d128` from
`main_workloads` to `sanity_workloads_list`, mirroring d64_long.

#### iter-11 d128 outcome — what failed

The iter-11 d128 result, with the iter-11 tightened policy
(`candidate_count=2`, `num_warps=[4]`):

| metric                                | iter-11 actual | gate ceiling | verdict                                |
|---------------------------------------|---------------:|-------------:|----------------------------------------|
| perf_geomean                          | 0.806          | 0.95         | passes by 0.144 (strong)               |
| time_geomean                          | **1.080**      | 1.02         | **FAILS by 0.060**                     |
| per-shape perf max                    | 0.806          | 0.95         | passes by 0.144                        |
| per-shape median time                 | 1.125          | 1.05         | **FAILS by 0.075**                     |
| max_repeat_perf                       | 0.822          | (1.05 noise) | passes                                 |
| max_repeat_time                       | **1.198**      | (1.05 noise) | **outside** repeat-noise band          |
| compile_time_total_cost               | **1.523**      | (≤1.20 desired) | **MATERIAL regression** per ≥20% rule (52% slower) |
| compile_time_max_cost                 | 1.065          | (≤1.20 desired)| inside                                 |
| compile_time_mean_cost (per-config)   | **1.444**      | (≤1.20 desired)| **MATERIAL** (44% slower per config)   |
| benchmark_time_total_cost             | 1.050          | (≤1.20 desired)| inside                                 |

The replication gate fails on **time** (1.080 > 1.02) AND **per-shape
median time** (1.125 > 1.05). Compile cost is **MORE elevated than iter-10**
(`compile_total: 1.523` vs iter-10's 1.250). Per-config compile cost mean
ratio is 1.444 — i.e., the configs the LLM proposes under the tightened
prompt take 44% LONGER to compile per-config than baseline configs.

Per-repeat:

| repeat   | perf  | time  | compile_total | compile_mean |
|----------|------:|------:|--------------:|-------------:|
| repeat_01| 0.800 | 1.198 | 1.465         | 1.353        |
| repeat_02| 0.822 | 0.935 | 1.338         | 1.235        |
| repeat_03| 0.798 | 1.125 | **1.801**     | **1.801**    |

repeat_03 had per-config compile cost 80% slower than baseline. Variance
across repeats is high (std 0.299 on compile_mean ratio).

#### Why tightening did not work

Per the iter-11 stability classifier:

| iter-11 outcome                                                     | classification         | iter-12 path per iter-11 plan                                                                                  |
|---------------------------------------------------------------------|------------------------|----------------------------------------------------------------------------------------------------------------|
| 3 mandatory pass; d128 fails on time only (perf ≤0.95, time in (1.02, 1.10]) | **PARTIAL — TIME**     | Tighten further (l2_groupings → [16]) OR REMOVE                                                                |

iter-11 result: 3 mandatory pass; d128 perf=0.806 (≤0.95), time=1.080 (in
(1.02, 1.10]). This puts iter-11 squarely in the **PARTIAL — TIME** path.
iter-11's plan offered "further tighten" or "REMOVE" as options; the user's
iter-12 brief explicitly chooses REMOVE unless a clearly different minimal
mechanism is justified.

#### Why "further tighten" is NOT a clearly different minimal mechanism

The iter-11 telemetry rules out the next-most-obvious tightening axes:

- **`candidate_count` is already at 2.** The matcher feeds this as
  guidance, not a hard cap; the iter-11 cfg_cost=1.055 shows the LLM
  proposed only ~5% more configs than baseline. Halving candidate_count
  again to 1 would be aggressive guidance for a single-sample bucket and
  would not meaningfully reduce compile work (cfg_cost is already near
  neutral).
- **`num_warps` is already narrowed to [4].** The Codex iter-11 review
  caveat noted: "candidate_count=2 is prompt guidance, not a hard execution
  cap." iter-11 prompt logs (per the same caveat) likely still showed some
  non-4-warp configs in the d128 CSV — the LLM does not strictly comply
  with the range. So further constraint via prompt guidance has diminishing
  returns.
- **`l2_groupings [[4, 8, 16]] → [[16]]`** (the iter-11 PATH B fallback)
  is the next obvious axis to narrow. But the iter-11 telemetry says
  per-config compile mean is 1.444, even with two of the three other axes
  already tightened. That makes l2 narrowing a hypothesis-driven attempt to
  identify a compile-cost mechanism the data has not yet confirmed.
  Importantly, **the user's brief specifically says** "do not keep d128 as
  a useful heuristic unless proposing a clearly different minimal
  mechanism" — it's not asking for one more round of the same prompt-only
  tightening. l2 narrowing is the same mechanism class as iter-11's
  num_warps narrowing; both are "narrow the prompt range and hope the LLM
  complies." That class of mechanism has now been tested twice (iter-9..10
  with broad ranges; iter-11 with two narrowed axes). Both rounds failed
  the wall-time/compile objective.
- **A hard-execution cap** (e.g., cap `candidate_count` at the matcher
  level so the LLM's first round cannot exceed N total configs) would be a
  clearly different mechanism but requires a code change. The iter-11 plan
  explicitly classified this as not minimal: it would change the LLM
  guidance contract from "guidance" to "hard constraint" across all
  policies, with implications for narrow / mid / RMS that we have not
  studied. iter-12 is not the right place to introduce that.
- **An anti_ranges entry** that tells the LLM to AVOID specific
  compile-expensive configs is plausible but has no data to drive it. The
  iter-11 telemetry shows compile_mean is high (1.444) but does not
  attribute the cost to specific (block_sizes, num_stages, num_warps,
  l2_grouping, pid_type) tuples. Without that attribution, an anti_ranges
  entry would be a guess.

So the "clearly different minimal mechanism" the user's brief asks for is
not available without either (a) a code change we have decided is not
warranted by one heuristic, or (b) per-config compile-time attribution
telemetry that we do not currently have.

#### Why DEMOTE-only is also not enough

DEMOTE-only is what iter-11 already chose (lifecycle=diagnostic with the
tightening). The iter-11 outcome shows that lifecycle alone does not
address the user's objective: even diagnostically, applying the d128
prompt costs 52% more compile time. The user's brief explicitly weights
wall-time and compile alongside kernel perf; running the d128 prompt as
diagnostic still incurs that cost on every benchmark. REMOVE is the only
option that stops paying the compile bloat.

#### What about the perf signal we are giving up?

iter-11 d128 perf=0.806 represents a 19.4% kernel perf win — material per
the ≥5% rule, comfortably above the d128 4-iter (iter-5..iter-8) band of
{0.776, 0.781, 0.799, 0.800} and inside the 7-iter band {0.776, 0.781,
0.799, 0.800, 0.806, 0.822, 0.876}. Removing the d128 prompt means
attention_2k_d128 returns to baseline (LLM with no Range-Based Heuristics
section).

This is the user's explicit tradeoff: a 19.4% kernel perf win on a single
shape is not worth a 52% compile-time regression on the same shape, given
the user weights wall-time/compile equally with kernel perf. The 5
iterations of perf wins are not erased — they remain in the provenance
record. iter-12 simply stops applying the d128 prompt.

There is one residual question: **is the iter-11 perf win an artifact of
the prompt or would the LLM find similar configs unprompted?** iter-11
canot answer that directly because the d128 prompt was applied in the
range_prompt arm. iter-12 will answer it: with no prompt for either d128
or d64, attention_2k_d128 in the range_prompt arm should be statistically
equivalent to baseline modulo LLM nondeterminism. If d128 sanity perf is
in [0.95, 1.05] (matching the d64 sanity band), the iter-3..iter-11 perf
wins were prompt-attributable. If d128 sanity perf is below 0.95 (e.g.,
0.85-0.95), then the LLM is finding strong d128 configs without the
prompt — the iter-3..iter-11 wins were partially LLM-baseline and the
prompt's marginal contribution was smaller than reported.

#### Why a swap to a sanity workload (not full deletion)

`attention_2k_d128` stays in the harness as a sanity workload because:

1. **Preserves the 7-iter perf series for trend-tracking.** Iteration N+k
   can compare attention_2k_d128 baseline vs range_prompt and confirm both
   converge once no prompt is applied.
2. **Detects if d64_long sanity behavior is GPU/B200-attention-specific.**
   If iter-12's d128 sanity is wildly unstable (e.g., time_geo > 1.10) but
   d64 sanity is stable, that's a head_dim-128 attention compile issue
   independent of any prompt. If both are stable, sanity testing is
   working.
3. **Cheap.** The sanity workload runs the same autotune as baseline; it
   adds no LLM prompt machinery. Cost is one extra workload-arm per
   benchmark.
4. **Mirrors d64_long pattern.** d64_long has been a no-policy sanity
   since iter-9 (3 consecutive in-band geomeans). d128_short joins the
   same pattern for symmetry.

iter-13 may drop d128 sanity once 2-3 in-band passes accumulate, freeing a
benchmark slot for breadth on a different class.

### Decision 4 — GPU 2 canonical?

**Decision: yes.** iter-11 was the first run on GPU 2 and produced a
clean, full set of results. iter-12 continues on GPU 2.

### Decision 5 — First-round prompt-only?

**Decision: yes.** No iter-11 evidence justifies re-enabling exact-template
prompt or seeds. iter-12's experimental question (does removing d128 leave
RMS/narrow/mid as a clean, stable validated set?) does not require any
re-enablement.

### Recap

| Decision        | Verdict                                                                       | What changes in JSON                                                                                                                                                                            |
|-----------------|-------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| RMS             | validated_preserved (unchanged from iter-11)                                  | none (provenance has iter-11 entry added)                                                                                                                                                       |
| Narrow softmax  | validated_preserved (unchanged from iter-11)                                  | none (provenance has iter-11 entry added)                                                                                                                                                       |
| Mid softmax     | validated_preserved (unchanged from iter-11)                                  | none (provenance has iter-11 entry added)                                                                                                                                                       |
| Attention d128  | **REMOVE**. Bucket → blocked_classes; attention_2k_d128 → sanity_workloads_list | d128_short policy entry deleted; new blocked_classes entry; new sanity_workloads entry; new gate `attention_d128_short_unprompted_sanity`; gate `attention_d128_short_replication_gate` deleted; headline group renamed `main_attention_d128_short` → `sanity_attention_d128_short_unprompted` |
| Attention d64   | excluded from policy list (unchanged from iter-11)                            | none (sanity provenance updated with iter-11 result; 3rd in-band note added)                                                                                                                     |
| Other classes   | block list grows to 14                                                        | one new entry: attention_d128_short                                                                                                                                                              |
| Exact-template  | prompt and seeds remain OFF                                                   | none                                                                                                                                                                                             |
| Telemetry       | iter-7/iter-8 reporting stays                                                 | none (already in harness)                                                                                                                                                                        |
| Workloads       | unchanged 16-workload set; one moved main → sanity                             | main_workloads drops attention_2k_d128; sanity_workloads_list adds attention_2k_d128                                                                                                            |
| Canonical GPU   | GPU 2                                                                         | none (iter-11 already used GPU 2)                                                                                                                                                                |

## What stays the same (vs iter-11)

- Experimental design: first-round prompt-only, two arms (baseline,
  range_prompt), `llm_max_rounds=1`, `repeats=3`, model `gpt-5-2`,
  autotuner `LLMGuidedSearch`, `CUDA_VISIBLE_DEVICES=2`.
- Range prompt arm: range section ON, exact-template prompt OFF, exact
  seeds OFF. Baseline: all three OFF.
- Section title `Range-Based Heuristics`.
- Range JSON values for RMS / narrow / mid: byte-identical to iter-11.
- `bucket_match` for RMS / narrow / mid: byte-identical to iter-11.
- `candidate_count` for RMS / narrow / mid: byte-identical to iter-11.
- Workload set: identical to iter-11 (16 total: 5 RMS + 6 narrow + 3 mid +
  1 attention d128 + 1 attention d64); only the headline assignment of
  attention_2k_d128 changes (main → sanity).
- Per-repeat (p25, p75) reporting introduced in iter-7 stays.
- Per-config compile_time / per-batch benchmark_time aggregation
  introduced in iter-8 stays.
- Materiality rule: a few percent is noise; ≥5% perf or ≥20% wall-time
  for "real".
- 13 of 14 `blocked_classes` entries identical to iter-11; new entry is
  `attention_d128_short`.

## Iteration 12 acceptance gates

| Gate                                          | Bound (geomean / per-shape)                                                                | Role in iter-12                                                                                       |
|-----------------------------------------------|--------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------|
| `directional_gate`                            | perf≤0.99, time≤0.95, per-shape perf≤1.05                                                  | Informational; "worth another iteration", not "works".                                                |
| `material_gate`                               | perf≤0.95 OR (time≤0.80 AND perf≤1.02), per-shape perf≤1.02                                | Required to call a class/group's range guidance real.                                                 |
| `rms_regression_guard`                        | RMS time≤0.85, perf≤1.02, per-shape perf≤1.05                                              | RMS no-regression flag. **MANDATORY** for iter-12 success.                                            |
| `rms_material_reproduction`                   | RMS time≤0.80, perf≤1.02, per-shape perf≤1.02                                              | REPORTED ALONGSIDE.                                                                                   |
| `softmax_narrow_regression_guard`             | narrow time≤0.85, perf≤1.02, per-shape perf≤1.05                                           | Narrow no-regression flag. **MANDATORY** for iter-12 success.                                         |
| `softmax_narrow_breadth_gate`                 | narrow time≤0.80, perf≤1.02, per-shape perf≤1.02, min_shapes=6                             | REPORTED ALONGSIDE.                                                                                   |
| `softmax_mid_regression_guard`                | mid time≤0.85, perf≤1.02, per-shape perf≤1.05, min_shapes=3                                | Mid no-regression flag. **MANDATORY** for iter-12 success.                                            |
| `softmax_mid_breadth_gate`                    | mid time≤0.80, perf≤1.02, per-shape perf≤1.02, min_shapes=3                                | REPORTED ALONGSIDE.                                                                                   |
| `attention_d128_short_unprompted_sanity`      | expected time ∈ [0.90, 1.10], expected perf ∈ [0.95, 1.05]                                 | **NEW**. REPORTED ALONGSIDE. attention_2k_d128 has no policy entry; this gate is sanity, not gating.  |
| `attention_d64_long_unprompted_sanity`        | expected time ∈ [0.90, 1.10], expected perf ∈ [0.95, 1.05]                                 | REPORTED ALONGSIDE; not gating iter-12 success.                                                        |

iter-12 success definition (3 mandatory gates, same count as iter-11):

- `rms_regression_guard` MUST hold **AND**
- `softmax_narrow_regression_guard` MUST hold **AND**
- `softmax_mid_regression_guard` MUST hold.

Reported-alongside (do not gate iter-12 success):

- `rms_material_reproduction` (iter-11 passed cleanly)
- `softmax_narrow_breadth_gate` (iter-11 passed)
- `softmax_mid_breadth_gate` (iter-11 passed; perf no longer at ceiling)
- **`attention_d128_short_unprompted_sanity` (NEW)** — informs whether
  the iter-3..iter-11 d128 perf wins were prompt-attributable.
- `attention_d64_long_unprompted_sanity` — fourth consecutive in-band
  geomean expected; the d64_long block is empirically permanent.
- per-config compile_time / per-batch benchmark_time stats per
  workload-arm — including the new d128 sanity row, which is the headline
  signal for "did REMOVE actually fix the compile cost?"

If any mandatory gate fails:

- RMS / narrow / mid regression — real regression. Roll back to iter-11
  JSON and investigate before further iteration.

If d128 sanity is out of band:

- Time > 1.10 or perf < 0.95 → unexpected. The unprompted LLM should
  produce baseline-like results since both arms have no Range-Based
  Heuristics section for this shape. Investigate harness state, LLM API
  drift, B200 attention compile behavior on bs=[1,128,128] shapes, or
  random-seed effect.
- Time and perf both in band → REMOVE worked as expected. The iter-11
  d128 wall-time/compile regression was prompt-attributable.

If d64 sanity is out of band:

- Time > 1.10 → harness/LLM drift suspected. d64 has now had 3 in-band
  passes; a 4th out-of-band would be the first sign of drift since iter-9.

## Validation design (unchanged framework, no harness changes)

```text
arms:                 baseline, range_prompt
llm_max_rounds:       1
repeats:              3
model:                gpt-5-2
autotuner:            LLMGuidedSearch
GPU:                  CUDA_VISIBLE_DEVICES=2
range_prompt arm:     range section ON, exact-template prompt OFF, exact seeds OFF
baseline arm:         range section OFF, exact-template prompt OFF, exact seeds OFF

main workloads (14):
  rms_norm_2048x4096, rms_norm_4k, rms_norm_8192x2048,
  rms_norm_1024x16384, rms_norm_1024x8192,
  softmax_4k_1k, softmax_4k_2k, softmax_1k_1k, softmax_1k_2k,
  softmax_2k_1k, softmax_2k_2k,
  softmax_4k, softmax_1k_4k, softmax_2k_4k

sanity workloads (2):
  attention_2k_d128 (NEW: was main in iter-9..11; sanity from iter-12 onward)
  attention_4k_d64

headline groups:
  main_rms                                = 5 RMS shapes               (validated_preserved, mandatory)
  main_softmax_narrow                     = 6 narrow softmax shapes    (validated_preserved, mandatory)
  main_softmax_mid                        = 3 mid softmax shapes       (validated_preserved, mandatory)
  sanity_attention_d128_short_unprompted  = 1 attention d128 shape     (sanity tracking only; NEW)
  sanity_attention_d64_long_unprompted    = 1 attention d64 shape      (sanity tracking only)
```

Total: 14 main + 2 sanity = 16 shapes, identical to iter-11 in count.
Note: the harness should bucket attention_2k_d128 under the new
`sanity_attention_d128_short_unprompted` headline group. If the harness's
attention markdown auto-rendering is workload-name-prefix-based, the row
will still print under the existing attention block; only the policy
attribution changes (no policy → no Range-Based Heuristics section in the
prompt for this shape).

## Required harness changes

**None.** iter-12 introduces zero matcher / harness / autotuner code
changes. The matcher already supports buckets with no policy entry as
no-guidance (the d64_long pattern since iter-9). attention_2k_d128 will
join attention_4k_d64 in receiving no Range-Based Heuristics section.

The only operational requirement is to refresh
`helion/autotuner/llm/data/range_heuristics_b200.json` from the iter-12
policy JSON before running the benchmark, identical to iter-11.

If Codex's evaluation script uses headline-group keys to pick which gates
to apply, iter-12's headline group rename (`main_attention_d128_short` →
`sanity_attention_d128_short_unprompted`) is the only signal-routing
change. If the evaluation script auto-derives headline group from
`headline_group` in the policy JSON, no script change is needed. If the
evaluation script has a hardcoded list of headline groups, the change
required is one keys list edit.

## Matcher sanity-check before benchmarking

iter-12 has **one** bucket_match deletion relative to iter-11
(d128_short policy entry removed). The matcher routing should produce:

```text
softmax_4k_1k      -> ['narrow']
softmax_4k_2k      -> ['narrow']
softmax_1k_1k      -> ['narrow']
softmax_1k_2k      -> ['narrow']
softmax_2k_1k      -> ['narrow']
softmax_2k_2k      -> ['narrow']
softmax_4k         -> ['mid_probe']
softmax_1k_4k      -> ['mid_probe']
softmax_2k_4k      -> ['mid_probe']
attention_2k_d128  -> []   (NEW: was ['d128_short'] in iter-11)
attention_4k_d64   -> []
```

Additionally, manually verify the Range-Based Heuristics section of the
repeat_01 log for `attention_2k_d128` is **absent** (or empty), matching
the iter-9..iter-11 behavior for `attention_4k_d64`. If a Range-Based
Heuristics section appears for attention_2k_d128 in repeat_01, the JSON
refresh did not pick up the iter-12 deletion.

## Per-class status (iteration 12)

| Class             | Bucket                                           | Status     | Lifecycle              | Headline                                  | Iter-11 → Iter-12                                                                            |
|-------------------|--------------------------------------------------|------------|------------------------|--------------------------------------------|----------------------------------------------------------------------------------------------|
| row_norm_rms      | any fp16/bf16                                    | validate   | validated_preserved    | main_rms                                   | unchanged                                                                                    |
| row_softmax       | narrow (cols<=2048, rows<=4096)                  | validate   | validated_preserved    | main_softmax_narrow                        | unchanged                                                                                    |
| row_softmax       | mid_probe (cols=`<=4096` exact, rows<=4096)      | validate   | validated_preserved    | main_softmax_mid                           | unchanged                                                                                    |
| attention         | d128_short (head_dim<=128, seq<=2048)            | (no entry) | excluded → blocked     | sanity_attention_d128_short_unprompted     | **REMOVED from active policy list**. Moved to blocked_classes. attention_2k_d128 → sanity.   |
| attention         | d64_long (head_dim<=64, seq<=4096)               | (no entry) | excluded               | sanity_attention_d64_long_unprompted       | unchanged; iter-11 sanity geomean cleanly in band (3rd consecutive)                          |
| attention         | other (small_short, large_or_long)               | hold       | blocked                | —                                          | unchanged                                                                                    |
| row_norm_layer    | any                                              | hold       | blocked                | —                                          | unchanged                                                                                    |
| row_cross_entropy | any                                              | hold       | blocked                | —                                          | unchanged                                                                                    |
| matmul / fp8 / bmm / split_k / grouped | any                         | hold       | blocked                | —                                          | unchanged                                                                                    |
| elementwise       | any                                              | hold       | blocked                | —                                          | unchanged                                                                                    |
| row_softmax_wide  | cols>=4096 (excluding mid_probe)                 | hold       | blocked                | —                                          | unchanged                                                                                    |
| sum / null class  | any                                              | hold       | blocked                | —                                          | unchanged                                                                                    |

## Iter-13 paths conditional on iter-12 outcome

| iter-12 outcome                                                                                                                       | iter-13 path                                                                                                                                                                                                                                                                                                                  |
|---------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| All 3 mandatory gates pass; both d128 sanity AND d64 sanity in band                                                                   | **CLEAN STABLE STATE**. iter-13 holds the validated set steady (RMS / narrow / mid) and explores ONE incremental probe: either widen mid_probe to cols=`<=8192` (data gap closure) OR drop one sanity workload (d64 has 4 in-band passes; freeing a slot for a new probe). Probe selection driven by AOT data, not by guess. |
| All 3 mandatory pass; d128 sanity in band; d64 sanity time > 1.10                                                                     | **D64 DRIFT**. iter-13 investigates harness/LLM API drift. d64 has been in band 3 iters; sudden out-of-band suggests external drift, not heuristic. Roll forward iter-12 JSON unchanged.                                                                                                                                       |
| All 3 mandatory pass; d128 sanity time > 1.10                                                                                          | **D128 SANITY DRIFT**. The unprompted LLM is producing wall-time-bad configs on this shape. iter-13 rules out the prompt as the cause and confirms d128 attention is intrinsically harder for the LLM. No policy change; sanity observation continues.                                                                          |
| All 3 mandatory pass; d128 sanity perf < 0.95 (the LLM finds strong configs unprompted)                                                | **D128 PROMPT WAS NOT THE PERF DRIVER**. iter-3..iter-11 perf wins were partially LLM-baseline; the prompt's marginal contribution was smaller than reported. iter-13 marks d128 as permanently REMOVE — no mechanism work needed. The 5 iters of perf wins were partly an LLM seed/baseline effect, not a heuristic effect.   |
| All 3 mandatory pass; d128 sanity perf in [0.95, 1.05] AND d128 sanity time in [0.90, 1.10]                                            | **EXPECTED**. The iter-3..iter-11 d128 perf wins WERE prompt-attributable. REMOVE means we lose that 19% perf win; the user has accepted that tradeoff for the wall-time/compile objective. iter-13 holds.                                                                                                                       |
| RMS / narrow / mid regression guard fails                                                                                              | **REAL REGRESSION**. Roll back to iter-11 JSON. Investigate before further iteration.                                                                                                                                                                                                                                          |
| Mid softmax perf_geo trends back up to ~1.005 (toward the regression-guard ceiling 1.02)                                              | **MID PERF DRIFT WATCH**. iter-11 mid perf=0.999 was a step DOWN from iter-9/iter-10's 1.002. If iter-12 reverts to ~1.005, that's noise. If it climbs above 1.02, that's a real regression.                                                                                                                                  |
| Both sanities in band for 3+ consecutive iters (iter-13/14 timeframe)                                                                  | **DROP BOTH ATTENTION SANITIES**. iter-15+ frees 2 workload slots for either narrow row expansion (row>4096) once AOT data is available, or for a new class probe.                                                                                                                                                              |

## Ready for Codex review — checklist

- [x] JSON validates (`json.load` syntactically valid).
- [x] No experimental design change (still first-round prompt-only, two
      arms, `llm_max_rounds=1`, `repeats=3`, model `gpt-5-2`, autotuner
      `LLMGuidedSearch`).
- [x] Exact-template prompt and exact-template seeds remain disabled.
- [x] No range edits to RMS / narrow / mid policies.
- [x] No bucket_match edits to RMS / narrow / mid policies.
- [x] No candidate_count edits to RMS / narrow / mid policies.
- [x] **d128_short policy entry deleted**: bucket
      `head_dim_bin_in: ["<=128"], seq_bin_in: ["<=2048"]` moved to
      `blocked_classes` with full provenance of iter-3..iter-11
      attempts.
- [x] **attention_2k_d128 demoted to sanity workload** with new
      `attention_d128_short_unprompted_sanity` reported-alongside gate.
- [x] **`attention_d128_short_replication_gate` removed** (no longer
      applies; bucket has no policy).
- [x] No new workloads (16 total: 14 main + 2 sanity, count unchanged).
- [x] No matcher / harness / autotuner code changes required.
- [x] Matcher routing in iter-12: `attention_2k_d128 -> []` (NEW: was
      `['d128_short']`); `attention_4k_d64 -> []` (unchanged); RMS /
      narrow / mid routing unchanged.
- [x] RMS class entry: lifecycle=`validated_preserved`;
      rms_regression_guard mandatory; rms_material_reproduction
      reported_alongside.
- [x] Narrow softmax: lifecycle=`validated_preserved`;
      softmax_narrow_regression_guard mandatory;
      softmax_narrow_breadth_gate reported_alongside.
- [x] Mid softmax: lifecycle=`validated_preserved`;
      softmax_mid_regression_guard mandatory;
      softmax_mid_breadth_gate reported_alongside.
- [x] Attention d128_short: NO policy entry; sanity workload defined;
      sanity gate `attention_d128_short_unprompted_sanity` defined with
      same band as d64 sanity; `blocked_classes` entry added with
      iter-3..iter-11 history.
- [x] Attention d64_long: no policy entry; `blocked_classes` entry
      retained from iter-11; iter-11 sanity result rolled into
      provenance (3rd consecutive in-band).
- [x] `attention_4k_d64` retained in harness as
      `sanity_attention_d64_long_unprompted` workload.
- [x] **`attention_2k_d128` retained in harness as new
      `sanity_attention_d128_short_unprompted` workload.**
- [x] `blocked_classes` has 14 entries (was 13 in iter-11; new entry is
      `attention_d128_short`).
- [x] `prompt_mode: "range_only"` and `arm_definitions` make
      exact-template prompt and seed suppression unambiguous.
- [x] Section title is `Range-Based Heuristics`.
- [x] Materiality rule applied: a few percent is noise; ≥5% perf or
      ≥20% wall-time for "real". iter-11 d128 perf=0.806 is material
      (19.4% win); iter-11 d128 time=1.080 is NOT material per the
      ≥20% rule (8.0% regression) but is past the 1.02 gate ceiling;
      iter-11 d128 compile=1.523 IS material per the ≥20% rule (52%
      compile regression — worse than iter-10's 25%).
- [x] iter-13 stability classifier and path table are concrete and
      actionable for all 7 outcome cases.
- [x] Canonical GPU: `CUDA_VISIBLE_DEVICES=2`. iter-11 was first GPU 2
      run; iter-12 continues.

## Remaining disagreements with Codex

None expected. iter-12 implements the user's iter-12 brief: REMOVE d128
because no clearly different minimal mechanism is justified by the iter-11
telemetry. Specifically:

- **iter-11 tightening trial classification**: PARTIAL—TIME (perf
  preserved at 0.806, time still failing at 1.080, compile materially
  worse at 1.523). Per the iter-11 plan's PATH B, the choice was
  "further tighten" or "REMOVE"; the user's brief explicitly chooses
  REMOVE.
- **No clearly different mechanism available**: candidate_count is
  already at 2 (further reduction is meaningless guidance for
  single-shape); num_warps is already [4]; l2_groupings narrowing is the
  same prompt-tightening mechanism class that has now been tested twice;
  hard-execution-cap or anti_ranges would require code or telemetry not
  currently in scope.
- **REMOVE preserves the option to revisit**: iter-13+ may add d128 back
  if (a) per-config compile-time attribution telemetry identifies a
  specific compile-cost mechanism, or (b) the iter-12 d128 sanity result
  shows the iter-3..iter-11 perf signal was not actually
  prompt-attributable (i.e., REMOVE costs us nothing).

If Codex prefers a different option:

- **Codex prefers further tightening (l2_groupings → [16])**: defensible,
  but per the user's brief, this is the same mechanism class as iter-11's
  tightening, not a clearly different mechanism. iter-11 already showed
  prompt-only narrowing has limits because the LLM does not strictly
  comply (cfg_cost=1.055, prompt logs likely still show non-4-warp
  configs). I prefer REMOVE because the iter-11 outcome is now the second
  failed prompt-only tightening on this bucket.
- **Codex prefers keeping d128 as diagnostic with no prompt change**:
  iter-11 already did this and the result was time_geo=1.080 with
  compile_total=1.523. The user's brief says "do not keep d128 unless
  proposing a clearly different mechanism" — diagnostic-with-same-prompt
  is not a different mechanism; it just relabels.
- **Codex prefers keeping the sanity workload count at 1 (drop d64)**:
  defensible. iter-11 d64 sanity was the cleanest of 3 iters
  (max=1.050), so the d64_long block is empirically permanent. But
  iter-12 is making a structural change (REMOVE d128 → sanity); keeping
  d64 sanity preserves prior baseline comparability and gives iter-12 a
  cross-check (both attention shapes should be neutral). iter-13+ can
  drop d64 once d128 sanity has its own in-band history.
- **Codex prefers a code-side hard cap on candidate_count**: a
  reasonable mechanism but not "minimal" — it changes the LLM guidance
  contract globally. The user's brief explicitly limits iter-12 to
  prompt/policy edits; a code change would be iter-13+ if telemetry
  supports it.
