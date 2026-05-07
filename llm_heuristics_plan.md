# Helion LLM Heuristics Gate Plan

This is the living plan for improving Helion LLM heuristic usefulness.

## Primary Objective and Score

Primary objective: improve circuit/kernel performance of the best config
produced after LLM round 0 when heuristics are enabled, compared against the
non-heuristics baseline arm from the same experiment.

Exact metric: `round0_best_geo` is the geometric mean over workload/repeat
pairs of:

```text
min(perf_ms where generation == 0 and status == ok in heuristics arm autotune CSV)
/
min(perf_ms where generation == 0 and status == ok in baseline arm autotune CSV)
```

Lower is better. A material win requires at least 10% better circuit/kernel
performance, i.e. `round0_best_geo <= 0.90`, unless a specific experiment sets
a stricter gate.
Do not continuously raise the base acceptance threshold as results improve.
Keep `round0_best_geo <= 0.90` as the material-win floor unless a gate
explicitly sets a stricter target. Track stretch levels separately, such as
strong `<=0.85` and excellent `<=0.80`, especially for a kernel class already
clearing the base gate.

This is not comparing against PyTorch, AOT pretune CSVs, or final full-autotune
winners. The AOT CSVs are input data for deriving heuristics; the score compares
heuristics vs non-heuristics arms produced by the live experiment.

For hybrid/LFBO, score `_stage1_llm.csv` when it exists because that is the LLM
seed batch handed to LFBO. Compile time and wall time are secondary diagnostics
unless an experiment is explicitly about compile-time/resource reduction; still
report them, but do not optimize against them over `round0_best_geo` in this
loop.
Serious compile failures, missing rows, illegal-memory failures, or accuracy
failures also block promotion until rerun or diagnosed.

All future benchmark work should use GPU 3 through `CUDA_VISIBLE_DEVICES=3` and
`--gpu 3`. GPU 3 supersedes older GPU 2 references in copied `/tmp`
snapshots, shared context, and previous reports.

## Archive State

This fork/data/archive checkout is
`/home/jongsokchoi/helion_2_aot_pretune_data_all` on branch
`choijon5/aot-pretune-data`. Before this docs/artifact update, the latest local
commit was `88e71c4b`. The branch
archives 149 AOT data files, about 69 MB, under `aot_pretune_data/` for B200
kernels: attention, cross_entropy, fp8_gemm, grouped_gemm, layer_norm, matmul,
rms_norm, softmax, and vector_add.

Current live experiments may still run from
`/home/jongsokchoi/helion_2_llm_priors` until scripts are consolidated. This
archive checkout is for data, manager docs, and derived heuristic artifacts
only.

## Current State

Data says global observed heuristics are not enough by themselves: guided
iteration 11 `heuristics` reached `round0_best_geo=0.953`, about a 5% geomean
win over baseline and below the current material-win gate. The useful signal is
bucketed:

| workload | best observed signal | round0 ratio | interpretation |
|---|---:|---:|---|
| attention_1k_d64 | heuristics/seeds | 0.802 | strong win |
| attention_2k_d128 | heuristics/range | 0.841 | strong win |
| attention_4k_d64 | heuristics/seeds | 0.722 | strong win |
| attention_4k_d128 | avoid current heuristics | 1.194 | strong regression |
| bmm_8x256x384x512 | heuristics | 0.895 | useful guided signal |
| rms_norm_2048x4096 | heuristics/range | 0.908/0.904 | useful |
| softmax_4k_2k | heuristics/range | 0.902/0.904 | useful |
| softmax_2k_4k | heuristics | 0.942 | modest useful |

Most matmul, layer norm, cross entropy, and several softmax/RMS shapes are
neutral. Iteration 12 range prompt preserved RMS and softmax value but removed
active attention guidance. Hybrid/LFBO handoff showed `heuristics`
`round0_best_geo=0.949`, but BMM regressed in stage-1/handoff and needs a
targeted explanation before product use.

The seed-only attention d128 check is closed as a failed direction. Seeds on
`attention_2k_d128` plus `attention_4k_d128` did not improve the aggregate
handoff target: aggregate perf/round0 was not useful, around `1.03`, and
`attention_4k_d128` was about `1.058`. Do not repeat seed-only as H2.

Latest gate decisions:

- H1 PASS on 2026-05-07. The verification subagent recomputed the corrected
  objective from metadata and `_stage1_llm.csv` selection with zero trace or
  ratio errors.
- H2 PASS on 2026-05-07 for primary objective and routing safety. The filtered
  observed-heuristics router reached corrected `round0_best_geo=0.802236`
  overall on attention, with `attention_4k_d128` unmatched by any observed
  rule in all repeats.
- H3d PASS on 2026-05-07. Paired-no-match scored corrected overall
  `round0_best_geo=0.822`, matched geo `0.791`, no-match
  `attention_4k_d128` geo `1.001`, no-match median `0.9996`, and max repeat
  `1.0048`. Baseline round 0 was recorded, matched arms did not replay, and
  no-match `attention_4k_d128` replayed the paired baseline with candidate
  overlap exact enough and `same_best=true`.
- H4 HOLD on 2026-05-07. Broad RMS/softmax active non-attention policy scored
  corrected overall `round0_best_geo=0.993`, matched active non-attention
  `0.989`, and no-match guardrails `0.998`; `row_norm_rms=0.989` and
  `row_softmax=0.988`. Best buckets were `rms_norm_1024x16384=0.958` and
  `softmax_2k_4k=0.955`, both below the material-win threshold.
- H5 HOLD on 2026-05-07. Attention hybrid/LFBO paired-no-match preserved a
  material aggregate win but weakened the round-0 advantage and introduced a
  final verified regression on `attention_512_d64`. Round-0 overall was
  `0.830`, matched attention was `0.799`, and no-match `attention_4k_d128` was
  `1.001`. Final verified overall was `0.896`, matched attention was `0.882`,
  and no-match `attention_4k_d128` was `0.971`, but `attention_512_d64`
  regressed to `1.025`. H5b must diagnose LFBO handoff before packaging.
- H5b HOLD for packaging on 2026-05-07. Adding `l2_groupings=[1]` to the d64
  `seq<=2048` attention family improved the corrected seed objective and fixed
  `attention_512_d64`, but weakened final hybrid/LFBO compared with H5. H5b
  scored round-0 overall `0.815`, matched `0.782`, no-match `1.001`; final
  verified overall `0.928`, matched `0.915`, no-match `0.991`.
  `attention_512_d64` improved from H5 `1.025` to `0.948`, but H5b final
  aggregate is worse than H5 final `0.896`. H5c should split `seq<=1024` from
  `seq<=2048` so `l2=1` does not apply to `attention_2k_d64`.

Current generality assessment:

- The only active candidate worth preserving is the attention policy from
  `h3b_attention_strict_family_no_r4_observed_heuristics_b200.json`.
- The policy is bucket-based, not exact workload-name based: it routes by
  attention kernel class, batch-head count, sequence length, head dimension,
  and dtype.
- Current validated scope is B200, fp16/bf16, standard attention, sequence
  lengths through 4k for d64 and through 2k for d128.
- Clean heldout-ish evidence exists for `attention_512_d64` and
  `attention_2k_d64`; `attention_4k_d128` is intentionally a no-match
  guardrail.
- It is not yet validated for long d128, 8k/16k sequence lengths, causal
  variants, GQA/MQA, paged attention, cross-attention, or other GPUs.
- Some selected-oracle sections in the JSON report `shape_coverage=2`; before
  packaging, add more AOT or live shapes per bucket so the rule is not anchored
  to two shape points.
- Do not keep RMS/softmax active now. Their H4 signals were below the material
  threshold and should remain archived diagnostics unless a new derivation
  produces a clearly stronger policy.

## Data Sources

- `/tmp/helion_heuristics_loop/input/shared_context.md`
- `/tmp/helion_round0_objective_20260505_230436/guided_round0_iter11_policy/round0_summary.md`
- `/tmp/helion_round0_objective_20260505_230436/guided_round0_iter12_policy/round0_summary.md`
- `/tmp/helion_round0_objective_20260505_230436/hybrid_lfbo_round0_handoff/round0_summary.md`
- `helion/autotuner/llm/data/observed_heuristics_b200.json`
- Archive copies in `llm_heuristics_artifacts/`, including the observed
  heuristics JSON, runtime snapshot, iteration 12/13 range policies, corrected
  round-0 summaries, and shared context.
- H2 env-path prototype:
  `llm_heuristics_artifacts/h2_attention_router_observed_heuristics_b200.json`
- H1 verification:
  `llm_heuristics_artifacts/h1_round0_verification_20260507.md`
- H2 benchmark archive:
  `llm_heuristics_artifacts/h2_attention_router_20260507/aggregate_summary.md`
  and
  `llm_heuristics_artifacts/h2_attention_router_20260507/aggregate_results.json`
- H3d paired-no-match archive:
  `llm_heuristics_artifacts/h3d_attention_paired_no_match_20260507/aggregate_summary.md`
  and
  `llm_heuristics_artifacts/h3d_attention_paired_no_match_20260507/aggregate_results.json`
- H4 paired-no-match archive:
  `llm_heuristics_artifacts/h4_non_attention_paired_no_match_20260507/aggregate_summary.md`
  and
  `llm_heuristics_artifacts/h4_non_attention_paired_no_match_20260507/aggregate_results.json`
- H5 hybrid/LFBO paired-no-match archive:
  `llm_heuristics_artifacts/h5_attention_hybrid_lfbo_paired_no_match_20260507/aggregate_summary.md`
  and
  `llm_heuristics_artifacts/h5_attention_hybrid_lfbo_paired_no_match_20260507/aggregate_results.json`
- H5b hybrid/LFBO archive:
  `llm_heuristics_artifacts/h5b_attention_hybrid_lfbo_l2_1_8_16_paired_no_match_20260507/aggregate_summary.md`
  and
  `llm_heuristics_artifacts/h5b_attention_hybrid_lfbo_l2_1_8_16_paired_no_match_20260507/aggregate_results.json`
- H2 policy critique:
  `llm_heuristics_artifacts/claude_h2_policy_critique.md`
- Derived snapshot, if useful for policy review:
  `/tmp/helion_heuristics_loop/codex/range_policy_data_snapshot/runtime_observed_heuristics_b200.json`
- Existing policy artifacts under `/tmp/helion_heuristics_loop/claude` and
  reviews under `/tmp/helion_heuristics_loop/codex`

## Repository State

The live experiment workspace can contain unrelated dirty implementation files.
This plan does not claim ownership of those changes. Future managers and
subagents must avoid reverting, staging, or summarizing unrelated changes as
part of this documentation or heuristic-planning work. This archive checkout
should stay limited to data/docs/artifact commits.

## Acceptance Rules

- Primary metric: `round0_best_geo` as defined above.
- A material win requires at least 10% better circuit/kernel performance, i.e.
  `round0_best_geo <= 0.90`, unless a specific experiment sets a stricter gate.
- Do not continuously raise the base threshold as results improve. Keep
  stretch levels separate from the base gate: strong `<=0.85`, excellent
  `<=0.80`.
- Do not score against PyTorch, AOT pretune CSVs, or final full-autotune
  winners; score live heuristics arms against live non-heuristics baseline arms.
- For hybrid/LFBO, score `_stage1_llm.csv` when it exists because that is the
  LLM seed batch handed to LFBO.
- Every report must include overall geomean, per-kernel-class geomeans, and
  per-workload geomeans.
- No promoted policy may have a promoted/matched workload geomean `>1.03`
  unless a focused rerun proves it is measurement noise.
- No guardrail/no-match workload geomean should exceed `1.05`.
- Any repeat-level ratio `>1.10` on a guardrail/no-match workload triggers a
  focused rerun before broadening or promotion.
- If the focused rerun still has repeat-level ratio `>1.10` or workload geo
  `>1.05`, treat the policy as HOLD unless the heuristic did not fire and the
  experiment can prove the swing is baseline/LLM stochasticity, not policy
  leakage.
- Attention acceptance must explicitly protect `attention_4k_d128`.
- Compile/wall time must stay within diagnostic guardrails, normally `<=1.20x`;
  still report them, but do not optimize against them over `round0_best_geo`
  unless the experiment is explicitly about compile-time/resource reduction.
- Repeats must use one GPU for the whole run. Going forward that GPU is 3.

## Gates

### H1: Reproduce and Lock Corrected Objective and Harness

Hypothesis: the team can make decisions from one corrected objective and one
repeatable harness contract.

Work:

- Reconfirm that `round0_best_geo` is computed from CSV metadata, not filename
  parsing.
- Document how hybrid/LFBO selects `_stage1_llm.csv`.
- Re-score existing corrected outputs only using metadata paths; no new
  benchmark and no GPU required.
- Make every future report include `round0_best_geo`, `verified_geo`,
  `wall_time_geo`, and compile diagnostics.

Acceptance:

- Existing corrected summaries reproduce the known aggregate values:
  iter11 guided `heuristics=0.953`, iter12 guided `range_prompt=0.974`,
  hybrid `heuristics=0.949`.
- A subagent can identify the CSV source for every row through JSON metadata.
- The manager has an accepted report template and gate decision format.

Status: PASS on 2026-05-07.

Artifact: `llm_heuristics_artifacts/h1_round0_verification_20260507.md`.

Exact computed aggregates:

| run | arm | n | computed round0_best_geo | rounded |
|---|---|---:|---:|---:|
| guided iter11 | heuristics | 57 | 0.952684747 | 0.953 |
| guided iter11 | range_prompt | 57 | 0.959271664 | 0.959 |
| guided iter11 | seeds | 57 | 0.968517931 | 0.969 |
| guided iter12 | range_prompt | 57 | 0.974486583 | 0.974 |
| hybrid LFBO | heuristics | 24 | 0.948877958 | 0.949 |
| hybrid LFBO | range_prompt | 24 | 0.999182870 | 0.999 |
| hybrid LFBO | seeds | 24 | 0.987971432 | 0.988 |

The verification inspected 438 metadata rows, resolved 438 CSV paths, used 96
hybrid `_stage1_llm.csv` rows, and reported zero trace or ratio errors.

### H2: Design a Bucket Router Starting with Attention

Hypothesis: observed heuristics become useful when routed by kernel and shape
bucket, especially for attention.

H2 is the attention bucket router. The first implementation should avoid source
edits: create a filtered observed-heuristics JSON and run the existing
`heuristics` arm with `HELION_LLM_OBSERVED_HEURISTICS_PATH` pointing at that
file. The filtered JSON should keep the winning attention buckets and exclude
or route away the known bad `attention_4k_d128` bucket. Source edits are only
needed if the env-path approach cannot express the router or if validation
shows the existing matching logic still gives observed guidance to the bad
bucket.

Initial attention policy direction:

- Enable heuristic help for `attention_1k_d64`, `attention_2k_d128`, and
  `attention_4k_d64`.
- Route `attention_4k_d128` to baseline/no-guidance until a safe candidate
  family exists.
- Add holdout attention shapes only after the four known buckets reproduce.
- Use `llm_heuristics_artifacts/h2_attention_router_observed_heuristics_b200.json`
  as the archived prototype filtered JSON unless a proposal agent produces a
  stricter reviewed replacement.

H2 GPU 3 benchmark command template:

```bash
cd /home/jongsokchoi/helion_2_llm_priors
HELION_LLM_OBSERVED_HEURISTICS_PATH=/home/jongsokchoi/helion_2_aot_pretune_data_all/llm_heuristics_artifacts/h2_attention_router_observed_heuristics_b200.json \
CUDA_VISIBLE_DEVICES=3 /home/jongsokchoi/.conda/envs/helion_2/bin/python \
  scripts/llm_heuristics_autoresearch.py \
  --gpu 3 \
  --suite core_rows \
  --workloads attention_1k_d64,attention_2k_d128,attention_4k_d64,attention_4k_d128 \
  --arms baseline,heuristics \
  --autotuner LLMGuidedSearch \
  --model gpt-5-2 \
  --llm-max-rounds 1 \
  --repeats 3 \
  --output-root /tmp/helion_llm_autoresearch_attention_router_h2
```

Acceptance:

- On `attention_1k_d64`, `attention_2k_d128`, `attention_4k_d64`,
  `attention_4k_d128`, the routed arm has attention `round0_best_geo <= 0.90`.
- `attention_4k_d128 <= 1.02` and no repeat has a large unexplained
  regression.
- Compile/wall diagnostics are `<=1.20x` unless explained by a better primary
  objective and accepted by the user.

Status: PASS on 2026-05-07 for primary objective and routing safety, with
caveats.

Artifacts:

- `llm_heuristics_artifacts/h2_attention_router_20260507/aggregate_summary.md`
- `llm_heuristics_artifacts/h2_attention_router_20260507/aggregate_results.json`
- `llm_heuristics_artifacts/claude_h2_policy_critique.md`

Corrected H2 `round0_best_geo` results:

| workload | corrected round0_geo | observed-rule match |
|---|---:|---|
| all attention | 0.802236 | mixed |
| attention_1k_d64 | 0.805943 | true |
| attention_2k_d128 | 0.822007 | true |
| attention_4k_d64 | 0.728283 | true |
| attention_4k_d128 | 0.858477 | false in all repeats |

Script aggregate diagnostics: `perf_geo=0.804`, `time_geo=0.802`,
`cfg_geo=0.993`.

Caveats:

- Repeat 1 had compile/wall spikes despite aggregate time improving:
  `attention_4k_d128` wall `1.632x`, compile_total `2.619x`;
  `attention_2k_d128` wall `1.549x`, compile_total `3.622x`.
- `attention_4k_d128` improved while `observed_rule_match=false`. This is safe
  for routing, because the bad bucket did not receive observed-rule guidance,
  but H3 must attribute whether the improvement came from baseline noise,
  prompt/seeding side effects, or unrelated first-round LLM behavior.

Matcher review recommended PASS: `attention_4k_d128` did not match under the
current semantics. Claude Opus 4.7 also recommended H2 PASS with the caveat
above and advised staying on attention for H3 coverage/attribution before
broadening.

### H3: Attribute Attention Router and Generalize Held-Out Shapes

Hypothesis: the H2 attention win is real, but must be attributed by
observed-rule matched versus unmatched behavior before converting exact observed
winners into candidate families or broadening beyond attention.

Attention is first because it has the strongest signal and tests the routing
mechanism. This is not a plan to focus only on attention forever; H4 still
broadens to RMS, softmax, and BMM after H3 resolves coverage and attribution.

Work:

- Split H2 corrected `round0_best_geo` contribution by
  `matched_observed_rule=true` versus `false`.
- Re-run or analyze `attention_4k_d128` unmatched behavior to explain why it
  improved with no observed-rule match. Treat the explanation as required
  before promoting broader router logic.
- For attention buckets that pass H2, extract common config features:
  block sizes, warps, stages, indexing, pid type, range parameters, and
  num-sm multiplier.
- Convert exact configs into small families or constrained candidate ranges.
- Include nearby holdouts such as `attention_512_d64`, `attention_2k_d64`, and
  additional d128 lengths when available.

Acceptance:

- Candidate families retain at least 80% of the H2 bucket win on known shapes.
- Matched versus unmatched attribution accounts for the H2 aggregate and does
  not hide a bad routed bucket.
- `attention_4k_d128` remains routed away from observed-rule guidance or has a
  data-backed safe mechanism before any promotion.
- Holdouts do not regress by more than 2% round-0 unless explicitly demoted.
- The rule is explainable by workload features, not just shape names.

Status: PASS via H3d paired-no-match on 2026-05-07. H3d scored corrected
overall `round0_best_geo=0.822`, matched geo `0.791`, no-match
`attention_4k_d128` geo `1.001`, no-match median `0.9996`, and max repeat
`1.0048`. Replay validation showed baseline round 0 recorded, matched arms did
not replay, no-match `attention_4k_d128` replayed the paired baseline, and
candidate overlap was exact enough with `same_best=true`. This unlocks H4
non-attention broadening while preserving the regression guardrail.

### H4: Expand Non-Attention Policies for RMS and Softmax

Hypothesis: the non-attention wins are smaller but easier to make robust.

Candidate artifact:

- `llm_heuristics_artifacts/h4_non_attention_observed_heuristics_b200.json`
  exists for paired-no-match testing. It keeps only `row_norm_rms` and
  narrow/mid `row_softmax` active; BMM, cross-entropy, matmul, layer norm, and
  attention should be benchmarked as guardrails/no-match classes.

Target buckets:

- RMS: start with `rms_norm_2048x4096`, then include `1024x16384` and
  `8192x2048`.
- Softmax: start with `softmax_4k_2k` and `softmax_2k_4k`; preserve neutral
  behavior on `softmax_4k` and `softmax_1k_1k`.
- Guardrails/no-match: include BMM, cross-entropy, matmul/split-k matmul, layer
  norm, and attention to prove the artifact does not leak active guidance into
  blocked classes.

Acceptance:

- Use paired-no-match mode with RMS/softmax active and the classes above as
  guardrails.
- Target non-attention `round0_best_geo <= 0.90`.
- No neutral guardrail workload exceeds `1.03`.
- Do not promote BMM, cross-entropy, matmul, layer norm, attention, elementwise,
  FP8, or grouped matmul from this H4 artifact.

Status: HOLD on 2026-05-07. The broad RMS/softmax policy is not promotable.

Artifacts:

- `llm_heuristics_artifacts/h4_non_attention_paired_no_match_20260507/aggregate_summary.md`
- `llm_heuristics_artifacts/h4_non_attention_paired_no_match_20260507/aggregate_results.json`

Corrected H4 results:

| scope | round0_best_geo | decision |
|---|---:|---|
| overall | 0.993 | below material win |
| matched active non-attention | 0.989 | below material win |
| no-match guardrails | 0.998 | neutral |
| row_norm_rms | 0.989 | active, not material |
| row_softmax | 0.988 | active, not material |
| rms_norm_1024x16384 | 0.958 | best RMS bucket, not promotable |
| softmax_2k_4k | 0.955 | best softmax bucket, not promotable |

Replay validation passed: baseline records 80/80, no-match replays 40/40,
matched arms using `off_matched_heuristic` 40/40, and no fatal errors. Do not
keep the broad RMS/softmax policy. Next non-attention work should be H4b narrow
diagnostics around `softmax_2k_4k` and `rms_norm_1024x16384`, or better policy
derivation from AOT data. Attention remains the only clean material win.

### H5: Validate Hybrid LFBO Handoff

Hypothesis: a good first LLM round should improve the downstream
`LLMSeededLFBOTreeSearch` handoff, not only the guided-only score.

Workloads:

- `attention_512_d64`
- `attention_1k_d64`
- `attention_2k_d64`
- `attention_2k_d128`
- `attention_4k_d64`
- `attention_4k_d128`

Acceptance:

- Hybrid `round0_best_geo <= 0.90` on promoted buckets.
- Hybrid final verified geomean should preserve a material aggregate win and
  avoid promoted-workload regressions over `1.03`.
- No promoted workload has round-0 or verified regression over `1.03`.
- No-match guardrails, especially `attention_4k_d128`, remain neutral.

Status: HOLD on 2026-05-07. H5 used paired-no-match mode with
`LLMSeededLFBOTreeSearch` and the H3b strict attention family. Round-0 scoring
from `_stage1_llm.csv` showed the heuristic still materially improves the LLM
seed batch:

| scope | round0_best_geo | decision |
|---|---:|---|
| overall attention | 0.830 | material win |
| matched attention | 0.799 | strong win |
| no-match `attention_4k_d128` | 1.001 | neutral guardrail |

Final verified performance after LFBO remained an aggregate win, but the
handoff weakened the signal and regressed one short-sequence d64 workload:

| workload | final verified geo | decision |
|---|---:|---|
| overall attention | 0.896 | material aggregate win |
| matched attention | 0.882 | material aggregate win |
| attention_512_d64 | 1.025 | HOLD; investigate |
| attention_1k_d64 | 0.879 | win |
| attention_2k_d64 | 0.791 | strong win |
| attention_2k_d128 | 0.907 | near material |
| attention_4k_d64 | 0.826 | strong win |
| attention_4k_d128 | 0.971 | no-match guardrail ok |

Artifacts:

- `llm_heuristics_artifacts/h5_attention_hybrid_lfbo_paired_no_match_20260507/aggregate_summary.md`
- `llm_heuristics_artifacts/h5_attention_hybrid_lfbo_paired_no_match_20260507/aggregate_results.json`

H5b status: HOLD for packaging. It fixes `attention_512_d64` and improves
round-0, but final hybrid/LFBO is weaker than H5.

H5b corrected results:

| workload | stage1 | stage2 | final verified |
|---|---:|---:|---:|
| attention_512_d64 | 0.832 | 0.959 | 0.948 |
| attention_1k_d64 | 0.786 | 0.907 | 0.918 |
| attention_2k_d64 | 0.740 | 0.875 | 0.887 |
| attention_2k_d128 | 0.827 | 0.914 | 0.927 |
| attention_4k_d64 | 0.733 | 0.880 | 0.898 |
| attention_4k_d128 | 1.001 | 0.996 | 0.991 |
| overall | 0.815 | 0.921 | 0.928 |
| matched | 0.782 | 0.907 | 0.915 |
| no-match | 1.001 | 0.996 | 0.991 |

H5c next: create a split d64 candidate with an exact `seq_bin="<=1024"` rule
using `l2_groupings=[1,8,16]`, restore the `seq_bin="<=2048"` d64 rule to H3b
`l2_groupings=[8,16]`, leave d64 4k and d128 short rules unchanged, and keep
`attention_4k_d128` as no-match. This tests whether the `l2=1` benefit is
specific to the short d64 bucket without diluting `attention_2k_d64`.

### H6: Package Prompt and Seed Policy if Material Wins Hold

Hypothesis: after routed buckets and candidate families pass, the project can
package a small product policy without broad risk.

Work:

- Freeze the promoted router/family policy.
- Define default prompt, seed, and demotion behavior.
- Produce a concise implementation plan and rollback rule.
- Run one final full matrix on GPU 3 through subagents.

Acceptance:

- Full matrix `round0_best_geo <= 0.90`.
- No known bad bucket is promoted, especially `attention_4k_d128`.
- Guardrail workloads remain within `1.03`.
- Artifacts include policy JSON/markdown, corrected round-0 summary, and a
  short source-change plan for user approval.

## Benchmark Matrix

| gate | workloads | arms | autotuner | repeats | purpose |
|---|---|---|---|---:|---|
| H1 | existing `/tmp` outputs | existing | existing | existing | objective lock |
| H2 | attention_1k_d64, attention_2k_d128, attention_4k_d64, attention_4k_d128 | baseline plus `heuristics` with env-path filtered observed JSON | LLMGuidedSearch | 3 | route attention buckets |
| H3 | H2 plus attention_512_d64, attention_2k_d64, nearby d128 holdouts when available | baseline plus router/family candidates | LLMGuidedSearch | 3 | attribute matched/unmatched attention wins and test holdouts |
| H4 | RMS/softmax active plus BMM, cross-entropy, matmul, layer norm, and attention guardrails | baseline plus candidate | LLMGuidedSearch | 5 | broad non-attention HOLD |
| H5 | 6 attention workloads, including no-match `attention_4k_d128` | baseline plus H3b strict attention family | LLMSeededLFBOTreeSearch | 5 | validate handoff; HOLD |
| H5b | H5 attention matrix | baseline plus H5b d64 `seq<=2048` l2 1/8/16 candidate | LLMSeededLFBOTreeSearch | 5 | fixed 512 and round0; HOLD for final LFBO |
| H5c | H5 attention matrix | baseline plus split d64 `seq<=1024` l2 1/8/16 and `seq<=2048` l2 8/16 candidate | LLMSeededLFBOTreeSearch | 5 | keep 512 fix without diluting 2k |
| H6 | corrected broad/full matrix | baseline plus final policy | selected final autotuner | 3 | package decision |

## Risks

- Overfitting exact shapes instead of reusable feature buckets.
- Filename parsing can corrupt arm/workload attribution.
- Prompt-only ranges do not hard constrain generated configs.
- Seed-batch wins can disappear after the LLM first-round candidates are added.
- GPU changes can dominate small deltas; future runs must use GPU 3.
- Copied artifacts may mention GPU 2 historically; those references are not
  instructions for new runs.
- Compile/wall-time noise can distract from the primary objective, but large
  compile spikes still indicate system or mechanism problems.
- Unmatched improvements can make a router look better than it is; H3 must
  split matched versus unmatched contribution before broadening.
- Existing local source changes mean implementation work must avoid unrelated
  edits and needs explicit approval.

## Next Unit

Do not promote or keep the broad H4 RMS/softmax policy. The next active unit is
H5c: split the d64 attention family by `seq_bin` and rerun the paired hybrid
handoff matrix.

H5c should:

- Preserve the H3d/H5b round-0 attention win in the seed batch passed to LFBO.
- Preserve the H5b `attention_512_d64` final verified fix.
- Avoid diluting `attention_2k_d64` by restoring the d64 `seq<=2048` rule to
  the H3b l2 8/16 family.
- Keep `attention_4k_d128` as a no-match guardrail unless new data supports a
  safe d128 long-sequence rule.
- Prefer more general attention families over exact shape configs, but gather
  additional AOT/live shapes if a bucket has only two supporting shapes.
- Keep RMS/softmax and other non-attention kernels archived only until a new
  policy derivation gives a clear `round0_best_geo <= 0.90` candidate.
