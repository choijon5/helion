# Helion LLM Heuristics Gate Plan

This is the living plan for improving Helion LLM heuristic usefulness. The
current objective is first LLM round handoff quality:
`round0_best_geo` from autotune CSV rows where `generation == 0` and
`status == ok`. Lower is better. Compile and wall time are secondary diagnostics
unless the explicit experiment is compile-time reduction.

All future benchmark work should use GPU 3 through `CUDA_VISIBLE_DEVICES=3` and
`--gpu 3`. GPU 3 supersedes older GPU 2 references in copied `/tmp`
snapshots, shared context, and previous reports.

## Archive State

This fork/data archive branch is `choijon5/aot-pretune-data`. Before this
docs/artifact update, the latest local commit was `88e71c4b`. The branch
archives 149 AOT data files, about 69 MB, under `aot_pretune_data/` for B200
kernels: attention, cross_entropy, fp8_gemm, grouped_gemm, layer_norm, matmul,
rms_norm, softmax, and vector_add.

The live experiment workspace remains `/home/jongsokchoi/helion_2_llm_priors`.
This archive checkout is for data, manager docs, and derived heuristic
artifacts only.

## Current State

Data says global observed heuristics are not enough by themselves: guided
iteration 11 `heuristics` reached `round0_best_geo=0.953`, about a 5% geomean
win over baseline. The useful signal is bucketed:

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

- Primary metric: `round0_best_geo`.
- A material performance win is about `>=5%` geomean, or better when scoped to
  a bucket.
- No accepted policy may hide a known large per-workload regression.
- Attention acceptance must explicitly protect `attention_4k_d128`.
- Compile/wall time must stay within diagnostic guardrails, normally
  `<=1.20x`, unless the experiment is explicitly about compile reduction.
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

### H4: Expand Non-Attention Policies for RMS, Softmax, and BMM

Hypothesis: the non-attention wins are smaller but easier to make robust.

Target buckets:

- RMS: start with `rms_norm_2048x4096`, then include `1024x16384` and
  `8192x2048`.
- Softmax: start with `softmax_4k_2k` and `softmax_2k_4k`; preserve neutral
  behavior on `softmax_4k` and `softmax_1k_1k`.
- BMM: investigate `bmm_8x256x384x512`, but require hybrid handoff safety
  because the hybrid stage result regressed.

Acceptance:

- Target non-attention `round0_best_geo <= 0.95`.
- No neutral guardrail workload exceeds `1.03`.
- BMM is not promoted unless both guided round-0 and hybrid handoff are safe.

### H5: Validate Hybrid LFBO Handoff

Hypothesis: a good first LLM round should improve the downstream
`LLMSeededLFBOTreeSearch` handoff, not only the guided-only score.

Workloads:

- `attention_1k_d64`
- `attention_2k_d128`
- `bmm_8x256x384x512`
- `cross_entropy_32k`
- `layer_norm_4k`
- `matmul_1k`
- `rms_norm_2048x4096`
- `softmax_4k_2k`

Acceptance:

- Hybrid `round0_best_geo <= 0.95` on promoted buckets.
- Hybrid `verified_geo <= 0.97` or no verified regression on the full handoff
  matrix.
- No promoted workload has round-0 or verified regression over `1.03`.
- BMM stage-1 regression is explained or BMM is demoted.

### H6: Package Prompt and Seed Policy if Material Wins Hold

Hypothesis: after routed buckets and candidate families pass, the project can
package a small product policy without broad risk.

Work:

- Freeze the promoted router/family policy.
- Define default prompt, seed, and demotion behavior.
- Produce a concise implementation plan and rollback rule.
- Run one final full matrix on GPU 3 through subagents.

Acceptance:

- Full matrix `round0_best_geo <= 0.95`.
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
| H4 | RMS, softmax, BMM targets plus neutral controls | baseline plus candidate | LLMGuidedSearch | 3 | broaden non-attention value |
| H5 | 8-workload hybrid handoff matrix | baseline plus promoted candidate | LLMSeededLFBOTreeSearch | 3 | validate handoff |
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

Run H3 attention attribution and coverage before broadening:

1. Analyze the archived H2 output root and split corrected `round0_best_geo` by
   `matched_observed_rule=true` and `false`.
2. Attribute `attention_4k_d128` improvement while unmatched; rerun only if
   analysis cannot explain it from existing metadata and CSVs.
3. Test held-out attention shapes such as `attention_512_d64` and
   `attention_2k_d64` if the harness exposes them.
4. After H3, broaden in H4 to RMS, softmax, and BMM. Do not skip H4; attention
   is first because it validates the routing mechanism and has the strongest
   signal.
