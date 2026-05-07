# Helion LLM Heuristics Gate Plan

This is the living plan for improving Helion LLM heuristic usefulness. The
current objective is first LLM round handoff quality:
`round0_best_geo` from autotune CSV rows where `generation == 0` and
`status == ok`. Lower is better. Compile and wall time are secondary diagnostics
unless the explicit experiment is compile-time reduction.

All future benchmark work should use GPU 3 through `CUDA_VISIBLE_DEVICES=3` and
`--gpu 3`.

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

## Data Sources

- `/tmp/helion_heuristics_loop/input/shared_context.md`
- `/tmp/helion_round0_objective_20260505_230436/guided_round0_iter11_policy/round0_summary.md`
- `/tmp/helion_round0_objective_20260505_230436/guided_round0_iter12_policy/round0_summary.md`
- `/tmp/helion_round0_objective_20260505_230436/hybrid_lfbo_round0_handoff/round0_summary.md`
- `helion/autotuner/llm/data/observed_heuristics_b200.json`
- Derived snapshot, if useful for policy review:
  `/tmp/helion_heuristics_loop/codex/range_policy_data_snapshot/runtime_observed_heuristics_b200.json`
- Existing policy artifacts under `/tmp/helion_heuristics_loop/claude` and
  reviews under `/tmp/helion_heuristics_loop/codex`

## Repository State

The current branch already has unrelated dirty implementation files. This plan
does not claim ownership of those changes. Future managers and subagents must
avoid reverting, staging, or summarizing those unrelated changes as part of this
documentation or heuristic-planning work.

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
- Re-score existing corrected outputs only; no new benchmark required.
- Make every future report include `round0_best_geo`, `verified_geo`,
  `wall_time_geo`, and compile diagnostics.

Acceptance:

- Existing corrected summaries reproduce the known aggregate values:
  iter11 guided `heuristics=0.953`, iter12 guided `range_prompt=0.974`,
  hybrid `heuristics=0.949`.
- A subagent can identify the CSV source for every row through JSON metadata.
- The manager has an accepted report template and gate decision format.

Status: docs/report template created, but H1 is not PASS yet. A subagent still
needs to explicitly re-score or verify the existing corrected summaries with
the corrected analyzer before H1 can be marked PASS.

### H2: Design a Bucket Router Starting with Attention

Hypothesis: observed heuristics become useful when routed by kernel and shape
bucket, especially for attention.

Initial attention policy direction:

- Enable heuristic help for `attention_1k_d64`, `attention_2k_d128`, and
  `attention_4k_d64`.
- Route `attention_4k_d128` to baseline/no-guidance until a safe candidate
  family exists.
- Add holdout attention shapes only after the four known buckets reproduce.
- Existing harness arms may only approximate this gate. A true routed policy
  may need a new `routed_heuristics` arm or equivalent mechanism. Any benchmark
  source edits to add that arm require explicit user approval before editing.

Acceptance:

- On `attention_1k_d64`, `attention_2k_d128`, `attention_4k_d64`,
  `attention_4k_d128`, the routed arm has attention `round0_best_geo <= 0.90`.
- `attention_4k_d128 <= 1.02` and no repeat has a large unexplained
  regression.
- Compile/wall diagnostics are `<=1.20x` unless explained by a better primary
  objective and accepted by the user.

Next experiment-sized unit:

- Proposal subagent designs the router behavior and whether it can be tested
  with existing arms as an approximation. If a true `routed_heuristics` or
  equivalent arm is needed, stop at an implementation design and ask for user
  approval before source edits.

### H3: Convert Exact Observed Configs into Candidate Families and Ranges

Hypothesis: exact observed winners can be generalized into candidate families
that preserve most wins without hardcoding individual shapes.

Work:

- For attention buckets that pass H2, extract common config features:
  block sizes, warps, stages, indexing, pid type, range parameters, and
  num-sm multiplier.
- Convert exact configs into small families or constrained candidate ranges.
- Include nearby holdouts such as `attention_512_d64`, `attention_2k_d64`, and
  additional d128 lengths when available.

Acceptance:

- Candidate families retain at least 80% of the H2 bucket win on known shapes.
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
| H2 | attention_1k_d64, attention_2k_d128, attention_4k_d64, attention_4k_d128 | baseline plus routed candidate, or existing-arm approximation | LLMGuidedSearch | 3 | route attention buckets |
| H3 | H2 plus attention_512_d64, attention_2k_d64, nearby d128 holdouts | baseline plus candidate families | LLMGuidedSearch | 3 | generalize exact configs |
| H4 | RMS, softmax, BMM targets plus neutral controls | baseline plus candidate | LLMGuidedSearch | 3 | broaden non-attention value |
| H5 | 8-workload hybrid handoff matrix | baseline plus promoted candidate | LLMSeededLFBOTreeSearch | 3 | validate handoff |
| H6 | corrected broad/full matrix | baseline plus final policy | selected final autotuner | 3 | package decision |

## Risks

- Overfitting exact shapes instead of reusable feature buckets.
- Filename parsing can corrupt arm/workload attribution.
- Prompt-only ranges do not hard constrain generated configs.
- Seed-batch wins can disappear after the LLM first-round candidates are added.
- GPU changes can dominate small deltas; future runs must use GPU 3.
- Compile/wall-time noise can distract from the primary objective, but large
  compile spikes still indicate system or mechanism problems.
- Existing local source changes mean implementation work must avoid unrelated
  edits and needs explicit approval.

## Next Unit

Finish H1/H2 planning before code work:

1. Have a proposal subagent design the attention bucket router and state whether
   it can be approximated with existing arms or needs a true `routed_heuristics`
   arm.
2. Have an independent review subagent check objective, GPU 3, harness arms, and
   `attention_4k_d128` protection.
3. If no new source is needed, delegate a GPU 3 H2 approximation benchmark
   through the harness subagent. If a new arm is needed, stop with a
   commit-sized implementation design and ask the user before editing source.
