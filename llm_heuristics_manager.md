# Helion LLM Heuristics Manager

This file defines the manager loop for the Helion LLM heuristic/autotuner
improvement project. Use `llm_heuristics_plan.md` as the living plan and gate
tracker.

## Role

The manager coordinates research. The manager does not directly implement
heuristics, debug source, run benchmarks, or tune kernels. Delegate those tasks
to subagents with narrow assignments and require artifact-backed reports.

The manager may run lightweight verification only:

- read repo and `/tmp` artifacts;
- inspect git status and diffs;
- check GPU state;
- validate JSON or markdown outputs;
- summarize existing CSVs when that is explicitly an analysis-only task.

## Objective and Score

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

## Regression Guardrails

- Every analysis must report overall geomean, per-kernel-class geomeans, and
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

## Hard Constraints

- Current live experiments may still run from
  `/home/jongsokchoi/helion_2_llm_priors` until scripts are consolidated. Use
  this for source inspection, harness commands, and delegated benchmark work.
- Fork/data/archive branch:
  `/home/jongsokchoi/helion_2_aot_pretune_data_all` on
  `choijon5/aot-pretune-data`. Use this for manager docs, AOT data, and
  archived derived heuristic artifacts. Do not run benchmarks from the archive
  checkout.
- Use GPU 3 for any delegated harness or benchmark work:
  `CUDA_VISIBLE_DEVICES=3` and `--gpu 3`.
- GPU 3 overrides older GPU 2 references in `shared_context.md`, copied `/tmp`
  artifacts, and previous reports for all future runs.
- Do not silently switch GPUs. If GPU 3 is busy or unhealthy, ask the user.
- Do not stage, commit, or push unless the user asks.
- Do not revert or disturb unrelated local changes.
- The current branch has existing unrelated dirty implementation files. These
  docs do not claim ownership of those changes and should not imply a clean
  worktree.
- Do not run installs, system package managers, broad test suites, or
  benchmarks as manager.
- Treat prompt-only range guidance as guidance, not a hard constraint.
- Use JSON metadata for workload, arm, and autotune CSV paths. Do not infer arms
  by splitting filenames, because workload names can contain arm-like suffixes.

## Canonical Inputs

- Shared context:
  `/tmp/helion_heuristics_loop/input/shared_context.md`
- Corrected round-0 summaries:
  `/tmp/helion_round0_objective_20260505_230436/guided_round0_iter11_policy/round0_summary.md`
  `/tmp/helion_round0_objective_20260505_230436/guided_round0_iter12_policy/round0_summary.md`
  `/tmp/helion_round0_objective_20260505_230436/hybrid_lfbo_round0_handoff/round0_summary.md`
- Current plan:
  `llm_heuristics_plan.md`
- Archive artifacts:
  `/home/jongsokchoi/helion_2_aot_pretune_data_all/llm_heuristics_artifacts/`
- Latest H3d archive:
  `/home/jongsokchoi/helion_2_aot_pretune_data_all/llm_heuristics_artifacts/h3d_attention_paired_no_match_20260507/`
- Latest H4 archive:
  `/home/jongsokchoi/helion_2_aot_pretune_data_all/llm_heuristics_artifacts/h4_non_attention_paired_no_match_20260507/`
- Latest H5 archive:
  `/home/jongsokchoi/helion_2_aot_pretune_data_all/llm_heuristics_artifacts/h5_attention_hybrid_lfbo_paired_no_match_20260507/`
- Latest H5b archive:
  `/home/jongsokchoi/helion_2_aot_pretune_data_all/llm_heuristics_artifacts/h5b_attention_hybrid_lfbo_l2_1_8_16_paired_no_match_20260507/`
- H2 preferred no-source-edit mechanism:
  set `HELION_LLM_OBSERVED_HEURISTICS_PATH` to a filtered observed-heuristics
  JSON and run the existing `heuristics` arm. Start with
  `/home/jongsokchoi/helion_2_aot_pretune_data_all/llm_heuristics_artifacts/h2_attention_router_observed_heuristics_b200.json`.

Latest known result: global observed heuristics help only about 5% geomean
round-0, below the current material-win gate, but useful buckets exist.
Attention wins are strong for
`attention_1k_d64` (+19.8%), `attention_2k_d128` (+15.9%), and
`attention_4k_d64` (+27.8%), while `attention_4k_d128` regresses (-19.4%).
Non-attention signals are smaller: BMM about 10.5%, `rms_norm_2048x4096` about
9.2%, `softmax_4k_2k` about 9.8%, `softmax_2k_4k` about 5.8%, and most others
neutral. The H4 paired-no-match test confirmed the broad RMS/softmax policy is
not a material win.

Latest gate state: H1 PASS, H2 PASS, H3d PASS, H4 HOLD, H5 HOLD, and H5b HOLD
for packaging on 2026-05-07. H2 corrected `round0_best_geo=0.802236` overall for attention, with
`attention_4k_d128=0.858477` while `observed_rule_match=false` in all repeats.
H3d paired-no-match scored corrected overall `round0_best_geo=0.822`, matched
geo `0.791`, no-match `attention_4k_d128` geo `1.001`, no-match median
`0.9996`, and max repeat `1.0048`. H4 scored overall `0.993`, matched active
non-attention `0.989`, no-match guardrails `0.998`, `row_norm_rms=0.989`, and
`row_softmax=0.988`; best active buckets were `rms_norm_1024x16384=0.958` and
`softmax_2k_4k=0.955`, below the material-win threshold. Replay validation
passed with baseline records 80/80, no-match replays 40/40, matched arms using
`off_matched_heuristic` 40/40, and no fatal errors. H5 attention hybrid/LFBO
paired-no-match scored round-0 overall `0.830`, matched attention `0.799`, and
no-match `attention_4k_d128` `1.001`; final verified overall `0.896`, matched
attention `0.882`, and no-match `attention_4k_d128` `0.971`, but
`attention_512_d64` regressed to `1.025`. Do not keep broad RMS/softmax.
H5b added `l2_groupings=[1]` to the d64 `seq<=2048` family. It improved the
corrected seed objective (round-0 overall `0.815`, matched `0.782`, no-match
`1.001`) and fixed `attention_512_d64` final verified performance (`0.948`),
but final hybrid/LFBO was weaker than H5 overall (`0.928` vs H5 `0.896`).
Attention remains the only clean material win, but H5c must polish the split
d64 policy before packaging.

Attention generality state: the current candidate is bucket-based, not
workload-name based, and is validated only for B200 fp16/bf16 standard
attention through 4k d64 and 2k d128. It is not validated for long d128,
8k/16k, causal variants, GQA/MQA, paged attention, cross-attention, or other
GPUs. Some selected-oracle evidence has only two supporting shapes, so gather
more shapes before turning it into product defaults.

Next candidate: split d64 attention by sequence bin. Add an exact
`seq_bin="<=1024"` d64 rule with `l2_groupings=[1,8,16]`; restore the
`seq_bin="<=2048"` d64 rule to H3b `l2_groupings=[8,16]`; leave d64 4k and
d128 short unchanged; keep `attention_4k_d128` as no-match.

Seed-only attention d128 is closed as a failed H2 direction: seeds on
`attention_2k_d128` plus `attention_4k_d128` had aggregate perf/round0 around
`1.03`, with `attention_4k_d128` around `1.058`. Do not repeat seed-only for
H2.

## Delegation Workflow

Run every cycle through gates from `llm_heuristics_plan.md`.

1. Sync context.
   Read the plan, shared context, latest summaries, archive README, and
   previous gate decision.
2. Define the next gate.
   State the hypothesis, workloads, arms, objective, acceptance criteria, and
   artifacts before delegating.
3. Delegate proposal.
   Ask Claude or Codex to propose a policy or analysis plan. Proposal agents do
   not run benchmarks.
4. Delegate review.
   Use a separate reviewer to check the proposal for objective drift, invalid
   harness assumptions, overfitting, missing controls, and GPU/reporting gaps.
   After every subagent return, run lightweight state verification:
   `git status --short` and `git diff --cached --stat`. If anything is staged
   or unrelated files changed unexpectedly, stop and ask for direction.
5. Delegate harness work only after review passes.
   The harness subagent runs the accepted command on GPU 3 and writes results
   under `/tmp`. The manager does not run the benchmark.
6. Delegate analysis.
   A separate analysis subagent computes `round0_best_geo` from autotune CSVs
   using metadata paths and reports both aggregate and per-workload results.
7. Decide the gate.
   Mark PASS, FAIL, or BLOCKED with exact metrics and artifact paths.
8. Update the plan.
   Record the decision, risks, and the next experiment-sized unit.

Do not let one subagent both propose and validate its own policy.

## Claude and Codex Collaboration

Use Claude for broad policy generation, pattern synthesis, and prompt/seed
design. Give it the canonical inputs and ask for markdown plus JSON when a
machine-readable policy is needed.

Use Codex for codebase-aware review, harness validation, JSON/schema checks,
metric correction, and small implementation proposals. Codex may inspect source
and artifacts, but source edits should be delegated only after a gate explicitly
requires implementation and the user has approved that scope.

For H2, prefer the no-source-edit env-var path before any implementation
proposal: review or create a filtered observed-heuristics JSON, set
`HELION_LLM_OBSERVED_HEURISTICS_PATH`, and use the existing `heuristics` arm.

Recommended pairing:

- Claude proposal: "Given Hn, propose the smallest policy or mechanism change.
  Do not run benchmarks. Include expected failure modes."
- Codex review: "Review the proposal against the corrected objective, current
  harness arms, GPU 3 constraint, and overfitting risk. Do not run benchmarks."
- Harness subagent: "Run exactly this accepted command on GPU 3. Do not change
  source. Return logs, output root, and failures."
- Analysis subagent: "Compute corrected `round0_best_geo` from CSV metadata.
  Do not infer arms from filenames."

## Assignment Format

```text
Task: <one sentence>
Gate: H<n>
Role: proposal | review | harness | analysis | implementation-design
Repo: /home/jongsokchoi/helion_2_llm_priors
Archive: /home/jongsokchoi/helion_2_aot_pretune_data_all
GPU: 3 required for harness, otherwise not used
Objective: round0_best_geo from generation==0,status==ok heuristics/baseline
  min(perf_ms); lower is better; material win <=0.90 unless stricter gate
Inputs:
- <artifact paths>
Allowed actions:
- <explicitly allowed actions>
Forbidden actions:
- source edits unless explicitly allowed
- staging, commits, pushes
- benchmark runs unless role is harness
Required output:
- <paths or report format>
Acceptance criteria:
- <numeric gate>
```

## Subagent Report Format

```text
## Result
status: PASS | FAIL | BLOCKED
gate: H<n>
role: proposal | review | harness | analysis | implementation-design
gpu: 3 | not_used
output_root: /tmp/<path or none>

## Objective
primary: round0_best_geo
definition: geomean over workload/repeat of generation==0,status==ok
  heuristics arm min(perf_ms) divided by matching baseline arm min(perf_ms)

## Summary
- <highest-signal finding>
- <highest-signal risk>

## Metrics
| arm | round0_best_geo | verified_geo | wall_time_geo | compile_total_geo | n |
|---|---:|---:|---:|---:|---:|

## Per Kernel Class
| kernel_class | arm | round0_best_geo | promoted_or_guardrail | note |
|---|---|---:|---|---|

## Per Workload
| workload | arm | matched_observed_rule | round0_best_geo | round0_range | verified_geo | note |
|---|---|---|---:|---|---:|---|

## H3 Matched Attribution
| match group | n | round0_best_geo | workloads | note |
|---|---:|---:|---|---|
| matched_observed_rule=true | <n> | <value> | <list> | <note> |
| matched_observed_rule=false | <n> | <value> | <list> | <note> |

## Gate Decision Input
- pass criteria met: yes | no
- blockers:
- caveats:

## Artifacts
- <path>

## Next
- <one concrete next action>
```

For proposal-only tasks, replace metric tables with:

```text
## Proposed Change
- mechanism:
- workloads:
- arms:
- expected win:
- expected risk:
- smallest source change needed: none | <files and reason>

## Validation Plan
- command template:
- acceptance criteria:
- rollback/demotion rule:
```

## Gate Decision Format

```text
Gate H<n>: PASS | FAIL | BLOCKED
Date:
Decision owner:
Artifacts:
- <paths>
Primary metric:
- <arm>: round0_best_geo=<value>, baseline=1.000
Secondary diagnostics:
- verified_geo=<value>
- wall_time_geo=<value>
- compile_total_geo=<value>
Per-workload regressions:
- <workload>: <value and threshold>
Reason:
- <short explanation>
Plan update:
- <what changes in llm_heuristics_plan.md>
Next unit:
- <next experiment-sized or commit-sized task>
```

## Harness Command Template

The manager writes command templates for subagents; the manager does not run
them.

For H2 attention-router runs, set the observed-heuristics path explicitly and
use `baseline,heuristics`:

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

```bash
CUDA_VISIBLE_DEVICES=3 /home/jongsokchoi/.conda/envs/helion_2/bin/python \
  scripts/llm_heuristics_autoresearch.py \
  --gpu 3 \
  --suite <suite> \
  --workloads <comma-separated-workloads> \
  --arms baseline,<candidate-arm> \
  --autotuner LLMGuidedSearch \
  --model gpt-5-2 \
  --llm-max-rounds 1 \
  --repeats 3 \
  --output-root /tmp/<descriptive-run-name>
```

For hybrid handoff gates, use paired-no-match mode when guardrails are present:

```bash
HELION_LLM_OBSERVED_HEURISTICS_PATH=/home/jongsokchoi/helion_2_aot_pretune_data_all/llm_heuristics_artifacts/h3b_attention_strict_family_no_r4_observed_heuristics_b200.json \
CUDA_VISIBLE_DEVICES=3 /home/jongsokchoi/.conda/envs/helion_2/bin/python \
  scripts/llm_heuristics_autoresearch.py \
  --gpu 3 \
  --workloads <comma-separated-workloads> \
  --arms baseline,<candidate-arm> \
  --autotuner LLMSeededLFBOTreeSearch \
  --model gpt-5-2 \
  --llm-max-rounds 1 \
  --llm-round0-mode paired-no-match \
  --repeats 3 \
  --output-root /tmp/<descriptive-run-name>
```

## Lightweight Verification Commands

Manager-safe checks:

```bash
pwd
git status --short
git diff --cached --stat
nvidia-smi -i 3 --query-gpu=index,name,memory.used,utilization.gpu --format=csv
git diff -- llm_heuristics_manager.md llm_heuristics_plan.md
LC_ALL=C grep -n '[^ -~]' llm_heuristics_manager.md llm_heuristics_plan.md
rg -n 'CUDA_VISIBLE_DEVICES=|--gpu ' llm_heuristics_manager.md llm_heuristics_plan.md
```

When summarizing an existing output root without new benchmark work, use
`--skip-run` only against an existing complete run:

```bash
/home/jongsokchoi/.conda/envs/helion_2/bin/python \
  scripts/llm_heuristics_autoresearch.py \
  --skip-run \
  --output-root /tmp/<existing-run-root>
```

## When to Ask for Help

Ask the user before proceeding if:

- GPU 3 is unavailable and a benchmark is needed;
- the next step requires source edits, staging, committing, or a new harness
  arm;
- corrected objective data disagrees across artifacts;
- a proposed gate changes the primary objective, benchmark matrix, model, or
  GPU;
- a benchmark would be expensive, long-running, or likely to contend with other
  local work;
- local source changes block a clean implementation design.

## Repeated Gate Cycle

For each H gate:

1. Restate the hypothesis and known data.
2. Confirm the metric and GPU constraint.
3. Get a proposal.
4. Get an independent review.
5. After each subagent return, check `git status --short` and
   `git diff --cached --stat`.
6. Run only the accepted benchmark through a harness subagent.
7. Analyze corrected round-0 from CSV metadata.
8. Decide PASS, FAIL, or BLOCKED.
9. Update `llm_heuristics_plan.md`.
10. Stop or ask for approval before any implementation beyond documentation or
   coordination changes.
