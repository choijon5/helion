# Row-Reduction Heuristic Hill-Climbing Plan

Living plan for iteratively improving `LLMSeededSearch` round-0 performance
on the row-reduction kernel family (`layer_norm`, `rms_norm`, `softmax`)
using Claude subagents.

## Primary Objective and Score

For each live benchmark we run two arms on the same kernel + shape set with
the same autotuner (`LLMSeededSearch` or `LLMSeededLFBOTreeSearch`):

- **baseline arm**: the autotuner as-is, no heuristic guidance to round 0.
- **heuristics arm**: the same autotuner with the candidate heuristic policy
  active (either an AOT heuristic file that injects a round-0 seed config
  per shape, or an observed-heuristics JSON that steers the LLM prompt / seed
  set).

Score is exactly the metric used in the attention loop:

```text
round0_best_geo =
  geomean over (workload, repeat) of
    min(perf_ms where generation==0 and status==ok in heuristics arm)
  / min(perf_ms where generation==0 and status==ok in baseline arm)
```

Lower is better; floor is `0.0`; baseline arm is `1.000` by construction.

**Terminal goal:** a *general* heuristic that reaches
`round0_best_geo ≤ 0.80` across the combined norm family
(layer_norm + rms_norm + softmax) overall and on at least two of the three
kernels individually. Lower-is-better: a 20% speedup vs. the same autotuner
with no heuristic seeding, purely from round 0.

"General" here is a first-class acceptance criterion, not a nice-to-have.
A candidate heuristic that hits the target on the training shape grid but
not on held-out shapes does not PASS; it is archived as a data point and
the loop continues.

Generality is measured three ways:

- **Held-out shapes, same kernels.** For each kernel we split the full
  shape grid into a train set and a held-out set (60/40). The heuristic is
  built from train-set CSVs only; the gate scores on held-out shapes. A
  candidate PASSes only if `round0_best_geo_heldout ≤ 0.80` AND
  `round0_best_geo_heldout − round0_best_geo_train ≤ 0.05`.
- **Held-out kernel, same family.** Build the heuristic from two of the
  three norms and score on the third. PASS requires
  `round0_best_geo_heldout_kernel ≤ 0.85`. (Looser than same-kernel
  held-out; the family generalization gate is real but strict.)
- **Feature discipline.** No rule in the heuristic may key off the kernel
  name or a feature that is unique to one CSV (e.g. a config hash, a
  run_id). Features must be functions of the kernel arguments only:
  shape, dtype, strides, numel. Enforced by inspection of the generated
  `heuristic_*.py` file and of any observed-heuristics JSON.

Stretch levels (track separately, don't move the floor):

- Strong: `round0_best_geo_heldout ≤ 0.75`
- Excellent: `round0_best_geo_heldout ≤ 0.70`, held-out kernel ≤ 0.80

Secondary metrics reported on every gate but not optimized against:

- `verified_geo` — geomean of the final best config after all stages.
- `wall_time_geo` and `compile_total_geo` — cost diagnostics; must stay
  `≤ 1.20` unless the user accepts the trade.

## Regression Guardrails

- No per-kernel `round0_best_geo` > `1.03` on train or held-out shapes. If
  a promoted policy regresses any of the three kernels > 3%, it must fix
  that before promotion.
- Any repeat-level ratio > `1.10` triggers a focused rerun before the gate
  passes.
- If a repeat-level regression > `1.10` survives the rerun, the policy is
  HOLD.
- **Overfit alarm:** `round0_best_geo_heldout − round0_best_geo_train >
  0.05` on any kernel is a FAIL, even if both numbers are below `0.90`. The
  gate demotes the candidate and goes back to propose.

## Autotuner Configuration

The live benchmark arms both use:

- autotuner: `LLMSeededSearch` (stage 1 = `LLMGuidedSearch`, stage 2 =
  `LFBOTreeSearch`) unless a gate explicitly chooses `LLMGuidedSearch`
  only.
- LLM provider: Anthropic via `HELION_LLM_PROVIDER=anthropic`.
- LLM model: `claude-opus-4-7` via `HELION_LLM_MODEL=claude-opus-4-7`.
- Reasoning effort: maximum. Helion's transport does not yet pass an
  extended-thinking flag to the Anthropic Messages API (see
  `helion/autotuner/llm/transport.py::_anthropic_payload`). Before N1 runs,
  a harness subagent adds `thinking={"type":"enabled","budget_tokens":...}`
  to the Opus-4.7 branch. Until that lands, we use whatever reasoning the
  current payload gets at max `max_tokens`.
- `llm_max_rounds=1` on both arms so scoring is directly on round 0.
- Repeats: 3 per workload per gate to detect per-repeat noise.

Baseline arm `HELION_LLM_*` env vars are identical except whatever
mechanism we use to disable the heuristic (either no env-var pointing at the
observed-heuristics JSON, or the heuristic file removed from the AOT cache
search path).

## Hill-Climbing Metric, offline

The offline AOT score from `heuristic_generator.py` is **input to the
heuristic**, not the loop metric. We still track it (`N0_baseline.json`)
because a better AOT heuristic (lower `max_slowdown`) should correlate with
a better `round0_best_geo`, but the acceptance gate is always the live
`round0_best_geo`.

## Data Sources

- Offline (heuristic construction): archived measurement CSVs and tuned
  configs under `aot_pretune_data/b200/{layer_norm,rms_norm,softmax}/`.
- Live (scoring): `LLMSeededSearch` run on each kernel example file in
  `examples/{layer_norm,rms_norm,softmax}.py` on this machine's B200.

Shape list per kernel for the live harness (proposed, adjust in N0):

- `layer_norm`: 12 shapes covering `(rows, cols)` with rows in
  `{256, 1024, 2048, 8192}` and cols in `{1024, 4096, 16384}`.
- `rms_norm`: same grid.
- `softmax`: 12 shapes across `(rows, cols)` and one batched `(B, rows,
  cols)` group.

### Train / held-out split

For every kernel, the 12-shape grid is split into a **train set** of 7
shapes and a **held-out set** of 5 shapes, committed in N0-live under
`iterations/N0_live/shape_grid.json`. The split must:

- cover both rows and cols axes in each half (no axis-only leakage);
- include at least one "corner" shape (smallest or largest numel) in each
  half;
- be deterministic (fixed seed) so later gates can reproduce it.

Rules for every gate after N0-live:

- Heuristic construction (whether AOT `heuristic_*.py`, new feature
  extractor, observed-heuristics JSON, or nearest-neighbor backend) uses
  train-set CSV rows only. If pooled CSVs are needed, they are filtered to
  train shapes before passing to `heuristic_generator.generate_heuristic`.
- Scoring reports three numbers per kernel: `train`, `heldout`, and the
  overfit delta `heldout − train`.
- The **held-out kernel** slot rotates per N4 run: build from two norms,
  score on the third, and tag which kernel was held out.

## Current Baseline (N0 offline) and Live Placeholder

Offline AOT max_slowdown (archive-reproduced, target of heuristic
construction):

| source | configs | max_slowdown | geomean | accuracy |
|---|---:|---:|---:|---:|
| layer_norm single run | 5 | 1.0769 | 1.0117 | 0.974 |
| layer_norm 5-run pooled | 9 | 1.0932 | 1.0172 | 0.829 |
| rms_norm single run | 7 | 1.0052 | 1.0006 | 1.000 |
| softmax single run | 5 | 1.0288 | 1.0014 | 1.000 |
| softmax 4-run pooled | 7 | 1.0962 | 1.0052 | 0.904 |

Live `round0_best_geo` (to be measured in N0-live):

| kernel | scope | baseline | heuristics | round0_best_geo |
|---|---|---:|---:|---:|
| layer_norm | train | | | |
| layer_norm | heldout | | | |
| rms_norm | train | | | |
| rms_norm | heldout | | | |
| softmax | train | | | |
| softmax | heldout | | | |
| norm family (geomean) | train | | | |
| norm family (geomean) | heldout | | | |

## Gates

### N0 offline: lock AOT baselines — DONE

Status: PASS. Artifact: `N0_baseline.json`.

### N0 live: pick a baseline shape list and measure `round0_best_geo = 1.000`

Work:

- Pick a shape grid per kernel (propose in a subagent, confirm with user).
- Run the baseline arm only (no heuristic) on this machine's B200.
- Confirm CSV emission format matches what the scoring harness expects.
- Sanity-check noise: 3 repeats should give per-shape variance < 3%.

Acceptance: shape list committed, baseline run archived, and per-kernel
variance within threshold.

### N1: Plumb Opus-4.7 extended thinking

Hypothesis: without max reasoning effort, Opus-4.7 round-0 responses are
weaker than `gpt-5-2` on the same kernels, biasing the comparison.

Work (implementation-design subagent):

- Review `helion/autotuner/llm/transport.py::_anthropic_payload`.
- Propose the smallest patch to pass `thinking={"type":"enabled",
  "budget_tokens":<large>}` on the Claude Opus 4.7 branch, gated by an env
  var (e.g. `HELION_LLM_ANTHROPIC_THINKING_BUDGET`).
- Do not land the patch until user approval.

Acceptance: patch proposal, reviewer approval, and a 1-shape sanity run
showing the payload reaches Anthropic unmodified.

### N2: Convert the existing AOT heuristics into a round-0 seed mechanism

Hypothesis: `heuristic_*.py` files in `aot_pretune_data/b200/<kernel>/runs/`
already pick a near-oracle config per shape for the archived measurements.
Injecting that as the first round-0 seed (alongside the default config)
should beat an untuned baseline by at least `round0_best_geo ≤ 0.85`.

Work:

- Implement a seed-config injector: a thin hook that, given the current
  kernel args, calls `autotune_<kernel>(*args)` from the latest AOT
  heuristic file and inserts the returned config at the front of
  `_build_seed_configs` in `LLMGuidedSearch`.
- Prefer an env-var-gated path (e.g.
  `HELION_LLM_ROUND0_HEURISTIC_PATH=<heuristic_file.py>`) so the baseline
  arm stays unpatched.
- Plumb this into the heuristics arm only.

Acceptance: the heuristics arm consistently emits the injected config in
`generation==0` rows; `round0_best_geo_train ≤ 0.85` overall and ≤ 0.88
per kernel; `round0_best_geo_heldout − round0_best_geo_train ≤ 0.05`
per kernel.

### N3: Hill-climb the AOT heuristic content

Hypothesis: the pooled `heuristic_*.py` already has ~9.3% max_slowdown room
on layer_norm and 9.6% on softmax; closing that gap should directly reduce
`round0_best_geo`.

Work (per kernel, one at a time):

1. Analysis subagent identifies the worst-slowdown shapes in the archived
   CSV and what config separates them.
2. Proposal subagent proposes one change — a new feature, an added config
   to the selected set, or a backend swap (`nearest_neighbor`).
3. Review subagent checks for overfitting the single run and leaking
   config-side features into the shape-side features.
4. Rebuild the heuristic from the updated inputs; rescore offline
   `max_slowdown` first.
5. If offline score improves, run live on the shape grid and compare to
   N2 champion.

Acceptance (per attempt): both offline `max_slowdown` (computed on train
shapes only) and live `round0_best_geo_train` strictly improve over the
current champion; `round0_best_geo_heldout` does not get worse by more
than 1%; no kernel regresses > 1%.

### N4: Broaden to observed-heuristics-style routing

Hypothesis: per-shape seed injection is the strongest, but the attention
loop succeeded more with observed-heuristics prompt guidance. Test whether
that mechanism adds or subtracts for norms.

Work:

- Build a filtered observed-heuristics JSON for the norm family from
  `aot_pretune_data/b200/<kernel>/runs/*/tuned_configs_*.json`.
- Run the heuristics arm with both mechanisms (seed + observed JSON) and
  each alone. Compare.

Acceptance: the best-of-three heuristics arm
`round0_best_geo_heldout ≤ 0.80` overall AND on at least two of the three
kernels; held-out kernel rotation (build from two norms, score on the
third) reaches `round0_best_geo ≤ 0.85`.

### N4b: Generality stress test

Hypothesis: the N4 champion holds when we stretch the shape range beyond
the training grid, when we perturb dtype, and when we build the heuristic
from one kernel's data and apply it to a sibling kernel.

Work:

- **Shape stretch.** Add 4 shapes per kernel that lie outside the N0 grid
  range (smaller/larger numel, non-power-of-two dimensions). Score the
  current champion; demand `round0_best_geo ≤ 0.85` on the stretched
  set.
- **Dtype swap.** The archived CSVs are bf16-dominant. Rerun a 4-shape
  subset with fp16 and fp32 variants. Demand `round0_best_geo ≤ 0.90`
  (weaker, because the archive has little non-bf16 signal).
- **Cross-kernel rotation.** Build the heuristic from two of the three
  norms and evaluate on the third. Report
  `round0_best_geo_crosskernel` per rotation.
- **Feature audit.** Grep the generated heuristic code and any observed
  JSON for kernel-name strings or run/config hashes. Any match is an
  automatic FAIL.

Acceptance: all four sub-checks pass their thresholds above; the feature
audit is clean. Any shape-family where the heuristic fires but regresses
is documented as a known limitation in the packaged policy.

### N5: Final packaging

Hypothesis: the champion that survived N4b is a narrow, general policy
ready for packaging; it should not depend on any kernel-name string and
should have a documented scope.

Work:

- Freeze the champion's seed mechanism and (if used) observed-heuristics
  JSON content.
- Write a short policy doc listing: applicable kernels, applicable
  dtypes, applicable shape ranges, applicable hardware (B200 only, for
  now), and the demotion rule for unsupported shapes.
- Produce a minimal source-change plan for user approval. The plan must
  keep the policy behind an opt-in env var or config flag; no silent
  default flip.

Acceptance: policy JSON/markdown + source-change plan reviewed by a
separate Claude subagent; all live numbers (train, heldout, stretch,
dtype, cross-kernel) reproduce within 1% of the N4/N4b runs.

## Workflow

Each gate:

1. Manager defines the hypothesis, the exact change, what PASS/FAIL means.
2. **Proposal subagent** (Claude) drafts the change. No edits, no runs.
3. **Review subagent** (a different Claude instance) checks it.
4. **Implementation subagent** makes the reviewed edits. Limited file
   scope.
5. **Harness subagent** runs the accepted commands on B200 and writes
   artifacts under `iterations/N<n>_<slug>/`.
6. **Analysis subagent** computes `round0_best_geo` from the CSV
   metadata (not from filenames) and writes the per-gate report.
7. Manager decides PASS/FAIL/BLOCKED and updates this file.

## Hard Constraints

- Do not run `git commit` or `git push` as part of any subagent.
- Do not run `pip install` or system package managers.
- Live benchmarks use GPU 0 on this machine (the only B200 here) unless the
  user says otherwise.
- The score always comes from `generation==0,status==ok` rows in the CSV,
  not from filenames. CSV metadata paths are the source of truth.
- Do not modify `heuristic_generator.py`'s public API. Extensions live in
  new files in `helion/autotuner/llm/` or in a wrapper script.

## Out of Scope

- Changing the LLM provider away from Anthropic Opus-4.7.
- Scoring against PyTorch eager, AOT CSVs directly, or the final
  full-autotune winner.
- Running live benchmarks on any hardware other than this B200.
