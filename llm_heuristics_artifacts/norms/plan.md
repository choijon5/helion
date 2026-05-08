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
- **Integrity check:** on any heuristics-arm run, the seeded config hash
  must appear in at least 80% of `(shape, repeat)` benchmark sets. If
  it's missing in most repeats, the heuristic is failing silently — the
  run is marked BLOCKED and the bug is fixed before any score is
  trusted.

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

Live `round0_best_geo` — best measured across gates (see "Running log"
for history, and `iterations/*/scores.json` for per-gate breakdown):

| kernel | scope | best round0_best_geo | gate that produced it | notes |
|---|---|---:|---|---|
| layer_norm | train | 0.9291 | N3b | heldout flat (1.000) |
| layer_norm | heldout | 1.0002 | N3b | overfit delta +0.071 |
| rms_norm | train | 0.9512 | N2b | clean win |
| rms_norm | heldout | 0.9232 | N2b | best heldout, only promotable kernel |
| softmax | train | 0.8905 | N3d | train wins 11% |
| softmax | heldout | 0.9993 | N3d | overfit delta +0.109 |
| norm family | train | 0.9264 | N2b | |
| norm family | heldout | 0.9740 | N3b | terminal goal 0.80 not reached |

Terminal goal `round0_best_geo ≤ 0.80` not reached. Best family
heldout so far is `0.974` (about 2.6% win vs baseline). See the
"What it would take to reach 0.80" section at the end of the
Running log for the archive-expansion + prompt-context follow-ons.

## Running log

Every gate updates this log with what ran, what we learned, and what
changed in the plan. The plan above is the latest-truth; this log is the
history behind it.

- **2026-05-08 N0-live baseline** — ran 3 × 12 × 3 = 108 autotuning runs
  on B200 via `LLMSeededSearch(max_rounds=1)` with Opus 4.7 + adaptive
  reasoning. Reproducibility noise per shape:
  layer_norm max spread 0.45%, rms_norm max 21% (one shape), softmax
  max 92%. Softmax noise is high because the LLM picks different configs
  each repeat; the ratio metric should tolerate it in expectation, but
  some ratios will be noisy.

- **2026-05-08 softmax kernel mismatch discovered** — `workloads.py`
  wired softmax to `softmax_two_pass` (2 block_sizes), but the archive
  was tuned against the simpler `softmax` (1 block_size). Fixed workloads
  to use `softmax`, re-ran softmax baseline, dropped the old softmax
  heuristics CSV. **Plan change**: added a compile-filter step to the
  heuristic-build pipeline (N2) so stale/incompatible configs cannot
  silently poison the seed.

- **2026-05-08 N2 v1 failed — seed injection bug** — first N2 run produced
  tiny changes (+/- 1-5%) rather than the expected win. Audit found the
  heuristic seed was never in any benchmarked config: the `run_live.py`
  seed-loader called `autotune_<kernel_fn.name>` where `kernel_fn.name`
  is the Python function name (e.g. `rms_norm_fwd`), while the generated
  heuristic exports `autotune_<csv_kernel_name>` (e.g. `autotune_rms_norm`).
  Silent `None` return meant the heuristics arm effectively ran as
  baseline + noise. **Plan change**: runner now falls back to the first
  `autotune_*` callable in the module and raises if none found. New gate
  acceptance bullet: "heuristics arm seed config hash must appear in at
  least 80% of (shape, repeat) benchmark sets; otherwise treat as
  invalid run."

- **2026-05-08 Triton compile failures on archive configs** — some
  archived configs with `num_stages=6 + indexing=tensor_descriptor`
  fail Triton MLIR pipelining on the current nightly
  (`ttg.local_alloc op does not have expected attribute ttg.partition`).
  Compile-filter step catches and drops them before they reach the
  seed path. rms_norm + layer_norm all compile; softmax archive uses
  the 1-block-size `softmax` kernel so those configs also compile once
  workloads.py points at the right entry.

- **2026-05-08 N2 v2 (bugfix rerun) — measured** —
  family train `0.972`, heldout `0.987`. Per kernel:
  - layer_norm: train `0.949`, heldout `1.014`, delta `+0.065`
    **overfit FAIL**.
  - rms_norm: train `0.967`, heldout `0.948`, delta `−0.019` (clean
    train+heldout win, but only 3–5%).
  - softmax: train/heldout both `1.000` (no signal; the heuristic's
    selected configs are not preferred over what the LLM finds).
  Seed integrity: 100% of pairs contained the heuristic seed. Target
  is `0.80` heldout family; we are at `0.987`. Gap = ~19 percentage
  points.

  **Takeaways:**
  1. Single-seed injection is too weak: round 0 still benchmarks the
     default + 3 random + 5 LLM configs, so the heuristic seed is
     one of ~9 candidates. If any of the other 8 is near-oracle, the
     heuristic contributes little.
  2. layer_norm fails the 0.05 overfit rule: train and heldout have
     different best-config patterns, suggesting the generated tree is
     memorizing the archive's training shape set rather than
     learning a transferable rule.
  3. softmax archive-selected configs do not beat what the live LLM
     picks on any shape. This matches the archived max_slowdown of
     ~1.09 — the archive's "best" configs only win by 1% or so on
     archive shapes, not enough to beat LLM-proposed configs on our
     different grid.

  **Plan change for N2b**: inject all configs from the heuristic's
  selected set as seeds (not just the single tree-picked winner), and
  reduce `initial_random_configs` to zero when the heuristic is
  active. This shifts the heuristics arm from "one good guess + LLM"
  to "good config library + LLM". Expected direction: strictly
  better than N2 because the heuristic-picked tree leaf is *always*
  in the seed library, plus the other 6–8 library configs give
  broader coverage.

- **2026-05-08 N2b (library injection) — measured** —
  family train `0.926`, heldout `0.976`. Per kernel:
  - layer_norm: train `0.929`, heldout `1.008`, delta `+0.079`
    **overfit, worse than N2**.
  - rms_norm: train `0.951`, heldout `0.923`, delta `−0.028`
    **clean win, best heldout so far**.
  - softmax: train `0.899`, heldout `0.999`, delta `+0.100`
    **big train win, no heldout transfer**.

  Library mode beats tree mode on all three kernels for train
  performance, but held-out performance only improves meaningfully on
  rms_norm. The problem is feature coverage: the 7–9 configs the
  generator selects cover archived-training shapes well, but the
  held-out slot in our grid hits shapes that need a different
  config not in the library.

  **Plan change for N3**: two parallel directions.
    (a) **More diverse library** — re-run `heuristic_generator` with
    `max_configs=20` so the library has richer shape-coverage.
    (b) **NearestNeighbor backend** — swap
    `backend='decision_tree'` for `backend='nearest_neighbor'`. NN
    maps each new shape to the closest archive shape's winning
    config; expected to generalize better on held-out than trees do.
    Run (a) and (b) independently; pick whichever performs better on
    layer_norm heldout (the weakest kernel).

- **2026-05-08 N3a skipped** — raising `max_configs=10→20` produced
  identical heuristics. The greedy set-cover already stopped at 7–9
  because threshold=1.10 was satisfied; bigger budget didn't add
  configs. No re-run needed; N3a=N2b.

- **2026-05-08 N3b (NN, library) — measured** —
  family train `0.932`, heldout `0.974`. Per-kernel nearly identical
  to N2b (delta < 0.003). Conclusion: backend swap doesn't change
  outcomes when the library is the same 7–9 configs — all library
  configs are benchmarked, so whichever is best wins regardless of
  selection rule.

- **2026-05-08 N3c (tight threshold=1.01) + N3d (top-30 archive
  configs) — measured** —
  - N3c: layer_norm and rms_norm pools are *already* set-cover
    minimal; tightening threshold added no configs. softmax grew
    7 → 14 configs.
  - N3d: dumped all unique configs from archive CSVs.
    layer_norm has 9 unique total, rms_norm 7 total, softmax 256.
    Taking top 30 for softmax: N3d softmax train `0.891`, heldout
    `1.000`. No meaningful change over N2b.

  **Finding that stops the loop**: the layer_norm archive only
  contains **9 unique configs** post compile-filter; rms_norm only 7.
  Those libraries have already been exhausted (both tree and NN
  return the same library; tight threshold doesn't add to it; direct
  dump tops out at 9/7). The LLM round-0 baseline already finds
  configs that tie the library on most held-out shapes. The only
  shapes where heuristics win decisively are the 2–3 shapes where
  the library contains a specific good config the LLM doesn't
  propose. The family heldout floor with this archive is about
  **0.95**, not `0.80`.

## What it would take to reach 0.80

The bottleneck is archive size, not mechanism. A follow-on effort
would need one of:

1. **Expand the archive.** Run fresh autotune on this B200 / Triton
   nightly for a broader shape grid (both train and extra held-out
   "archive" shapes), producing 30–50 unique configs per kernel. The
   existing `heuristic_generator` machinery consumes this directly.
   Expected time: ~hours per kernel.
2. **Give the LLM the archive-measurement CSV in the prompt** so
   Opus 4.7 can propose library-aware configs on live shapes. This
   is a core-source change to `helion/autotuner/llm/prompting.py`
   and requires user approval.
3. **Combine the library with 1-2 LFBO refinement rounds** (move
   from `LLMGuidedSearch` to `LLMSeededLFBOTreeSearch` and let
   stage-2 surrogate search explore around library configs). Not
   a round-0 metric win, but drives final verified perf down.

## Current best policy (end of this session)

For packaging, **rms_norm** is the only kernel with a clean,
promotable N2b heuristic (train 0.951, heldout 0.923, delta
−0.028). layer_norm and softmax show train wins but heldout is
flat; they should not be promoted without more archive data.

Not at the 0.80 terminal goal — current family heldout best is
**0.974** (N3b). Honest stopping point; next unit is archive
expansion.

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

## Archive Expansion Rule

When generating extra measurement data to feed the heuristic (e.g. N4a and
any future archive-expansion gate), always use the **default autotuner
(`LFBOTreeSearch`) with full autotuning effort**. Do not substitute
`PatternSearch`, `DifferentialEvolutionSearch`, or any reduced-effort
configuration just because it produces more raw (config, timing) pairs
faster. The goal of archive expansion is to capture configs that full
autotuning would actually select on this hardware; using a lighter
autotuner biases the archive toward configs the lighter search happens
to visit.

Concretely, for archive expansion:

- Autotuner: `LFBOTreeSearch` (Helion's default when nothing is overridden).
- Effort: full (no `HELION_AUTOTUNE_EFFORT=none` or `=quick`).
- Capture every benchmarked (config, timing) pair, not just the final best,
  so downstream `heuristic_generator` has a full measurement matrix.

N4a originally ran with `PatternSearch(max_generations=8, copies=4)` and
has been re-run with full `LFBOTreeSearch`. The rule applies to all
future archive-expansion gates.

## Out of Scope

- Changing the LLM provider away from Anthropic Opus-4.7.
- Scoring against PyTorch eager, AOT CSVs directly, or the final
  full-autotune winner.
- Running live benchmarks on any hardware other than this B200.
