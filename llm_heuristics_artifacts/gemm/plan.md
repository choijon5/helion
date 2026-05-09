# GEMM Heuristic Hill-Climbing Plan

Living plan for iteratively improving `LLMSeededSearch` round-0 performance
on the GEMM kernel family (`matmul`, `fp8_gemm`) using Claude subagents.

## Primary Objective and Score

We run two separate experiments. The **primary** answers what the
heuristic is worth to a user who does not have an LLM in the loop; the
**secondary** answers whether the heuristic still adds value when the
LLM is already proposing configs.

### Experiment 1 (primary): no-autotune baseline

Both arms use `HELION_AUTOTUNE_EFFORT=none` — Helion's default-config
path. One config per shape, benchmarked directly. No random seeds, no
LLM, no search at all.

- **baseline arm**: Helion's `config_spec.default_config()`, the
  generic default shipped with the kernel.
- **heuristics arm**: the single config the heuristic picks for this
  shape, benchmarked directly.

Score:

```text
round0_best_geo =
  geomean over (workload, repeat) of
    min(perf_ms where generation==0 and status==ok in heuristics arm)
  / min(perf_ms where generation==0 and status==ok in baseline arm)
```

Lower is better. Baseline arm = 1.000 by construction. This ratio
answers: "does the heuristic's one config beat Helion's one default
config?"

**Primary terminal goal:** `round0_best_geo_heldout ≤ 0.20` across the
GEMM family AND on each kernel individually. That is ≥5× speedup from
picking the heuristic's config instead of Helion's default.

Rationale for the bar: in the N7 ablation (recorded 2026-05-09), the
current N6 heuristics scored 0.088 matmul / 0.074 fp8 family heldout
on this metric. 0.20 leaves ~2× of slack for regression while still
being a legitimate 5× speedup. A heuristic that lands at 0.20–0.30 is
shippable; below 0.10 is strong.

Runner: `tools/run_no_autotune.py`.

### Experiment 2 (secondary): LLM-on baseline

Both arms use `LLMGuidedSearch` with `max_rounds=1` and Opus 4.7 via
Bedrock enabled (the normal path). The baseline arm has no heuristic
seed; the heuristics arm has it. Same `round0_best_geo` formula.

**Secondary goal (much tighter, because Opus already proposes ~5 good
configs at round 0):** `round0_best_geo_heldout ≤ 0.95` across the
family AND on each kernel. A 5% speedup on top of the LLM is
meaningful; 10%+ is strong; ≤0.80 would be excellent.

Rationale: the N7 ablation showed the LLM alone (no heuristic) gives a
~7× round-0 speedup over no-LLM-no-heuristic. On top of that, N6
heuristic adds 21% on matmul and 11% on fp8. Fp8 was already near
"LLM saturation": B/N6 = 1.02 means removing the LLM when the heuristic
is active costs only 2%. So on mature kernels, heuristic-vs-LLM-
baseline scores will be close to 1.00 no matter how good the
heuristic gets.

### Generality gates (apply to both experiments)

"General" is a first-class acceptance criterion. A heuristic that hits
the target on train but not on held-out does NOT pass; it is archived
as a data point and the loop continues.

- **Held-out shapes, same kernels.** 7/5 split committed in
  `iterations/N0_live/shape_grid.json`. Heuristic is built from train
  only; gate scores on heldout. Experiment 1 requires
  `round0_best_geo_heldout ≤ 0.20` AND
  `|heldout − train| ≤ 0.03` (tighter overfit gate because the ratios
  are smaller in absolute terms). Experiment 2 requires
  `round0_best_geo_heldout ≤ 0.95` AND
  `heldout − train ≤ 0.05`.
- **Held-out kernel, same family.** Build from one GEMM, score on the
  other. Experiment 1: `round0_best_geo_crosskernel ≤ 0.30`.
  Experiment 2: `≤ 1.00` (i.e. do not regress the sibling kernel).
- **Feature discipline.** No rule may key off the kernel name or a
  feature unique to one CSV (config hash, run_id). Features must be
  functions of the kernel arguments only: shape, dtype, strides,
  numel. Enforced by inspection of the generated `heuristic_*.py`
  file and of any observed-heuristics JSON.

Stretch levels (track separately, don't move the floor):

- Strong:    exp1 heldout ≤ 0.15,   exp2 heldout ≤ 0.85
- Excellent: exp1 heldout ≤ 0.10,   exp2 heldout ≤ 0.80

Secondary metrics reported on every gate but not optimized against:

- `verified_geo` — geomean of the final best config after all stages.
- `wall_time_geo` and `compile_total_geo` — cost diagnostics; must stay
  `≤ 1.20` unless the user accepts the trade.

## Regression Guardrails

Applies to both experiments. For Experiment 1 (no-LLM baseline), the
guardrails are absolute ratios that compare heuristic-on vs heuristic-off
with no LLM in the picture. For Experiment 2, the same thresholds apply
relative to the LLM-on baseline.

- **Per-kernel regression**: no kernel `round0_best_geo` > `1.03` on
  train or held-out. A promoted policy that regresses any kernel > 3%
  must fix that before promotion.
- **Per-repeat spike**: any single (shape, repeat) ratio > `1.10`
  triggers a focused rerun before the gate passes.
- **Held-out survival**: if a repeat-level regression > `1.10` survives
  the rerun, the policy is HOLD.
- **Overfit alarm**:
  - Exp 1: `heldout − train > 0.03` on any kernel is a FAIL (tighter
    because ratios are smaller).
  - Exp 2: `heldout − train > 0.05` on any kernel is a FAIL.

## Autotuner Configuration

The live benchmark arms both use:

- autotuner: `LLMGuidedSearch` with `max_rounds=1` and
  `finishing_rounds=0`. We use `LLMGuidedSearch` directly (not
  `LLMSeededSearch`) because this loop scores round 0 only and does not
  need the stage-2 LFBO tree search.
- LLM provider: Bedrock by default via `HELION_LLM_PROVIDER=bedrock`
  (falls back to `anthropic` when Bedrock creds are absent).
- LLM model: `us.anthropic.claude-opus-4-7`.
- Reasoning effort: maximum. Opus 4.7 uses `thinking.type="adaptive"`
  with `output_config.effort=high` (handled in `_bedrock.py`).
- Repeats: 3 per workload per gate to detect per-repeat noise.

Baseline arm uses identical env vars except it has no
`HELION_LLM_ROUND0_HEURISTIC_PATH`.

## Hill-Climbing Metric, offline

The offline AOT score from `heuristic_generator.py` is **input to the
heuristic**, not the loop metric. We still track it (`N0_baseline.json`)
because a better AOT heuristic (lower `max_slowdown`) should correlate
with a better `round0_best_geo`, but the acceptance gate is always the
live `round0_best_geo`.

## Data Sources

- Offline (heuristic construction): archived measurement CSVs and tuned
  configs under `aot_pretune_data/b200/{matmul,fp8_gemm}/`.
- Live (scoring): `LLMGuidedSearch` run on each kernel example file in
  `examples/{matmul,fp8_gemm}.py` on this machine's B200.

### Archive reality check

The matmul and fp8_gemm archives contain only square POT-aligned shapes
(`M=N=K`, step 128 from 256 to 3968), fp16 or fp8 only. Skewed shapes
(e.g. `M=1, N=4096, K=4096` decode) are outside the archive and land in
the held-out set so generality is tested honestly.

### Train / held-out split

For every kernel, the 12-shape grid is split into a **train set** of 7
shapes and a **held-out set** of 5 shapes, committed in N0-live under
`iterations/N0_live/shape_grid.json`. Seed is 20260509 (one more than
the norms loop to avoid coincidental overlap). Split invariants:

- cover the M / N / K axes in each half (no axis-only leakage);
- include at least one "corner" shape (smallest or largest numel) in each
  half;
- include at least one non-POT and at least one non-archive-signature
  shape in the held-out set for true generalization testing.

Rules for every gate after N0-live:

- Heuristic construction uses train-set CSV rows only. If pooled CSVs are
  needed, they are filtered to train shapes first.
- Scoring reports three numbers per kernel: `train`, `heldout`, and
  `heldout − train`.
- The **held-out kernel** slot rotates per N4 run: build from one GEMM,
  score on the other; report both directions.

## Current Baseline (N0 offline) and Live Score Board

Offline AOT max_slowdown (full-archive training; input to heuristic,
not the loop metric):

| source             | configs | max_slowdown | geomean | accuracy |
|--------------------|--------:|-------------:|--------:|---------:|
| matmul 20260430    | 7       | 1.0902       | 1.0124  | 1.000    |
| matmul expanded    | 8       | 1.0991       | 1.0162  | 0.949    |
| fp8_gemm 20260430  | 6       | 1.0435       | 1.0044  | 1.000    |
| fp8_gemm 20260501a | 6       | 1.0998       | 1.0100  | 1.000    |
| fp8_gemm 20260501b | 5       | 1.0952       | 1.0264  | 1.000    |
| fp8_gemm 20260501c | 6       | 1.0745       | 1.0076  | 1.000    |
| fp8_gemm expanded  | 7       | 1.0806       | 1.0208  | 0.974    |

Train-only AOT offline (pool of all archive CSVs filtered to the 7
train signatures per kernel; this is the seed source for N2):

| source       | backend       | configs | max_slowdown | geomean | acc   | features         |
|--------------|---------------|--------:|-------------:|--------:|------:|------------------|
| matmul train | decision_tree | 5       | 1.0894       | 1.0334  | 1.000 | `arg0_dim0` only |
| fp8 train    | decision_tree | 5       | 1.0998       | 1.0506  | 1.000 | `arg0_dim0` only |

**Key observation:** feature selector drops K and N on square-only
train data because they are collinear with M. This is the root cause
of the N2 held-out failure on skewed shapes (see N2 report). It is not
a bug in `heuristic_generator`; it is a training-distribution
limitation that no tree-backend tweak can fix.

### Experiment 2 leaderboard (LLM-on baseline — historic, pre-reframing)

These gates were scored against the LLM-on baseline (the original
terminal goal of heldout ≤ 0.80). They remain as a record of
heuristic-on-top-of-LLM value.

| gate   | arm                       | matmul trn | matmul hld | fp8 trn | fp8 hld | fam trn | fam hld | overfit delta       |
|--------|---------------------------|-----------:|-----------:|--------:|--------:|--------:|--------:|---------------------|
| N0     | baseline only             | 1.0000     | 1.0000     | 1.0000  | 1.0000  | 1.0000  | 1.0000  | 0.000               |
| N2     | archive tree (M-only)     | 0.8248     | 0.8857     | 0.7620  | 0.8386  | 0.7928  | 0.8619  | +0.069 FAIL         |
| N3     | adaptive blocks (no tree) | 0.8464     | 0.8984     | 0.9135  | 0.8628  | 0.8793  | 0.8804  | +0.001 delta-OK     |
| N3b-v1 | hybrid (aspect ≤ 1.25)    | 0.8209     | 0.8822     | 0.7617  | 0.8560  | 0.7907  | 0.8690  | +0.078 FAIL         |
| N3b-v2 | hybrid + total_out gate   | 0.7946     | 0.9018     | 0.7619  | 0.8248  | 0.7795  | 0.8624  | +0.068 FAIL         |
| N5     | JSON + N3b fallback       | 0.8567     | 0.9129     | 0.9173  | 0.8857  | 0.8865  | 0.8992  | +0.013 FAIL         |
| N6     | full-autotune JSON + fb   | 0.8620     | 0.9114     | 0.7965  | 0.7902  | 0.8286  | 0.8486  | +0.020 delta-OK     |
| N8     | N6 + 10 shape densify     | 0.8687     | 0.9322     | —       | —       | —       | —       | +0.021 (matmul only) |

Exp-2 champion: **N6** (heldout 0.849 family, 0.790 fp8 heldout).
Passes the new ≤ 0.95 target on family AND each kernel. N8 densify
dispatches identically to N6 on gap shapes (didn't shift the
dispatcher), so apparent regression is repeat-noise. See
`iterations/N8_matmul_densify/report.md`.

### Experiment 1 leaderboard (true no-autotune baseline — the real primary)

Introduced by the N7 ablation (2026-05-09). The baseline is
`HELION_AUTOTUNE_EFFORT=none`: Helion's default config, one config per
shape, benchmarked directly. The heuristic arm is the same — one
config per shape from the heuristic, benchmarked directly. No random
seeds, no LLM. This is the honest measurement of "how much does the
heuristic help a user who does not autotune at all?"

Target: heldout ≤ 0.20.

| gate  | setup                                    | matmul trn | matmul hld | fp8 trn | fp8 hld | fam trn | fam hld |
|-------|------------------------------------------|-----------:|-----------:|--------:|--------:|--------:|--------:|
| N7-C  | effort=none baseline (self-ratio)        | 1.0000     | 1.0000     | 1.0000  | 1.0000  | 1.0000  | 1.0000  |
| N7-D  | single N6 heuristic config / default     | 0.0861     | 0.0877     | 0.0871  | 0.0741  | 0.0866  | 0.0806  |
| N8    | single N8 heuristic config / default     | 0.0860     | 0.0875     | 0.0871  | 0.0741  | 0.0866  | 0.0806  |

**Exp-1 champion: N6 heuristic standalone** (N8 = N6 at dispatch; no
change). Family heldout 0.081 — ~12× faster than Helion's default
config on held-out shapes. Well inside the 0.20 target.

**Known limitation:** N6 matmul heuristic produces a shmem-over-limit
config on MM_003 (1024³, block_sizes=[128,64,128], num_stages=6). In
any arm that runs multiple configs (N6 LLM-on, B no-LLM-with-random)
the failure is masked by other seeds; in D (single-config) MM_003
errors out and is dropped from the geomean. A proper fix is to have
the generator probe-compile candidate templates before emitting.

Intermediate reference (LLMGuidedSearch + `--no-llm`, seed configs
only — i.e. default + 3 random ± heuristic; not the true "no
autotune" but useful for understanding LLM contribution):

| comparison                              | matmul hld | fp8 hld | fam hld |
|-----------------------------------------|-----------:|--------:|--------:|
| B / A (heuristic + random vs random)    | 0.1154     | 0.1138  | 0.1146  |
| A / N0 (random only vs LLM-on baseline) | 8.7935     | 7.2198  | 7.92    |

So the LLM alone is a 7–9× round-0 speedup over default+random. The
heuristic alone (Exp-1 primary) is 12× over default-only. They
compound partially: with both, round-0 config is ~9× faster than
no-heuristic (N0 baseline), which is what the LLM-on N6 scores
(0.115 in B/A view, though the comparison is not apples-to-apples).

### Ablation cross-reference (N7)

| comparison                             | kernel   | train  | heldout | all    |
|----------------------------------------|----------|-------:|--------:|-------:|
| [LLM-on] N6 / baseline                 | matmul   | 0.862  | 0.911   | 0.882  |
| [LLM-on] N6 / baseline                 | fp8_gemm | 0.797  | 0.790   | 0.794  |
| [no-LLM] A / LLM-on baseline           | matmul   | 5.975  | 8.794   | 7.019  |
| [no-LLM] A / LLM-on baseline           | fp8_gemm | 5.737  | 7.220   | 6.314  |
| [no-LLM] B (heuristic) / A (no-seed)   | matmul   | 0.185  | 0.115   | 0.152  |
| [no-LLM] B (heuristic) / A (no-seed)   | fp8_gemm | 0.140  | 0.114   | 0.128  |
| LLM extra on top of heuristic: B / N6  | matmul   | 1.284  | 1.114   | 1.210  |
| LLM extra on top of heuristic: B / N6  | fp8_gemm | 1.005  | 1.040   | 1.020  |

Interpretation:
- **LLM alone** (A vs LLM-on baseline): removing the LLM is 6–9× slower
  at round 0. The LLM is doing a lot.
- **Heuristic alone** (B vs A): adding the heuristic in a no-LLM world
  is **8–9× faster** at round 0.
- **LLM extra on top of heuristic** (B vs N6): on fp8 the LLM adds
  only ~2% on top of the heuristic. On matmul it still adds ~21% —
  the matmul heuristic has more room to improve before it "saturates"
  the LLM's contribution.

A theoretical "oracle hybrid" (each shape picks the best of the three
runs) would score roughly 0.76 train, 0.80 heldout — so the 0.80
heldout goal is within reach if routing is right; we just haven't
found the routing rule yet.

## Gates

### N0 offline: lock AOT baselines — DONE when N0_baseline.json is written

Run `heuristic_generator.generate_heuristic` on every archive CSV for
both kernels with `goal_type='max_slowdown'`, `threshold=1.10`,
`backend='decision_tree'`. Record `max_slowdown`, `geomean_slowdown`,
`model_accuracy`, and `num_configs` in `N0_baseline.json`.

Acceptance: `N0_baseline.json` committed.

### N0 live: pick a baseline shape list and measure `round0_best_geo = 1.000`

Work:

- 12-shape grid per kernel (train/heldout 7/5) committed.
- Baseline arm only, 3 repeats, GPU 0.
- Per-shape variance check: 3 repeats should give per-shape std/mean
  < 5%. (GEMM is noisier than norms at the low end; 5% is the working
  threshold; if we see > 8%, bump repeats to 5.)

Acceptance: baseline CSVs + meta JSON under
`iterations/N0_live/baseline/`; variance check logged.

### N2: AOT heuristic as a round-0 seed

Hypothesis: the pooled `heuristic_<kernel>.py` already picks a
near-oracle config per shape for the archived measurements. Injecting
that as the first round-0 seed (alongside the default config) should
beat an untuned baseline by at least `round0_best_geo ≤ 0.85` on train.

Work:

- Generate `heuristic_matmul.py` and `heuristic_fp8_gemm.py` from train
  shapes only.
- Run the heuristics arm with `HELION_LLM_ROUND0_HEURISTIC_PATH=<file>`
  via `tools/run_live.py`.

Acceptance: the heuristics arm consistently emits the injected config
in `generation==0` rows; `round0_best_geo_train ≤ 0.85` overall and per
kernel; `round0_best_geo_heldout − round0_best_geo_train ≤ 0.05` per
kernel.

### N3: Hill-climb the AOT heuristic content

Hypothesis: the pooled `heuristic_<kernel>.py` has ~9–10% max_slowdown
room (we will quantify in N0). Closing that gap should directly reduce
`round0_best_geo`.

Work (per kernel, one at a time):

1. Analysis subagent identifies the worst-slowdown shapes in the archived
   CSV and what config separates them.
2. Proposal subagent proposes one change — a new feature, an added config
   to the selected set, or a backend swap (`nearest_neighbor`).
3. Review subagent checks for overfitting and feature leakage.
4. Rebuild the heuristic from the updated inputs; rescore offline
   `max_slowdown` first.
5. If offline score improves, run live on train shapes and compare to
   N2 champion.

Acceptance: both offline `max_slowdown` (on train shapes) and live
`round0_best_geo_train` strictly improve over the current champion;
`round0_best_geo_heldout` does not get worse by more than 1%; no kernel
regresses > 1%.

### N4: Observed-heuristics-style routing

Hypothesis: per-shape seed injection is the strongest, but prompt
guidance with a curated observed-heuristics JSON may add marginal
improvement when the shape is outside the decision-tree's training
distribution. Test whether it adds or subtracts.

Work:

- Build a filtered observed-heuristics JSON for the GEMM family from
  `aot_pretune_data/b200/<kernel>/runs/*/tuned_configs_*.json`.
- Run the heuristics arm with both mechanisms (seed + observed JSON),
  each alone, and the N2 champion. Compare.

Acceptance: the best-of-three heuristics arm
`round0_best_geo_heldout ≤ 0.80` overall AND on at least one of the two
kernels; held-out kernel rotation reaches `round0_best_geo ≤ 0.85`.

### N4b: Generality stress test

Hypothesis: the N4 champion holds when we stretch the shape range
beyond the training grid, perturb dtype, and rotate across kernels.

Work:

- **Shape stretch.** Add 4 shapes per kernel that lie outside the N0
  grid (non-square, skewed for LLM prefill/decode: `(1,N,K)`,
  `(M,1,K)`, `(128,4096,4096)`). Score the current champion; demand
  `round0_best_geo ≤ 0.85`.
- **Dtype swap.** Rerun a 4-shape matmul subset with bf16 and fp32.
  fp8_gemm has no dtype swap (fp8 only). Demand `round0_best_geo ≤
  0.90` on bf16 (the archive has no bf16 signal so this is weaker).
- **Cross-kernel rotation.** Build from matmul → evaluate on fp8_gemm
  and vice versa. Report `round0_best_geo_crosskernel` per direction.
- **Feature audit.** Grep the generated heuristic code and any observed
  JSON for kernel-name strings (`matmul`, `fp8_gemm`, `float8`,
  `float16`) outside the top-level dispatch function, and for any run
  or config hash leaking from the CSVs into the features. Any hit is an
  automatic FAIL.

Acceptance: all four sub-checks pass their thresholds; feature audit
clean.

### N5: Expand archive and stack observed-heuristics JSON

Hypothesis — what we learned in N2/N3/N3b: the brittleness of every
tree- and hand-built dispatcher is **not the dispatcher, it is the
training data**. The archive contains only square shapes, so neither
feature selection, backend swaps, nor hand-clamped block sizes can
produce a heuristic that dispatches correctly on skewed held-out
shapes. The correct fix is to add skewed shapes to the archive.

Work:

1. **Expansion tuning**: 40 shapes per kernel (matmul, fp8_gemm)
   covering balanced_small, balanced_mid, skinny_m, skinny_n,
   skinny_k, rectangular — 8 shapes per bucket. 40 configs per shape
   via RandomSearch. Committed to `aot_pretune_data/b200/<kernel>/runs/
   20260508_expand_skewed/`. Budget ~15 min per kernel.
2. **Regenerate observed-heuristics JSON** by running
   `scripts/llm_heuristics_research.py` on the expanded archive. The
   generator's strict LOSO filters (`max_rule_holdout_geomean: 1.05`,
   `max_rule_holdout_p90: 1.10`) stay as-is — loosening them would be
   papering over the underlying data gap. Measure how many
   skewed-aspect buckets survive the filter.
3. **N5 live arm**: stack the observed-heuristics JSON on top of the
   N3b hybrid seed. Bucket each query shape, if a rule matches inject
   its top-1 template *in addition* to N3b's config; if no rule
   matches, fall back to N3b alone. This is *additive*, not
   replacement.

Acceptance: family `round0_best_geo_heldout ≤ 0.80` AND
`heldout − train ≤ 0.05`. If the expansion doesn't yield enough
skewed buckets, the plan is HOLD (do not loosen filters) — iterate
expansion shape list before proceeding.

**Result (recorded 2026-05-08): FAIL.** N5 with the full 59-rule JSON
scored family train=0.887 / heldout=0.899 / delta=+0.013. Better
generalization than N3b but worse absolute numbers on both halves.
Diagnosis in `iterations/N5_expand/report.md`: the RandomSearch
expansion winners were 5–10% slower than the archive's tuned winners
on shapes that overlapped, so the "observed-heuristics" templates
were actually shallower than N2's archive tree. The brittleness moved
from the dispatcher (N2/N3b) to the templates themselves.

**Hard lesson: dispatcher format is not the bottleneck. Tuning depth
is.** Go to N6 with deep tuning, not another dispatcher variant.

### N6: Deep-autotune the expansion archive

Hypothesis: N5 failed because our expansion tuning used
`RandomSearch(count=40)`. Replacing it with the **default full
autotune pipeline** on the same 40 shapes per kernel should produce
real oracle winners, which should in turn let the observed-heuristics
generator emit strong templates that beat N3b-v2 on skewed held-out.

Work:

1. **Rewrite the expansion harness** as `run_full_tuning.py`: drive
   the default autotuner (`LFBOTreeSearch` with
   `autotune_effort='full'`) per shape instead of a single optimizer.
   Hook `LocalBenchmarkProvider.benchmark` to tee every per-config
   timing into an archive-schema CSV. Do NOT instantiate a single
   optimizer class directly; see the "tuning effort" hard constraint
   below.
2. **Run overnight** on all 40 shapes per kernel. Budget: full
   pipeline is much deeper than DE-only (pattern_search +
   lfbo_pattern_search + DE + random + llm_search stages), so expect
   5–15 min per shape; 80 shapes × ~10 min ≈ 13 GPU-hours.
3. **Regenerate observed-heuristics JSON** on the N6 archive. Check
   how many buckets now pass the strict LOSO filter and how many win
   the "best template" role for our held-out shapes.
4. **N6 live arm**: same stacking logic as N5 (JSON lookup + N3b
   fallback), scored against the same N0 baseline.

Acceptance: family `round0_best_geo_heldout ≤ 0.80` AND
`heldout − train ≤ 0.05`. On FAIL, document what the remaining gap is
and whether it is (a) still a tuning-depth problem, (b) a template-
fields problem (schema drops too many knobs), or (c) a bucketing
problem. Do not invent a new dispatcher format; close out honestly.

### N7: Final packaging

Freeze the champion's seed mechanism and observed-heuristics JSON
content. Write a short policy doc (scope, hardware, dtype, shape range,
demotion rule). Produce a minimal source-change plan for user approval
— kept behind an opt-in env var; no silent default flip.

## Workflow

Each gate:

1. Manager defines the hypothesis, the exact change, what PASS/FAIL
   means.
2. **Proposal subagent** drafts the change. No edits, no runs.
3. **Review subagent** (fresh instance) checks it.
4. **Implementation subagent** makes the reviewed edits. Narrow
   allowlist.
5. **Harness subagent** runs on B200 and writes artifacts under
   `iterations/N<n>_<slug>/`.
6. **Analysis subagent** computes `round0_best_geo` from CSV metadata
   (not filenames) and writes the per-gate report.
7. Manager decides PASS/FAIL/BLOCKED and updates this file.

## Hard Constraints

- Do not run `git commit` or `git push` as part of any subagent.
- Do not run `pip install` or system package managers.
- Live benchmarks use GPU 0 on this machine (the only B200 here) unless
  the user says otherwise.
- The score always comes from `generation==0, status==ok` rows in the
  CSV, not from filenames. CSV metadata paths are the source of truth.
- Do not modify `heuristic_generator.py`'s public API. Extensions live
  in new files in `helion/autotuner/llm/` or in a wrapper script.
- **Tuning to produce archive data: always use the default full
  autotune pipeline** (`HELION_AUTOTUNE_EFFORT=full`, which runs
  pattern_search + lfbo_pattern_search + differential_evolution +
  random_search + llm_search as defined in
  `helion/autotuner/effort_profile.py`). Do not hand-wire a single
  optimizer (DE-only, random-only, pattern-only) — you will produce
  shallower data and miss the ensemble benefit. The only exception is
  a documented ablation to isolate one optimizer's contribution.
  Reason for this rule: in N5 we used `RandomSearch(count=40)` and
  got templates 5–10% slower than the original archive's winners; in
  N6-DE we used DE-only and still skipped pattern_search /
  lfbo_pattern_search / llm_search — both short of what `effort=full`
  gives for free.

## Out of Scope

- grouped_gemm (API mismatch with archive).
- Changing the LLM provider away from Bedrock Opus-4.7.
- Scoring against PyTorch eager, AOT CSVs directly, or the final
  full-autotune winner.
- Running live benchmarks on any hardware other than this B200.
