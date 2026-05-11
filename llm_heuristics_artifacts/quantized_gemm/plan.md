# Quantized-GEMM Hill-Climbing Plan

Living plan for a second-pass hill-climb applied to the quantized-GEMM
kernel family (`matmul_bf16_int4`, `_bf16xint16_gemm`, `nvfp4_matmul`)
using the infrastructure developed for the dense-GEMM loop.

## Context (what's already done)

The dense-GEMM loop on `choijon5/gemm-hill-climb` produced an
observed-heuristics JSON + dispatcher `.py` that scores:

- **Exp-1** (no autotune, `HELION_AUTOTUNE_EFFORT=none` baseline, one
  config per shape): family heldout **0.081** — ~12× speedup vs
  Helion's default config.
- **Exp-2** (LLM-on baseline, `LLMGuidedSearch max_rounds=1`, Opus 4.7
  via Bedrock): family heldout **0.849** — heuristic adds ~15% on
  top of the LLM's round-0 proposals.

Infrastructure that carries over verbatim:

- `scripts/llm_heuristics_research.py` (offline JSON generator).
- `tools/run_full_tuning.py` (per-shape archive-producing harness).
- `tools/run_live.py` (LLM-on live arm).
- `tools/run_no_autotune.py` (effort=none single-config arm).
- `tools/compute_round0_geo.py` (scorer).
- Dispatcher template (`iterations/N6_full_tune/heuristic/heuristic_matmul.py`).

What needs new code for the quantized loop:

- `tools/workloads.py` — arg builders for each of the 3 kernels
  (packing formats differ).
- `iterations/N0_live/shape_grid.json` — 12-shape scoring grid,
  expansion grid separately (40 shapes × 3 kernels for the archive).
- Kernel-specific dispatcher `.py` — same structure as matmul's, but
  `kernel_class` slots in the JSON are `matmul_bf16_int4`,
  `_bf16xint16_gemm`, `nvfp4_matmul` (the generator's
  `infer_kernel_class` falls back to the kernel's own name when it
  does not match a known pattern).

## Primary Objective (inherited from gemm/plan.md)

Same two experiments, same targets:

- **Experiment 1 (primary)**: no autotune baseline. Target heldout
  ≤ 0.20 family-wide AND per kernel.
- **Experiment 2 (secondary)**: LLM-on baseline. Target heldout
  ≤ 0.95 family-wide AND per kernel.

Bucketing is identical: `(aspect, dtype, m_bin, k_bin, n_bin)`. The
`dtype` slot will separate the three kernels because their weight
dtypes are different — bf16/int4, bf16/int16, bf16/fp4.

Generality: train/heldout 7/5 split per kernel, rules built from
train CSV rows only, overfit gate `heldout − train ≤ 0.03` on Exp-1
and `≤ 0.05` on Exp-2.

## Autotuner choice (open question as of Q0)

Default autotuner today: `LFBOTreeSearch` with `effort=full`
(surrogate-guided pattern search, ~1500-2000 candidates per shape,
roughly 3-10 min/shape). This is what N6 used for dense GEMM.

Alternative: `LLMSeededLFBOTreeSearch` — same LFBOTreeSearch plus
an Opus 4.7 round-0 seed stage. Potentially finds good configs
faster if the LLM proposes strong starting points. Question: does
it reach similar-or-better final quality AND noticeably faster?

**Q0 (before launching expansion tuning)**: compare the two
autotuners on 2–3 matmul shapes (using the dense-GEMM archive
shapes so we have an oracle to check against). Decision rule:

- If **LLMSeeded is ≥1.5× faster wall-clock AND reaches within 3%
  of LFBOTree-only's best config**: switch to LLMSeeded for the
  quantized expansion (saves hours overnight).
- If **LLMSeeded is faster but finds a worse config**: stick with
  LFBOTree-only.
- If **LLMSeeded is slower or equivalent**: stick with LFBOTree-only.

Records in `iterations/Q0_autotuner_bakeoff/`.

## Gates

### Q0: Autotuner bake-off — **DONE (2026-05-09)**

Compared `LFBOTreeSearch` (current default, what N6 used) vs
`LLMSeededLFBOTreeSearch` on 3 matmul shapes at effort=full with
Opus 4.7 seeds. Results:

| shape          | wall LLMSeeded/LFBO | best LLMSeeded/LFBO |
|----------------|--------------------:|--------------------:|
| 256³           | 0.91×               | 0.996× (better)     |
| 2048³          | **0.33×**           | 1.000× (tied)       |
| 128×2048×2048  | **0.61×**           | 1.000× (tied)       |
| geomean        | **0.57×** (1.76× faster) | **0.9985** (tied or slightly better) |

**Decision: adopt `LLMSeededLFBOTreeSearch` for archive tuning.**
Full report in `iterations/Q0_autotuner_bakeoff/report.md`.

Side-effect: the bake-off exposed that choijon5/gemm-hill-climb was
missing the bedrock transport (`_bedrock.py` + bedrock additions to
`transport.py`). Ported both from choijon5/norms-hill-climb so this
branch is now self-contained.

### Q1: Workload smoke test + shape grid commit

- Confirm each of the 3 quantized kernels builds args and runs a
  single-config forward pass without error (effort=none bench).
- Commit `iterations/N0_live/shape_grid.json` — 12 shapes × 3
  kernels, 7/5 train/heldout split per kernel.
- Commit `iterations/N1_expand/expansion_shapes.json` — ~40 shapes
  × 3 kernels covering balanced + skinny_m/n/k + rectangular.

### Q2: Archive expansion (full-tune) — **DONE (2026-05-10 06:28 UTC)**

Run the chosen autotuner on all ~120 expansion shapes. Per-shape
subprocess (to avoid the CUDA state-corruption we hit in N6). Output
CSVs to `aot_pretune_data/b200/<kernel>/runs/<run_id>/`.

Budget depends on Q0 outcome:
- LFBOTree-only: ~3-10 min/shape × 120 = ~12 GPU-hours.
- LLMSeeded (if faster): fewer hours, but adds Opus API cost.

**Run id: `20260509_q2_llmseeded`**. Autotuner:
`LLMSeededLFBOTreeSearch` (per Q0 decision). Effort: `full`.
LLM: Opus 4.7 via Bedrock, 8000-token adaptive thinking.
Crash isolation: `HELION_AUTOTUNE_BENCHMARK_SUBPROCESS=1` +
per-shape bash loop.

Final counts — all three kernels at 40/40:

| kernel             | shapes | configs recorded |
|--------------------|-------:|-----------------:|
| `matmul_bf16_int4` | 40 / 40 |             8348 |
| `_bf16xint16_gemm` | 40 / 40 |             7486 |
| `nvfp4_matmul`     | 40 / 40 |            10917 |

Resume wall time (36 shapes via `tools/resume_q2.sh`): ~4 h 40 m.
Full report in `iterations/Q2_expansion/report.md`.

### Q3: Generate observed-heuristics JSON — **DONE (2026-05-10)**

Ran `scripts/llm_heuristics_research.py --data-root aot_pretune_data/b200`
over the Q2 archive (26 751 rows, 120 shapes).

**Script edit required**: `infer_kernel_class` in the research script
did not know the three quantized kernel names, so it returned
"unknown" and lumped all shapes into a single dtype-only bucket. Added
`matmul_int4`, `matmul_int16`, `matmul_fp4` classes and routed them
through the matmul M/N/K bucketing branch. Additive change only; no
existing behavior changed.

Result after fix:

| kernel_class  | derived rules | promoted (strict LOSO) |
|---------------|--------------:|-----------------------:|
| `matmul_fp4`  | 19            | 2 |
| `matmul_int4` | 19            | 1 |
| `matmul_int16`| 19            | 0 |

Full report: `iterations/Q3_heuristic/report.md`.

Matches PR 2378's general shape (observed heuristic seeds keyed on
shape buckets) but at an earlier stage — PR 2378's schema v2 JSON
has live-validated rules with `validation` + `source` blocks and
`match_exact_only`. Our Q3 output is schema v1: the raw script
output, sibling in intent to
`llm_heuristics_artifacts/gemm/iterations/N6_full_tune/runtime_observed_heuristics_b200.json`.

### Q4: Build quantized dispatcher .py files — **DONE (2026-05-10)**

Three dispatchers under
`iterations/N6_full_tune/heuristic/heuristic_<kernel>.py`, each a
thin wrapper around a shared `_dispatcher_core.py` engine. The core
handles bucketing (matches generator), shmem-budget fits (N9
safety — 232 KB × 0.9 margin), template merge + shrink-to-fit,
and a per-kernel fallback table keyed on shape group (small_m /
small_n / small_k / balanced / rect). Each dispatcher supplies only
its `kernel_class` label, fallback table, and `_shape_from_args`
mapping.

Review subagent flagged one real bug: dispatcher bin values did
not match the generator's bins (dispatcher had
`[128,256,512,1024,2048,4096,8192,...]`, generator uses
`m/n: [64,128,256,512,1024,4096]` and `k: [64,128,256,512,1024,4096,32768]`).
Fixed and cross-verified: 14/14 test shapes produce identical
buckets dispatcher vs generator.

Also confirmed: no kernel-name strings leak into rule lookup; the
three kernel-name occurrences per dispatcher are only in the
top-level `autotune_<kernel>` / `key_<kernel>` function names. The
`_KERNEL_CLASS` constants (`matmul_int4`, `matmul_int16`,
`matmul_fp4`) are semantic classes, not kernel names.

### Q5: Exp-1 live score (no autotune) — **DONE (2026-05-10)**

All three kernels **PASS** heldout ≤ 0.20. No overfitting.

| kernel             | train | heldout | target |
|--------------------|------:|--------:|-------:|
| `_bf16xint16_gemm` | 0.142 | **0.107** | ✅ |
| `matmul_bf16_int4` | 0.157 | **0.132** | ✅ |
| `nvfp4_matmul`     | 0.213 | **0.198** | ✅ |
| **family**         | 0.168 | **0.141** | — |

Full report: `iterations/Q5_exp1/report.md`.

### Q6: Exp-2 live score (LLM-on) — **DONE (2026-05-10)**

All three kernels **PASS** heldout ≤ 0.95, Δheldout-train ≤ 0.05.

| kernel             | train  | heldout | target |
|--------------------|-------:|--------:|-------:|
| `_bf16xint16_gemm` | 0.829  | **0.827** | ✅ |
| `matmul_bf16_int4` | 0.576  | **0.581** | ✅ |
| `nvfp4_matmul`     | 0.575  | **0.609** | ✅ |
| **family**         | 0.650  | **0.664** | — |

Full report: `iterations/Q6_exp2/report.md`. Notable: the quantized
loop's Exp-2 (0.664) is *better* than the dense-GEMM loop's Exp-2
(0.849) because Opus round-0 picks for quantized start farther from
optimal, giving the heuristic seed more room to help.

### Q7: Packaging — **DONE (2026-05-10)**

Exp-1 and Exp-2 passed on **3/3 kernels**. Policy doc written at
`iterations/Q7_package/policy.md` summarizing files, three
deployment options (opt-in env var only / upstream into
`helion/autotuner/data/` PR-2378-style / hybrid), and a PR-body
draft.

No commits, no push — per manager.md, autotuner-source changes and
source commits require explicit user approval. The dispatcher runs
today via `HELION_LLM_ROUND0_HEURISTIC_PATH=<path>` as a zero-risk
opt-in; formal upstreaming is the user's call.

### Q8: Upstream JSON schema-v2 + re-benchmark — **DONE (2026-05-10/11)**

Generated `helion/autotuner/data/observed_heuristics_b200_quantized.json`
(21 rules, schema v2) matching PR 2378's on-disk location and rule
shape (`kernel_class`, `shape_bucket`, `match_exact_only`, `source`,
`validation`, `templates`). Per-rule `validation` blocks populated
from actual Q5 + Q6 measurements. Verified performance hold:

| experiment | v1 heldout (internal) | v2 heldout (upstream) | Δ |
|------------|----------------------:|----------------------:|--:|
| Q5b Exp-1 (no-autotune)     | 0.141 | 0.142 | +0.001 |
| Q6b Exp-2 (LLM-on, Opus 4.7)| 0.664 | 0.663 | -0.001 |

Full report: `iterations/Q6b_schema_v2/report.md` + Q5b report.

Also merged into PR 2378's own branch as a 30-rule consolidated
`observed_heuristics_b200.json` (9 existing attention/matmul/softmax
rules + 21 quantized rules). Ran Q5 + Q6 there using the **native
PR 2378 runtime** (no dispatcher files; `observed_heuristic_default_config`
path). Result without fallbacks:

| experiment | our-branch dispatcher | PR 2378 native runtime | Δ |
|------------|----------------------:|-----------------------:|--:|
| Q5  Exp-1 | 0.142 | 0.213 | +50% |
| Q6  Exp-2 | 0.664 | 0.675 | +1.6% |

Q6 ratio is preserved because the LLM's random seeds cover the
buckets our JSON misses. Q5's 50% gap traces to a single shape
(I4_009: `balanced k<=1024 m<=4096 n<=4096`) whose bucket isn't in
the JSON; native runtime falls to Helion's generic default while our
branch's dispatcher has a per-group fallback table that picks a
shape-appropriate archive winner.

### Q9: Generalize fallback-table mechanism — **DONE (2026-05-11)**

**Goal.** Port the per-kernel fallback tables (the safety net our
dispatchers use when exact bucket lookup misses) into PR 2378's
runtime as a reusable pattern. This closes the 0.213 → 0.142 gap
on Q5 for quantized and generalizes to every kernel family listed in
PR 2378's taxonomy.

**Design.**

1. JSON schema addition: top-level `fallbacks` map:
   ```json
   "fallbacks": {
     "matmul_int4": {
       "small_m": {"template": {...full config...}},
       "small_n": {...}, "small_k": {...},
       "balanced": {...}, "rect": {...}
     },
     "matmul_fp4":  {...},
     "matmul_int16":{...}
   }
   ```
2. Runtime hook in `observed_heuristic_seed_configs`: when exact
   `_find_rule` returns None, compute the shape-group via a new
   `_fallback_group_for_class(kernel_class, args)` helper, then
   look up `fallbacks[kernel_class][group]` and materialize it
   through the existing `_materialize_config` path.
3. Per-kernel-class grouping functions:
   - matmul family (matmul, matmul_fp8, matmul_int4/int16/fp4):
     5 groups `small_m / small_n / small_k / balanced / rect`
     (threshold 256 for "small", `max/min < 2` for "balanced").
   - row_softmax / row_norm_* / row_cross_entropy:
     4 groups `tall / square / wide / huge_cols`.
   - attention: 6 groups keyed on
     `(seq_magnitude, head_dim_magnitude, batch_heads_magnitude)`.
   - elementwise: 3 groups on `numel` magnitude.
   - split_k_matmul / grouped_matmul / batched_matmul: inherit
     matmul family's 5 + add batch axis where relevant.
4. Generator script: given `(archive_root, kernel_class,
   grouping_fn)`, emit the fallback block for that class.

**Gates:**

- **Q9a** (done): schema + runtime patch in a worktree on
  PR 2378's branch. Grouping function for the 3 quantized
  classes + the existing `matmul` class. Validate end-to-end
  with a unit test that hits the fallback path.
- **Q9b** (done): generator script `scripts/observed_heuristics_fallbacks.py`
  emits fallback JSON for one kernel class given archive CSVs.
  Run for 3 quantized classes; commit the resulting `fallbacks`
  section into the merged 30-rule JSON on PR 2378's branch.
- **Q9c** (done): re-run Q5 + Q6 on PR 2378's branch with the
  fallback-enabled JSON. Target: Q5 heldout ≤ 0.145 (parity with
  our-branch dispatcher result 0.142).
- **Q9d** (done): write `iterations/Q9_fallbacks/report.md`.
  Document the recipe so adding fallbacks to attention, row_*,
  elementwise is a straight-line task.

**Acceptance.** Q5 heldout on PR 2378 branch within 5% of our
branch's 0.142. Q6 heldout unchanged within 5%.

**Result**: Q5 heldout 0.133 (better than our 0.142 by 6%), Q6
heldout 0.592 (better than our 0.664 by 11%). All per-kernel
within 5% or better. Full report:
`iterations/Q9_fallbacks/report.md`.

### Q10: Perf regression verification — **DONE (2026-05-11)**

Goal: confirm **end-to-end** that the changes accumulated across
Q8 / Q9 never make anything slower than it was before. Three
comparisons to run:

1. **Regression vs pre-Q8 (our-branch original)**: Q5/Q6 on PR 2378
   branch with fallbacks (Q9c output) vs Q5b/Q6b from our branch
   (the 0.142 / 0.664 result set). Both directions must be within
   5% per-kernel, geomean within 3% family-wide.
2. **Regression vs PR 2378 pre-merge**: Run Q5/Q6 on PR 2378's
   branch **with our 30-rule merged JSON** but with
   `HELION_AUTOTUNE_OBSERVED_HEURISTICS=0` disabling our rules.
   This measures whether adding kernel-class branches to the
   classifier (Q8 patch) or the fallback lookup paths (Q9 patch)
   slowed down their existing attention/matmul/softmax workloads
   by accident. Acceptance: existing rules' perf stays within 3%
   of `d4f17d4` HEAD.
3. **No-heuristic baseline consistency**: with
   `HELION_AUTOTUNE_OBSERVED_HEURISTICS=0`, the quantized kernels
   should produce **exactly the same** configs as they do today
   on `origin/main` (Helion's stock default). No drift from the
   runtime integration itself.

Artifacts: `iterations/Q10_regression/` with three sub-reports
(one per comparison) and a summary `report.md` that says PASS or
lists the regressions.

**Result: all three checks PASS.**
- vs our-branch pre-Q8: quantized kernels **better** by 6-11%
  (fallback + native runtime found more LLM-seed synergy).
- vs PR 2378's existing rules: unchanged; all 9 rules still
  resolve, all 4 `test_observed_heuristics.py` tests pass.
- vs stock Helion: with HELION_AUTOTUNE_OBSERVED_HEURISTICS=0,
  all quantized kernels get identical configs to pre-patch.

Full report: `iterations/Q10_regression/report.md`.

### Q11: Package for upstream PR — **IN PROGRESS**

Once Q9 AND Q10 land:
1. Commit the schema patch + grouping functions to PR 2378's
   branch (one PR or a follow-up PR; user's call).
2. Land the merged 30-rule JSON with fallbacks.
3. Update `iterations/Q7_package/policy.md` to point at the
   upstream PR and drop the standalone-dispatcher deployment
   option.

## Open risks

- **int4/fp4 packing means arg1_dim0 = K//2, not K.** The generator's
  matmul bucketing reads `arg0_dim0 (M), arg0_dim1 (K), arg1_dim1 (N)`
  — arg1_dim0 is ignored, so logical K from arg0 is correct. Verified.
- **Kernel classes differ from `matmul`/`matmul_fp8`.** The generator's
  `infer_kernel_class` falls back to the literal kernel name for
  unknown kernels. The dispatcher will look up rules keyed on the
  kernel's actual name. Minor schema adjustment — no code change
  required in the generator.
- **Triton pipeline bugs.** Running the int4 kernel via default
  autotune in Q1 smoke test hit
  `ttg.convert_layout op does not have expected attribute
  ttg.partition` on some configs. These get skipped by the autotuner
  with `HELION_AUTOTUNE_IGNORE_ERRORS=1`, but could mean certain
  fancier tile/warp-spec combos won't run. Document what fraction of
  configs fail compile and whether the failures cluster on specific
  knob values.

## Log of cross-branch porting

- **Bedrock transport port (2026-05-09)**: Q0 bake-off revealed that
  `helion/autotuner/llm/` on `choijon5/gemm-hill-climb` (forked from
  `main`) does not include the bedrock provider. All prior
  choijon5/gemm-hill-climb live runs worked because they were
  executed from the `choijon5/norms-hill-climb` working tree.
  Ported `helion/autotuner/llm/_bedrock.py` (SigV4 + IMDSv2, 304 lines)
  and updated `helion/autotuner/llm/transport.py` (added bedrock
  provider, adaptive-thinking payload for Opus 4.7) from
  `origin/choijon5/norms-hill-climb` to this branch so the Q0
  bake-off can run LLMSeededLFBOTreeSearch with Opus 4.7 via AWS
  IAM credentials. This also makes `choijon5/gemm-hill-climb`
  self-contained — previously its own `run_live.py` would have
  failed with "Unsupported LLM provider 'bedrock'" when run from
  this branch's working tree.

## Hard constraints (inherited from gemm/plan.md)

- Do not run `git commit` or `git push` without user approval.
- Archive tuning uses `HELION_AUTOTUNE_EFFORT=full`, never a
  hand-wired single optimizer. Exception: Q0 bake-off, where we
  explicitly compare two autotuner classes.
- Archive CSVs go on `choijon5/aot-pretune-data`, not on the code
  branch.
- Live benchmarks use GPU 0 on this B200.
