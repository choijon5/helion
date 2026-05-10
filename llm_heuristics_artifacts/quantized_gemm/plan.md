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
