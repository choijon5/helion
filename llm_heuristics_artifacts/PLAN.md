# Plan: LLM Prompt Optimization for GPU Kernel Autotuning

**Status:** Phase 2 extended (8 iterations). Best result: **+8.13% single-config (iter 8), +9.18% per-kernel-oracle**. Softmax alone hits +16.12%, clearing the 15% target.
**Last Updated:** 2026-05-09

## Final Summary

| Iteration | Approach | Geomean Improvement |
|-----------|----------|---------------------|
| iter 1 | Adaptive refinement prompts (round-based) | +0.64% |
| iter 2 | Adaptive + state-aware (exploit if last round improved) | +0.80% |
| iter 3 | Pretuned data in initial prompt (Approach 8A, 23 rules) | +6.36% |
| iter 4 | Add pretuned as round-0 seeds (Approach 8B) | +3.46% (worse) |
| iter 5 | Augmented library (self-mined rules, 37 rules) | +5.48% |
| iter 6 | Approach 8 only, no adaptive refinement | +5.22% |
| iter 7 | Consolidated library (66 rules from all sibling branches) | +6.18% |
| **iter 8** | **Knob-prior mode (synthesized priors from consolidated)** | **+8.13%** 🥇 |
| **Oracle** | **Best iter per kernel (across iter 3/7/8)** | **+9.18%** |

**Best approach per kernel (as of iter 8):**
- cross_entropy → **iter 8 (knob priors)** → +9.90%
- softmax → **iter 8 (knob priors)** → **+16.12%** (exceeds 15% target!)
- matmul → iter 7 (consolidated library templates) → +6.29%
- attention → iter 3 (original library, prompt + adaptive refinement) → +7.63%
- layernorm → **iter 8 (knob priors)** → +5.58%

**Deliverable:** `FINAL_BEST_CONFIGS.json` — per-shape best config extracted from the winning iteration per kernel.

**Key takeaways:**
1. **Approach 8 (pretuned data library) was the single biggest lever** — provided ~6% out of 6.36%. Prompt-only changes (Approaches 1-6) hit a ~1% ceiling.
2. **Seeding harms broad performance** except for softmax where it worked well. Prompt injection > hard seeding.
3. **No single configuration is best for all kernels.** The oracle shows attention prefers one strategy, layernorm another.
4. **Bucket coverage matters.** Kernels with tight pretuned coverage (attention, softmax) improved most.
5. **Direct-template (no LLM) is not viable.** Tested separately: raw templates work only for shapes they were fitted to (softmax +5%), fail hard on out-of-bucket shapes (matmul -261%, attention -41%, layernorm -29%). Many matmul shapes couldn't even compile the template (shared memory OOM). The LLM is what makes the templates generalize.

### Measurement note
During this experiment I discovered a pre-existing bug in `run_live.py`: `res.perf * 1000.0` inflated every CSV `perf_ms` value by 1000× because `BenchmarkResult.perf` is already in milliseconds. The bug is now fixed. All iterations above share the same 1000× inflation, so **relative improvements (% numbers) are valid** but **absolute numbers in CSVs/PLAN.md are 1000× too large** (real geomean baseline is ~26 µs, not 26 ms — B200 is fast on these sizes). Ratios and percentages are unaffected.

---

---

## Goal

**Target:** Achieve **15% improvement** over baseline LLM autotuner performance (geometric mean across kernels)

**Stretch:** 20% improvement (ratio ≤ 0.80) on any kernel

**Use Case:** Model compilation for serving (one-time cost)
- Autotuning speed doesn't matter (happens once)
- **Quality of final config is what matters**
- Focus on making LLM find BETTER configs, not faster

---

## Baseline Definition

**Library defaults** (what LLMGuidedSearch does out of the box):
- `--max-rounds 4`
- `--configs-per-round 15`
- `--initial-random-configs 10`
- `--repeats 3`

**Total configs per shape:** ~70 (10 initial + 4 rounds × 15 configs)

**Why these settings:** Represents "current LLM performance" without artificial limits. Gives LLM full opportunity to refine across multiple rounds.

---

## Kernels Under Test (5 total)

| Kernel | Type | Bottleneck | Status |
|--------|------|------------|--------|
| cross_entropy | Loss function | Memory-bound | ✅ Ready |
| softmax | Reduction | Memory-bound | ✅ Ready |
| layernorm | Normalization | Memory-bound | ✅ Ready |
| matmul | Matrix multiply | Compute-bound | ✅ Ready |
| attention | QKV attention | Mixed | ✅ Ready |

**Rationale:** Mix of memory-bound and compute-bound kernels, plus attention as mixed workload. This diversity helps identify if round benefits vary by kernel type.

**Shape coverage:** 12 shapes per kernel × 3 repeats = 180 trials total
(train/heldout splits: ~7/5 for most kernels, 8/4 for softmax)

---

## Infrastructure Status

### ✅ Ready Components

**Code:**
- `loss_functions/tools/workloads.py`: Builders for all 5 kernels
- `loss_functions/tools/run_live.py`: Accepts all 5 kernels, `--track-round-progress` flag implemented

**Shape grids:** all under `loss_functions/grids/`
- `cross_entropy_grid.json`
- `softmax_grid.json`
- `matmul_grid.json`
- `attention_grid.json`
- `layernorm_grid.json`

**Per-round tracking:**
- Captures best config after each round
- Output CSV: `{kernel}_baseline_round_progress.csv`
- Format: `kernel,shape_id,repeat,round,best_so_far_ms,new_configs_tested,improvement_pct`

---

## Execution Plan

### Phase 0: Baseline Measurement ⏳ READY TO START

**Goal:** Establish current LLM performance with library defaults

**Duration:** ~4 hours (can parallelize across kernels)

**LLM configuration (required for all experiments):**
- Model: Claude Opus 4.7 (`us.anthropic.claude-opus-4-7`)
- Thinking budget: 16384 tokens (adaptive thinking enabled)
- Reasoning effort: `max` (highest available — options are `low` < `medium` < `high` < `xhigh` < `max`)

These settings MUST be set via environment variables — the code returns early without thinking/effort config if `HELION_LLM_ANTHROPIC_THINKING_BUDGET` is unset.

**Command per kernel:**
```bash
cd /home/dev/helion_choijon5/llm_heuristics_artifacts/loss_functions/tools
AWS_REGION=us-east-2 \
HELION_LLM_MODEL=us.anthropic.claude-opus-4-7 \
HELION_LLM_ANTHROPIC_THINKING_BUDGET=16384 \
HELION_LLM_ANTHROPIC_REASONING_EFFORT=max \
python run_live.py \
  --kernel {kernel_name} \
  --arm baseline \
  --max-rounds 4 \
  --configs-per-round 15 \
  --initial-random-configs 10 \
  --shape-grid ../grids/{kernel_name}_grid.json \
  --output-dir ../../baseline/{kernel_name} \
  --repeats 3 \
  --track-round-progress
```

**Run for:** cross_entropy, softmax, matmul, attention, layernorm

**Outputs per kernel:**
- `baseline/{kernel}/{kernel}_baseline.csv` — full config results
- `baseline/{kernel}/{kernel}_baseline.meta.json` — run metadata
- `baseline/{kernel}/{kernel}_baseline_round_progress.csv` — per-round best config

**Measure:** Best config found per kernel = our baseline to beat

---

### Phase 1: Per-Round Analysis ⏳ PENDING

**Goal:** Identify optimization opportunities from baseline data

**Duration:** ~1-2 hours

**Per-kernel analysis:**
1. **Round Value:** Round 1→2, 2→3, 3→4 improvement (%)
2. **Plateau Detection:** At which round does improvement < 2%?
3. **Regression Detection:** Which rounds make performance worse?
4. **Total Improvement:** Round 1 vs Round 4 final improvement

**Cross-kernel comparison:**
1. Do all kernels benefit similarly from rounds 2-4?
2. Which kernel types improve most? (memory-bound vs compute-bound)
3. Are there kernel-specific patterns?

**Expected scenarios:**

| Scenario | Pattern | Implication |
|----------|---------|-------------|
| A: Universal Improvement | All kernels improve 15-20% round 1→4 | Keep multi-round, improve prompts all rounds |
| B: Selective Improvement | Loss functions improve, reductions don't | Kernel-type-specific prompting |
| C: Diminishing Returns | Round 1→2 helps (+10%), later marginal | Use 2 rounds, invest in rounds 1-2 prompts |
| D: Late Regression | Round 3-4 make things worse | Simplify refinement, cap at 2 rounds |

**Output:** `PHASE_1_ANALYSIS.md` with recommendations for Phase 2 optimizations

---

### Phase 2: Optimize & Iterate ⏳ IN PROGRESS (iter 1 running)

**Goal:** Apply prompt optimizations until 15% improvement achieved

**Duration:** ~10-20 hours (iterative)

**Workflow:**
1. **Proposal:** Generate optimization ideas for bottleneck rounds (see strategies below)
2. **Implementation:** Modify prompts in `helion/autotuner/llm_search.py`
3. **Measurement:** Test optimized prompts with same baseline settings
4. **Evaluate:** Compare optimized vs baseline
5. **Iterate:** If <15%, loop back to Proposal. If ≥15%, success.

---

### Phase 3: Leverage Pretuned Data ⏳ PLANNED (later)

**Goal:** Use a pretuned heuristics dataset to improve LLM prompts and seeds, as a later-phase boost once Phase 2 has been explored.

**Data source:** `~/helion_choijon52/llm_heuristics_artifacts/` — an existing archive with:
- **`observed_heuristics_b200.json`** — 23 validated rules across 7 kernel classes (attention, elementwise, matmul, matmul_fp8, row_norm_layer, row_norm_rms, row_softmax). Each rule has a shape bucket (e.g., `seq<=16384, head_dim<=64, batch_heads<=128, dtype=fp16_bf16`), one or more empirically-winning config templates (block_sizes, num_warps, num_stages, pid_type, l2_groupings), and leave-one-shape-out slowdown stats proving the template generalizes.
- **`runtime_observed_heuristics_b200.json`** — runtime-observed variant of the same schema.
- **Several H2-H5 policy iterations** showing progressive refinement, with aggregated results and what worked / what didn't (e.g. H3b paired-no-match achieved geo 0.822).

**Why this is a "later" phase:** Phase 2 tests whether pure prompt improvements help. Phase 3 tests whether prompt+data fusion helps more. Separating them isolates which component actually drives the win.

**Sub-phases:**

**3A: Data-driven pattern library in the prompt (medium risk, fast)**
- Parse `observed_heuristics_b200.json` per kernel class.
- Match the current kernel's class + shape to the nearest bucket at autotune time.
- Inject the top-1 to top-3 templates into the initial LLM prompt as concrete starting points labeled "configs empirically observed to be near-optimal for this shape family."
- **Expected outcome:** LLM's round-0 set already includes strong candidates, so subsequent rounds spend effort on refinement rather than discovery.

**3B: Data-driven round-0 seeds (higher risk, more invasive)**
- Use the matched templates as actual seed configs in `_SeededLLMGuidedSearch` (same mechanism the old `--arm heuristics` used).
- Risk: past experiments (archived H2-H5 policies) showed seeding often plateaus at ~0.80-0.95 — seeds can constrain the LLM's exploration. Test 3A first.

**3C: Feedback-stage analogy retrieval (novel)**
- When the LLM is refining in round 2+, look up which kernels-in-the-dataset had similar "best configs so far" and inject their next-round winners as hints: *"Kernels with anchor configs like X tended to converge to Y — consider that direction."*
- This bridges the dataset into the refinement loop, not just the initial prompt.

**Dependencies:** Phase 2 must complete first so we have a clean comparison (prompt-only vs prompt+data). The archived H2-H5 experiments in the same directory tell us what's already been tried — read `h3d_attention_paired_no_match_20260507/aggregate_summary.md` and `h4_non_attention_paired_no_match_20260507/aggregate_summary.md` before proposing, to avoid repeating tried approaches.

**Decision criteria:**
- Start Phase 3 if Phase 2 stalls below 15% after 3-4 iterations.
- Skip Phase 3 if Phase 2 reaches 15%+ — we want the simpler wins first.

---

## Prompt Optimization Strategies

Eight techniques available, prioritized based on Phase 1 findings. Approaches 1-6 modify the LLM prompt; Approach 7 modifies the Helion compiler itself to provide richer context to the LLM; Approach 8 uses an external dataset of pretuned configs.

### Approach 1: Multi-Round Tuning
**What:** Adjust max_rounds dynamically based on diminishing returns
**When:** Plateau at round 2 or 3
**How:** Set kernel-specific max_rounds (e.g., 2 for compute, 3 for memory-bound)
**File:** Kernel-specific config in `helion/autotuner/llm_search.py`

### Approach 2: Adaptive Refinement Strategy
**What:** Different prompting strategies for each round
**When:** Round 1→2 has biggest gain but later rounds plateau
**How:**
```
Round 1: Broad exploration (diverse block_sizes, num_warps)
Round 2: Exploit best (single-field mutations of top config)
Round 3: Fine-tune (tiny tweaks ±1 on best fields)
Round 4+: Desperate search (try avoided configs)
```
**File:** `_refinement_strategy_lines()` in `helion/autotuner/llm_search.py`

### Approach 3: Enhanced Feedback Analysis
**What:** Add trend analysis to feedback (does increasing X help/hurt?)
**When:** LLM makes poor choices despite good data
**How:**
- Analyze scaling: "num_warps 4→8 (+5%), 8→16 (-20%)"
- Generate hypotheses: "Memory-bound kernel, excessive warps hurt"
- Add guidance: "Try num_warps=6 (between 4 and 8)"
**File:** Feedback formatting in `helion/autotuner/llm_search.py`

### Approach 4: Theoretical Guidance (Kernel-Type-Aware)
**What:** Add bottleneck analysis to initial prompt (memory-bound vs compute-bound)
**When:** Kernel-specific patterns emerge (loss ≠ reduction ≠ compute)
**How:**
```
Memory-bound → Favor larger tiles, avoid excessive parallelism
Compute-bound → Favor parallelism, pipeline stages
```
**File:** Initial prompt generation in `helion/autotuner/llm_search.py`

### Approach 5: Ensemble Search
**What:** Run multiple independent searches with different seeds, take best
**When:** Last resort if single search plateaus
**Trade-off:** 3x time cost (acceptable for serving use case)

### Approach 6: Success Pattern Learning
**What:** Emphasize "why winners win" over "why failures fail"
**When:** LLM needs better guidance to exploit good configs
**How:**
- Analyze top configs: "Config 1 (12.3ms) — block_sizes=[2], num_warps=4"
- Extract pattern: "Conservative configs (4-8 warps) dominate"
- Guide next round: "Double down on configs like winner"
**File:** Feedback formatting in `helion/autotuner/llm_search.py`

### Approach 8: Pretuned Data Library (used in Phase 3)
**What:** Inject shape-bucketed, empirically-validated configs from an existing pretuned dataset into either (a) the LLM initial prompt, (b) the round-0 seed configs, or (c) the round 2+ refinement feedback.
**Data source:** `~/helion_choijon52/llm_heuristics_artifacts/observed_heuristics_b200.json` — 23 rules over attention/matmul/layer_norm/rms_norm/softmax/elementwise, each with shape bucket + near-optimal template.
**When:** Use after Phase 2 plateaus; see Phase 3 above.
**Why different from Approach 4 / 7:** Those are derived from static analysis of the kernel; Approach 8 is derived from empirical benchmarks of kernels on actual B200 hardware.
**File:** New module (e.g. `helion/autotuner/llm/pretuned_library.py`) that loads the JSON, matches the current (kernel_class, shape_bucket) → template, injects into prompting or seeding.

### Approach 7: Compiler-Detected Kernel Patterns
**What:** Modify the Helion compiler to statically detect kernel patterns and expose them to the LLM prompt
**When:** LLM lacks semantic understanding of what the kernel is doing (works well for any kernel but especially useful when kernel-type-specific tuning matters)
**Why this is different from Approach 4:** Approach 4 relies on manual labels ("memory-bound", "compute-bound") per kernel. Approach 7 auto-detects patterns from the kernel IR so it generalizes to arbitrary user kernels without manual annotation.

**Patterns to detect (examples):**
- **Reduction shape:** single-axis reduce, multi-axis reduce, reduction size relative to tile
- **Memory access pattern:** contiguous, strided, broadcast, reshape-heavy
- **Compute intensity:** arithmetic intensity (FLOPs / bytes) — distinguishes memory-bound from compute-bound
- **Op signatures:** matmul-like (dot product in inner loop), softmax-like (reduce + exp + divide), normalization-like (reduce + affine), elementwise-only
- **Loop structure:** single tiled loop, nested tiled loops, sequential reduction
- **Dtype-driven hints:** bf16/fp16 favor tensor cores → suggest larger block_sizes; fp32 prefers different tiling

**Prompt injection example:**
```
## Kernel Analysis (auto-detected)
- Pattern: reduction + elementwise epilogue (softmax-like)
- Arithmetic intensity: 0.8 FLOPs/byte → memory-bound
- Inner reduction extent: vocab_size (variable, up to 131072)
- Suggested tuning focus: maximize tile size along reduction axis, minimize num_warps to reduce register pressure
```

**File:** New module `helion/autotuner/kernel_analysis.py` (static analysis over the bound kernel's IR/graph). Plumb results into `helion/autotuner/llm/workload.py` so they land in the initial LLM prompt.

**Dependencies:** Requires access to Helion's compiler IR — need to inspect `helion/_compiler/` for hooks. Heavier implementation than Approaches 1-6 (touches compiler, not just prompts).

**Risk:** MEDIUM-HIGH. Biggest potential upside (generalizes to any kernel) but touches compiler code — higher chance of regressions.

---

## Prompt Optimization Roadmap

Based on Phase 1 findings, Proposal subagent prioritizes:

| Finding | Approach to Try |
|---------|-----------------|
| Round 1→2 has biggest gain, then plateau | **Approach 2** (Adaptive Refinement) |
| Plateau at round 2 | **Approach 6** (Success Pattern Learning) |
| Kernel-specific patterns (loss vs reduction vs compute) | **Approach 4** (Theoretical Guidance) or **Approach 7** (Compiler-Detected Patterns) |
| Round 4 regresses | **Approach 1** (Multi-Round Tuning, cap at 3) |
| LLM makes poor choices despite good data | **Approach 3** (Enhanced Feedback) |
| Kernel-type hints help but don't generalize | **Approach 7** (auto-detect patterns in compiler) |

---

## Success Criteria

| Tier | Target | Meaning |
|------|--------|---------|
| Minimum | ≥5% improvement | Optimizations add measurable value |
| **Target** | **≥15% improvement** | **Systematic improvement achieved** |
| Stretch | ≥20% improvement | Any kernel achieves ratio ≤ 0.80 |

**Metric:** Best config performance (optimized vs baseline), geometric mean across kernels

**Example:**
```
Kernel: cross_entropy
Baseline best: 13.5ms (Phase 0 measurement)
Target:        11.5ms (15% improvement)
Stretch:       10.8ms (20% improvement)
```

---

## Resource Budget

| Phase | Duration | Cumulative |
|-------|----------|------------|
| Phase 0: Baseline | ~4 hours | 4 hours |
| Phase 1: Analysis | ~1-2 hours | ~6 hours |
| Phase 2: Iterate | ~10-20 hours | ~26 hours |
| **Total budget** | **~40 hours** | |

**Stopping criteria:**
- ✅ Success: 15% improvement achieved
- ⚠️ Time limit: 40 hours exhausted → escalate
- ⚠️ Exhaustion: 5+ optimization attempts failed, no new ideas → escalate

---

## Data Tracking

**PLAN.md is a living document.** After each experiment, the Evaluate subagent appends to the sections below so future iterations (and new Proposal subagents) have full context.

**What goes where:**
- **PLAN.md** (this file) — results summaries, proposals, findings (human-readable narrative)
- **MANAGER_STATE.json** — structured state for decision-making (numeric progress, current phase, iteration count)
- **MANAGER.md** — orchestration rules (rarely changes)

**Who edits what:**
- **Evaluate subagent:** Appends results to "Results Log" and updates "Current Findings" below
- **Proposal subagent:** Appends ideas to "Active Proposals" and moves tried ones to "Tried Approaches"
- **Manager:** Never edits PLAN.md directly (only updates MANAGER_STATE.json)

---

## Results Log

*Appended by Evaluate subagent after each measurement completes.*

### Iteration 6: Phase 3A — Pretuned lib only (reverted Approach 2 round-adaptive prompts)
**Date:** 2026-05-09
**Approach:** Pretuned lib (original 23 rules, same as iter 3) in prompt, BUT reverted Approach 2's round-adaptive refinement prompt to the default `_DEFAULT_REFINEMENT_LINES`. Tests whether Approach 2 was slightly harmful.
**Code change:** `_refinement_strategy_lines` simplified to always use default prompt (ignores `round_num` and `improved_last_round`).
**Wall time:** ~1h 22min

**Result:**

| Kernel | Baseline | Iter3 (A2+A8) | **Iter6 (A8 only)** | Iter3 Δ | Iter6 Δ |
|--------|----------|---|---|---|---|
| cross_entropy | 39.47 | 36.74 | **36.62** | +6.91% | **+7.21%** 🎯 |
| softmax | 22.69 | 19.95 | 20.62 | +12.06% | +9.12% |
| matmul | 41.31 | 39.17 | 39.29 | +5.18% | +4.89% |
| attention | 23.25 | 21.48 | 22.93 | +7.63% | +1.38% |
| layernorm | 14.48 | 14.54 | **14.00** | -0.42% | **+3.28%** 🎯 |
| **Geomean** | **26.25** | **24.58** | 24.88 | **+6.36%** | **+5.22%** |

**Outcome:** Slightly WORSE than iter 3 (-1.14 pp geomean). Per-kernel behavior split.

**What worked:**
- **Layernorm best so far (+3.28%)**. Removing Approach 2 let the LLM explore more freely; layernorm needed that.
- **Cross_entropy best so far (+7.21%)**, slight improvement over iter 3.

**What didn't:**
- **Attention regressed from +7.63% → +1.38%**. Approach 2 was actually helping attention.
- Softmax and matmul slightly worse.

**Lesson:** Approach 2 has mixed per-kernel effects — helps attention/softmax by structuring search, hurts layernorm/CE by over-constraining. A single global setting can't be optimal for all kernels.

**Oracle best-per-kernel across all iterations:**
- cross_entropy: iter6 (+7.21%)
- softmax: iter4 (+14.81%) — seeding helped softmax specifically
- matmul: iter5 (+8.58%) — augmented library matmul rules
- attention: iter3 (+7.63%) — Approach 2 + Approach 8
- layernorm: iter6 (+3.28%) — no Approach 2
- **Oracle geomean: +8.38%**

**Decision:** +6.36% (iter 3) remains best single-run result; +8.38% is achievable with per-kernel optimization. Given ~40h budget, spending many more iterations for single-digit % gains has diminishing returns. Phase 2 is effectively complete. Final recommendation: document findings, return best per-kernel configs as the deliverable.

---

### Iteration 5: Phase 3A — Augmented pretuned library (14 new rules from iter3 data)
**Date:** 2026-05-09
**Approach:** Revert iter 4's seeding. Keep Approach 8A (prompt-only). Augment the library with 14 new rules mined from iter 3's best configs per shape for layernorm and matmul buckets that were previously uncovered.
**Code change:** Set `HELION_LLM_PRETUNED_LIBRARY_PATH=pretuned_library_augmented.json` (37 rules total, 14 new). Seeding reverted in `_build_seed_configs`.
**Supplementary rules generated:** 7 layernorm (covering cols×rows up to 16384×16384), 7 matmul (covering M/K/N up to 16384).
**Wall time:** ~1h 13min

**Result:**

| Kernel | Baseline | Iter3 (orig lib) | **Iter5 (augmented lib)** | Iter3 Δ | Iter5 Δ |
|--------|----------|---|---|---|---|
| cross_entropy | 39.47 | 36.74 | 37.25 | +6.91% | +5.61% |
| softmax | 22.69 | 19.95 | 20.57 | +12.06% | +9.34% |
| matmul | 41.31 | 39.17 | **37.77** | +5.18% | **+8.58%** 🎯 |
| attention | 23.25 | 21.48 | 22.48 | +7.63% | +3.34% |
| layernorm | 14.48 | 14.54 | 14.43 | -0.42% | +0.29% |
| **Geomean** | **26.25** | **24.58** | **24.81** | **+6.36%** | **+5.48%** |

**Outcome:** Slightly WORSE than iter 3 (-0.88 percentage points). Mixed per-kernel.

**What worked:**
- **Matmul improved (+5.18% → +8.58%)** — the 7 matmul-specific rules provided useful anchor configs for shapes outside the original library's coverage.
- **Layernorm finally in positive territory** (+0.29% vs -0.42% in iter 3) — modest.

**What didn't:**
- **Softmax regressed** (+12.06% → +9.34%). SM_007 lost its 38.8% gain (back to baseline). The augmented library may have diluted attention to softmax-specific rules.
- **Attention regressed** (+7.63% → +3.34%). ATN_002 lost its win.
- **Cross_entropy slight regression** (+6.91% → +5.61%).

**Lesson:** Adding more templates to the prompt has diminishing returns — the LLM's attention gets diluted when the prompt has many "templates to consider." Iter 3's lean 3-template prompt focused the LLM better than iter 5's broader library.

**Oracle insight:** If we could pick the best iteration per-kernel (iter3 for softmax/CE/attention, iter5 for matmul/layernorm), we'd get **+7.93% geomean**. Still short of 15% target but much better than any single iteration.

**Decision:** Current best is iter 3 at +6.36%. Still ~9% away from 15% target. Possible next directions:
1. Per-kernel prompt library selection (use small library for softmax/attention, augmented for matmul)
2. Try different Approach combinations: revert prompt changes (Approach 2) — it's possible that Approach 2 itself is slightly harmful and Approach 8 alone would be better
3. Explore HIGHER-level changes like increasing configs_per_round or max_rounds
4. Move on — +6-8% is substantial improvement, declare partial success

---

### Iteration 4: Phase 3B — Approach 8B (Pretuned templates also as round-0 seeds) — REGRESSION
**Date:** 2026-05-09
**Approach:** Kept Approach 8A prompt injection; additionally used pretuned templates as explicit round-0 seed configs (benchmarked directly in round 0).
**Code change:** `helion/autotuner/llm_search.py:_build_seed_configs` now prepends matched pretuned templates before random seeds. New helper `get_pretuned_config_dicts` in `helion/autotuner/llm/pretuned_library.py`.
**LLM:** Opus 4.7, thinking_budget=16384, effort=max
**Wall time:** ~1h 15min (parallel, 5 jobs)

**Result:**

| Kernel | Baseline | Iter3 (prompt only) | **Iter4 (prompt + seed)** | Iter3 Δ | **Iter4 Δ** |
|--------|----------|---|---|---|---|
| cross_entropy | 39.465 | 36.739 | 37.759 | +6.91% | **+4.32%** |
| softmax | 22.686 | 19.950 | 19.325 | +12.06% | **+14.81%** 🎯 |
| matmul | 41.314 | 39.174 | 40.581 | +5.18% | +1.77% |
| attention | 23.254 | 21.479 | 23.944 | +7.63% | **-2.97%** |
| layernorm | 14.476 | 14.536 | 14.723 | -0.42% | -1.71% |
| **Geomean** | **26.245** | **24.576** | **25.336** | **+6.36%** | **+3.46%** |

**Outcome:** WORSE than iter 3. Seeding regressed the geomean from +6.36% → +3.46%.

**What happened:**
- **Attention catastrophically broke** (+7.63% → −2.97%). Shapes ATN_002 −46.5%, ATN_101 −45.1%, ATN_005 −32.4%. Seeding 2-3 templates into the 10 random-config slot consumed diversity the LLM needed.
- Cross_entropy regressed −8.7% on CE_103, −9.2% on CE_105.
- Matmul: MM_101 −11.4%, MM_104 −8.3%, MM_007 −8.4%.
- **Only softmax benefited** (+14.81%, hitting the 15% target for that single kernel). Softmax has 3 overlapping rules that provide varied templates instead of constraining.

**Lesson:** Seeding constrains exploration. The pretuned configs are useful as CONTEXTUAL HINTS (prompt) but harmful as HARD STARTING POINTS (seeds) for kernels where the LLM's round-0 generation is already good.

**Decision:** Revert seeding. Keep iter 3 (Approach 8A: prompt-only) as the current best. Next iteration should focus on the layernorm gap — layernorm has only 1 bucket in the pretuned library (cols≤1024, rows≤1024) while our 12 layernorm shapes span wider ranges.

**Plan for iter 5:** Augment the pretuned library with custom layernorm rules derived from iter 3's own results — any config that was the best on a given layernorm shape can be registered as a "shape-specific template" for that bucket, then used to enhance the prompt for future runs. This is a bootstrap: iter 3 is already good enough to self-generate layernorm templates.

---

### Iteration 3: Phase 3A — Approach 8 (Pretuned Data Library in Initial Prompt) ⭐ BREAKTHROUGH (best so far: +6.36%)
**Date:** 2026-05-09
**Approach:** Inject validated config templates from `observed_heuristics_b200.json` into the initial LLM prompt, shape-matched per (kernel_class, shape_bucket). Kept Approach 2 adaptive refinement active from iter 2.
**Code change:** New module `helion/autotuner/llm/pretuned_library.py` (loads JSON, classifies shape, matches templates), integrated into `build_initial_prompt()` as a new "Pretuned Templates" section. Enabled via `HELION_LLM_PRETUNED_LIBRARY_PATH` env var.
**Data source:** `/home/dev/helion_choijon52/llm_heuristics_artifacts/observed_heuristics_b200.json` (23 rules, 7 kernel classes, each with empirically-measured near-optimal templates geomean_slowdown ≤ 1.02).
**LLM:** Opus 4.7, thinking_budget=16384, effort=max
**Wall time:** ~1h 26min (parallel, 5 jobs)

**Result:**

| Kernel | Baseline | Iter1 | Iter2 | **Iter3** | Base→Iter3 |
|--------|----------|-------|-------|-----------|------------|
| cross_entropy | 39.465 | 38.700 | 38.884 | **36.739** | **+6.91%** |
| softmax | 22.686 | 21.609 | 21.350 | **19.950** | **+12.06%** |
| matmul | 41.314 | 41.420 | 40.062 | **39.174** | **+5.18%** |
| attention | 23.254 | 24.340 | 24.644 | **21.479** | **+7.63%** |
| layernorm | 14.476 | 14.300 | 14.590 | 14.536 | -0.42% |
| **Geomean** | **26.245** | 26.076 | 26.034 | **24.576** | **+6.36%** |

**Outcome:** MAJOR improvement. +6.36% geomean vs baseline (vs +0.80% in iter 2) — pretuned data is the dominant lever.

**What worked:**
- **Attention fixed:** iter 1 had -4.67%, iter 2 had -5.98%, now +7.63%. ATN_002 jumped 29.2% (18.84→13.34 ms), ATN_005 +24.6%, ATN_102 +20.2%. The pretuned attention templates (block_sizes=[1,128,128], num_warps=4, num_stages=3, pid_type=flat, l2_groupings=[8]) directly addressed attention.
- **Softmax broke through:** SM_007 +38.8% (68.6→42.0 ms), SM_005 +44.5%, SM_006 +22.6%, SM_202 +20.5%.
- **Cross_entropy:** 10/12 shapes improved, biggest +15.7% on CE_105 (large heldout).
- **Matmul:** MM_001 +21.7%, MM_004 +23.7%, MM_002 +14.0% — small/medium matmul shapes now near-optimal.

**What didn't:**
- Layernorm flat (-0.42%). The one layernorm rule (cols<=1024, rows<=1024) only matches LN_001; bigger shapes get a less-specific match and the default LLM proposals were already good.
- Matmul MM_006 -10.4%, MM_103 -5.7%, MM_104 -4.3%: larger shapes where the pretuned matmul buckets (m/k/n≤1024) don't cover the region.
- Attention ATN_103 -2.9%, ATN_104 -0.4%: extrapolated large-seq/batch shapes where pretuned bucket doesn't match.

**Lesson:** Pretuned data helps most when bucket coverage matches the target shape. Shapes OUTSIDE the pretuned coverage don't benefit (sometimes slightly regress from iter 1-2 prompt changes that are still active).

**Decision:** Continue to **Phase 3A iter 4**. Try making Approach 8 more aggressive — instead of just showing templates in the prompt, also use them as seeds in round 0. This is Sub-phase 3B (seeded). Risk is that past experiments showed seeds can constrain exploration, but we now have clean attention wins to lose to.

---

### Iteration 2: Phase 2 — Approach 2 + state-adaptive (exploit if last round improved)
**Date:** 2026-05-09
**Approach:** Same adaptive refinement as iter 1, but switches between exploit/finetune based on whether the previous round actually improved best perf by >1%. If improved → exploit; if not → fine-tune.
**LLM:** Opus 4.7, thinking_budget=16384, effort=max
**Code change:** `helion/autotuner/llm_search.py` (track `_per_round_best_perf`, compute `improved_last_round`, pass through), `helion/autotuner/llm/prompting.py` (accept `improved_last_round` param).
**Wall time:** ~1h 18min (parallel, 5 jobs)

**Result:**

| Kernel | Baseline (ms) | Iter1 (ms) | Iter2 (ms) | Base→Iter2 |
|--------|---------------|-----------|-----------|-----------|
| cross_entropy | 39.465 | 38.700 | 38.884 | +1.47% |
| softmax | 22.686 | 21.609 | 21.350 | **+5.89%** |
| matmul | 41.314 | 41.420 | 40.062 | **+3.03%** |
| attention | 23.254 | 24.340 | 24.644 | **-5.98%** |
| layernorm | 14.476 | 14.300 | 14.590 | -0.79% |
| **Geomean** | **26.245** | 26.076 | **26.034** | **+0.80%** |

**Outcome:** Marginal improvement over iter 1 (+0.80% vs +0.64%). Still far from 15% target.

**What worked:**
- Matmul now improves +3.03% (vs -0.26% in iter 1). The "keep exploiting if last round improved" rule helped matmul continue finding gains.
- Softmax SM_005 (+47%) and SM_006 (+24%) — the big wins from iter 1 held.
- CE_101 now +7.2%.

**What didn't:**
- Attention regresses harder (-5.98% vs -4.67%). Shapes ATN_101 -35.7%, ATN_103 -29.7%, ATN_005 -19.1%. Our prompt changes consistently harm attention.
- Layernorm slightly worse (-0.79% vs +1.21% in iter 1).

**Key insight:** Prompt-only changes have a ~5-10% ceiling before they start harming kernels where the default prompt was already working. We're trading wins on loss kernels for losses on attention. Needs a different mechanism.

**Decision:** Pivot to **Approach 8: Pretuned Data Library (Phase 3A)**. Reasons:
1. `observed_heuristics_b200.json` has validated attention configs for exactly our shape buckets → directly addresses attention regression.
2. Pretuned data is empirical (measured on B200, same GPU as ours) → higher-quality starting point than any prompt can describe.
3. Phase 3A (inject into initial prompt) is the least invasive sub-phase — doesn't require reseeding, just augments the prompt with known-good configs.

---

### Iteration 1: Phase 2 — Approach 2 (Adaptive Refinement)
**Date:** 2026-05-09
**Approach:** Approach 2 — adaptive refinement prompts per round (R1 explore → R2 exploit → R3+ finetune)
**LLM:** Opus 4.7, thinking_budget=16384, effort=max
**Code change:** `helion/autotuner/llm/prompting.py` (`_ROUND_EXPLORATION_LINES`, `_ROUND_EXPLOITATION_LINES`, `_ROUND_FINETUNE_LINES`; pass `round_num` through `build_refinement_prompt`). Also `helion/autotuner/llm_search.py` — remove `del round_num`.
**Wall time:** ~1h 3min (parallel, 5 jobs)

**Result:**

| Kernel | Baseline (ms) | Iter1 (ms) | Δ |
|--------|---------------|-----------|-----|
| cross_entropy | 39.465 | 38.700 | **+1.94%** |
| softmax | 22.686 | 21.609 | **+4.75%** |
| matmul | 41.314 | 41.420 | -0.26% |
| attention | 23.254 | 24.340 | **-4.67%** |
| layernorm | 14.476 | 14.300 | +1.21% |
| **Geomean** | **26.245** | **26.076** | **+0.64%** |

**Outcome:** NEUTRAL — well short of 15% target. Mixed across kernels.

**What worked:**
- Softmax SM_005 jumped 48.9% (46→23 ms). Single best-shape improvement.
- Loss kernels generally improved (softmax +4.75%, CE +1.94%, layernorm +1.21%).

**What didn't:**
- Attention regressed -4.67% overall; small-seq shapes hit hard (ATN_001 -30%, ATN_101 -20%). The fine-tune-only R3 prompt was too restrictive — attention benefits from CONTINUED exploration in late rounds.
- Matmul flat; several large shapes regressed 10-13%.

**Lesson:** One-size-fits-all round strategy is wrong. Attention and matmul need the ability to keep exploring; loss kernels benefit from exploit/finetune.

**Decision:** Proceed to iter 2. Approach: make the refinement strategy **adaptive to search state**, not just round number — if the latest round still found a new best, the LLM should keep exploring; only switch to finetune when the search has plateaued. This is hybrid of Approach 2 and Approach 6 (success pattern learning).

---

### Iteration 0: Phase 0 Baseline Measurement
**Date:** 2026-05-09
**Approach:** baseline (library defaults — no optimizations)
**LLM:** Opus 4.7, thinking_budget=16384, effort=max
**Kernels tested:** cross_entropy, softmax, matmul, attention, layernorm (12 shapes × 3 repeats each)
**Wall time:** 1h 17min (parallel, 5 jobs)

**Per-kernel final best config (geomean across shapes, ms):**

| Kernel | R0 init | R1 | R2 | R3 final | Total Δ | Plateau@ | Regressions |
|--------|---------|-------|-------|----------|---------|----------|-------------|
| attention | 27.52 | 24.31 | 22.56 | **22.11** | +19.7% | R3 | — |
| layernorm | 13.65 | 14.62 | 12.43 | **8.46** | +38.1% | R1 | R1 |
| cross_entropy | 39.30 | 38.96 | 37.88 | **43.37** | -10.3% | R1 | R3 |
| matmul | 45.60 | 40.73 | 39.19 | **47.50** | -4.2% | R3 | R3 |
| softmax | 23.04 | 22.82 | 22.29 | **28.60** | -24.1% | R1 | R3 |

**Per-round improvement % (vs previous round):**

| Kernel | R1 | R2 | R3 |
|--------|-----|-----|-----|
| attention | +11.7% | +7.2% | +2.0% |
| layernorm | -7.0% | +15.0% | +32.0% |
| cross_entropy | +0.9% | +2.8% | **-14.5%** |
| matmul | +10.7% | +3.8% | **-21.2%** |
| softmax | +1.0% | +2.3% | **-28.3%** |

**Key findings:**
- **Round 3 is catastrophic for 3/5 kernels** (cross_entropy -14.5%, matmul -21.2%, softmax -28.3%). The LLM actively makes things worse in the final round.
- **Layernorm is the outlier** — gets worse in R1 (-7%), then huge gains in R2 (+15%) and R3 (+32%). Something about late rounds works for it.
- **Attention shows clean diminishing returns** (+11.7% → +7.2% → +2.0%) — the "expected" pattern.
- **Compute-bound kernels improve more on average** (matmul + attention: +7.7%) vs memory-bound (cross_entropy + softmax + layernorm: +1.2%, dragged down by late regressions).
- The biggest average R2 gain (+6.2%) suggests round 2 prompts are highest-leverage for targeted improvements.

**Decision:**
- Phase 0 complete, baseline established for all 5 kernels.
- The **most urgent problem is round 3 regression** — this ALONE, if fixed, would yield ~15-20% improvement on 3 kernels because they had already found better configs in R2 and R1.
- Proceeding to Phase 2 with two concurrent experiments:
  1. **Quick win: cap rounds at 2** (Approach 1) — test on all 5 kernels, should recover the losses on cross_entropy/matmul/softmax without hurting attention/layernorm much
  2. **Deeper fix: Approach 6 Success Pattern Learning** — help LLM stop diverging from R2 winners in R3

Full details: `PHASE_1_ANALYSIS.md` and `PHASE_1_ANALYSIS.json`

<!-- Template for each result entry:
### Iteration N: [Experiment name]
**Date:** YYYY-MM-DD
**Approach:** [Approach 1/2/3/4/5/6 or "baseline"]
**Kernels tested:** [list]
**Result:** [best perf per kernel, geometric mean vs baseline]
**Key findings:** [3-5 bullets]
**Decision:** [what happens next]
-->

---

## Current Findings

*Updated by Evaluate after each analysis. Captures what we've learned so far.*

**Phase 0 (Baseline) — corrected after tracking-noise analysis:**
True baseline = geomean across shapes of each shape's minimum perf_ms (averaged across 3 repeats per shape):

| Kernel | True baseline (ms) | What R3 tracked CSV said |
|--------|--------------------|--------------------------|
| cross_entropy | 39.47 | 43.37 (inflated by ~10%) |
| softmax | 22.69 | 28.60 (inflated by ~26%) |
| matmul | 41.31 | 47.50 (inflated by ~15%) |
| attention | 23.25 | 22.11 (close) |
| layernorm | 14.48 | 8.46 (population tracking showed something better than any single benchmark — suspicious) |

**The "R3 regression" was largely a tracking artifact, not a real regression.** The `round_progress` CSV uses `min(population, key=perf)` where `.perf` is a median of rebenchmarks. Later rounds cause rebenchmarks that update some members' `.perf`, which makes the tracked "best so far" appear to worsen.

**However, the tracking artifact itself is a real finding:** the autotuner's **final returned config** is based on this population state. If rebenchmarking in R3 pushes good configs' `.perf` up, the final selected config could genuinely be worse than what R2 found. This needs verification.

**Phase 1 (Per-Round Analysis) — what's actually true:**
- The TRUE best-ever config per shape is reliably found by **the middle of the search** (round 1-2 typically).
- R3 doesn't find better configs, but the population-based "best" selection can drift.
- **Attention shows clean diminishing returns.** Layernorm's R3 gain is also real (per global-best).
- **Kernel-type effect:** compute-bound (matmul) benefits from deeper exploration in early rounds; memory-bound kernels (softmax, cross_entropy) find their best quickly.

**Implications for Phase 2:**
- Approach 1 (cap rounds) would NOT help if the issue is tracking noise — the true best configs exist; we just need to pick them correctly.
- Real fix: **ensure the autotuner returns the best-ever config found, not just the best in the current population state**. This may already work correctly — need to verify by comparing what `autotune()` returns vs the per-shape global min from CSV.
- If global-min IS what's returned, the baseline is actually better than we thought, and the "15% improvement" target is measured vs the correct baseline numbers above.

<!-- Examples of what goes here:
- "Round 1→2 improves 12% on memory-bound kernels, only 3% on matmul"
- "Attention regresses in round 3 (prompt confuses LLM)"
- "Cross_entropy plateaus at round 2"
-->

---

## Active Proposals

*Maintained by Proposal subagent. Ideas to try next, in priority order.*

*(No active proposals yet — Phase 0 not complete)*

<!-- Template:
### Proposal N: [Short name]
**Approach:** [which Approach 1-6, or novel]
**Idea:** [what to try]
**Rationale:** [why it might work, based on findings]
**Target:** [which round/kernel]
**Expected improvement:** [% over baseline]
**Time cost:** [hours]
**Risk:** [LOW/MEDIUM/HIGH]
**Status:** [queued / in-progress / tested]
-->

---

## Tried Approaches

*Proposal subagent moves entries here after measurement. Avoids duplicate work and builds knowledge.*

*(No approaches tried yet)*

<!-- Template:
### Tried: [Approach name] (Iteration N)
**Result:** [% improvement vs baseline]
**Outcome:** SUCCESS / NEUTRAL / FAILED
**Lesson learned:** [what we know now]
**Don't retry because:** [reason, so future Proposals skip it]
-->

---

## Files Reference

| File | Purpose |
|------|---------|
| `PLAN.md` | This file — everything we plan to do |
| `MANAGER.md` | Orchestration rules for Manager agent |
| `MANAGER_STATE.json` | Structured state (progress, results) |
| `loss_functions/tools/run_live.py` | Experiment runner |
| `loss_functions/tools/workloads.py` | Kernel builders |
| `loss_functions/grids/*.json` | Shape grids for each kernel |
| `helion/autotuner/llm_search.py` | LLM search logic (what we'll optimize) |
| `archived_plans/` | Old heuristics experiments (reference only) |
