# Novel Proposals for Achieving round0_best_geo ≤ 0.80

**Date:** 2026-05-08  
**Context:** 3 consecutive failures (A: 1.0376, B: 1.0063, C: 1.0076)  
**Proposal Subagent:** ACTIVE

---

## Executive Summary

After reading EXPERIMENT_LOG.md completely, I've identified the **core problem**:

**All three approaches (A, B, C) use round-0 seeding strategies** - they inject configs before LLM starts searching. This has fundamental limitations:

1. **Small shapes (SM_001-004, dims ≤3072):** Baseline is already near-optimal. Seeds provide no value because LLM quickly finds good configs anyway.

2. **Large shapes (SM_005-008, dims ≥5120):** Seeds cause **8-16% degradation**. The heuristic/observed seeds are systematically worse than what random+LLM exploration finds.

3. **Interference effect:** Approach C (hybrid A+B) scored 1.0076, worse than B alone (1.0063). Combining two seed strategies caused interference, not synergy.

**Root cause:** Round-0 seeding has limited impact when:
- LLM is already strong (Opus 4.7 finds good configs in 5-6 tries)
- Seed quality is unverified (no guarantee seed is better than LLM's random start)
- Config space is large (1 or 3 seeds don't meaningfully guide search)

**Escalation justified:** After ~5 hours and 3 approaches, we've exhausted the "inject seeds before LLM" paradigm. Need fundamentally different strategies.

---

## Idea 1: Size-Dependent Strategy Switching

### Core Innovation
Don't use a universal heuristic. Instead, **route shapes to different strategies** based on workload size:

- **Small shapes (dim ≤1024):** Disable heuristics entirely. LLM baseline is already optimal.
- **Mid shapes (1024 < dim ≤ 4096):** Use Approach B (observed examples in prompt)
- **Large shapes (dim > 4096):** Use dedicated large-shape heuristic trained ONLY on large shapes

### Novelty Check
- **NOT like A:** A used universal decision tree. This uses NO heuristic for small shapes.
- **NOT like B:** B showed examples for all shapes. This selectively applies B only to mid shapes.
- **NOT like C:** C combined A+B universally. This uses different strategies per size class.

### How It Addresses Large Shape Problem
**Hypothesis:** Large shapes fail because:
1. Heuristic configs (from decision tree) are trained on diverse AOT data including small shapes
2. These configs optimize for small-shape patterns (low warps, simple indexing)
3. Large shapes need different configs (higher parallelism, different memory patterns)

**Solution:**
- Build separate `observed_heuristics_large.json` using ONLY shapes with dim≥5120
- Extract top-5 configs from SM_005, SM_006, SM_007, SM_008 (train set large shapes)
- Show LLM these examples ONLY when dim>4096

**For small shapes:** Skip heuristics entirely. Baseline already achieves ~1.0 ratio.

### Implementation Plan

1. **Modify prompting code** to check dimension:
   ```python
   def _build_observed_heuristics_section(args):
       dim = args[0].shape[1]
       
       if dim <= 1024:
           return ""  # No heuristics for small shapes
       elif dim <= 4096:
           # Use existing observed_heuristics_softmax.json
           return _load_mid_examples(dim)
       else:
           # Use new observed_heuristics_large.json
           return _load_large_examples(dim)
   ```

2. **Build observed_heuristics_large.json:**
   - Extract best 5 configs from baseline runs of SM_005-008
   - Format same as Approach B
   - Ensure configs are actually optimal for large dims

3. **Test with routing logic:**
   ```bash
   # Small shapes: no env vars (pure baseline)
   python run_live.py --kernel softmax --shapes SM_001,SM_002,SM_003,SM_004 --arm baseline
   
   # Mid shapes: Approach B
   export HELION_LLM_OBSERVED_HEURISTICS_PATH=observed_heuristics_softmax.json
   python run_live.py --kernel softmax --shapes SM_101,SM_102 --arm heuristics
   
   # Large shapes: dedicated large-shape heuristics
   export HELION_LLM_OBSERVED_HEURISTICS_PATH=observed_heuristics_large.json
   python run_live.py --kernel softmax --shapes SM_005,SM_006,SM_007,SM_008 --arm heuristics
   ```

### Success Criteria
- **Target score:** ≤0.90 (milestone toward 0.80)
- **Key metric:** Large shape ratio (SM_005-008 geo mean)
  - Current: ~1.10 (10% degradation)
  - Target: ≤1.00 (neutral or better)
- **Small shape preservation:** SM_001-004 remain ~1.00 (no regression)

### Risks
- **Risk:** Large-shape examples might still be suboptimal
  - **Mitigation:** Extract from BEST baseline runs, not heuristic runs
- **Risk:** Implementation complexity (3 code paths)
  - **Mitigation:** Simple if/elif/else, well-tested
- **Risk:** Mid shapes might regress without decision tree seed
  - **Mitigation:** Approach B already scored 1.0063, acceptable

### Effort Estimate
**4 hours**
- 1h: Build observed_heuristics_large.json
- 1h: Modify prompting.py routing logic
- 2h: Run experiment (12 shapes × 3 repeats)

---

## Idea 2: Multi-Stage LLM Search with Progressive Refinement

### Core Innovation
Instead of seeding round-0, **run the autotuner multiple times** with different configs_per_round:

**Stage 1 (Exploration):** 
- configs_per_round=10, rounds=1
- High diversity, explore config space broadly
- Save top-5 configs

**Stage 2 (Refinement):**
- Start from Stage 1 top-5 as seeds
- configs_per_round=5, rounds=2
- LLM refines promising configs, explores nearby

**Stage 3 (Validation):**
- Benchmark Stage 1+2 winners again (reduce variance)
- Pick best across all stages

### Novelty Check
- **NOT like A:** A seeds once. This runs LLM multiple times with different strategies.
- **NOT like B:** B shows examples in prompt. This actually changes the search process.
- **NOT like C:** C combines A+B. This is a workflow change, not a seed change.

### How It Addresses Large Shape Problem
**Current issue:** LLM only tries 5-6 configs in round-0. For large shapes, this isn't enough exploration.

**Solution:** 
- Stage 1 explores 10 configs (2x current)
- Stage 2 refines top-5, adds another 10 configs (5 per round × 2 rounds)
- Total: 20 configs explored vs current 5-6
- More exploration → higher chance of finding optimal config for large dims

### Implementation Plan

1. **Create multi_stage_search.py:**
   ```python
   def run_multi_stage_autotune(kernel, shape):
       # Stage 1: Broad exploration
       stage1_configs = run_autotuner(
           configs_per_round=10,
           rounds=1,
           initial_random_configs=10
       )
       top5 = select_top_n(stage1_configs, n=5)
       
       # Stage 2: Refinement
       stage2_configs = run_autotuner(
           configs_per_round=5,
           rounds=2,
           seed_configs=top5
       )
       
       # Stage 3: Validate
       all_candidates = top5 + get_top_n(stage2_configs, n=5)
       best = benchmark_and_select(all_candidates)
       return best
   ```

2. **Integrate with run_live.py:**
   ```bash
   python run_live.py \
     --kernel softmax \
     --arm multistage \
     --shape-grid shape_grid.json \
     --stage1-configs 10 \
     --stage2-rounds 2 \
     --repeats 3
   ```

3. **Measure improvement:**
   - Compare multistage vs baseline
   - Focus on large shapes: do they benefit from more exploration?

### Success Criteria
- **Target score:** ≤0.85 (significant improvement)
- **Key metric:** Config diversity
  - Current: ~15-20 unique configs per shape
  - Target: ~30-40 unique configs per shape
- **Large shape improvement:** SM_005-008 ratio ≤1.05

### Risks
- **Risk:** 3x more LLM calls = 3x longer runtime
  - **Mitigation:** Worth it if achieves goal; can optimize later
- **Risk:** Diminishing returns (Stage 2 doesn't improve over Stage 1)
  - **Mitigation:** Monitor scores per stage, skip Stage 2 if no improvement
- **Risk:** Variance increases with more configs
  - **Mitigation:** Stage 3 validation re-benchmarks top candidates

### Effort Estimate
**6 hours**
- 2h: Build multi_stage_search.py
- 1h: Integrate with run_live.py
- 3h: Run experiment (longer due to multiple stages)

---

## Idea 3: Negative Heuristics (Avoid Known Bad Configs)

### Core Innovation
Instead of seeding GOOD configs, **filter out BAD configs** before LLM tries them.

**Current approach:** LLM suggests configs, we benchmark ALL of them.

**New approach:** LLM suggests configs → filter out known-bad patterns → benchmark only plausible ones.

**Example bad patterns from analysis:**
- Large dims (>4096) + persistent_blocked mode → always slower
- Large dims + num_warps>16 → regression
- Large dims + tensor_descriptor indexing → overhead
- Any dim + num_stages>4 → diminishing returns

### Novelty Check
- **NOT like A:** A injects good seeds. This REJECTS bad candidates.
- **NOT like B:** B shows good examples. This teaches LLM what to avoid.
- **NOT like C:** C combines A+B. This is orthogonal (works with or without seeds).

### How It Addresses Large Shape Problem
**Root cause:** Heuristic/observed configs guide LLM toward complex configs (high warps, persistent mode). These perform poorly on large shapes.

**Solution:** 
- Build `bad_config_patterns.json` from past failures:
  ```json
  {
    "large_dims": {
      "condition": "dim > 4096",
      "reject_if": [
        {"field": "pid_type", "value": "persistent_blocked"},
        {"field": "num_warps", "operator": ">", "value": 16},
        {"field": "indexing", "contains": "tensor_descriptor"}
      ]
    }
  }
  ```
- Before benchmarking, check each config against reject rules
- If matched, skip config (don't waste time benchmarking known-bad)
- LLM learns: "I suggested persistent mode for dim=8192, but it was rejected. Try simpler configs."

### Implementation Plan

1. **Analyze failures from Approaches A/B/C:**
   ```python
   # Extract configs that caused regression
   bad_configs = []
   for shape in large_shapes:
       baseline_time = get_baseline_time(shape)
       for config in heuristic_configs:
           if config.time > baseline_time * 1.05:  # 5% slower
               bad_configs.append({
                   'shape': shape,
                   'config': config,
                   'slowdown': config.time / baseline_time
               })
   
   # Find common patterns in bad configs
   patterns = analyze_patterns(bad_configs)
   # Example: 90% of bad configs for large dims use persistent_blocked
   ```

2. **Build bad_config_patterns.json:**
   ```json
   {
     "rules": [
       {
         "name": "large_dim_no_persistent",
         "condition": {"dim": {">": 4096}},
         "reject": {"pid_type": "persistent_blocked"},
         "reason": "Persistent mode 10-15% slower for large dims"
       },
       {
         "name": "large_dim_low_warps",
         "condition": {"dim": {">": 4096}},
         "reject": {"num_warps": {">": 16}},
         "reason": "High warp count causes overhead"
       }
     ]
   }
   ```

3. **Add filtering to autotuner:**
   ```python
   def should_benchmark_config(config, shape, bad_patterns):
       for rule in bad_patterns['rules']:
           if matches_condition(shape, rule['condition']):
               if matches_reject(config, rule['reject']):
                   logger.info(f"Skipping config: {rule['reason']}")
                   return False
       return True
   
   # In LLM search loop:
   suggested_configs = llm.suggest_configs()
   valid_configs = [c for c in suggested_configs if should_benchmark_config(c, shape, bad_patterns)]
   benchmark_results = benchmark(valid_configs)
   ```

4. **Test negative heuristics:**
   ```bash
   export HELION_LLM_NEGATIVE_PATTERNS_PATH=bad_config_patterns.json
   python run_live.py --kernel softmax --shape-grid shape_grid.json --arm negative_heuristics
   ```

### Success Criteria
- **Target score:** ≤0.88 (improvement over B's 1.0063)
- **Key metric:** Config filter rate
  - Target: 20-30% of LLM suggestions filtered out
  - Should filter MORE for large shapes (where bad patterns concentrated)
- **Large shape ratio:** ≤1.00 (no more regression)

### Risks
- **Risk:** Over-filtering (reject good configs by accident)
  - **Mitigation:** Start with conservative rules, monitor false positive rate
- **Risk:** LLM keeps suggesting same bad patterns
  - **Mitigation:** Add feedback to prompt: "These patterns rejected: [list]"
- **Risk:** Patterns extracted from A/B/C might not generalize
  - **Mitigation:** Only use patterns with 80%+ confidence (appear in majority of failures)

### Effort Estimate
**5 hours**
- 2h: Analyze A/B/C failures, extract patterns
- 1h: Build bad_config_patterns.json
- 1h: Implement filtering logic
- 1h: Test and measure

---

## Idea 4: Ensemble Autotuning (Run Multiple Parallel Searches)

### Core Innovation
Run 3-5 **independent autotuner instances** in parallel, each with different strategy:
- Instance 1: Pure baseline (no heuristics)
- Instance 2: Approach A (decision tree)
- Instance 3: Approach B (observed examples)
- Instance 4: Random seed variation 1
- Instance 5: Random seed variation 2

**Final step:** Select best config across ALL instances.

### Novelty Check
- **NOT like A/B/C:** Those are single-strategy. This runs multiple strategies in parallel.
- Combines strengths: If B works for mid shapes and A works for small shapes, ensemble picks best for each.

### How It Addresses Large Shape Problem
**Insight:** Different shapes benefit from different strategies.
- Small shapes: Baseline wins
- Mid shapes: Approach B wins
- Large shapes: Unknown (maybe random exploration wins?)

**Solution:** Run all strategies, pick per-shape winner.

### Implementation Plan

1. **Create ensemble_autotune.py:**
   ```python
   def run_ensemble_autotune(kernel, shape):
       strategies = [
           {'name': 'baseline', 'heuristic': None, 'observed': None},
           {'name': 'decision_tree', 'heuristic': 'A.py', 'observed': None},
           {'name': 'observed', 'heuristic': None, 'observed': 'B.json'},
           {'name': 'random1', 'heuristic': None, 'observed': None, 'seed': 42},
           {'name': 'random2', 'heuristic': None, 'observed': None, 'seed': 123},
       ]
       
       results = []
       for strategy in strategies:
           config = run_autotuner(**strategy)
           results.append({
               'strategy': strategy['name'],
               'config': config,
               'time': benchmark(config, shape)
           })
       
       best = min(results, key=lambda x: x['time'])
       return best
   ```

2. **Parallelize for speed:**
   - Use multiprocessing to run 5 instances concurrently
   - Each uses different GPU stream
   - Total time ~= single run time (if 5 GPUs available)

3. **Measure ensemble benefit:**
   ```bash
   python ensemble_autotune.py --kernel softmax --shape-grid shape_grid.json
   ```

### Success Criteria
- **Target score:** ≤0.85
- **Key metric:** Per-shape strategy distribution
  - Expect: Small shapes pick baseline, mid shapes pick B, large shapes pick ???
- **Overhead:** Ensemble should not be >2x slower than single run

### Risks
- **Risk:** 5x more compute required
  - **Mitigation:** Only worth it if significantly improves score
- **Risk:** Variance across runs makes comparison noisy
  - **Mitigation:** Use median of 3 runs per strategy
- **Risk:** Best strategy might still not hit goal
  - **Mitigation:** This exposes which strategy works for which shapes → guide future proposals

### Effort Estimate
**8 hours**
- 3h: Build ensemble framework
- 2h: Parallelize (handle GPU allocation)
- 3h: Run experiment (5 strategies × 12 shapes × 3 repeats)

---

## Idea 5: Autotuner Meta-Optimization

### Core Innovation
**Tune the autotuner itself**, not just the configs.

Current autotuner settings:
- configs_per_round=3
- initial_random_configs=2
- rounds=1 (for most shapes)

**Hypothesis:** These hyperparameters are not optimal for all shapes.

**Approach:** Grid search over autotuner hyperparameters:
- configs_per_round: [3, 5, 8, 10]
- initial_random_configs: [2, 4, 8]
- rounds: [1, 2, 3]

For each combination, measure round0_best_geo. Find best hyperparams.

### Novelty Check
- **NOT like A/B/C:** Those change what configs LLM sees. This changes how many configs LLM explores.
- Meta-level: Optimizing the optimization process.

### How It Addresses Large Shape Problem
**Hypothesis:** Large shapes need more exploration.
- Small shapes: 3 configs enough (space is simple)
- Large shapes: 10+ configs needed (space is complex)

**Solution:** Size-dependent hyperparameters:
```python
if dim <= 1024:
    configs_per_round = 3  # Fast convergence
elif dim <= 4096:
    configs_per_round = 5  # Moderate exploration
else:
    configs_per_round = 10  # Deep exploration
```

### Implementation Plan

1. **Grid search script:**
   ```python
   hyperparams = {
       'configs_per_round': [3, 5, 8, 10],
       'initial_random': [2, 4, 8],
       'rounds': [1, 2]
   }
   
   results = []
   for cpr in hyperparams['configs_per_round']:
       for ir in hyperparams['initial_random']:
           for r in hyperparams['rounds']:
               score = run_autotuner(
                   configs_per_round=cpr,
                   initial_random_configs=ir,
                   rounds=r,
                   shape=test_shape
               )
               results.append({
                   'cpr': cpr, 'ir': ir, 'rounds': r,
                   'score': score, 'time': elapsed
               })
   
   best_hyperparams = min(results, key=lambda x: x['score'])
   ```

2. **Test on large shapes:**
   - Run grid search on SM_005 (5120 dim)
   - Find best hyperparams for large dims
   - Apply to all large shapes, measure improvement

3. **Adaptive tuning:**
   - Use different hyperparams for small vs mid vs large
   - Baseline: small shapes use (3,2,1), large shapes use best from grid search

### Success Criteria
- **Target score:** ≤0.88
- **Key metric:** Large shape improvement
  - Current: SM_005-008 average 1.10
  - Target: ≤1.02 with optimized hyperparams
- **Time budget:** Grid search should complete in <2 hours

### Risks
- **Risk:** Optimal hyperparams might still not improve score
  - **Mitigation:** Low cost to try; provides data on sensitivity
- **Risk:** Overfitting to test shapes
  - **Mitigation:** Test on held-out shapes
- **Risk:** Increased runtime
  - **Mitigation:** Only increase params for large shapes (small shapes stay fast)

### Effort Estimate
**5 hours**
- 2h: Grid search implementation
- 2h: Run grid search on 2-3 large shapes
- 1h: Apply best hyperparams, measure full run

---

## Idea 6: Prompt Engineering - Explicit Dimension-Aware Guidance

### Core Innovation
**Modify LLM prompt to include dimension-specific advice**, not just examples.

Current prompt:
```
Kernel: softmax
Workload: batch=4096, dim=8192
[Generic config space description]
Task: Suggest configs
```

**New prompt:**
```
Kernel: softmax
Workload: batch=4096, dim=8192

Dimension analysis:
- This is a LARGE dimension (>4096)
- Large dimensions have different performance characteristics:
  * Prefer flat pid_type over persistent_blocked (-10% latency)
  * Use num_warps ≤ 8 (higher causes overhead)
  * block_sizes=[1] often optimal (process entire dim in one block)
  * Avoid tensor_descriptor indexing (adds overhead)

[Config space + examples]
Task: Suggest configs optimized for large dimensions
```

### Novelty Check
- **NOT like A:** A seeds configs. This teaches LLM principles.
- **NOT like B:** B shows examples. This explains WHY examples work.
- **NOT like C:** C combines A+B. This adds explanatory text to prompt.

### How It Addresses Large Shape Problem
**Current issue:** LLM doesn't understand size-specific patterns. It suggests same config types for all sizes.

**Solution:** Explicitly teach LLM the rules:
- Small dims: X patterns work
- Mid dims: Y patterns work
- Large dims: Z patterns work

LLM can then reason: "This is large dim, so I should avoid persistent mode."

### Implementation Plan

1. **Extract dimension-specific rules from data:**
   ```python
   # Analyze: What configs win for small vs mid vs large dims?
   small_winners = analyze_configs(shapes_with_dim_le_1024)
   mid_winners = analyze_configs(shapes_with_dim_1024_4096)
   large_winners = analyze_configs(shapes_with_dim_ge_4096)
   
   # Summarize patterns:
   # Small: {'num_warps': 4, 'pid_type': 'flat', 'block_sizes': [16]}
   # Mid: {'num_warps': 8, 'pid_type': 'flat', 'block_sizes': [32]}
   # Large: {'num_warps': 4, 'pid_type': 'flat', 'block_sizes': [1]}
   ```

2. **Add dimension guidance to prompt:**
   ```python
   def _build_dimension_guidance(dim):
       if dim <= 1024:
           return """
           Small dimension optimization guide:
           - Fast convergence possible with simple configs
           - Prefer: num_warps=4, pid_type=flat, block_sizes=[16]
           - Avoid: Complex indexing, high num_stages
           """
       elif dim <= 4096:
           return """
           Medium dimension optimization guide:
           - Balance between parallelism and overhead
           - Prefer: num_warps=8, pid_type=flat, block_sizes=[32]
           - Experiment with: Different block sizes, stages=2
           """
       else:
           return """
           Large dimension optimization guide:
           - Simplicity wins over complexity for large workloads
           - Prefer: num_warps=4-8, pid_type=flat, block_sizes=[1]
           - Avoid: persistent_blocked (-10%), high warps (>16), tensor_descriptors
           - Key insight: Processing entire dim in one block often optimal
           """
   ```

3. **Integrate into prompting.py:**
   ```python
   def build_initial_prompt(...):
       dim = args[0].shape[1]
       dimension_guidance = _build_dimension_guidance(dim)
       
       return _join_sections(
           describe_kernel(kernel, args),
           dimension_guidance,  # ADD THIS
           describe_config_space(config_spec),
           observed_examples,
           task_section
       )
   ```

4. **Test dimension-aware prompting:**
   ```bash
   python run_live.py --kernel softmax --shape-grid shape_grid.json --arm dimension_aware_prompt
   ```

### Success Criteria
- **Target score:** ≤0.87
- **Key metric:** Config alignment with guidance
  - For large shapes: % of LLM-suggested configs that follow guidance (avoid persistent, low warps)
  - Target: 70%+ alignment
- **Large shape improvement:** SM_005-008 ratio ≤1.00

### Risks
- **Risk:** LLM ignores guidance, still suggests bad configs
  - **Mitigation:** Make guidance more emphatic, use examples alongside
- **Risk:** Guidance is wrong (our rules are incorrect)
  - **Mitigation:** Extract rules from empirical data, not assumptions
- **Risk:** Prompt becomes too long (token limit)
  - **Mitigation:** Keep guidance concise (<200 tokens per section)

### Effort Estimate
**3 hours**
- 1h: Extract dimension-specific rules from data
- 1h: Implement prompt modification
- 1h: Run experiment and measure

---

## Idea 7: Shape-Specific Fine-Tuning (Micro-Heuristics)

### Core Innovation
Instead of one universal heuristic, build **12 separate micro-heuristics** (one per shape).

Each micro-heuristic:
- Trained on AOT data for that exact shape only
- Returns top-3 configs specific to that shape
- No generalization needed

**For held-out shapes:** Use nearest-neighbor interpolation:
- SM_101 (dim=512) → average of SM_001 (256) and SM_002 (896) heuristics
- SM_202 (dim=12672) → use SM_007 (11264) heuristic

### Novelty Check
- **NOT like A:** A uses universal decision tree. This is shape-specific.
- **NOT like B:** B uses dim-bucket examples. This is exact-shape matching.
- **NOT like C:** C combines A+B. This eliminates generalization entirely.

### How It Addresses Large Shape Problem
**Root cause of failures:** Heuristics try to generalize from small shapes to large shapes. This fails because performance patterns differ.

**Solution:** Don't generalize. Train separate heuristics for each large shape.
- SM_005 heuristic trained only on dim=5120 data
- SM_006 heuristic trained only on dim=8192 data
- etc.

**No cross-contamination** from small-shape patterns.

### Implementation Plan

1. **Extract shape-specific configs:**
   ```python
   # For each shape in train set
   for shape in train_shapes:
       # Get baseline CSV for this shape
       shape_data = baseline_csv[baseline_csv['shape_id'] == shape.id]
       
       # Extract top-5 configs by timing
       top5 = shape_data.nsmallest(5, 'timing_ms')
       
       # Save as micro-heuristic
       micro_heuristics[shape.id] = {
           'shape': shape,
           'top_configs': top5['config'].tolist()
       }
   ```

2. **Build interpolation for held-out:**
   ```python
   def get_heuristic_for_shape(target_shape, train_heuristics):
       # Find nearest train shape by dimension
       distances = [(s, abs(s.dim - target_shape.dim)) for s in train_heuristics.keys()]
       nearest = min(distances, key=lambda x: x[1])[0]
       
       # Use nearest neighbor's heuristic
       return train_heuristics[nearest]
   ```

3. **Test micro-heuristics:**
   ```bash
   export HELION_LLM_MICRO_HEURISTICS_PATH=micro_heuristics_per_shape.json
   python run_live.py --kernel softmax --shape-grid shape_grid.json --arm micro_heuristics
   ```

### Success Criteria
- **Target score:** ≤0.85
- **Key metric:** Per-shape heuristic match rate
  - For train shapes: 100% (exact match)
  - For held-out shapes: Measure how well interpolation works
- **Large shape preservation:** SM_005-008 ratio ≤1.00 (no regression)

### Risks
- **Risk:** Overfitting to train shapes
  - **Mitigation:** Test on held-out; if fails, expand train set
- **Risk:** Interpolation doesn't work for held-out shapes
  - **Mitigation:** Use wider interpolation (average of 2-3 nearest neighbors)
- **Risk:** Maintenance burden (12 separate heuristics)
  - **Mitigation:** Automate extraction from AOT data

### Effort Estimate
**4 hours**
- 1h: Extract micro-heuristics from baseline data
- 1h: Build interpolation logic
- 2h: Run experiment and measure

---

## Idea 8: Learned Reward Model (Config Scoring Before Benchmarking)

### Core Innovation
Train a **lightweight ML model** to predict config performance before benchmarking.

**Workflow:**
1. LLM suggests 10 configs
2. Reward model scores each config: predicted_time = f(config, shape)
3. Benchmark only top-5 predicted configs
4. Use actual timings to update reward model
5. Repeat

**Model:** Simple gradient boosting (sklearn) or neural net
- Input: Config features (num_warps, num_stages, block_sizes, ...) + shape features (batch, dim)
- Output: Predicted timing (ms)

### Novelty Check
- **NOT like A/B/C:** Those use heuristics/examples. This uses ML model.
- Active learning: Model improves as we collect more data.

### How It Addresses Large Shape Problem
**Current issue:** We benchmark ALL LLM suggestions, even bad ones. Wastes time.

**Solution:** 
- Reward model learns: "Config X on large dim → slow"
- Filters out bad configs before benchmarking
- LLM gets feedback: "Your suggestion scored poorly, try something else"

For large shapes: Model learns patterns like "persistent mode → slow" and guides search away.

### Implementation Plan

1. **Train initial model on AOT data:**
   ```python
   from sklearn.ensemble import GradientBoostingRegressor
   
   # Load AOT data
   X = []  # Features: [batch, dim, num_warps, num_stages, block_sizes, ...]
   y = []  # Targets: timing_ms
   
   for row in aot_data:
       X.append(extract_features(row))
       y.append(row['timing_ms'])
   
   model = GradientBoostingRegressor()
   model.fit(X, y)
   ```

2. **Integrate into autotuner:**
   ```python
   def llm_search_with_reward_model(shape):
       for round in range(num_rounds):
           # LLM suggests configs
           suggested = llm.suggest_configs(n=10)
           
           # Reward model scores
           predictions = [reward_model.predict(c, shape) for c in suggested]
           
           # Benchmark top-5 predicted
           top5 = select_top_n_by_prediction(suggested, predictions, n=5)
           actual_timings = benchmark(top5)
           
           # Update model with new data
           reward_model.update(top5, actual_timings)
           
           # Give feedback to LLM
           llm.add_context(f"Configs {top5} achieved {actual_timings}")
   ```

3. **Test reward-guided search:**
   ```bash
   python run_live.py --kernel softmax --shape-grid shape_grid.json --arm reward_model
   ```

### Success Criteria
- **Target score:** ≤0.82
- **Key metric:** Model prediction accuracy
  - Target: R²>0.7 (explain 70% of timing variance)
- **Efficiency:** Benchmark 50% fewer configs (filter out bad ones)
- **Large shape improvement:** Model should predict large-shape failures accurately

### Risks
- **Risk:** Model underfits (not enough training data)
  - **Mitigation:** Use AOT data (1000+ examples) + active learning
- **Risk:** Model overfits to training distribution
  - **Mitigation:** Regularization, cross-validation
- **Risk:** Prediction is slower than benchmarking
  - **Mitigation:** Use simple model (GBM inference <1ms), still net win

### Effort Estimate
**7 hours**
- 2h: Train initial model on AOT data
- 2h: Integrate into autotuner (prediction + active learning)
- 3h: Run experiment and measure

---

## Idea 9: Compositional Search (Decompose Config into Sub-Decisions)

### Core Innovation
Instead of searching the full config space, **decompose into orthogonal sub-decisions**:

1. **Parallelism level:** num_warps [2, 4, 8, 16, 32]
2. **Pipeline depth:** num_stages [1, 2, 3, 4]
3. **Memory pattern:** pid_type [flat, persistent_blocked, persistent_streaming]
4. **Block granularity:** block_sizes [1, 16, 32, 64, 128, 256]
5. **Indexing mode:** indexing [pointer, tensor_descriptor]

**Search strategy:**
1. Fix all but one dimension (e.g., optimize num_warps, fix others to defaults)
2. LLM searches 1D space (num_warps only)
3. Fix num_warps to best, optimize num_stages
4. Repeat for all 5 dimensions
5. Final config: best from each dimension

### Novelty Check
- **NOT like A/B/C:** Those search full config space. This decomposes into 1D searches.
- Similar to coordinate descent in optimization.

### How It Addresses Large Shape Problem
**Current issue:** Config space is huge (5 dimensions × 3-5 options each = 100s of combos). LLM explores randomly.

**Solution:** 
- 1D searches are easier: LLM only decides num_warps ∈ [2,4,8,16,32]
- For large shapes: Quickly learn "num_warps=4 best, persistent mode bad"
- Build optimal config incrementally

### Implementation Plan

1. **Define sub-search spaces:**
   ```python
   sub_searches = [
       {'name': 'num_warps', 'options': [2, 4, 8, 16, 32], 'default_context': {...}},
       {'name': 'num_stages', 'options': [1, 2, 3, 4], 'default_context': {...}},
       {'name': 'pid_type', 'options': ['flat', 'persistent_blocked'], 'default_context': {...}},
       {'name': 'block_sizes', 'options': [[1], [16], [32], [64]], 'default_context': {...}},
       {'name': 'indexing', 'options': ['pointer', 'tensor_descriptor'], 'default_context': {...}},
   ]
   ```

2. **Coordinate descent search:**
   ```python
   def compositional_search(shape):
       config = default_config()
       
       for sub in sub_searches:
           best_option = None
           best_time = float('inf')
           
           for option in sub['options']:
               test_config = config.copy()
               test_config[sub['name']] = option
               time = benchmark(test_config, shape)
               
               if time < best_time:
                   best_time = time
                   best_option = option
           
           config[sub['name']] = best_option
       
       return config
   ```

3. **LLM-guided version:**
   - Instead of trying all options, LLM suggests 2-3 options per dimension
   - Still decomposed (easier for LLM than full space)

### Success Criteria
- **Target score:** ≤0.88
- **Key metric:** Search efficiency
  - Current: ~15-20 configs benchmarked per shape
  - This: ~10-15 configs (fewer, but more targeted)
- **Large shape improvement:** Quickly identify "flat mode + low warps" pattern

### Risks
- **Risk:** Assumes dimensions are independent (may not be true)
  - **Mitigation:** Add final "joint optimization" stage
- **Risk:** Greedy search gets stuck in local optimum
  - **Mitigation:** Try multiple initialization points
- **Risk:** More complex implementation
  - **Mitigation:** Worth it if improves efficiency

### Effort Estimate
**6 hours**
- 2h: Implement compositional search framework
- 1h: Define sub-search spaces
- 3h: Run experiment and measure

---

## Idea 10: Transfer Learning from Cross-Entropy

### Core Innovation
**Use cross-entropy AOT data to guide softmax search**.

**Insight:** Cross-entropy and softmax are similar kernels:
- Both have reduction operations
- Both use exp/log
- Similar shapes (batch × vocab)

**Hypothesis:** Configs that work well for cross-entropy might transfer to softmax.

**Approach:**
1. Extract top-10 configs from cross-entropy AOT data
2. Use these as additional seeds for softmax
3. LLM explores: "These configs worked for similar kernel, try adapting them"

### Novelty Check
- **NOT like A/B/C:** Those use softmax-only data. This uses cross-kernel transfer.
- Novel: Leveraging similarity between kernels.

### How It Addresses Large Shape Problem
**Current issue:** Limited AOT data for softmax large shapes.

**Solution:**
- Cross-entropy has comprehensive AOT data (12 shapes, 1200+ configs)
- If cross-entropy found good configs for large dims, transfer to softmax
- More diverse seed pool → better chance of finding optimal for large softmax shapes

### Implementation Plan

1. **Analyze cross-entropy AOT data:**
   ```python
   # Load cross-entropy AOT
   ce_data = load_aot_data('cross_entropy')
   
   # Extract configs for large shapes (dim>4096)
   large_ce_configs = ce_data[ce_data['dim'] > 4096]
   
   # Get top-10 configs
   top10_ce = large_ce_configs.nsmallest(10, 'timing_ms')
   ```

2. **Transfer configs to softmax:**
   ```python
   # Adapt cross-entropy configs to softmax
   # (May need minor adjustments for kernel differences)
   softmax_seeds = adapt_configs(top10_ce, target_kernel='softmax')
   ```

3. **Use as seeds for softmax:**
   ```bash
   export HELION_LLM_TRANSFER_CONFIGS_PATH=cross_entropy_to_softmax.json
   python run_live.py --kernel softmax --shape-grid shape_grid.json --arm transfer_learning
   ```

### Success Criteria
- **Target score:** ≤0.86
- **Key metric:** Transfer effectiveness
  - Measure: How many cross-entropy configs work well for softmax?
  - Target: 40%+ of transferred configs in top-20 for softmax
- **Large shape focus:** Cross-entropy large-dim configs should help softmax large dims

### Risks
- **Risk:** Cross-entropy and softmax too different (configs don't transfer)
  - **Mitigation:** Analyze similarity first; if <30% overlap, skip this idea
- **Risk:** Cross-entropy AOT data may not cover same shapes
  - **Mitigation:** Check shape distribution first
- **Risk:** Overhead of adapting configs
  - **Mitigation:** Simple mapping (no complex transformations needed)

### Effort Estimate
**5 hours**
- 1h: Analyze cross-entropy AOT data
- 1h: Extract and adapt configs
- 1h: Implement transfer logic
- 2h: Run experiment and measure

---

## Rankings and Recommendations

### Ranking by Likelihood of Success

1. **Idea 1: Size-Dependent Strategy Switching** (HIGH: 70% confidence ≤0.90)
   - **Rationale:** Directly addresses root cause (large shapes fail with universal heuristics). Small shapes don't need heuristics; large shapes need specialized ones.
   - **Risk:** LOW - Simple to implement, clear logic
   - **Expected score:** 0.85-0.92

2. **Idea 6: Prompt Engineering - Dimension-Aware Guidance** (HIGH: 65% confidence ≤0.90)
   - **Rationale:** Cheapest to implement, teaches LLM principles instead of injecting seeds. Addresses large shape problem by explicit guidance.
   - **Risk:** MEDIUM - LLM might ignore guidance
   - **Expected score:** 0.87-0.93

3. **Idea 3: Negative Heuristics** (MEDIUM: 60% confidence ≤0.90)
   - **Rationale:** Novel approach (filter bad instead of seed good). Directly prevents known-bad configs for large shapes.
   - **Risk:** MEDIUM - Need accurate bad-pattern extraction
   - **Expected score:** 0.88-0.94

4. **Idea 2: Multi-Stage LLM Search** (MEDIUM: 55% confidence ≤0.90)
   - **Rationale:** More exploration → better configs for large shapes. Proven strategy in optimization.
   - **Risk:** MEDIUM - Longer runtime (3x), diminishing returns possible
   - **Expected score:** 0.85-0.95

5. **Idea 7: Shape-Specific Fine-Tuning** (MEDIUM: 50% confidence ≤0.90)
   - **Rationale:** Eliminates generalization problem entirely. Each shape gets dedicated heuristic.
   - **Risk:** MEDIUM - Held-out interpolation might fail
   - **Expected score:** 0.83-0.90

6. **Idea 5: Autotuner Meta-Optimization** (LOW-MEDIUM: 45% confidence ≤0.90)
   - **Rationale:** Systematic approach to finding best search hyperparameters. Large shapes might benefit from more exploration.
   - **Risk:** MEDIUM - May not improve score significantly
   - **Expected score:** 0.88-0.95

7. **Idea 9: Compositional Search** (LOW-MEDIUM: 40% confidence ≤0.90)
   - **Rationale:** More efficient search. Easier for LLM to optimize 1D subspaces.
   - **Risk:** HIGH - Assumes independence (may not hold)
   - **Expected score:** 0.86-0.96

8. **Idea 8: Learned Reward Model** (LOW: 35% confidence ≤0.90)
   - **Rationale:** ML-guided search is powerful. Could learn large-shape patterns.
   - **Risk:** HIGH - Complex implementation, may need lots of data
   - **Expected score:** 0.82-0.94 (high variance)

9. **Idea 4: Ensemble Autotuning** (LOW: 30% confidence ≤0.90)
   - **Rationale:** Combines strengths of multiple strategies. Exposes per-shape best strategy.
   - **Risk:** HIGH - 5x compute, unclear if ensemble beats best single strategy
   - **Expected score:** 0.85-0.98

10. **Idea 10: Transfer Learning from Cross-Entropy** (LOW: 25% confidence ≤0.90)
    - **Rationale:** Interesting idea, but cross-entropy and softmax may be too different.
    - **Risk:** HIGH - Transfer might not work
    - **Expected score:** 0.90-1.05 (wide range, uncertain)

---

## Top 3 Recommendations

### #1: Idea 1 - Size-Dependent Strategy Switching

**Why try first:**
- **Highest likelihood of success** (70% confidence)
- **Directly addresses the core problem:** Universal heuristics fail on large shapes because they're trained on mixed data
- **Clear action plan:** Build `observed_heuristics_large.json` from SM_005-008 only
- **Low risk:** Simple routing logic, no complex implementation
- **Fast iteration:** 4 hours total
- **Strong theoretical foundation:** Small shapes don't need help (baseline is optimal), large shapes need specialized patterns

**Expected outcome:** 0.85-0.92 (significant progress toward goal)

**If this fails:** We learn that even large-shape-specific heuristics don't help → problem is deeper than data distribution

---

### #2: Idea 6 - Dimension-Aware Prompt Engineering

**Why try second (or in parallel with #1):**
- **Lowest implementation cost** (3 hours)
- **Orthogonal to Idea 1:** Can combine (Idea 1 provides shape-specific examples, Idea 6 explains principles)
- **Teaches LLM patterns** instead of just showing examples
- **Addresses understanding gap:** LLM currently doesn't know large dims need simple configs

**Expected outcome:** 0.87-0.93 (modest improvement, but cheap)

**If this fails:** We learn that LLM can't follow abstract guidance → needs concrete examples only

**Synergy:** If both #1 and #2 work individually, combine them:
- #1 provides large-shape-specific examples
- #2 explains why those examples work
- Combined might achieve 0.80-0.85 (close to or hitting goal)

---

### #3: Idea 3 - Negative Heuristics

**Why try third:**
- **Novel approach:** All A/B/C tried positive seeds. This is fundamentally different (reject bad configs).
- **Complementary:** Works WITH any seed strategy (can combine with #1 or #2)
- **Data-driven:** Extract bad patterns from actual failures in A/B/C
- **Efficiency gain:** Skip known-bad configs → faster convergence

**Expected outcome:** 0.88-0.94 standalone, 0.82-0.88 combined with #1 or #2

**If this fails:** We learn that bad-pattern filtering doesn't help → config space exploration isn't the bottleneck

---

## Suggested Testing Order

### Sequential Testing (Lower Risk):
1. **Week 1:** Test Idea 1 (Size-Dependent Switching) - 4 hours
   - If ≤0.80 → SUCCESS, expand to cross-entropy
   - If 0.80-0.90 → Continue to Idea 6
   - If >0.90 → Continue to Idea 6

2. **Week 1:** Test Idea 6 (Dimension-Aware Prompts) - 3 hours
   - If ≤0.80 → SUCCESS
   - If 0.80-0.90 → Combine Ideas 1+6, test hybrid
   - If >0.90 → Continue to Idea 3

3. **Week 2:** Test Idea 3 (Negative Heuristics) - 5 hours
   - If ≤0.80 → SUCCESS
   - If 0.80-0.90 → Combine with best from 1/6
   - If >0.90 → Test Idea 2 (Multi-Stage) or Idea 5 (Meta-Optimization)

### Parallel Testing (Higher Risk, Faster Results):
- **Run Ideas 1 and 6 in parallel** (both cheap, orthogonal)
  - 4 hours to test both
  - If either ≤0.80 → SUCCESS
  - If both fail individually, test combination (1+6 hybrid)
  - Best case: 0.80-0.85 from synergy

---

## Contingency Plans

### If Top 3 All Fail (Score Still >0.90):
- **Pivot to Idea 2 (Multi-Stage)** or **Idea 8 (Reward Model)**
  - Both are more complex but higher potential upside
  - Multi-Stage: 6 hours, 55% confidence
  - Reward Model: 7 hours, 35% confidence but could hit 0.82

### If Score Improves to 0.85-0.90 Range:
- **Combination strategies:**
  - Size-dependent switching (#1) + dimension-aware prompts (#6)
  - Size-dependent switching (#1) + negative heuristics (#3)
  - All three combined (#1 + #6 + #3)
- **Fine-tuning:**
  - Expand large-shape examples from top-3 to top-10
  - Add more detailed dimension guidance
  - Stricter negative pattern filtering

### If No Progress After 5 Ideas:
- **Escalate to user with analysis:**
  - "Round-0 seeding paradigm fundamentally limited for this problem"
  - "LLM baseline (Opus 4.7) is already within 5-10% of optimal"
  - "20% improvement goal may require: (1) Better LLM model, (2) Multi-round tuning instead of round-0 focus, (3) Different kernel family"

---

## Key Insights from Historical Analysis

### What I Learned from EXPERIMENT_LOG.md:

1. **Approach A (1.0376) failed because:**
   - LLM-generated decision tree used overly complex configs
   - Heuristics suggested persistent mode, high warps for large dims
   - Actual winners were simpler: flat mode, low warps

2. **Approach B (1.0063) improved 3% because:**
   - Observed examples more reliable than synthetic rules
   - LLM saw actual winning configs with timing data
   - But still failed on large shapes (7-10% degradation)

3. **Approach C (1.0076) regressed 0.13% from B because:**
   - Combining decision tree + examples caused interference
   - Decision tree seed pulled LLM toward complex configs
   - Examples showed simple configs
   - LLM confused by conflicting signals

4. **Common pattern across all three:**
   - Small shapes: Already optimal, seeds don't help
   - Large shapes: Seeds make it WORSE (8-16% slower)
   - Problem is NOT lack of data; problem is round-0 seeding approach itself

### Why These Proposals Are Different:

1. **Idea 1 (Size-Dependent):** Doesn't use universal heuristic (root cause of A/B/C failures)
2. **Idea 6 (Prompt Engineering):** Doesn't inject seeds, teaches principles instead
3. **Idea 3 (Negative Heuristics):** Doesn't try to pick winners, rejects losers (inverted approach)
4. **Idea 2 (Multi-Stage):** Changes search workflow, not just seed content
5. **Ideas 7-10:** More exploratory, try if top 3 fail

**Critical lesson:** Don't propose more round-0 seeding variants. Need different paradigms.

---

## Conclusion

**Recommended immediate action:** Test Idea 1 (Size-Dependent Strategy Switching) ASAP.

**Rationale:**
- Highest success probability (70%)
- Clearest path to addressing large shape regression
- Fast implementation (4 hours)
- Low risk

**Backup plan:** If Idea 1 doesn't reach ≤0.80, combine with Idea 6 (dimension-aware prompts) for synergy.

**Success criteria:** Achieve 0.90 or better within 2 ideas tested. If score improves to 0.85-0.90 range, test combinations.

**Escalation trigger:** If 5 ideas tested and still >0.90, escalate to user with recommendation to either:
1. Adjust goal to more realistic target (0.85-0.90)
2. Invest in multi-round tuning instead of round-0 focus
3. Switch to different kernel family with more improvement headroom
