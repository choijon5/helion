# LLM Prompt Optimization Plan

**Context Shift:** Use case is model compilation for serving (one-time cost)
- Autotuning speed doesn't matter (happens once)
- **Quality of final config is ALL that matters**
- Focus on making LLM find BETTER configs, not faster

---

## Goal

**Old Goal:** `round0_best_geo ≤ 0.80` (heuristics make autotuning faster)

**New Goal:** Maximize quality of LLM-found configs for serving workloads

**Success Metric:** Best config performance after search (regardless of search time)

---

## Key Insight from Previous Work

**Heuristics (round-0 seeds) often HURT performance:**
- Softmax small shapes: Pure LLM (no seeds) = 0.78 (22% faster)
- Softmax with seeds: ~1.0 (neutral)
- Seeds constrain LLM exploration space

**Implication:** Focus on improving LLM search quality, not seeding strategies

---

## Experimental Approaches

### Approach 1: Multi-Round Search

**Hypothesis:** More LLM rounds → better final configs

**Test:**
- Baseline: `--max-rounds 1 --configs-per-round 5` (5 configs total)
- Test A: `--max-rounds 3 --configs-per-round 5` (15 configs total)
- Test B: `--max-rounds 5 --configs-per-round 10` (50 configs total)

**Metric:** Best config performance at end of search

**Expected:** Diminishing returns curve (1→3 rounds helps, 3→5 marginal)

**Implementation:**
- Run cross_entropy with different round counts
- Compare best config from each setting
- Measure: time-to-best and final quality

---

### Approach 2: Adaptive Refinement Strategy

**Hypothesis:** Different rounds need different strategies

**Current:** Same strategy all rounds (2/3 mutations of anchor 1, 1/3 of anchor 2)

**Proposed:**
```
Round 1: Broad exploration
  - 60% diverse families (3+ different block_sizes)
  - 30% balanced configs
  - 10% aggressive

Round 2: Exploit best
  - 70% single-field mutations of best anchor
  - 20% two-field mutations
  - 10% try opposite extreme

Round 3: Fine-tune
  - 80% tiny tweaks (±1 on best fields)
  - 20% interpolate between top-2 configs

Round 4+: Desperate search
  - Try configs LLM avoided (high num_warps, aggressive pipelining)
  - Sample from failure boundary
```

**Implementation:**
- Modify `_refinement_strategy_lines()` in `prompting.py`
- Add round-number-aware guidance
- Test on cross_entropy

---

### Approach 3: Enhanced Feedback Analysis

**Hypothesis:** Better feedback → LLM makes smarter choices

**Current feedback shows:**
- List of results (best first)
- Failed config patterns
- Top config patterns (generic)

**Enhanced feedback:**
```
## Performance Analysis
- num_warps scaling: 4→8 (+5%), 8→16 (-20%)
- Optimal point: num_warps=8
- Hypothesis: Memory-bound kernel, excessive warps hurt

## Configuration Insights
- block_sizes=[2] best for this shape
- num_stages>2 causes register spill
- tensor_descriptor 30% slower than pointer

## Next Round Guidance
- Explore: num_warps=6 (between 4 and 8)
- Avoid: num_stages>2, tensor_descriptor
- Try: block_sizes=[1,2,3] with num_warps=4-8
```

**Implementation:**
- Enhance `format_results_for_llm()` in `feedback.py`
- Add trend analysis (does increasing X improve/hurt?)
- Add hypothesis generation

---

### Approach 4: Theoretical Bounds Guidance

**Hypothesis:** Hardware-aware guidance improves LLM choices

**Current:** LLM sees GPU model, SM count
**Missing:** Theoretical bottleneck analysis

**Add to prompt:**
```
## Kernel Characteristics
- Operation: softmax reduction
- Memory traffic: 8 MB/call
- Compute ops: 100K FLOPs
- Memory bandwidth: 900 GB/s (GPU spec)
- Compute throughput: 20 TFLOPS (GPU spec)

## Bottleneck Analysis
- Memory time: 8.9 ms (bandwidth bound)
- Compute time: 0.005 ms (not compute bound)
- **Bottleneck: Memory bandwidth**

## Optimization Strategy
- Favor: Larger tiles (reduce memory accesses)
- Avoid: Excessive parallelism (doesn't help bandwidth)
- Focus: Memory access patterns, cache utilization
```

**Implementation:**
- Add `compute_kernel_bottleneck()` to `workload.py`
- Include in initial prompt
- Test if LLM uses this info effectively

---

### Approach 5: Ensemble Search

**Hypothesis:** Multiple independent searches → better configs

**Method:**
- Run LLM search 3 times with different random seeds
- Each search explores different config space regions
- Take best config from all 3 runs

**Implementation:**
```bash
for seed in 1 2 3; do
  python run_live.py --seed $seed --output-dir results_seed_$seed
done
# Merge results, take best overall
```

**Trade-off:** 3x time, but for serving (one-time cost) this is acceptable

---

### Approach 6: Success Pattern Learning

**Hypothesis:** Learning from successes > avoiding failures

**Current emphasis:** "Failed Config Patterns" (what not to do)
**Add emphasis:** "Why Winners Win" (what to repeat)

**Enhanced prompt:**
```
## Top Configs Analysis

Config 1 (12.3ms - BEST):
  - block_sizes=[2], num_warps=4, num_stages=1
  - Why it works: Small tiles reduce overhead, low warp count avoids contention
  - Key trait: Minimal resource usage

Config 2 (13.1ms):
  - block_sizes=[4], num_warps=8, num_stages=2
  - Why slower: Larger tiles increase latency, more warps compete for bandwidth
  - Lesson: This shape prefers simplicity over parallelism

## Winning Pattern
- Conservative configs (4-8 warps, 1-2 stages) dominate
- Small block_sizes ([1-4]) work best
- Hypothesis: Overhead-sensitive workload

## Next Round Strategy
- Double down: More configs like winner (block_sizes=[1,2,3])
- Avoid: Aggressive pipelining (num_stages>2)
- Explore: Intermediate num_warps=6
```

**Implementation:**
- Add `analyze_winner_traits()` to `feedback.py`
- Generate hypotheses about why configs succeed
- Guide next round based on winning patterns

---

## Execution Plan

### Phase 1: Quick Wins (2-3 hours)

**Experiment A1: Multi-Round Search**
1. Run cross_entropy with 1/3/5 rounds
2. Compare best config quality
3. Measure time-to-best

**Expected Result:**
- 1 round: baseline (score ~1.0)
- 3 rounds: 5-10% improvement (score ~0.90-0.95)
- 5 rounds: marginal (score ~0.88-0.93)

**Decision:**
- If 3 rounds >> 1 round: adopt 3 rounds as default
- If 5 rounds >> 3 rounds: increase to 5
- If no improvement: investigate why (LLM not learning?)

---

### Phase 2: Prompt Engineering (4-6 hours)

**Experiment A2: Enhanced Feedback**
1. Implement trend analysis in feedback
2. Add hypothesis generation
3. Test on cross_entropy

**Experiment A3: Adaptive Strategy**
1. Implement round-specific strategies
2. Test on cross_entropy
3. Compare to uniform strategy

**Decision Criteria:**
- If score improves >5%: keep enhancement
- If neutral (<2% change): too complex, revert
- If degrades: LLM confused by extra info

---

### Phase 3: Advanced Techniques (6-8 hours)

**Experiment A4: Theoretical Guidance**
1. Implement bottleneck analysis
2. Add to prompt
3. Test on multiple kernels

**Experiment A5: Ensemble Search**
1. Run 3 parallel searches
2. Merge results
3. Compare best vs single search

---

## Success Criteria

### Minimum Viable Success
- Any approach achieves `score ≤ 0.90` on cross_entropy
- Validates that prompt optimization helps

### Strong Success
- Achieve `score ≤ 0.85` on cross_entropy
- Approach generalizes to kl_div, jsd

### Stretch Goal
- Achieve `score ≤ 0.80` on multiple kernels
- Document systematic prompt optimization methodology

---

## Autonomous Manager Integration

Use the Manager.md framework to iterate:

1. **Measurement:** Run experiment with specific prompt modifications
2. **Evaluate:** Analyze results, compare to baseline
3. **Decide:**
   - If improvement >5%: Adopt change, test next approach
   - If neutral: Try variant or move to next approach
   - If regression: Revert, understand why
4. **Implement:** Build next variant
5. **Proposal:** When stuck, generate new ideas

**Manager tracks:**
- Best score achieved so far
- Which prompt modifications helped
- Time budget remaining
- When to escalate to user

---

## Key Differences from Previous Experiments

| Old Approach | New Approach |
|--------------|--------------|
| Build heuristics to seed LLM | Improve LLM search quality |
| Focus on round-0 speed | Focus on final config quality |
| Test on 12 shapes, 1 round | Test on fewer shapes, more rounds |
| Measure `round0_best_geo` | Measure best config after full search |
| Seeds constrain LLM | Let LLM explore freely |
| Optimize for fast autotuning | Optimize for serving performance |

---

## Implementation Checklist

- [ ] Create experiment runner for multi-round tests
- [ ] Implement trend analysis in feedback.py
- [ ] Add round-adaptive strategies to prompting.py
- [ ] Build bottleneck analysis for workload.py
- [ ] Create ensemble search script
- [ ] Update MANAGER_STATE.json with new goal
- [ ] Set up autonomous iteration loop

---

## Expected Timeline

- **Phase 1 (Multi-round):** 2-3 hours
- **Phase 2 (Prompt engineering):** 4-6 hours
- **Phase 3 (Advanced):** 6-8 hours
- **Total:** 12-17 hours of GPU time

For overnight run: Start with Phase 1 (multi-round), auto-advance to Phase 2 if successful.

---

## Fallback Plan

If all approaches fail to improve over baseline:

**Hypothesis 1:** LLM already at peak performance
- Current prompts are near-optimal
- Additional rounds don't help (diminishing returns at round 1)

**Hypothesis 2:** Test harness issues
- Measurement variance obscures real improvements
- Need more repeats or different shapes

**Hypothesis 3:** Wrong optimization target
- Optimizing average score, but serving cares about tail latency
- Should optimize P95/P99 instead

**Action:** Escalate to user with findings
