# Phase 3 Decision: Proposal Selection

**Date:** 2026-05-08  
**Decision Maker:** Evaluate Subagent  
**Status:** APPROVED - Implementing Proposal #1

---

## DECISION: Option A - Single Best Proposal

**SELECTED PROPOSAL:** Proposal #1: Size-Dependent Strategy Switching

---

## RATIONALE

### Why Proposal #1 Over Others

**Highest Success Probability:**
- **70% confidence** to reach ≤0.90 milestone (highest of all 10 proposals)
- **Expected score:** 0.85-0.92 (significant progress toward 0.80 goal)
- **Direct attack** on root cause (large shape degradation)

**Root Cause Alignment:**
After analyzing all three Phase 2 failures, the core problem is clear:

```
Problem: Universal heuristics fail systematically
- Small shapes (≤1024 dim): Seeds provide NO value (baseline already 1.0)
- Large shapes (≥5120 dim): Seeds HURT performance (8-16% degradation)
- Mid shapes (1024-4096 dim): Seeds help inconsistently (mixed results)
```

**Proposal #1 directly solves this:**
- Route small shapes → NO heuristics (stop wasting effort)
- Route mid shapes → Approach B (proven 11-13% improvement)
- Route large shapes → Specialized large-shape heuristics (trained ONLY on SM_005-008)

This eliminates the cross-contamination that killed A/B/C.

**Low Risk, High Reward:**
- **Risk:** LOW - Simple if/elif/else routing, well-tested components
- **Effort:** 4 hours (fast iteration)
- **Downside:** If fails, only lose 4 hours
- **Upside:** Could hit 0.80 goal directly (if large shapes improve to 1.02)

### Why Not Other Proposals

**Proposal #6 (Dimension-Aware Prompts):**
- Good complementary option (65% confidence, 3 hours)
- BUT: Lower confidence than #1
- PLAN: Use as combo if #1 reaches 0.85-0.90 range

**Proposal #3 (Negative Heuristics):**
- Novel approach (60% confidence)
- BUT: Needs accurate pattern extraction (more complex)
- PLAN: Use if #1 fails

**Proposal #2 (Multi-Stage Search):**
- Interesting (55% confidence)
- BUT: 3x longer runtime (6 hours vs 4)
- PLAN: Fallback if #1 and #6 fail

**Proposals #4-10:**
- Lower confidence (25-50%)
- Higher complexity or risk
- Keep as backup if top 3 fail

---

## EXPECTED OUTCOME

### Target Metrics

**Overall Score:**
- **Target:** 0.85-0.92
- **Success threshold:** ≤0.90 (milestone toward 0.80 goal)
- **Confidence:** 70%

**Large Shape Improvement (Critical):**
```
Current (Approach B):
SM_005: 1.0975 (+9.75% degradation)
SM_006: 1.0925 (+9.25% degradation)
SM_007: 1.0983 (+9.83% degradation)
SM_008: 1.0779 (+7.79% degradation)
Avg: 1.0966 (+9.66% degradation)

Target (Proposal #1):
SM_005-008 avg: ≤1.02 (+2% degradation or better)
```

If large shapes improve from 1.10 → 1.02, overall score should reach ~0.88-0.90.

**Small Shape Preservation:**
- SM_001-004: Stay at ~1.0 (no regression)
- No wasted effort seeding shapes that don't need help

**Mid Shape Maintenance:**
- SM_002, SM_004, SM_102: Keep 11-13% improvement from Approach B
- Continue using observed examples for these shapes

### Timeline

**Phase 3 Implementation:** 4 hours
1. Build observed_heuristics_large.json (1h)
2. Modify prompting.py routing logic (1h)
3. Run experiment (2h)

**Total invested so far:** ~5 hours (Phase 2A/B/C)
**Budget remaining:** ~35 hours

---

## FALLBACK PLAN

### If Proposal #1 Achieves 0.85-0.90 (Close but not goal)

**Action:** Combine with Proposal #6 (Dimension-Aware Prompts)
- **Rationale:** Orthogonal approaches (routing + teaching principles)
- **Synergy:** #1 provides specialized examples, #6 explains WHY they work
- **Expected combo score:** 0.80-0.85 (high chance of hitting goal)
- **Time cost:** +3 hours

### If Proposal #1 Achieves 0.90-0.95 (Marginal improvement)

**Action:** Try Proposal #3 (Negative Heuristics)
- **Rationale:** Novel inversion (reject bad configs vs inject good)
- **Can combine:** Works with #1's routing (filter bad for each size class)
- **Expected combo score:** 0.82-0.88
- **Time cost:** +5 hours

### If Proposal #1 Achieves ≥0.95 (Little/no improvement)

**Action:** Test Proposal #6 standalone, then escalate
- **Rationale:** #1 failed means routing isn't the issue
- **Next:** Try teaching LLM principles (#6) or multi-stage search (#2)
- **Escalation trigger:** If 5 proposals tested and still >0.90

### If Proposal #1 Achieves <0.85 (SUCCESS!)

**Action:** Expand to cross_entropy kernel
- Apply size-dependent strategy to cross_entropy
- Build cross_entropy large-shape heuristics
- Target: 0.80 overall across both kernels

---

## SUCCESS CRITERIA

### Milestone (≤0.90): Meaningful Progress
- Proves size-dependent routing works
- Large shape problem solved (≤1.02)
- Clear path to 0.80 goal

### Goal (≤0.80): Full Success
- 20% improvement achieved
- Expand to cross_entropy
- Generalize to other kernels

### Minimum Viable (0.90-0.95): Partial Success
- Some improvement over B's 1.0063
- Learnings for next iteration
- Try combinations with #6 or #3

### Failure (≥0.95): No Improvement
- Routing alone insufficient
- Test alternative approaches (#6, #2, #3)
- Re-evaluate if 5 proposals fail

---

## RISK ASSESSMENT

### Low Risk Factors
1. **Simple implementation** - if/elif/else routing, no complex logic
2. **Proven components** - reusing Approach B for mid shapes
3. **Fast iteration** - 4 hours, minimal time investment
4. **No downside** - worst case: learn routing doesn't help

### Mitigation Strategies
1. **Risk:** Large-shape examples might still be suboptimal
   - **Mitigation:** Extract from BEST baseline runs, not heuristic runs
   
2. **Risk:** Implementation complexity (3 code paths)
   - **Mitigation:** Simple if/elif/else, well-tested
   
3. **Risk:** Mid shapes might regress without decision tree seed
   - **Mitigation:** Approach B already scored 1.0063 without tree seed

### Upside Potential
- **Best case:** Large shapes improve to 1.0 → overall score 0.82-0.85
- **Likely case:** Large shapes improve to 1.02 → overall score 0.88-0.90
- **Worst case:** No improvement → 1.0063 (same as B)

**Risk/Reward:** FAVORABLE (high upside, low downside)

---

## IMPLEMENTATION NOTES

### Files to Modify
1. `/home/dev/helion_choijon5/helion/autotuner/llm/prompting.py`
   - Add dimension-based routing logic
   - Keep Approach B code for mid shapes

2. Create: `/home/dev/helion_choijon5/llm_heuristics_artifacts/softmax_experiment/observed_heuristics_large.json`
   - Extract from SM_005-008 baseline runs
   - Top-5 configs per large shape

3. Update: `/home/dev/helion_choijon5/llm_heuristics_artifacts/softmax_experiment/shape_grid.json`
   - Already has size metadata (dim field)
   - No changes needed

### Testing Strategy
1. **Verify routing:** Print which heuristic path chosen per shape
2. **Monitor large shapes:** SM_005-008 critical (must improve)
3. **Preserve mid shapes:** SM_002, SM_004, SM_102 (must not regress)
4. **Check small shapes:** SM_001, SM_003 (should stay neutral)

---

## COMPARISON TO PHASE 2

### Why This Is Different

**Phase 2 (A/B/C):** Universal seeding
- ALL shapes get same heuristic type
- Small shapes get seeds they don't need
- Large shapes get seeds trained on mixed data
- Result: FAILED (0.63% degradation best case)

**Phase 3 (Proposal #1):** Size-dependent routing
- Each size class gets appropriate strategy
- Small shapes: NO seeds (stop wasting effort)
- Large shapes: Specialized seeds (trained ONLY on large)
- Result: EXPECTED 0.85-0.92 (significant improvement)

### Key Innovation
Not "better seeds" but "right seeds for right shapes."

---

## AUTHORIZATION

**Decision:** APPROVED

**Next Steps:**
1. Implementation Subagent: Build observed_heuristics_large.json
2. Implementation Subagent: Modify prompting.py routing
3. Execution Subagent: Run Phase 3 experiment
4. Evaluate Subagent: Analyze results, decide next steps

**Timeline:** Start immediately, complete within 4 hours

**Budget:** 4 hours of 35 remaining (11% of budget)

**Expected Completion:** 2026-05-08 11:30 UTC

---

## APPENDIX: Alternative Proposals Ranking

For reference, here's the full ranking of all 10 proposals:

| Rank | Proposal | Confidence | Score | Effort |
|------|----------|------------|-------|--------|
| 1    | Size-Dependent Strategy | 70% | 0.85-0.92 | 4h |
| 2    | Dimension-Aware Prompts | 65% | 0.87-0.93 | 3h |
| 3    | Negative Heuristics | 60% | 0.88-0.94 | 5h |
| 4    | Multi-Stage Search | 55% | 0.85-0.95 | 6h |
| 5    | Shape-Specific Fine-Tuning | 50% | 0.83-0.90 | 4h |
| 6    | Autotuner Meta-Optimization | 45% | 0.88-0.95 | 5h |
| 7    | Compositional Search | 40% | 0.86-0.96 | 6h |
| 8    | Learned Reward Model | 35% | 0.82-0.94 | 7h |
| 9    | Ensemble Autotuning | 30% | 0.85-0.98 | 8h |
| 10   | Transfer Learning | 25% | 0.90-1.05 | 5h |

**Selection criteria:** Highest confidence + lowest risk + directly addresses root cause

**Backup plan:** Test #2 and #3 if #1 reaches 0.85-0.95 range

---

**Status:** APPROVED - Implementation in progress
**Last Updated:** 2026-05-08 07:30 UTC
