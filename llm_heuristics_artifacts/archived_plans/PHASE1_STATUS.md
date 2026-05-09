# Phase 1: Multi-Round Search - Status Report

## Manager: Prompt Optimization Orchestration

**Started:** 2026-05-08 22:50 UTC
**Current Status:** Running A1_baseline experiment
**Goal:** Determine if multi-round LLM search improves config quality

---

## Infrastructure Updates

### Modified Files
1. **run_live.py** - Added `--max-rounds` parameter support
   - Modified `_run_one_shape()` to accept `max_rounds` parameter
   - Added CLI argument `--max-rounds` (default=1)
   - Updated metadata tracking to include `max_rounds`
   - **Purpose:** Enable testing different numbers of LLM refinement rounds

---

## Experiment Timeline

### A1_baseline (IN PROGRESS)
**Objective:** Establish baseline with 1 round
**Command:**
```bash
python tools/run_live.py \
  --kernel cross_entropy \
  --arm baseline \
  --max-rounds 1 \
  --configs-per-round 5 \
  --repeats 3 \
  --shape-grid cross_entropy/shape_grid.json \
  --output-dir prompt_opt_experiments/A1_baseline
```

**Parameters:**
- Kernel: cross_entropy
- Shapes: 12 (7 train, 5 heldout)
- Rounds: 1
- Configs per round: 5
- Total configs per shape: ~5
- Repeats: 3
- Expected duration: 10-15 minutes
- Expected score: ~1.0 (baseline)

**Background Job:** bxavk27y0
**Output:** /home/dev/helion_choijon5/llm_heuristics_artifacts/loss_functions/prompt_opt_experiments/A1_baseline/

**Status:** Running (started 22:51 UTC)

---

### A1_round3 (QUEUED)
**Objective:** Test if 3 rounds improve over baseline
**Parameters:**
- Rounds: 3
- Configs per round: 5
- Total configs per shape: ~15
- Expected duration: 30-40 minutes
- Expected score: 0.90-0.95 (5-10% improvement expected)

**Decision Criteria:**
- If A1_round3 shows >5% improvement: Proceed to A1_round5
- If improvement <5%: Spawn Evaluate subagent to analyze patterns
- If no improvement: Consider alternative approaches (Proposal subagent)

---

### A1_round5 (QUEUED)
**Objective:** Test diminishing returns with 5 rounds
**Parameters:**
- Rounds: 5
- Configs per round: 10
- Total configs per shape: ~50
- Expected duration: 60-80 minutes
- Expected score: 0.88-0.93 (marginal improvement expected)

**Decision Criteria:**
- If score ≤ 0.90: SUCCESS - advance to Phase 2
- If score > 0.90 but improving: Continue testing
- If plateaued: Spawn Proposal for new ideas

---

## Success Criteria

### Minimum Viable Success
- **Score ≤ 0.90** on cross_entropy (10% improvement)
- Validates multi-round search helps

### Strong Success
- **Score ≤ 0.85** on cross_entropy (15% improvement)
- Clear trend: more rounds → better configs

### Stretch Goal
- **Score ≤ 0.80** on cross_entropy (20% improvement)
- Ready to advance to Phase 2 (Prompt Engineering)

---

## Evaluation Metrics

For each experiment, we compute:
1. **Best config quality** (geomean of best perf per shape)
2. **Per-shape analysis** (train vs heldout)
3. **Round-to-best** (which round found the winner)
4. **Convergence patterns** (diminishing returns curve)

**Key Insight:** We don't care about search speed - only final config quality matters (one-time compilation cost for serving)

---

## Next Steps

### After A1_baseline Completes:
1. Check output CSV and metadata
2. Compute baseline score (geomean of best configs)
3. Launch A1_round3 immediately
4. Compare results when A1_round3 completes

### After A1_round3 Completes:
1. Compute improvement over baseline
2. **If >5% improvement:** Launch A1_round5
3. **If <5% improvement:** Analyze patterns, decide next action
4. **If regression:** Investigate issues

### After A1_round5 Completes:
1. Plot diminishing returns curve (1 round → 3 rounds → 5 rounds)
2. Determine optimal round count
3. **If goal achieved (≤0.90):** Advance to Phase 2
4. **If not:** Spawn Proposal subagent for new ideas

---

## Resource Tracking

**Time Budget:** 20 hours total
**Invested So Far:** ~0.5 hours (setup + A1_baseline running)
**Remaining:** 19.5 hours

**Phase 1 Expected Total:** 2-3 hours
- A1_baseline: 15 minutes
- A1_round3: 40 minutes
- A1_round5: 80 minutes
- Analysis: 30 minutes

---

## Manager State

**Current Iteration:** 1
**Active Task:** A1_baseline measurement
**Subagent:** Measurement (self-executing)
**Next Action:** Wait for completion, then evaluate results

**Autonomous Operation:** ENABLED
- Manager will automatically launch next experiment based on results
- No user approval needed unless blocked or escalation criteria met
- Continuous iteration until Phase 1 complete or success achieved

---

## Contact/Escalation

**Escalate if:**
- Infrastructure failures (CUDA errors, import failures)
- All experiments show regression (scores worsen)
- Resource limits exceeded (>40 hours invested)

**Otherwise:** Manager continues autonomous iteration toward goal
