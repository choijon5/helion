# Manager: LLM Heuristics Project

**Goal:** Achieve 20% performance improvement (round0_best_geo ≤ 0.80) through iterative experimentation

**Manager Role:** Orchestrate subagents, track progress, ensure continuous iteration toward goal

---

## Manager Responsibilities

### ✅ DO:
1. **Spawn subagents** for specific tasks (measurement, evaluate, proposal, implementation)
2. **Track progress** toward 20% improvement goal using MANAGER_STATE.json
3. **Keep momentum** - continuously spawn next subagent based on results
4. **Execute recommendations** from Evaluate subagent
5. **Push forward** without waiting for user approval (unless truly blocked)
6. **Maintain context** across multiple agent runs
7. **Update state after each iteration:**
   - MANAGER.md (human-readable status)
   - MANAGER_STATE.json (structured state for decisions)

### ❌ DON'T:
1. **Don't run experiments directly** - delegate to Measurement subagent
2. **Don't analyze data yourself** - delegate to Evaluate subagent
3. **Don't make strategic decisions** - delegate to Evaluate subagent
4. **Don't update the plan** - delegate to Evaluate subagent
5. **Don't document experiments** - delegate to Evaluate subagent
6. **Don't stop for approval** - keep iterating until goal reached or truly blocked
7. **Don't lose sight of goal** - every action should move toward 0.80 target

**Manager is a dumb orchestrator** - spawns subagents, executes recommendations, nothing more.

### When Manager Escalates to User (ONLY):

**Infrastructure issues:**
- Critical files missing/corrupted
- Environment broken (CUDA, PyTorch, dependencies)
- Persistent import/compilation errors across multiple approaches

**Exhaustion criteria (all 3 must be true):**
- Tested 5+ different approaches (A, B, C, D, E...)
- Score stuck >0.90 for 3+ consecutive approaches
- Proposal subagent out of ideas (returns "no new proposals")

**Resource limits:**
- >40 hours invested without hitting goal
- Storage/compute budget exceeded
- User-specified constraints violated

**Strategic pivots (require user input):**
- Abandon kernel family entirely
- Change goal metric (e.g., from 0.80 to 0.85)
- Switch to different hardware/platform

**Otherwise: KEEP ITERATING** - Manager doesn't escalate prematurely

---

## Subagent Architecture

### Measurement Subagent
**Purpose:** Execute experiments, collect data
**Input:** Experiment design (which approach to test, parameters)
**Output:** Raw CSV data, timing measurements
**Tools:** Bash (run_live.py), file writes

**Example tasks:**
- Run baseline measurement on 12 shapes
- Run heuristics_A with decision tree seed
- Run heuristics_B with observed examples
- Benchmark held-out shapes

**Error handling:**
- Compilation failures (15-20%): EXPECTED, save to CSV with status='error'
- Import errors: Report to Manager (may need Implementation to fix)
- Infrastructure failures: Report to Manager (escalate if blocking)
- Success rate <50%: Flag in report, Evaluate will analyze

**CRITICAL: Data Collection Method**
- For **live experiments** (baseline vs heuristics comparison): Use `run_live.py` with LLMGuidedSearch
  - Fast: ~5-6 configs per shape, 2-5 minutes per shape
  - Purpose: Compare LLM with/without heuristic seeds
  
- For **gathering new AOT training data**: Use LFBOTreeSearch with full autotuning
  - Command: `HELION_FORCE_AUTOTUNE=1 python script_that_calls_kernel.py`
  - Comprehensive: ~100-200 configs per shape, 30-60 minutes per shape
  - Purpose: Generate training data for building new heuristics
  - See: `AOT_DATA_GENERATION.md` for detailed methodology

### Evaluate Subagent
**Purpose:** Analyze experimental results AND decide next action
**Input:** 
- CSV files from measurements (current iteration)
- **EXPERIMENT_LOG.md** (past learnings - MUST READ FIRST)
- MANAGER_STATE.json (current progress, budget)
- Current plan, resource constraints
**Output:** 
- Scores and analysis (round0_best_geo, per-shape breakdown, patterns)
- Decision on next experiment
- Updated documentation (EXPERIMENT_LOG.md + PLAN.md + MANAGER_STATE.json)
**Tools:** Python, Read, Write, Edit

**CRITICAL: Learning from History**
Before making decisions, ALWAYS:
1. **Read EXPERIMENT_LOG.md completely**
2. **Extract key learnings:** What worked? What failed? Why?
3. **Check for repeated mistakes:** Am I about to try something that already failed?
4. **Identify patterns:** Do similar approaches have similar outcomes?

**Example learnings to extract:**
- "Approach A failed because LLM-generated configs don't match actual winners"
- "Large dims (>5120) consistently regress with approach X"
- "Held-out extrapolation works well when we have Y"
- "High error rates indicate Z problem"

**Responsibilities:**
1. **Analyze results:**
   - Compute round0_best_geo (baseline vs heuristics)
   - Per-shape breakdown (train, interp, extrap)
   - Check error rates (if >50%, flag for Implementation retry)
   - Identify patterns (which shapes improved/regressed, why)
   - **Compare to past attempts from EXPERIMENT_LOG**

2. **Make decision WITH HISTORICAL CONTEXT:**
   - If goal reached (≤0.80) → "SUCCESS, expand to cross_entropy"
   - If high error rate (>50%) → "Retry Implementation of [approach]"
   - If pre-planned approach exists AND not tried before → "Test Approach B"
   - If approach similar to past failure → Skip or modify
   - If stuck → "Spawn Proposal for new ideas"
   - Weigh tradeoffs: time, risk, expected ROI, budget remaining, past results

3. **Update documentation:**
   - **EXPERIMENT_LOG.md:** Add new findings AND reference past learnings
   - **LLM_HEURISTICS_REVISED_PLAN.md:** Update next steps, mark phases complete
   - **MANAGER_STATE.json:** Update iteration count, best score, time invested

**Decision output options:**
- "SUCCESS" - Goal achieved
- "Test Approach [X]" - Run next pre-planned approach (not tried before)
- "Retry Implementation of [X]" - Fix broken implementation
- "Skip Approach [X], try [Y]" - Learned from past that X won't work
- "Spawn Proposal" - Need new ideas (existing approaches exhausted/learned from)

**Example tasks:**
- "Read EXPERIMENT_LOG: A failed (LLM configs wrong), B uses real winners - try B"
- "Check history: Is Approach D similar to failed A? If yes, skip to E"
- "Past shows large dims always fail with X pattern - avoid similar approaches"
- "Measurement had 90% error rate - check log if this happened before"

**Anti-patterns (DON'T DO):**
- ❌ Make decisions without reading EXPERIMENT_LOG
- ❌ Suggest approach that's conceptually similar to documented failure
- ❌ Ignore patterns from past iterations
- ❌ Forget why previous approaches failed

### Proposal Subagent
**Purpose:** Generate new ideas when existing approaches exhausted
**Input:** 
- Evaluate output (scores, patterns, failure modes)
- **EXPERIMENT_LOG.md** (past attempts - MUST READ)
- Current plan
**Output:** 2-3 concrete experiment proposals with expected outcomes
**Tools:** Read existing data, reason about patterns

**When to spawn:** Only when Evaluate decides "Spawn Proposal"

**CRITICAL: Learn from Past Failures**
Before proposing, MUST:
1. **Read EXPERIMENT_LOG.md completely**
2. **Identify what's been tried:** Don't propose similar ideas
3. **Understand why past approaches failed:** Address root causes
4. **Find unexplored directions:** Novel angles, not variations of failures

**Output format:**
Each proposal must include:
- **Idea:** What to try (e.g., "Use top-10 configs instead of top-3")
- **Rationale:** Why this might work (e.g., "More diversity in examples")
- **Novelty check:** "Different from past attempts because..." (reference EXPERIMENT_LOG)
- **Expected score:** Predicted round0_best_geo (e.g., "0.75-0.85")
- **Time cost:** Implementation + measurement time (e.g., "2 hours")
- **Risk:** LOW/MEDIUM/HIGH

**What happens after:**
- Manager spawns Evaluate with Proposal output
- Evaluate picks best proposal based on expected ROI + novelty
- Manager executes Evaluate's choice

**Example tasks:**
- "Read log: A/B/C all used seeds, all failed. Propose: no seeds, pure LLM prompt engineering"
- "Log shows large dims always fail. Propose: train separate heuristic for large dims"
- "All heuristic-based approaches failed. Propose: use ensemble of random LLMs"

**Anti-patterns (DON'T DO):**
- ❌ Propose variations of documented failures
- ❌ Ignore root causes from EXPERIMENT_LOG
- ❌ Generate ideas without reading history

### Implementation Subagent
**Purpose:** Build new tools, approaches, and infrastructure from proposals/decisions
**Input:** Decision/Proposal output, technical requirements, existing codebase
**Output:** Working code (scripts, configs, data files), implementation report
**Tools:** Read, Write, Edit, Bash

**Example tasks:**
- "Build observed_heuristics.json with top-5 configs per dim bucket from AOT data"
- "Modify prompting.py to inject observed examples into LLM prompt"
- "Extract and format comprehensive AOT data into training-ready format"
- "Fix infrastructure bug: add softmax support to run_live.py"
- "Create script to extract heuristic configs from baseline winners"

**When to spawn:**
- Decision recommends approach that needs new code (e.g., "Test Approach B")
- Proposal generates idea requiring implementation
- Infrastructure issue blocks progress
- Data needs preprocessing/formatting

**Success criteria:**
- Code runs without errors
- Produces expected outputs (files, configs, etc.)
- Documented and ready for Measurement subagent to use


---

## Workflow Loop

```
START
  ↓
Manager: Spawn Measurement Subagent
  ↓
Measurement: Run experiment, save CSV
  ↓
Manager: Spawn Evaluate Subagent
  ↓
Evaluate: Analyze scores + Decide next action + Update docs
  ├─ Computes round0_best_geo, per-shape patterns
  ├─ Updates EXPERIMENT_LOG.md (what happened)
  ├─ Updates PLAN.md (what's next)
  └─ Returns decision:
      ├─ "SUCCESS" → expand to cross_entropy
      ├─ "Test Approach B" → run next experiment
      └─ "Spawn Proposal" → need new ideas
      ↓
Manager: Execute Evaluate's decision
  ├─ If needs new code → Spawn Implementation
  │   ↓
  │   Implementation: Build tools/data/configs
  │   ↓
  │   Manager: Spawn Measurement
  │   ↓
  │   Measurement: Run experiment
  │   ↓
  │   [If high error rate >50%]
  │   Manager: Spawn Evaluate (with error report)
  │   ↓
  │   Evaluate: Analyze errors, decide:
  │     ├─ "Retry Implementation with fixes" → Manager spawns Implementation again
  │     ├─ "Errors acceptable, continue" → Manager continues normal flow
  │     └─ "Try different approach" → Manager executes new decision
  │
  ├─ If ready to run → Spawn Measurement directly
  │
  ├─ If "Spawn Proposal" → 
  │   ↓
  │   Manager: Spawn Proposal
  │   ↓
  │   Proposal: Generate 2-3 new experiment ideas with expected outcomes
  │   ↓
  │   Manager: Spawn Evaluate (with Proposal output as input)
  │   ↓
  │   Evaluate: Pick best proposal + decide next action
  │   ↓
  │   (back to "Manager: Execute Evaluate's decision")
  │
  └─ If "SUCCESS" → Document, spawn cross_entropy expansion
      ↓
LOOP back to Manager
```

**Key principles:**
- **Manager:** Pure orchestrator - spawns, executes, tracks (NO strategic decisions)
- **Evaluate:** Makes ALL decisions (strategic) + analysis + documentation
- **Manager escalates to user ONLY if:** Infrastructure broken, all approaches exhausted, resource limits hit

---

## Decision Criteria

### When to iterate current approach:
- Score improving but not yet at goal (e.g., 0.85 → 0.82)
- Clear next step exists (e.g., "increase K in top-K from 3 to 5")
- Low risk, incremental improvement expected

### When to try new approach:
- Current approach plateaued (e.g., stuck at 0.87 for 3 iterations)
- Analysis reveals fundamental limitation (e.g., "decision tree only uses 5 configs")
- Proposal shows high expected value (e.g., "hybrid approach could combine strengths")

### When to expand scope:
- Goal achieved on current kernel (e.g., softmax ≤ 0.80)
- Approach proven robust (train + held-out both good)
- Ready to test generalization (apply to cross_entropy, kl_div)

### When to gather new AOT data:
- Need heuristic for kernel without existing AOT data
- Existing AOT data doesn't cover test shape distribution
- Held-out extrapolation fails badly (>1.0) - may need more training shapes

**Method: Use LFBOTreeSearch with full autotuning**
```bash
HELION_FORCE_AUTOTUNE=1 python3 << 'EOF'
from examples.{kernel} import {fn}
import torch

# Call kernel to trigger autotuning
x = torch.randn(shape, device='cuda', dtype=dtype)
result = {fn}(x)
# LFBOTreeSearch explores ~100-200 configs, saves to cache
EOF
```
Time: 30-60 min per shape, but provides comprehensive training data

### When to ask user:
- Truly blocked (infrastructure issue, need new data not available)
- Strategic pivot needed (abandon current kernel, change goal metric)
- Resource decision (spend 12+ hours on comprehensive AOT vs move on)

---

## Progress Tracking

### Current Status (Updated after each iteration):

**See MANAGER_STATE.json for structured state tracking**

**Iteration:** 2
**Kernel:** Softmax (primary focus)
**Best Score:** 1.0376 (Approach A - failed)
**Current Task:** Testing Approach B (observed examples)
**Active Subagent:** Measurement (a45504a5a651b41b0)

**Progress:**
- ✅ Phase 1: Baseline complete (1.0 baseline)
- ✅ Phase 2A: Approach A tested (1.0376 - failed)
- 🔄 Phase 2B: Approach B testing (in progress)

**Paused/Deferred:**
- ⏸️ Cross-entropy - AOT complete, waiting for softmax winning approach

**Key Decisions:**
- **Decision 1:** Test Approach B after A failed
  - Reason: A used LLM-generated heuristic (wrong patterns), B uses actual train winners
  - Expected: 0.85-0.95 (better than A but uncertain if hits 0.80)
  - Backup: Try Approach C (hybrid) or spawn Proposal

### Key Metrics:
- **Target:** round0_best_geo ≤ 0.80
- **Current Best:** TBD
- **Gap to Goal:** TBD
- **Iterations Run:** 0
- **Time Invested:** ~1 hour (planning)

---

## Iteration Log

### Iteration 1: Baseline Measurement
**Goal:** Establish baseline performance without heuristics
**Kernel:** Softmax
**Subagent:** Measurement (ae6a6d0382c62e42b)
**Actions:**
- Fixed run_live.py to support softmax
- Create shape_grid.json (12 shapes: 8 train, 2 held-out interp, 2 held-out extrap)
- Run baseline on all 12 shapes × 3 repeats
**Expected Output:** baseline.csv with ~180-200 rows
**Expected Time:** 2-3 hours
**Status:** IN PROGRESS (started 2026-05-08 ~06:00)

### Iteration 2: Test Decision Tree (Approach A)
**Goal:** Test existing decision tree heuristic
**Subagent:** Measurement
**Actions:**
- Run heuristics_A with existing softmax heuristic
**Dependencies:** Iteration 1 complete
**Status:** PENDING

### Iteration 3: Analyze A vs Baseline
**Goal:** Quantify improvement from decision tree
**Subagent:** Analysis
**Actions:**
- Compute round0_best_geo (baseline vs A)
- Break down by train/interp/extrap
- Identify best/worst shapes
**Output:** scores_A.json, analysis report
**Status:** PENDING

### Iteration 4+: TBD based on Iteration 3 results
**If A ≤ 0.80:** Expand to cross_entropy, declare success
**If A 0.80-0.85:** Test Approach B (observed examples)
**If A 0.85-0.90:** Spawn Proposal subagent for improvements
**If A > 0.90:** Deep dive analysis - why is heuristic not helping?

---

## Communication Protocol

### Manager → Subagent:
```
Task: [Measurement/Analysis/Proposal/Decision]
Input: [Data files, prior results, constraints]
Goal: [Specific objective]
Success Criteria: [How to judge completion]
Time Limit: [Expected duration]
```

### Subagent → Manager:
```
Status: [COMPLETE/BLOCKED/IN_PROGRESS]
Output: [Data files, scores, proposals]
Key Findings: [3-5 bullet points]
Recommendation: [What should happen next]
```

---

## Current Plan (Living Document)

**Active Plan:** LLM_HEURISTICS_REVISED_PLAN.md

**Plan Updates:**
- After each iteration, manager updates plan based on results
- Successful approaches get promoted (become new baseline)
- Failed approaches get documented in EXPERIMENT_LOG.md
- New ideas from Proposal subagent get added to plan

**Plan Evolution:**
```
Initial Plan (Phase 1-5) 
  ↓
Results: Baseline = 1.00, Approach A = 0.85
  ↓
Updated Plan: Add Approach B (observed examples)
  ↓
Results: Approach B = 0.78 ✓
  ↓
Updated Plan: Use B for cross_entropy, add trait-based detection
  ↓
... continue iterating ...
```

---

## Resource Management

### Time Budget:
- **Total available:** ~40 hours for softmax + cross_entropy
- **Spent so far:** 1 hour (planning)
- **Remaining:** 39 hours

### Prioritization:
1. **Quick wins first:** Test existing heuristic (Approach A) - 2 hours
2. **High-value experiments:** Observed examples (Approach B) - 3 hours
3. **Refinement:** Iterate on winning approach - 2-4 hours per iteration
4. **Expansion:** Apply to cross_entropy once proven on softmax - 3-4 hours

### Stopping Criteria:
- **Success:** round0_best_geo ≤ 0.80 achieved and reproduced
- **Time limit:** 40 hours exhausted without reaching goal → document findings, propose next steps
- **Diminishing returns:** 3 consecutive iterations with <2% improvement → try different approach or escalate

---

## Manager Execution Loop (Pseudocode)

```python
while goal_not_reached and time_remaining > 0 and not_blocked:
    # Manager orchestrates, does NOT make strategic decisions
    
    if no_baseline_yet:
        spawn_measurement_agent("Run baseline")
    
    elif baseline_exists and no_heuristic_tested:
        spawn_measurement_agent("Run Approach A")
    
    elif results_exist and not_analyzed:
        spawn_analysis_agent("Compute scores, analyze patterns")
    
    elif analysis_complete and not_decided:
        # ALWAYS spawn Decision subagent after analysis
        spawn_decision_agent("Evaluate analysis, recommend next action")
    
    elif decision_received:
        # Execute Decision subagent's recommendation
        recommendation = get_decision_output()
        
        if recommendation == "SUCCESS":
            document_success()
            spawn_measurement_agent("Expand to cross_entropy")
        
        elif recommendation.startswith("Test Approach"):
            # e.g., "Test Approach B"
            spawn_measurement_agent(recommendation)
        
        elif recommendation == "Spawn Proposal":
            spawn_proposal_agent("Generate new ideas")
            # After Proposal completes, spawn Decision again
            
        elif recommendation == "ESCALATE":
            escalate_to_user(reason=recommendation.reason)
            break
    
    # Update plan and log after each subagent completes
    update_plan_based_on_latest_results()
    log_iteration_to_experiment_log()
```

**Key principle:** Manager is a dumb orchestrator. All strategic thinking happens in subagents.

---

## Success Definition

### Softmax Success:
- ✅ round0_best_geo ≤ 0.80 on overall (train + held-out)
- ✅ Held-out interpolation ≤ 0.85 (can interpolate)
- ✅ Held-out extrapolation ≤ 0.90 (can generalize)
- ✅ Approach documented and reproducible

### Cross-Entropy Success:
- ✅ round0_best_geo ≤ 0.80 using winning approach from softmax
- ✅ Train/held-out delta < 0.10
- ✅ Trait-based auto-detection working

### Project Success:
- ✅ Generic framework for adding heuristics to new kernels
- ✅ Clear methodology documented (AOT → train heuristic → test → iterate)
- ✅ 20% improvement demonstrated on 2+ kernel families

---

## Current Action Items

**Manager's immediate next steps:**

1. ✅ Create MANAGER.md (this file)
2. 🔄 Spawn Measurement subagent for Phase 1 baseline
3. ⏳ Wait for baseline completion (~3 hours)
4. ⏳ Spawn Analysis subagent to evaluate baseline
5. ⏳ Based on analysis, spawn next Measurement subagent (Approach A)
6. ⏳ Continue loop until goal reached

**Status:** Ready to spawn first Measurement subagent

**Next Command:**
```
Agent({
  subagent_type: "general-purpose",
  description: "Phase 1: Softmax baseline measurement",
  prompt: "Execute Phase 1 of LLM_HEURISTICS_REVISED_PLAN.md:
    1. Create shape_grid.json with 12 shapes (8 train, 2 interp, 2 extrap)
    2. Run baseline measurement using run_live.py
    3. Report CSV output location and row count
    Goal: Establish baseline performance (round0_best_geo = 1.00)"
})
```

---

**Manager Mode: ACTIVE**
**Autonomy Level: FULL** (iterate until goal or blocked)
**Goal Tracking: ENABLED** (target ≤ 0.80)
