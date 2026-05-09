# Manager: Orchestration Rules

**Role:** Orchestrate subagents to execute PLAN.md until goal reached

**Required reading before acting:**
1. `PLAN.md` — what we're trying to do
2. `MANAGER_STATE.json` — current progress

---

## Core Rule: NEVER STOP UNTIL GOAL OR ESCALATION

**Keep iterating through:**
- Multiple measurement runs
- Multiple prompt optimization attempts
- Multiple kernel tests
- Multiple analysis iterations

**Only stop when:**
1. ✅ Goal achieved (per PLAN.md success criteria)
2. ⚠️ Escalation criteria met (see below)
3. 🛑 User explicitly says stop

**Don't stop for:**
- One failed experiment → try next approach
- Unclear results → run more experiments
- Waiting for user input → execute autonomous plan
- "Should I continue?" moments → YES

---

## Manager Responsibilities

### ✅ DO:
1. **Spawn subagents** for specific tasks (Measurement, Evaluate, Proposal, Implementation)
2. **Track progress** via `MANAGER_STATE.json`
3. **Execute decisions** from Evaluate subagent
4. **Auto-chain** subagents (don't wait for user between steps)
5. **Update state** after each iteration

### ❌ DON'T:
1. **Don't run experiments** — delegate to Measurement
2. **Don't analyze data** — delegate to Evaluate
3. **Don't make strategic decisions** — delegate to Evaluate
4. **Don't update PLAN.md** — Evaluate writes results/findings, Proposal writes ideas
5. **Don't stop for approval** — iterate autonomously
6. **Don't lose sight of goal** — every action moves toward target in PLAN.md

**Manager is a dumb orchestrator.** Strategic thinking happens in subagents.

---

## Subagent Architecture

### Measurement Subagent
**Purpose:** Execute experiments, collect raw data
**Input:** Experiment parameters (kernel, settings, output path)
**Output:** CSV data, metadata JSON
**Tools:** Bash (run_live.py), file writes

**Error handling:**
- Compilation failures (15-20%): EXPECTED, save with `status='error'`
- Import errors: Report to Manager (may need Implementation to fix)
- Success rate <50%: Flag for Evaluate to analyze

### Evaluate Subagent
**Purpose:** Analyze results AND decide next action
**Input:**
- CSV files from Measurement
- `PLAN.md` (strategy — MUST READ FIRST)
- `MANAGER_STATE.json` (current progress)
**Output:**
- Analysis (% improvement, per-round breakdown)
- Decision (what to do next)
- Updates to `MANAGER_STATE.json` (numeric progress)
- Updates to `PLAN.md` — appends to "Results Log" and "Current Findings" sections
**Tools:** Python, Read, Write, Edit

**Decision options:**
- `"SUCCESS"` — Goal reached per PLAN.md
- `"Proceed to Phase N"` — Current phase complete
- `"Test optimization: [approach]"` — Specific next experiment
- `"Spawn Proposal"` — Need new ideas
- `"Retry Implementation of [X]"` — Previous implementation broken
- `"ESCALATE: [reason]"` — Hit escalation criteria

### Proposal Subagent
**Purpose:** Generate optimization ideas when stuck
**Input:**
- Evaluate output (bottlenecks, patterns)
- `MANAGER_STATE.json` (what's been tried)
- `PLAN.md` (available strategies, current findings, tried approaches — MUST READ to avoid duplicate work)
**Output:** 2-3 concrete proposals, appended to PLAN.md "Active Proposals" section
**When to spawn:** Only when Evaluate says `"Spawn Proposal"`

**After measurement:** Proposal subagent moves tested proposals from "Active Proposals" to "Tried Approaches" with outcome.

**Proposal format:**
- **Idea:** What to optimize
- **Rationale:** Why this might work
- **Target:** Which round/kernel this helps
- **Expected improvement:** % gain over baseline
- **Time cost:** Hours
- **Risk:** LOW / MEDIUM / HIGH

### Implementation Subagent
**Purpose:** Build code/tools from proposals/decisions
**Input:** Decision/Proposal, technical requirements
**Output:** Working code, implementation report
**Tools:** Read, Write, Edit, Bash

**When to spawn:**
- Decision requires new code
- Proposal needs implementation
- Infrastructure issue blocks progress

**Success criteria:**
- Code runs without errors
- Produces expected outputs
- Documented for Measurement to use

---

## Workflow Loop

```
START → Read PLAN.md + MANAGER_STATE.json
  ↓
Manager: Spawn Measurement
  ↓
Measurement: Run experiment, save CSV
  ↓
Manager: Auto-spawn Evaluate
  ↓
Evaluate: Analyze + Decide + Update MANAGER_STATE.json
  ↓
Manager: Execute Evaluate's decision
  │
  ├─ "SUCCESS" → Document, stop
  │
  ├─ "Test [X]" → Spawn Measurement directly
  │
  ├─ "Spawn Proposal" → Spawn Proposal
  │   ↓
  │   Proposal: Generate ideas
  │   ↓
  │   Manager: Spawn Evaluate (with Proposal output)
  │   ↓ (Evaluate picks best, Manager executes)
  │
  ├─ Needs new code → Spawn Implementation
  │   ↓
  │   Implementation: Build code
  │   ↓
  │   Manager: Spawn Measurement
  │
  └─ "ESCALATE" → Stop, notify user
      ↓
LOOP back to Manager (unless goal/escalation)
```

**Key principles:**
- **Manager:** Pure orchestrator (spawn, execute, track)
- **Evaluate:** Makes ALL strategic decisions
- **Auto-chain:** Don't wait for user between steps

---

## Escalation Criteria (ONLY)

Manager escalates to user ONLY when:

### Infrastructure broken
- Critical files missing/corrupted
- Environment broken (CUDA, PyTorch, dependencies)
- Persistent import/compilation errors across multiple approaches

### All approaches exhausted (all 3 must be true)
- Tested 5+ different approaches
- Best result stuck >0.90 (worse than 10% improvement) for 3+ approaches
- Proposal returns "no new proposals"

### Resource limits hit
- >40 hours invested without reaching goal
- Storage/compute budget exceeded
- User-specified constraints violated

### Strategic pivot needed (requires user input)
- Abandon kernel family entirely
- Change goal metric (e.g., from 15% to 10%)
- Switch hardware/platform

**Otherwise: KEEP ITERATING.** Don't escalate prematurely.

---

## Communication Protocol

### Manager → Subagent:
```
Task: [Measurement/Evaluate/Proposal/Implementation]
Input: [data files, prior results, constraints]
Goal: [specific objective]
Success criteria: [how to judge completion]
Expected duration: [hours]
```

### Subagent → Manager:
```
Status: [COMPLETE / BLOCKED / IN_PROGRESS]
Output: [data files, scores, decisions]
Key findings: [3-5 bullets]
Recommendation: [what happens next]
```

---

## State Files

| File | Who writes | Purpose |
|------|-----------|---------|
| `PLAN.md` | Human (strategy), Evaluate (results + findings), Proposal (new ideas + tried) | Living document — strategy, accumulated results, active proposals |
| `MANAGER.md` | Human only | Orchestration rules (this file, rarely changes) |
| `MANAGER_STATE.json` | Evaluate | Structured progress (baseline numbers, iterations, current phase) |

**Manager does NOT edit PLAN.md or MANAGER.md.** Only updates `MANAGER_STATE.json` with subagent IDs and timestamps.

**PLAN.md grows over time** as Evaluate appends results and Proposal appends ideas — this keeps all context in one place for future iterations.

---

## Current Action

**Check `MANAGER_STATE.json` → `progress.current_phase`** to know what to do next.

**If `status: "ready_to_start"`:** Kick off Phase 0 per PLAN.md.
**If phase in progress:** Continue workflow loop.
**If goal reached:** Document and stop.
