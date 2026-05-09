# LLM Heuristics: Concrete Execution Plan

**Goal:** Achieve 20% better performance (round0_best_geo ≤ 0.80) using LLM heuristics

**Test Case:** Softmax kernel with 100 shapes of existing AOT data

---

## Phase 0: Understand Current State

### Existing Assets:
1. **Raw AOT data (C):** `/aot_pretune_data/b200/softmax/.../measurements_cuda_NVIDIA_B200_13.0.csv`
   - 100 shapes, 8300 configs explored
   - Full benchmark results from LFBOTreeSearch

2. **Decision tree heuristic (A):** `pretuned_kernels/softmax/_helion_aot_softmax_cuda_sm100.py`
   - 5 configs selected by sklearn decision tree
   - if/else logic based on (arg0_dim0, arg0_dim1)

3. **Tuned configs JSON:** `tuned_configs_cuda_NVIDIA_B200_13.0.json`
   - 100 best configs (1 per shape)
   - Shape features + winning config

### Question to Answer:
**"Which input format helps LLM generate better round-0 configs?"**

---

## Phase 1: Baseline Measurement (2-3 hours)

**Goal:** Measure current LLM performance WITHOUT heuristics

### Step 1.1: Create softmax shape grid
```bash
cd /home/dev/helion_choijon5/llm_heuristics_artifacts
mkdir -p softmax_experiment/shapes
```

**Extract 12 representative shapes from 100:**
- 4 small (dim1 ≤ 512): Test low-overhead configs
- 4 medium (512 < dim1 ≤ 4096): Test balanced configs
- 4 large (dim1 > 4096): Test high-throughput configs
- Split 8 train / 4 held-out

```python
# Create shape_grid.json with 12 shapes
{
  "kernels": {
    "softmax": {
      "shapes": [
        {"id": "SM_001", "args": {"batch_size": 256, "dim": 128}, "split": "train"},
        {"id": "SM_002", "args": {"batch_size": 1024, "dim": 256}, "split": "train"},
        ...
      ]
    }
  }
}
```

### Step 1.2: Run baseline (LLM only, no heuristic)
```bash
python tools/run_live.py \
  --kernel softmax \
  --arm baseline \
  --shape-grid softmax_experiment/shapes/shape_grid.json \
  --repeats 3 \
  --configs-per-round 3 \
  --initial-random-configs 2 \
  --output-dir softmax_experiment/baseline
```

**Expected output:** CSV with generation=0 configs from LLM

**Time:** ~2 hours (12 shapes × 3 repeats × 5-10 min each)

---

## Phase 2: Test Three Heuristic Approaches (3-4 hours)

### Approach A: Decision Tree Seed (Current Method)

**What:** Use existing `_helion_aot_softmax_cuda_sm100.py`

**How it works:**
1. Heuristic selects 1 config from 5 pre-defined configs
2. Config injected as seed to LLM
3. LLM never sees the logic, just benchmarks the config

**Run:**
```bash
export HELION_LLM_ROUND0_HEURISTIC_PATH=/home/dev/helion_choijon5/pretuned_kernels/softmax/_helion_aot_softmax_cuda_sm100.py

python tools/run_live.py \
  --kernel softmax \
  --arm heuristics_A \
  --shape-grid softmax_experiment/shapes/shape_grid.json \
  --repeats 3 \
  --output-dir softmax_experiment/heuristics_A
```

**Hypothesis:** Should improve over baseline (seed helps), but limited to 5 configs

---

### Approach B: Observed Examples JSON (NEW)

**What:** Give LLM top-K configs for similar shapes with performance data

**Implementation steps:**

#### Step 2.1: Build observed_heuristics.json
```python
# Script: build_observed_heuristics.py
import json, csv
from collections import defaultdict

# Load tuned configs (100 best configs, 1 per shape)
tuned = json.load(open('tuned_configs_cuda_NVIDIA_B200_13.0.json'))

# Load measurements (8300 configs with timing)
measurements = defaultdict(list)
with open('measurements_cuda_NVIDIA_B200_13.0.csv') as f:
    for row in csv.DictReader(f):
        shape_hash = row['shape_hash']
        config = json.loads(row['config'])
        timing = float(row['timing_ms'])
        features = json.loads(row['shape_features'])
        measurements[shape_hash].append({
            'config': config,
            'timing_ms': timing,
            'batch': features['arg0_dim0'],
            'dim': features['arg0_dim1']
        })

# Group by dim ranges, select top-3 per range
observed = {}
for dim_range in [(0, 512), (512, 1024), (1024, 4096), (4096, float('inf'))]:
    range_key = f"{dim_range[0]}-{dim_range[1]}"
    configs_in_range = []
    for shape_hash, configs in measurements.items():
        for c in configs:
            if dim_range[0] <= c['dim'] < dim_range[1]:
                configs_in_range.append(c)
    
    # Sort by timing, take top-3
    top3 = sorted(configs_in_range, key=lambda x: x['timing_ms'])[:3]
    observed[range_key] = top3

# Save
json.dump(observed, open('observed_heuristics_softmax.json', 'w'), indent=2)
```

**Output format:**
```json
{
  "0-512": [
    {"config": {...}, "timing_ms": 0.015, "batch": 256, "dim": 128},
    {"config": {...}, "timing_ms": 0.018, "batch": 1024, "dim": 256}
  ],
  "512-1024": [...],
  ...
}
```

#### Step 2.2: Modify LLM prompt to include observed examples

**File to modify:** `helion/autotuner/llm/prompting.py`

Add new function:
```python
def build_observed_heuristics_section(
    args: Sequence[object],
    observed_heuristics_path: str | None
) -> str:
    """Load and format observed heuristics for LLM prompt."""
    if observed_heuristics_path is None:
        return ""
    
    # Extract workload features
    batch = args[0].shape[0]
    dim = args[0].shape[1]
    
    # Load JSON
    observed = json.load(open(observed_heuristics_path))
    
    # Find matching range
    for range_key, examples in observed.items():
        dim_min, dim_max = map(float, range_key.split('-'))
        if dim_min <= dim < dim_max:
            # Format examples
            lines = [
                f"Previously observed configs for similar shapes (dim={range_key}):",
                ""
            ]
            for i, ex in enumerate(examples, 1):
                cfg = ex['config']
                lines.append(f"{i}. Config (batch={ex['batch']}, dim={ex['dim']} → {ex['timing_ms']:.3f}ms):")
                lines.append(f"   num_warps={cfg['num_warps']}, num_stages={cfg['num_stages']}, block_sizes={cfg['block_sizes']}")
                lines.append(f"   indexing={cfg['indexing'][:3]}...")
                lines.append("")
            
            return "\n".join(lines)
    
    return ""
```

Modify `build_initial_prompt`:
```python
def build_initial_prompt(..., observed_heuristics_path: str | None = None):
    ...
    observed_section = build_observed_heuristics_section(args, observed_heuristics_path)
    
    return _join_sections(
        describe_kernel(kernel, args),
        _section("Configuration Space", describe_config_space(config_spec)),
        default_section,
        observed_section if observed_section else None,  # ADD THIS
        guidance,
        _section("Task", task_section),
    )
```

#### Step 2.3: Add env var support

**File:** `helion/autotuner/llm_search.py`

```python
def guided_search_kwargs_from_config(...):
    ...
    observed_path = os.getenv("HELION_LLM_OBSERVED_HEURISTICS_PATH")
    if observed_path:
        kwargs["observed_heuristics_path"] = observed_path
    ...
```

#### Step 2.4: Run with observed heuristics
```bash
export HELION_LLM_OBSERVED_HEURISTICS_PATH=/home/dev/helion_choijon5/softmax_experiment/observed_heuristics_softmax.json

python tools/run_live.py \
  --kernel softmax \
  --arm heuristics_B \
  --shape-grid softmax_experiment/shapes/shape_grid.json \
  --repeats 3 \
  --output-dir softmax_experiment/heuristics_B
```

**Hypothesis:** LLM sees examples with perf data, can extrapolate patterns better than single seed

---

### Approach C: Raw CSV Feed (Experimental)

**What:** Give LLM raw CSV data for nearest shapes

**Implementation:** Similar to B, but instead of top-3, give ALL configs (20-80 per shape)

**Challenge:** Context length - might need to:
1. Filter to nearest 2-3 shapes only
2. Sample configs (top-5 + bottom-5 + 5 random)
3. Compress format (remove redundant fields)

**Run:**
```bash
export HELION_LLM_RAW_CSV_PATH=/home/dev/helion_choijon5/softmax_experiment/filtered_measurements.json

python tools/run_live.py \
  --kernel softmax \
  --arm heuristics_C \
  --shape-grid softmax_experiment/shapes/shape_grid.json \
  --repeats 3 \
  --output-dir softmax_experiment/heuristics_C
```

**Hypothesis:** Maximum information, but may overwhelm LLM or waste context

---

## Phase 3: Evaluation & Analysis (1 hour)

### Step 3.1: Compute scores
```bash
python tools/compute_round0_geo.py \
  --baseline softmax_experiment/baseline \
  --heuristics-a softmax_experiment/heuristics_A \
  --heuristics-b softmax_experiment/heuristics_B \
  --heuristics-c softmax_experiment/heuristics_C \
  --output softmax_experiment/comparison.json
```

**Output:**
```json
{
  "baseline": {"overall": 1.00},
  "approach_A": {"overall": 0.85, "train": 0.83, "heldout": 0.88},
  "approach_B": {"overall": 0.78, "train": 0.76, "heldout": 0.81},
  "approach_C": {"overall": 0.82, "train": 0.80, "heldout": 0.85}
}
```

### Step 3.2: Per-shape analysis
```python
# Which approach wins for each shape?
# Is there a pattern (small/medium/large shapes)?
# Does heldout generalize well?
```

### Step 3.3: Decision
- **If any approach ≤ 0.80:** SUCCESS! Use that approach
- **If all > 0.80 but < 0.90:** Iterate (Phase 4)
- **If all > 0.90:** Deep investigation needed

---

## Phase 4: Kernel Pattern Recognition (2-3 hours)

**Goal:** Solve "how does Helion know it's a softmax kernel?"

### Challenge:
Currently, heuristic requires explicit kernel name:
```python
export HELION_LLM_ROUND0_HEURISTIC_PATH=.../heuristic_SOFTMAX.py
```

But LLM autotuner gets called on ANY kernel. How to detect pattern?

### Approach 1: Kernel Name Matching
```python
# In llm_search.py
def _load_heuristic_for_kernel(kernel_name: str):
    heuristic_dir = os.getenv("HELION_LLM_HEURISTICS_DIR", "pretuned_kernels/")
    heuristic_path = f"{heuristic_dir}/{kernel_name}/_helion_aot_{kernel_name}_cuda_sm100.py"
    if os.path.exists(heuristic_path):
        return load_heuristic(heuristic_path)
    return None
```

**Pros:** Simple, works for exact name matches
**Cons:** Fails if user names kernel `my_custom_softmax`

### Approach 2: Code Pattern Detection
```python
# Analyze kernel source code for patterns
def detect_kernel_pattern(kernel: _AutotunableKernel) -> str | None:
    source = inspect.getsource(kernel.fn)
    
    # Softmax pattern: exp, sum, divide
    if 'torch.exp' in source and 'torch.sum' in source:
        if '/' in source or 'torch.div' in source:
            return "softmax"
    
    # Cross-entropy pattern: log, gather, mean
    if 'torch.log' in source and 'torch.gather' in source:
        return "cross_entropy"
    
    # FlashAttention pattern: matmul, softmax, matmul
    if source.count('torch.matmul') >= 2 or '@' in source:
        if 'softmax' in source.lower():
            return "attention"
    
    return None
```

**Pros:** Works for user-named kernels
**Cons:** Fragile (false positives/negatives), requires pattern library

### Approach 3: Workload Feature Matching
```python
# Match on tensor shapes + operations
def match_workload_to_pattern(
    args: Sequence[object],
    workload_traits: dict
) -> str | None:
    # Softmax: 2D input, reduction on dim=1
    if len(args) == 1 and args[0].ndim == 2:
        if workload_traits.get('has_reduction'):
            return "softmax"
    
    # Cross-entropy: 2D logits + 1D labels
    if len(args) == 2:
        if args[0].ndim == 2 and args[1].ndim == 1:
            if workload_traits.get('has_gather'):
                return "cross_entropy"
    
    return None
```

**Pros:** Based on runtime behavior, not brittle source parsing
**Cons:** May need more sophisticated feature extraction

### Recommended: Hybrid Approach
1. Try kernel name matching first (fast path)
2. Fallback to code pattern detection
3. Fallback to workload feature matching
4. If no match, use generic heuristic or skip

**Implementation:**
```python
def get_heuristic_for_kernel(
    kernel: _AutotunableKernel,
    args: Sequence[object],
    workload_traits: dict
) -> dict | None:
    # 1. Exact name match
    if kernel.fn.__name__ in PRETUNED_KERNELS:
        return load_heuristic(kernel.fn.__name__)
    
    # 2. Pattern detection
    pattern = detect_kernel_pattern(kernel)
    if pattern:
        return load_heuristic(pattern)
    
    # 3. Workload matching
    pattern = match_workload_to_pattern(args, workload_traits)
    if pattern:
        return load_heuristic(pattern)
    
    # 4. No match
    return None
```

---

## Phase 5: Integration & Productionization (3-4 hours)

### Step 5.1: Choose winning approach from Phase 3
- If Approach B (observed JSON) wins, implement fully
- Add to `helion/autotuner/llm_search.py`
- Add tests in `test/test_llm_autotuner.py`

### Step 5.2: Add kernel pattern detection
- Implement hybrid matching (Phase 4)
- Test on softmax, cross_entropy, attention
- Handle edge cases (no match, ambiguous match)

### Step 5.3: Create heuristic registry
```python
# helion/autotuner/heuristics/registry.py
HEURISTIC_REGISTRY = {
    "softmax": {
        "decision_tree": "pretuned_kernels/softmax/_helion_aot_softmax_cuda_sm100.py",
        "observed_json": "pretuned_kernels/softmax/observed_heuristics.json",
        "patterns": ["softmax", "batch_softmax"],  # Name variants
    },
    "cross_entropy": {
        "decision_tree": "...",
        "observed_json": "...",
        "patterns": ["cross_entropy", "nll_loss"],
    },
    ...
}
```

### Step 5.4: End-to-end test
```bash
# Test that heuristics auto-load without explicit env var
python examples/softmax.py  # Should use pretuned heuristic automatically
python examples/cross_entropy.py  # Should use pretuned heuristic
python examples/my_custom_kernel.py  # Should fallback to baseline LLM
```

---

## Phase 6: Expand to Loss Functions (2-3 hours)

### Step 6.1: Apply winning approach to cross_entropy
- Use comprehensive AOT data from overnight run (once complete)
- Build observed_heuristics_cross_entropy.json OR decision_tree (whichever won in Phase 3)
- Run same evaluation: baseline vs heuristics

### Step 6.2: Fix kl_div and jsd
- Generate proper AOT data for kl_div (configs from cross_entropy don't transfer)
- Fix jsd dtype bug OR skip if too complex

### Step 6.3: Measure loss_functions family
- Target: round0_best_geo ≤ 0.80 across all 3 kernels
- Report: train vs heldout, per-kernel breakdown

---

## Success Criteria

### Phase 2-3 (Softmax Proof of Concept):
- ✅ At least ONE approach achieves round0_best_geo ≤ 0.80
- ✅ Held-out delta < 0.10 (generalization works)
- ✅ Understand which input format (A/B/C) works best

### Phase 4-5 (Production Ready):
- ✅ Kernel pattern detection works for 3+ kernel types
- ✅ Heuristics auto-load without manual env vars
- ✅ No regressions on kernels without heuristics (baseline LLM still works)

### Phase 6 (Loss Functions Goal):
- ✅ round0_best_geo ≤ 0.80 for cross_entropy
- ✅ round0_best_geo ≤ 0.85 overall (cross_entropy + kl_div + jsd if fixed)
- ✅ Documented approach for future kernel families

---

## Timeline

| Phase | Task | Time | Dependencies |
|-------|------|------|--------------|
| 1 | Baseline measurement | 2-3h | None |
| 2 | Approach A (decision tree) | 1h | Phase 1 |
| 2 | Approach B (observed JSON) | 2h | Phase 1 |
| 2 | Approach C (raw CSV) | 2h | Phase 1 |
| 3 | Evaluation & decision | 1h | Phase 2 |
| 4 | Kernel pattern detection | 2-3h | Phase 3 |
| 5 | Integration | 3-4h | Phase 3, 4 |
| 6 | Loss functions expansion | 2-3h | Phase 5 |
| **Total** | | **15-20 hours** | |

---

## Risk Mitigation

### Risk 1: None of A/B/C achieve 0.80
**Mitigation:** 
- Analyze WHY (LLM baseline too strong? Heuristic too weak?)
- Try hybrid: Seed (A) + Examples (B)
- Try more configs per range in B (top-5 instead of top-3)

### Risk 2: Kernel pattern detection has false positives
**Mitigation:**
- Log all pattern matches for debugging
- Allow manual override: `HELION_LLM_FORCE_PATTERN=softmax`
- Conservative defaults: only match high-confidence patterns

### Risk 3: Observed JSON approach requires too much prompt engineering
**Mitigation:**
- Start with simple format (just configs + timing)
- Iterate on format based on LLM response quality
- A/B test different prompt phrasings

---

## Next Steps

1. **Immediate:** Create softmax shape_grid.json (12 shapes)
2. **Run Phase 1:** Baseline measurement (start now, 2-3 hours)
3. **While Phase 1 runs:** Implement observed_heuristics.json builder (Step 2.1)
4. **After Phase 1:** Run Approach A (quick, 1 hour)
5. **Then:** Implement and run Approach B (2-3 hours)
6. **Decision point:** After Phase 3 evaluation, decide next steps

**Manager:** Continue executing this plan until we achieve 20% improvement (round0_best_geo ≤ 0.80)
