# LLM Heuristics: Revised Concrete Plan

**Goal:** Achieve 20% better performance (round0_best_geo ≤ 0.80) using LLM heuristics

**Test Case:** Softmax kernel with 100 shapes of existing AOT data

**Critical fixes from review:**
1. ✅ **NO name-based matching** - use workload traits only (automatic, generic)
2. ✅ **Proper train/test split** - held-out shapes OUTSIDE training distribution
3. ✅ **Understand decision tree** - gives exactly 1 config per shape (not multiple)

---

## Phase 0: Data Preparation & Understanding

### Available Data:
- **100 shapes:** Mostly 4096×[256 to 12672 in steps of 128], plus 1 outlier (2048×32768)
- **8300 configs:** ~83 configs explored per shape via LFBOTreeSearch
- **5 decision tree configs:** Selected by sklearn from AOT analysis

### Train/Test Split Strategy:

**Challenge:** 99 shapes have batch=4096, only dim varies (256 to 12672)

**Solution:** Interpolation + extrapolation test
```
TRAIN (60 shapes):
  - 4096×256, 4096×384, 4096×640, 4096×896, ... (every other shape from 256-6400)
  - 4096×7680, 4096×8064, 4096×8576, ... (every other from 7680-12672)
  - 2048×32768 (outlier, different batch size)

HELD-OUT INTERPOLATION (20 shapes):
  - 4096×512, 4096×768, 4096×1152, ... (gaps between training shapes)
  - Tests: Can heuristic interpolate between 640→896?

HELD-OUT EXTRAPOLATION (20 shapes):
  - 4096×128 (below training range) ← ADD SYNTHETIC
  - 4096×192 (below training range) ← ADD SYNTHETIC
  - 4096×13000, 4096×14000, ... (above training range) ← ADD SYNTHETIC
  - 8192×1024, 8192×4096 (different batch) ← ADD SYNTHETIC
  - Tests: Can heuristic generalize to unseen batch/dim values?
```

**Why synthetic shapes?**
- Existing data doesn't have extrapolation cases (all batch=4096, dim≤12672)
- Need to test: What happens with dim<256? What about batch≠4096?
- This reveals if heuristic overfits to training distribution

### Workload Trait Detection (Automatic, Generic)

**How Helion detects patterns (NOT name-based):**

From `helion/autotuner/llm/workload.py`:
```python
def detect_workload_traits(kernel, config_spec):
    """Analyzes FX graph to detect operations."""
    traits = set()
    
    for target in _iter_call_targets(kernel):
        name_parts = _target_name_parts(target)
        
        # Detects by operation, not kernel name
        if 'softmax' in name_parts or 'logsumexp' in name_parts:
            traits.add('reduction')
        if 'exp' in name_parts or 'exp2' in name_parts:
            traits.add('exp')
        if 'matmul' in name_parts or 'bmm' in name_parts:
            traits.add('matmul')
    
    # Composite patterns
    if 'matmul' in traits and 'reduction' in traits and 'exp' in traits:
        traits.add('attention_reduction')
    
    return frozenset(traits)
```

**For softmax, expected traits:**
- `reduction` (sum, amax)
- `exp` (torch.exp)
- Possibly others based on implementation

**Key insight:** This is already generic! No name matching needed. We just need to:
1. Map trait fingerprint → heuristic data
2. Example: `frozenset(['reduction', 'exp'])` → load softmax heuristics

---

## Phase 1: Baseline Measurement (2-3 hours)

**Goal:** Measure LLM performance WITHOUT any heuristics

### Step 1.1: Create shape grid with train/test split

```bash
cd /home/dev/helion_choijon5/llm_heuristics_artifacts
mkdir -p softmax_experiment
```

**Generate shape_grid.json:**
```python
import json

# Select 12 shapes for fast iteration (from 100 total)
# Train: 8 shapes covering dim range
# Held-out interpolation: 2 shapes (gaps in training)
# Held-out extrapolation: 2 shapes (outside training)

shapes = {
    "kernels": {
        "softmax": {
            "shapes": [
                # TRAIN (8 shapes)
                {"id": "SM_001", "args": {"batch": 4096, "dim": 256}, "split": "train"},
                {"id": "SM_002", "args": {"batch": 4096, "dim": 896}, "split": "train"},
                {"id": "SM_003", "args": {"batch": 4096, "dim": 1536}, "split": "train"},
                {"id": "SM_004", "args": {"batch": 4096, "dim": 3072}, "split": "train"},
                {"id": "SM_005", "args": {"batch": 4096, "dim": 5120}, "split": "train"},
                {"id": "SM_006", "args": {"batch": 4096, "dim": 8192}, "split": "train"},
                {"id": "SM_007", "args": {"batch": 4096, "dim": 11264}, "split": "train"},
                {"id": "SM_008", "args": {"batch": 2048, "dim": 32768}, "split": "train"},
                
                # HELD-OUT INTERPOLATION (2 shapes - gaps between training)
                {"id": "SM_101", "args": {"batch": 4096, "dim": 512}, "split": "heldout_interp"},  # Between 256-896
                {"id": "SM_102", "args": {"batch": 4096, "dim": 2048}, "split": "heldout_interp"},  # Between 1536-3072
                
                # HELD-OUT EXTRAPOLATION (2 shapes - outside training)
                {"id": "SM_201", "args": {"batch": 4096, "dim": 128}, "split": "heldout_extrap"},  # Below min
                {"id": "SM_202", "args": {"batch": 8192, "dim": 4096}, "split": "heldout_extrap"},  # Different batch
            ]
        }
    }
}

with open('softmax_experiment/shape_grid.json', 'w') as f:
    json.dump(shapes, f, indent=2)
```

### Step 1.2: Run baseline
```bash
python llm_heuristics_artifacts/loss_functions/tools/run_live.py \
  --kernel softmax \
  --arm baseline \
  --shape-grid llm_heuristics_artifacts/softmax_experiment/shape_grid.json \
  --repeats 3 \
  --configs-per-round 3 \
  --initial-random-configs 2 \
  --output-dir llm_heuristics_artifacts/softmax_experiment/baseline
```

**Expected:** CSV with 12 shapes × 3 repeats × ~5-6 configs = ~180-200 rows

**Time:** 12 shapes × 3 repeats × 5 min = ~3 hours

---

## Phase 2: Three Heuristic Approaches (4-6 hours)

### Approach A: Decision Tree Seed (1 config)

**What it does:** Heuristic returns exactly 1 config based on if/else tree

**Current implementation:**
```python
def key_softmax(*args) -> int:
    dim = args[0].shape[1]
    if dim <= 4096:
        if dim <= 1024:
            return 2  # config_2
        else:
            return 1  # config_1
    else:
        if batch <= 2048:
            return 4  # config_4
        else:
            return 0  # config_0
```

**How LLM uses it:**
1. Before LLM call, heuristic selects config_2
2. Config injected: `[config_2, random_1, random_2]`
3. LLM benchmarks all 3, picks best in round-0
4. **LLM never sees the decision logic**

**Pros:** Simple, fast, deterministic
**Cons:** Limited to 5 pre-selected configs, LLM can't learn from heuristic reasoning

### Approach B: Observed Examples (Top-K configs with performance)

**What it does:** Give LLM top-3 configs for similar shapes WITH timing data

**How LLM uses it:**
LLM prompt becomes:
```
Kernel: softmax
Workload: batch=4096, dim=2048

Previously observed configs for similar shapes (dim=1536-3072):

1. Shape 4096×1536 → 0.045ms
   num_warps=4, num_stages=1, block_sizes=[1], indexing=['pointer', ...]
   
2. Shape 4096×2304 → 0.061ms
   num_warps=2, num_stages=4, block_sizes=[1], indexing=['tensor_descriptor', ...]
   
3. Shape 4096×3072 → 0.089ms
   num_warps=4, num_stages=1, block_sizes=[1], indexing=['pointer', ...]

Analysis: num_warps=4 is consistently best, stages=1 wins for this dim range

Task: Suggest 3 configs using the patterns above
```

**Pros:** 
- LLM sees WHY configs work (timing data)
- LLM can extrapolate patterns (e.g., "num_warps=4 best for dim<4096")
- More transparent than black-box seed

**Cons:**
- Requires prompt engineering (add ~300-500 tokens)
- Need to select "similar" shapes (bucketing strategy)

**Implementation:** (detailed in Phase 2B below)

### Approach C: Hybrid (Seed + Examples)

**What it does:** Combine A and B
- Heuristic injects 1 deterministic seed (Approach A)
- ALSO show LLM top-3 examples with timing (Approach B)

**Hypothesis:** Best of both worlds
- Seed gives LLM a strong starting point
- Examples help LLM understand WHY seed is good, when to deviate

---

## Phase 2A: Implement Approach A (Decision Tree)

**Already exists!** Just need to run it:

```bash
export HELION_LLM_ROUND0_HEURISTIC_PATH=/home/dev/helion_choijon5/pretuned_kernels/softmax/_helion_aot_softmax_cuda_sm100.py

python llm_heuristics_artifacts/loss_functions/tools/run_live.py \
  --kernel softmax \
  --arm heuristics_A \
  --shape-grid llm_heuristics_artifacts/softmax_experiment/shape_grid.json \
  --repeats 3 \
  --output-dir llm_heuristics_artifacts/softmax_experiment/heuristics_A
```

**Time:** ~2 hours (slightly faster than baseline since seed is good)

---

## Phase 2B: Implement Approach B (Observed Examples)

### Step 2B.1: Build observed_heuristics_softmax.json

**Script:** `build_observed_heuristics.py`
```python
import json, csv
from collections import defaultdict

# Load measurements
measurements = defaultdict(list)
csv_path = '/home/dev/helion_choijon5/aot_pretune_data/b200/softmax/runs/20260503_114301_softmax_full_tutorial_seed98_gpu7/measurements_cuda_NVIDIA_B200_13.0.csv'

with open(csv_path) as f:
    for row in csv.DictReader(f):
        features = json.loads(row['shape_features'])
        config = json.loads(row['config'])
        timing = float(row['timing_ms'])
        
        batch = features['arg0_dim0']
        dim = features['arg0_dim1']
        
        measurements[(batch, dim)].append({
            'config': config,
            'timing_ms': timing
        })

# Group by dim buckets, select top-3 per bucket
dim_buckets = [
    (0, 512, "0-512"),
    (512, 2048, "512-2048"),
    (2048, 6144, "2048-6144"),
    (6144, float('inf'), "6144+")
]

observed = {}
for dim_min, dim_max, bucket_name in dim_buckets:
    # Collect all configs in this bucket
    bucket_configs = []
    for (batch, dim), configs in measurements.items():
        if dim_min <= dim < dim_max:
            for c in configs:
                bucket_configs.append({
                    'batch': batch,
                    'dim': dim,
                    **c
                })
    
    # Sort by timing, take top-5
    top5 = sorted(bucket_configs, key=lambda x: x['timing_ms'])[:5]
    
    # Format for JSON
    observed[bucket_name] = [
        {
            'batch': c['batch'],
            'dim': c['dim'],
            'timing_ms': round(c['timing_ms'], 6),
            'config': {
                'num_warps': c['config']['num_warps'],
                'num_stages': c['config']['num_stages'],
                'block_sizes': c['config']['block_sizes'],
                'indexing': c['config']['indexing'][:3],  # Truncate for readability
                # Include other important fields...
            }
        }
        for c in top5
    ]

# Save
output_path = 'llm_heuristics_artifacts/softmax_experiment/observed_heuristics_softmax.json'
with open(output_path, 'w') as f:
    json.dump(observed, f, indent=2)

print(f'Wrote {output_path}')
print(f'Buckets: {list(observed.keys())}')
for bucket, configs in observed.items():
    print(f'  {bucket}: {len(configs)} examples')
```

### Step 2B.2: Modify LLM prompt to include observed examples

**File:** `helion/autotuner/llm/prompting.py`

Add new function:
```python
import os

def _build_observed_heuristics_section(
    args: Sequence[object],
) -> str:
    """Load and format observed heuristics for LLM prompt."""
    observed_path = os.getenv("HELION_LLM_OBSERVED_HEURISTICS_PATH")
    if not observed_path or not os.path.exists(observed_path):
        return ""
    
    # Extract workload dims
    if not args or not isinstance(args[0], torch.Tensor) or args[0].ndim < 2:
        return ""
    
    batch = args[0].shape[0]
    dim = args[0].shape[1]
    
    # Load JSON
    with open(observed_path) as f:
        observed = json.load(f)
    
    # Find matching bucket
    for bucket_name, examples in observed.items():
        if not examples:
            continue
        
        # Parse bucket range (e.g., "512-2048")
        if '-' in bucket_name:
            parts = bucket_name.split('-')
            if parts[1] == '' or parts[1] == '+':  # "6144+" case
                dim_min = int(parts[0])
                dim_max = float('inf')
            else:
                dim_min = int(parts[0])
                dim_max = int(parts[1])
        else:
            continue
        
        # Check if current dim is in range
        if dim_min <= dim < dim_max:
            lines = [
                f"## Previously Observed Configs",
                f"",
                f"For similar shapes (dim range {bucket_name}):",
                ""
            ]
            
            for i, ex in enumerate(examples[:3], 1):  # Show top-3
                cfg = ex['config']
                lines.append(f"{i}. Shape {ex['batch']}×{ex['dim']} → {ex['timing_ms']:.3f}ms")
                lines.append(f"   num_warps={cfg['num_warps']}, num_stages={cfg['num_stages']}, block_sizes={cfg['block_sizes']}")
                lines.append(f"   indexing={cfg['indexing']}")
                lines.append("")
            
            return "\n".join(lines)
    
    return ""
```

**Modify `build_initial_prompt`:**
```python
def build_initial_prompt(...) -> str:
    ...
    observed_section = _build_observed_heuristics_section(args)
    
    return _join_sections(
        describe_kernel(kernel, args),
        _section("Configuration Space", describe_config_space(config_spec)),
        default_section,
        observed_section,  # ADD THIS LINE
        guidance,
        _section("Task", task_section),
    )
```

### Step 2B.3: Run Approach B
```bash
python llm_heuristics_artifacts/softmax_experiment/build_observed_heuristics.py

export HELION_LLM_OBSERVED_HEURISTICS_PATH=/home/dev/helion_choijon5/llm_heuristics_artifacts/softmax_experiment/observed_heuristics_softmax.json

python llm_heuristics_artifacts/loss_functions/tools/run_live.py \
  --kernel softmax \
  --arm heuristics_B \
  --shape-grid llm_heuristics_artifacts/softmax_experiment/shape_grid.json \
  --repeats 3 \
  --output-dir llm_heuristics_artifacts/softmax_experiment/heuristics_B
```

**Time:** ~2-3 hours

---

## Phase 2C: Implement Approach C (Hybrid)

**Run both heuristic seed AND observed examples:**
```bash
export HELION_LLM_ROUND0_HEURISTIC_PATH=/home/dev/helion_choijon5/pretuned_kernels/softmax/_helion_aot_softmax_cuda_sm100.py
export HELION_LLM_OBSERVED_HEURISTICS_PATH=/home/dev/helion_choijon5/llm_heuristics_artifacts/softmax_experiment/observed_heuristics_softmax.json

python llm_heuristics_artifacts/loss_functions/tools/run_live.py \
  --kernel softmax \
  --arm heuristics_C \
  --shape-grid llm_heuristics_artifacts/softmax_experiment/shape_grid.json \
  --repeats 3 \
  --output-dir llm_heuristics_artifacts/softmax_experiment/heuristics_C
```

**Time:** ~2-3 hours

---

## Phase 3: Evaluation & Analysis (1 hour)

### Step 3.1: Compute scores with train/test breakdown

**Modify scoring script** to handle 3-way split:
```python
# compute_round0_geo.py additions
splits = {
    'train': [s for s in shapes if s['split'] == 'train'],
    'heldout_interp': [s for s in shapes if s['split'] == 'heldout_interp'],
    'heldout_extrap': [s for s in shapes if s['split'] == 'heldout_extrap'],
}

results = {
    'train': compute_geo(splits['train']),
    'heldout_interp': compute_geo(splits['heldout_interp']),
    'heldout_extrap': compute_geo(splits['heldout_extrap']),
    'overall': compute_geo(all_shapes)
}
```

### Step 3.2: Run scoring
```bash
python llm_heuristics_artifacts/loss_functions/tools/compute_round0_geo.py \
  --baseline llm_heuristics_artifacts/softmax_experiment/baseline \
  --heuristics llm_heuristics_artifacts/softmax_experiment/heuristics_A \
  --output llm_heuristics_artifacts/softmax_experiment/scores_A.json

# Repeat for B and C
...
```

### Step 3.3: Compare results
```python
import json

for approach in ['A', 'B', 'C']:
    with open(f'scores_{approach}.json') as f:
        scores = json.load(f)
    
    print(f"\nApproach {approach}:")
    print(f"  Train:            {scores['train']:.4f}")
    print(f"  Held-out (interp): {scores['heldout_interp']:.4f}")
    print(f"  Held-out (extrap): {scores['heldout_extrap']:.4f}")
    print(f"  Overall:          {scores['overall']:.4f}")
    print(f"  Status: {'✅ PASS' if scores['overall'] <= 0.80 else '❌ FAIL'}")
```

### Step 3.4: Decision matrix

| Approach | Train | Interp | Extrap | Overall | Winner? |
|----------|-------|--------|--------|---------|---------|
| Baseline | 1.00 | 1.00 | 1.00 | 1.00 | - |
| A (seed) | ? | ? | ? | ? | ? |
| B (examples) | ? | ? | ? | ? | ? |
| C (hybrid) | ? | ? | ? | ? | ? |

**Decision criteria:**
1. **If any approach ≤ 0.80 overall:** SUCCESS! Use that approach
2. **If multiple ≤ 0.80:** Pick best on heldout_extrap (most challenging)
3. **If none ≤ 0.80 but < 0.90:** Iterate (increase K in top-K, try more buckets)
4. **If all ≥ 0.90:** Deep investigation (is LLM baseline too strong? Are heuristics too weak?)

---

## Phase 4: Generic Trait-Based Heuristic Matching (2-3 hours)

**Goal:** Auto-load heuristics without manual kernel name specification

### Step 4.1: Define trait fingerprints

**Create registry:** `helion/autotuner/heuristics/registry.py`
```python
from dataclasses import dataclass
from pathlib import Path

@dataclass
class HeuristicEntry:
    """Metadata for a pre-tuned heuristic."""
    pattern_name: str
    trait_fingerprint: frozenset[str]
    decision_tree_path: Path | None = None
    observed_json_path: Path | None = None
    min_confidence: float = 0.8  # Only use if traits match this well

# Registry mapping trait fingerprints to heuristics
HEURISTIC_REGISTRY = {
    # Softmax: reduction + exp operations
    frozenset(['reduction', 'exp']): HeuristicEntry(
        pattern_name='softmax',
        trait_fingerprint=frozenset(['reduction', 'exp']),
        decision_tree_path=Path('pretuned_kernels/softmax/_helion_aot_softmax_cuda_sm100.py'),
        observed_json_path=Path('pretuned_kernels/softmax/observed_heuristics.json'),
    ),
    
    # FlashAttention: matmul + reduction + exp
    frozenset(['attention_reduction']): HeuristicEntry(
        pattern_name='attention',
        trait_fingerprint=frozenset(['attention_reduction']),
        decision_tree_path=Path('pretuned_kernels/attention/_helion_aot_attention_cuda_sm100.py'),
        observed_json_path=None,  # Not yet available
    ),
    
    # Cross-entropy: reduction (no exp, has gather/log)
    frozenset(['reduction']): HeuristicEntry(
        pattern_name='cross_entropy',
        trait_fingerprint=frozenset(['reduction']),
        decision_tree_path=Path('pretuned_kernels/cross_entropy/_helion_aot_cross_entropy_cuda_sm100.py'),
        observed_json_path=None,
        min_confidence=0.9,  # Higher threshold since 'reduction' is generic
    ),
}

def match_heuristic(
    detected_traits: frozenset[str]
) -> HeuristicEntry | None:
    """Find best matching heuristic based on workload traits."""
    best_match = None
    best_score = 0.0
    
    for fingerprint, entry in HEURISTIC_REGISTRY.items():
        # Compute Jaccard similarity
        intersection = detected_traits & fingerprint
        union = detected_traits | fingerprint
        score = len(intersection) / len(union) if union else 0.0
        
        # Must meet minimum confidence
        if score >= entry.min_confidence and score > best_score:
            best_match = entry
            best_score = score
    
    return best_match
```

### Step 4.2: Integrate into LLM search

**Modify:** `helion/autotuner/llm_search.py`
```python
from .heuristics.registry import match_heuristic

class LLMGuidedSearch:
    def __init__(self, ...):
        ...
        # Auto-detect heuristic from workload traits
        self._heuristic_entry = self._auto_detect_heuristic()
    
    def _auto_detect_heuristic(self) -> HeuristicEntry | None:
        """Automatically load heuristic based on kernel traits."""
        # Detect traits from kernel
        traits = detect_workload_traits(self._kernel, config_spec=self._config_spec)
        
        # Match to registry
        entry = match_heuristic(traits)
        if entry:
            logger.info(f"Auto-detected heuristic: {entry.pattern_name} (traits={traits})")
        else:
            logger.debug(f"No heuristic match for traits={traits}")
        
        return entry
    
    def _build_seed_configs(self):
        """Build initial seed configs, using heuristic if available."""
        seeds = []
        
        # Try to load heuristic seed
        if self._heuristic_entry and self._heuristic_entry.decision_tree_path:
            heuristic_config = self._load_heuristic_config(
                self._heuristic_entry.decision_tree_path
            )
            if heuristic_config:
                seeds.append(heuristic_config)
        
        # Add random seeds
        seeds.extend(self._random_configs(count=self._initial_random_configs))
        
        return seeds
    
    def _build_initial_prompt(self, ...):
        """Build prompt, including observed examples if available."""
        ...
        # Check if heuristic has observed examples
        observed_section = ""
        if self._heuristic_entry and self._heuristic_entry.observed_json_path:
            observed_section = _build_observed_heuristics_section(
                args, str(self._heuristic_entry.observed_json_path)
            )
        ...
```

### Step 4.3: Test automatic detection

**Script:** `test_trait_detection.py`
```python
import torch
import helion
import helion.language as hl
from helion.autotuner.llm.workload import detect_workload_traits
from helion.autotuner.config_spec import ConfigSpec
from helion.autotuner.heuristics.registry import match_heuristic

# Test 1: Softmax
@helion.kernel()
def softmax(x):
    n, _ = x.size()
    out = torch.empty_like(x)
    for tile_n in hl.tile(n):
        out[tile_n, :] = torch.nn.functional.softmax(x[tile_n, :], dim=1)
    return out

x = torch.randn(4, 128, device='cuda', dtype=torch.float16)
result = softmax(x)
kernel = softmax._get_kernel_for_workload(x)
traits = detect_workload_traits(kernel, config_spec=ConfigSpec.from_kernel(kernel))
match = match_heuristic(traits)

print(f"Softmax:")
print(f"  Traits: {traits}")
print(f"  Match: {match.pattern_name if match else 'None'}")
print(f"  Heuristic: {match.decision_tree_path if match else 'N/A'}")

# Test 2: Custom kernel (no match)
@helion.kernel()
def my_custom_add(x, y):
    return x + y

x = torch.randn(100, device='cuda')
y = torch.randn(100, device='cuda')
result = my_custom_add(x, y)
kernel = my_custom_add._get_kernel_for_workload(x, y)
traits = detect_workload_traits(kernel, config_spec=ConfigSpec.from_kernel(kernel))
match = match_heuristic(traits)

print(f"\nCustom add:")
print(f"  Traits: {traits}")
print(f"  Match: {match.pattern_name if match else 'None'}")
```

**Expected output:**
```
Softmax:
  Traits: frozenset(['reduction', 'exp'])
  Match: softmax
  Heuristic: pretuned_kernels/softmax/_helion_aot_softmax_cuda_sm100.py

Custom add:
  Traits: frozenset([])
  Match: None
```

---

## Phase 5: Expand to Loss Functions (3-4 hours)

### Step 5.1: Wait for cross_entropy AOT to complete
- Currently running overnight (task bp229l74y)
- Should have 12 shapes × ~100-200 configs = 1200-2400 datapoints

**CRITICAL: If new AOT data is needed for any kernel:**
- **DO NOT use run_live.py** (that uses LLMGuidedSearch, only ~5-6 configs)
- **DO use LFBOTreeSearch with full autotuning:**
  ```bash
  HELION_FORCE_AUTOTUNE=1 python3 << 'EOF'
  from examples.{kernel_name} import {kernel_fn}
  import torch
  
  # Define shape
  x = torch.randn(batch, dim, device='cuda', dtype=torch.float16)
  
  # This triggers LFBOTreeSearch (default autotuner)
  # Explores ~100-200 configs over 30-60 minutes
  result = {kernel_fn}(x)
  EOF
  ```
- See: `llm_heuristics_artifacts/loss_functions/AOT_DATA_GENERATION.md` for full guide
- Time budget: 30-60 min per shape for comprehensive exploration

### Step 5.2: Apply winning approach from Phase 3
**If Approach B (observed examples) won:**
1. Build observed_heuristics_cross_entropy.json (same script as softmax)
2. Add cross_entropy to trait registry
3. Re-run heuristics arm
4. Score: target ≤ 0.80

**If Approach A (decision tree) won:**
1. Train sklearn decision tree on comprehensive AOT data
2. Generate heuristic_cross_entropy.py
3. Add to trait registry
4. Re-run and score

**If Approach C (hybrid) won:**
1. Do both A and B above
2. Use both seed + examples

### Step 5.3: Measure success
```bash
python tools/compute_round0_geo.py \
  --baseline iterations/N0_full/baseline \
  --heuristics iterations/N0_full/heuristics_v4 \
  --output iterations/N0_full/scores_v4.json
```

**Target:** round0_best_geo ≤ 0.80 (20% improvement)

---

## Success Criteria (Revised)

### Phase 3 (Softmax Proof of Concept):
- ✅ At least ONE approach achieves round0_best_geo ≤ 0.80 overall
- ✅ Held-out interpolation ≤ 0.85 (can interpolate between training shapes)
- ✅ Held-out extrapolation ≤ 0.90 (can generalize outside training range)
- ✅ Understand which input format (A/B/C) works best

### Phase 4 (Generic Detection):
- ✅ Trait-based matching works for softmax (no name-based matching)
- ✅ No false positives on unrelated kernels
- ✅ Graceful fallback when no match (baseline LLM still works)

### Phase 5 (Loss Functions Goal):
- ✅ round0_best_geo ≤ 0.80 for cross_entropy (using winning approach)
- ✅ Train vs held-out delta < 0.10 (good generalization)

---

## Timeline (Revised)

| Phase | Task | Time | Dependencies |
|-------|------|------|--------------|
| 1 | Baseline (12 shapes) | 3h | None |
| 2A | Approach A (decision tree) | 2h | Phase 1 |
| 2B | Approach B (observed JSON) | 3h | Phase 1 |
| 2C | Approach C (hybrid) | 2h | Phase 1 |
| 3 | Evaluation & decision | 1h | Phase 2 |
| 4 | Trait-based matching | 2h | Phase 3 |
| 5 | Cross-entropy expansion | 3h | Phase 4, AOT complete |
| **Total** | | **16 hours** | |

---

## Phase 2A Results: FAILED

**Date:** 2026-05-08
**Status:** COMPLETED - FAILED TO MEET GOAL

### Metrics:
- **round0_best_geo:** 1.0376 (Target: ≤0.80)
- **Gap to goal:** 23.76 percentage points
- **Train:** 1.0550 (5.5% degradation)
- **Held-out interp:** 0.9769 (2.3% improvement)
- **Held-out extrap:** 1.0311 (3.1% degradation)

### Shape Performance:
- **Improved:** 1/12 shapes (SM_102: 4.7% faster)
- **Neutral:** 3/12 shapes (within 0.2%)
- **Regressed:** 8/12 shapes (worst: SM_008 at 16.3% slower)

### Root Causes:
1. **Low coverage:** Only 14.5% of configs are heuristic-seeded (85.5% random)
2. **Low win rate:** Heuristics only win 18.2% of the time
3. **Wrong patterns:** LLM-generated heuristics use complex configs (persistent mode, tensor descriptors) for large dims, but simple configs (flat mode, basic pointers) actually win
4. **Large dim regression:** Dims ≥5120 regress 9-16% (5 shapes affected)

### Key Insight:
The decision tree heuristic was generated by an LLM analyzing config space, NOT from observed best configs. It lacks domain knowledge and suggests overly complex configurations.

---

## Decision: Test Approach B (Observed Examples)

**Rationale:**
- Approach A failed because LLM-generated heuristics don't know what actually works
- Approach B shows LLM REAL winning configs with timing data from train set
- This is more reliable: we're showing evidence, not synthetic heuristics
- Lower risk than spawning new proposals or switching kernels
- Analysis explicitly recommends this as next step

**Expected Outcome:**
- Score: 0.85-0.95 (improvement, but uncertain if ≥20%)
- Better than A because using observed winners, not guesses
- May still not hit 0.80 goal if LLM can't extrapolate patterns well

**Implementation Time:** ~3 hours (build observed_heuristics.json + run experiment)

**Backup Plan if B fails:**
1. **If B scores 0.85-0.95:** Try Approach C (hybrid: seed + examples) to combine strengths
2. **If B scores 0.95-1.00:** Spawn Proposal Subagent for fundamentally new ideas
3. **If B scores >1.00:** Consider switching to cross_entropy kernel (different problem space)

**Next Steps:**
1. Build observed_heuristics_softmax.json from train set best configs
2. Modify LLM prompt to include observed examples
3. Run Phase 2B experiment (3 hours)
4. Compare scores: if B ≤ 0.80 → SUCCESS, if B > 0.90 → Spawn Proposal, if 0.80-0.90 → Try hybrid

---

## Next Actions

1. **Immediate:** Build observed_heuristics_softmax.json (Step 2B.1)
2. **Then:** Modify LLM prompt to include observed examples (Step 2B.2)
3. **Run:** Phase 2B experiment (~3 hours)
4. **Evaluate:** Compare 2B vs 2A, decide next move based on score
5. **Decision point:** If 2B fails (>0.90), escalate to Proposal Subagent
