# Norm Hill-Climbing Setup (portable)

Everything in this directory is designed to run on a fresh B200 box. These
notes capture the one-time setup the first author ran so a collaborator can
reproduce it on a different machine.

## Getting the AOT measurement data

The hill-climbing loop needs the archived B200 measurement CSVs under
`aot_pretune_data/b200/{layer_norm,rms_norm,softmax,...}/`. These are large
(~79 MB, 31 CSVs) and not on this branch. Pull them from the archive
branch:

```bash
cd /home/dev/helion_choijon5
git fetch origin choijon5/aot-pretune-data
git checkout origin/choijon5/aot-pretune-data -- aot_pretune_data/
```

This drops the data into your working tree but does not switch branches or
stage it. If you want to keep main clean after experiments,
`git clean -fd aot_pretune_data/` removes the checkout.

## Prerequisites

- A Linux GPU host (tested on NVIDIA B200, `sm_100`).
- An AWS IAM role (via instance profile or `AWS_ACCESS_KEY_ID` +
  `AWS_SECRET_ACCESS_KEY` env vars) with `bedrock:InvokeModel` permission
  on `us.anthropic.claude-opus-4-7` in some region (tested: `us-east-2`).
  Alternatively, set `ANTHROPIC_API_KEY` and use the existing `anthropic`
  provider — Bedrock is what this loop was authored on but the transport
  supports both.
- ~2 GB free disk for the conda env + nightly PyTorch wheel.

## One-time setup

```bash
# 1. Install Miniconda if you don't have conda
curl -L https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    -o /tmp/mc.sh && bash /tmp/mc.sh -b -p /home/dev/miniconda3

# 2. Activate + accept Anaconda ToS (only needed for default channels)
source /home/dev/miniconda3/etc/profile.d/conda.sh
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# 3. Create the env
conda create -n helion_choijon5 python=3.11 -y
conda activate helion_choijon5

# 4. Install PyTorch nightly with sm_100 support + Triton
pip install --pre \
    --index-url https://download.pytorch.org/whl/nightly/cu130 \
    --extra-index-url https://pypi.org/simple \
    torch triton

# 5. Install helion editable
pip install numpy pluggy
cd /home/dev/helion_choijon5
SETUPTOOLS_SCM_PRETEND_VERSION_FOR_HELION=0.0+dev pip install -e .

# 6. Verify B200 works
python - <<'PY'
import torch
print("arch list:", torch.cuda.get_arch_list())
assert any(a.startswith("sm_100") for a in torch.cuda.get_arch_list()), \
    "nightly did not install with sm_100 support"
x = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
print("bf16 matmul ok:", (x @ x).dtype)
PY
```

## Running the Bedrock LLM autotuner

```bash
source /home/dev/miniconda3/etc/profile.d/conda.sh && conda activate helion_choijon5

# Opus 4.7 on Bedrock, adaptive thinking at high effort.
export HELION_LLM_PROVIDER=bedrock
export HELION_LLM_MODEL=us.anthropic.claude-opus-4-7
export HELION_LLM_ANTHROPIC_THINKING_BUDGET=8000

# On EC2 / EKS: IMDSv2 credentials are picked up automatically.
# Elsewhere: also set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION.
```

Smoke test that Bedrock is reachable:

```bash
python - <<'PY'
import sys; sys.path.insert(0, "/home/dev/helion_choijon5")
from helion.autotuner.llm.transport import call_provider
print(call_provider(
    "bedrock",
    model="us.anthropic.claude-opus-4-7",
    api_base=None, api_key=None,
    messages=[{"role": "user", "content": "reply with just OK"}],
    max_output_tokens=64,
    request_timeout_s=30,
))
PY
```

## Running the hill-climb gates

```bash
cd /home/dev/helion_choijon5

# N0 baseline: 3 kernels x 12 shapes x 3 repeats via LLMGuidedSearch(max_rounds=1)
for K in rms_norm layer_norm softmax; do
  python llm_heuristics_artifacts/norms/tools/run_live.py \
    --kernel $K \
    --arm baseline \
    --shape-grid llm_heuristics_artifacts/norms/iterations/N0_live/shape_grid.json \
    --output-dir llm_heuristics_artifacts/norms/iterations/N0_live/baseline \
    --repeats 3
done

# Score baseline alone (per-shape noise + best-so-far)
python llm_heuristics_artifacts/norms/tools/compute_round0_geo.py \
  --baseline llm_heuristics_artifacts/norms/iterations/N0_live/baseline \
  --output   llm_heuristics_artifacts/norms/iterations/N0_live/baseline_summary.json
```

Heuristics arm for gate N2 (once an AOT heuristic is picked):

```bash
export HELION_LLM_ROUND0_HEURISTIC_PATH=/absolute/path/to/heuristic_layer_norm.py
python llm_heuristics_artifacts/norms/tools/run_live.py \
  --kernel layer_norm \
  --arm heuristics \
  --shape-grid llm_heuristics_artifacts/norms/iterations/N0_live/shape_grid.json \
  --output-dir llm_heuristics_artifacts/norms/iterations/N2_seed/heuristics \
  --repeats 3

python llm_heuristics_artifacts/norms/tools/compute_round0_geo.py \
  --baseline   llm_heuristics_artifacts/norms/iterations/N0_live/baseline \
  --heuristics llm_heuristics_artifacts/norms/iterations/N2_seed/heuristics \
  --output     llm_heuristics_artifacts/norms/iterations/N2_seed/scores.json
```

## Files in this tree

- `plan.md` — living plan + gate definitions + terminal goal.
- `manager.md` — subagent workflow, scoring harness, report formats.
- `N0_baseline.json` — offline AOT heuristic reproduction (input, not score).
- `iterations/N0_live/shape_grid.json` — committed 12-shape grid per kernel
  with 7/5 train/heldout split, fixed seed 20260508.
- `iterations/N0_live/proposal.md` — justification of the shape grid.
- `tools/workloads.py` — builds kernel + args for a shape entry.
- `tools/run_live.py` — runs one arm × one kernel across the grid.
- `tools/compute_round0_geo.py` — computes `round0_best_geo` from arm CSVs.

## Gotchas

- The `HELION_LLM_MODEL` env var must be an **inference-profile** ID (e.g.
  `us.anthropic.claude-opus-4-7`), not a bare model ID like
  `anthropic.claude-opus-4-7`. Bedrock rejects bare model IDs for Opus 4.7.
- Opus 4.7 on Bedrock uses `thinking.type = "adaptive"` + `output_config.effort`;
  older Opus models use `thinking.type = "enabled"` + `budget_tokens`. The
  `_bedrock.py` helper switches on the model name automatically.
- `run_live.py` runs three kernels sequentially to avoid GPU contention; do
  not run two invocations of it on the same GPU concurrently or timings
  will be corrupted.
