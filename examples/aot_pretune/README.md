# AOT Pretune for Triton-tutorial Kernels

Multi-GPU AOT autotuning workflow that mirrors the Triton tutorial benchmark
shape sweeps. Produces a `(kernel, shape) -> config` JSON database that ships
with the kernels, so users can copy-paste pretuned kernels and run them via
`HELION_AOT_MODE=evaluate` without re-running autotuning.

## Files

| File | Purpose |
|---|---|
| `tutorial_kernels.py` | 7 helion kernels (vector_add, matmul, softmax, layer_norm, attention, grouped_gemm, fp8_gemm), each with `@helion.experimental.aot_kernel(collect_fn=, measure_fn=)`. Shape sweeps match the corresponding Triton tutorial. |
| `pretune_runner.py` | Multi-GPU orchestrator. Detects unused GPUs (preferring later indices), spawns `aot_runner --phase all` per kernel, each pinned to its own GPU. Merges JSONs into `.helion_aot/` at the end. |
| `pick_best_configs.py` | Naive post-merge: scans every measurements CSV under one or more roots, picks the lowest-timing config per `(kernel, shape)` row. |
| `robust_pick.py` | Robust post-merge: takes top-K candidates per shape from the CSVs, re-benchmarks each with K_BENCH `do_bench` calls on real hardware, picks the candidate with the lowest median. |
| `run_3x.sh` | Wrapper that runs the orchestrator 3 times back-to-back, then runs `pick_best_configs.py`. |
| `run_remaining_3x.sh` | Per-GPU scheduler with sticky kernel→GPU mapping. Each GPU runs a queue of kernels sequentially; kernels that share a GPU don't conflict. |
| `run_fp8_3x.sh` / `run_fp8_retry.sh` | Targeted retry helpers for `fp8_gemm`, which on Blackwell hits Triton compile failures that benefit from `HELION_AUTOTUNE_PRECOMPILE_WORKERS` and `HELION_AUTOTUNE_BENCHMARK_SUBPROCESS`. |

## Tutorial-mirrored shape sweeps

Counts match the Triton tutorial source (verified via the actual `x_vals` lists):

| Kernel | Triton tutorial | Shapes | Count |
|---|---|---|---|
| `vector_add`   | 01-vector-add        | `2**i for i in range(12, 28)` | 16 |
| `softmax`      | 02-fused-softmax     | `(4096, 128*i) for i in range(2, 100)` | 98 |
| `matmul`       | 03-matrix-multiplication | square M=N=K = `128*i for i in range(2, 33)` | 31 |
| `layer_norm`   | 05-layer-norm        | `(4096, 512*i) for i in range(2, 32)` | 30 |
| `attention`    | 06-fused-attention   | `BATCH=4 H=32`, `HEAD_DIM ∈ {64, 128} × N_CTX = 2^i for i in range(10, 15)` | 10 |
| `grouped_gemm` | 08-grouped-gemm      | `G=4`, square `2^i for i in range(7, 11)` + `(M=2^i, 8192, 8192)` | 8 |
| `fp8_gemm`     | 10-block-scaled-matmul (adapted) | same as matmul (helion kernel doesn't do block scaling) | 31 |
| **Total**      |                      |  | **224** |

## How to run

```bash
# 1. Make sure you're on a branch that has the helion AOT autotune workflow
#    (choijon5/stack/31 or later) and the float8 hash fix in
#    helion/autotuner/aot_cache.py.

# 2. From the repo root, launch the orchestrator on idle GPUs:
PYTHONPATH=$(pwd) python examples/aot_pretune/pretune_runner.py \
    --gpus 7,6,5,4 --max-workers 4

# 3. After the runs complete, optionally re-benchmark top-K candidates per
#    shape with median-of-5 do_bench calls to filter measurement noise:
PYTHONPATH=$(pwd) python examples/aot_pretune/robust_pick.py --gpu 7

# 4. The final config database lands at
#    .helion_aot/best_configs.json (naive) and
#    .helion_aot/robust_best_configs.json (re-benchmarked).
```

## Sticky kernel→GPU mapping (`pretune_runner.py`)

Same kernel always runs on the same GPU across runs so the Triton compile
cache and L2 layout stay warm.  See `KERNEL_GPU_MAP` in `pretune_runner.py`
— update for your machine's GPU layout.

## Notes for fp8 / Blackwell

`fp8_gemm` on B200 hits recurring Triton compile failures that can hang the
main process.  Two settings help:

- `HELION_AUTOTUNE_PRECOMPILE_WORKERS=N` — long-lived precompile worker pool
  (from `choijon5/stack/31`).  Forces a positive worker count so the main
  process never blocks on compile.
- `HELION_AUTOTUNE_BENCHMARK_SUBPROCESS=1` — runs benchmark in a long-lived
  spawn subprocess.  Lets the autotuner kill a hung kernel and continue.

`run_fp8_retry.sh` enables both for the fp8_gemm kernel.

## Per-GPU scheduler (`run_remaining_3x.sh`)

For a 3-runs-each pretune campaign, this script schedules per GPU
sequentially: e.g. GPU 4 runs `layer_norm × 3` then `fp8_gemm × 3`.
Multiple GPUs run in parallel.  Each pipeline waits for its GPU to be idle
before starting (via `wait_gpu_idle` querying `nvidia-smi`).

## Output layout

```
.helion_aot/
├── job_<kernel>/
│   └── <run_id>/                                       # one per AOT run
│       ├── tuned_configs_<hardware_id>.json            # best config per shape from this run
│       ├── measurements_<hardware_id>.csv              # every (config, shape, timing) row
│       └── heuristic_<kernel>.py                       # auto-generated decision tree
├── logs/<kernel>.log                                   # per-kernel autotune log
├── logs/scheduler/gpu<N>.log                           # per-GPU pipeline log
├── best_configs.json                                   # naive merge (lowest CSV timing)
└── robust_best_configs.json                            # robust pick (median of 5)
```
