# Helion Pallas matmul perf harness

Vendored from
[`cota/Helion-Pallas-Kernels`](https://github.com/cota/Helion-Pallas-Kernels)
(commit `092ec89835a8a22bfb4cd9bef153f68e644fae7d`). Style-normalized to fit
Helion's lint rules; behavior is unchanged from upstream.

This directory contains the head-to-head matmul benchmark used to track how
close Helion's Pallas backend is to a hand-written Pallas baseline (and to
JAX) on a TPU. The kernels here are reference implementations — please don't
import them from production code paths.

## What's in here

| File | Purpose |
|------|---------|
| `matmul_configs.py` | Shapes / dtypes / block configs shared by every variant. |
| `matmul_bench.py` | Inner benchmark loop used by the JAX and Pallas variants. |
| `matmul_jax.py` | Plain `jnp.matmul` reference variant. |
| `matmul_pallas.py` | Hand-written Pallas TPU kernels (matmul, matvec, vecmat, outer). |
| `matmul_helion.py` | Helion (Pallas backend) variant we are trying to beat. |
| `run_variants.py` | Runs N variant scripts sequentially and prints a comparison table. |
| `filter_best_speedups.py` | Reads the `run_variants.py` table and emits one best-config row per shape. |
| `benchmark.sh` | Wrapper that sets TPU-friendly `LIBTPU_INIT_ARGS` before invoking Python. |

## How to run on the TPU pod

All measurements live on the `jongsokchoi-torchtpu` pod; the devserver does
not have a local TPU. Drive everything through `scripts/run-on-pod.sh`, which
tar-syncs the working tree to the pod and runs the command inside the
pre-built venv. Always pass both `HELION_BACKEND=pallas` (otherwise
`helion._testing.DEVICE` defaults to `cuda` and the tests die with
"no NVIDIA driver") and `TPU_VISIBLE_CHIPS=3` (pin to chip 3, 2 cores).

### Full-matrix sweep

Runs every shape / dtype / block from `matmul_configs.py` for JAX, Pallas,
and Helion, then filters to the best block per row. This is the canonical
end-to-end command — same as `plan.md` § 7.2.

```bash
./scripts/run-on-pod.sh HELION_BACKEND=pallas TPU_VISIBLE_CHIPS=3 \
  'examples/pallas_perf/benchmark.sh run_variants.py matmul_jax matmul_pallas matmul_helion > /tmp/results.txt && examples/pallas_perf/filter_best_speedups.py < /tmp/results.txt'
```

### Headline (bf16 1024×1024×1024 @ block 128) measurement

`matmul_bench.py` itself doesn't take CLI args; instead extract the headline
row from a full sweep. Run the sweep three times, take the median of the
`v7_bfloat16_1024x1024x1024_128x128x128` row, and record the 3-run spread.
This matches `plan.md` § 7.1.

```bash
for i in 1 2 3; do
  ./scripts/run-on-pod.sh HELION_BACKEND=pallas TPU_VISIBLE_CHIPS=3 \
    'examples/pallas_perf/benchmark.sh run_variants.py matmul_helion matmul_pallas matmul_jax' \
    | tee /tmp/headline_run_$i.txt
done
```

### Inspecting generated code

```bash
./scripts/run-on-pod.sh HELION_BACKEND=pallas TPU_VISIBLE_CHIPS=3 \
  HELION_PRINT_OUTPUT_CODE=1 HELION_LOGS=+all \
  'examples/pallas_perf/benchmark.sh matmul_helion.py'
```

Diff the printed kernel against `matmul_pallas.py` for the same shape to
verify the generated-code markers from `plan.md` § 9.
