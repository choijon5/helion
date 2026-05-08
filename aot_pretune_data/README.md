# AOT Pretune Data

This branch stores the B200 AOT pretuning datasets used to derive heuristics.
The data is intentionally kept outside the generated `.helion_aot/` run-cache
layout, but preserves the measured-run structure so the CSV/JSON from all
completed measurement runs can be inspected without duplicate file copies.

## Layout

```
aot_pretune_data/
└── b200/
    ├── attention/
    │   └── runs/<run_id>/
    ├── cross_entropy/
    ├── fp8_gemm/
    ├── grouped_gemm/
    ├── layer_norm/
    ├── matmul/
    ├── rms_norm/
    ├── softmax/
    ├── vector_add/
    ├── merged/
    └── run_index.json
```

Each kernel directory contains one subdirectory per AOT run that produced a
measurement CSV:

```
aot_pretune_data/b200/<kernel>/runs/<run_id>/
```

Run directories contain the CSV/JSON files that were produced by that measured
run:

- `measurements_cuda_NVIDIA_B200_13.0.csv`
- `tuned_configs_cuda_NVIDIA_B200_13.0.json`
- `run_metadata.json`
- `heuristic_summary_cuda_NVIDIA_B200_13.0.json`, when generated
- `evaluation_cuda_NVIDIA_B200_13.0.json`, when generated

`merged/` contains cross-run selection outputs when available:

- `best_configs.json`
- `robust_best_configs.json`

`run_index.json` summarizes every retained measured run, including row counts,
shape/config coverage, and repeated shape/config observations inside individual
CSVs.

## Dataset Summary

| Kernel | Measured runs | Measurement rows | Unique shapes | Unique configs |
|---|---:|---:|---:|---:|
| `attention` | 4 | 1000 | 20 | 40 |
| `cross_entropy` | 1 | 441 | 21 | 21 |
| `fp8_gemm` | 5 | 8989 | 39 | 131 |
| `grouped_gemm` | 5 | 898 | 20 | 44 |
| `layer_norm` | 5 | 8130 | 82 | 118 |
| `matmul` | 2 | 3126 | 39 | 64 |
| `rms_norm` | 1 | 839 | 30 | 28 |
| `softmax` | 4 | 58868 | 198 | 256 |
| `vector_add` | 4 | 1694 | 20 | 45 |

The expanded `fp8_gemm` run completed, but it does not contain every possible
shape/config timing cell: one config was only measured for one of the expanded
shapes, leaving 38 unavailable cells.  The earlier completed `fp8_gemm`
measurement runs are preserved as separate run directories.
