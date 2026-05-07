# Round-0 Objective Summary: `/tmp/helion_round0_objective_20260505_230436/hybrid_lfbo_round0_handoff`

Lower is better. `round0_best` is `min(perf_ms)` among autotune CSV rows with `generation == 0` and `status == ok`.

## By Arm

```text
arm                       round0 geo verified geo      round0 range    n
heuristics                     0.949        0.973   0.806-  1.400   24
range_prompt                   0.999        0.970   0.825-  1.235   24
seeds                          0.988        0.978   0.807-  1.400   24
```

## By Workload

```text
workload                           arm                       round0 geo verified geo      round0 range
attention_1k_d64                   heuristics                     0.824        0.899   0.806-  0.834
attention_1k_d64                   range_prompt                   1.037        0.911   1.014-  1.065
attention_1k_d64                   seeds                          0.824        0.882   0.807-  0.833
attention_2k_d128                  heuristics                     0.821        0.938   0.811-  0.842
attention_2k_d128                  range_prompt                   0.954        0.938   0.825-  1.235
attention_2k_d128                  seeds                          1.057        1.046   0.999-  1.141
bmm_8x256x384x512                  heuristics                     1.191        0.999   1.000-  1.400
bmm_8x256x384x512                  range_prompt                   1.201        1.001   1.200-  1.203
bmm_8x256x384x512                  seeds                          1.121        0.999   1.000-  1.400
cross_entropy_32k                  heuristics                     1.000        1.012   1.000-  1.001
cross_entropy_32k                  range_prompt                   1.000        0.994   1.000-  1.000
cross_entropy_32k                  seeds                          1.000        1.005   1.000-  1.000
layer_norm_4k                      heuristics                     1.001        1.000   0.998-  1.003
layer_norm_4k                      range_prompt                   1.001        0.969   1.000-  1.003
layer_norm_4k                      seeds                          1.001        1.000   0.998-  1.002
matmul_1k                          heuristics                     0.999        0.965   0.997-  1.000
matmul_1k                          range_prompt                   1.024        0.989   1.000-  1.070
matmul_1k                          seeds                          1.000        0.975   0.997-  1.003
rms_norm_2048x4096                 heuristics                     0.857        1.044   0.857-  0.857
rms_norm_2048x4096                 range_prompt                   0.857        1.025   0.857-  0.857
rms_norm_2048x4096                 seeds                          0.975        0.990   0.929-  1.000
softmax_4k_2k                      heuristics                     0.951        0.938   0.925-  0.996
softmax_4k_2k                      range_prompt                   0.951        0.935   0.925-  0.996
softmax_4k_2k                      seeds                          0.952        0.939   0.925-  0.999
```
