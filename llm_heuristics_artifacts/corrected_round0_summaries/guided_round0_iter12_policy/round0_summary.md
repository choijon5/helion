# Round-0 Objective Summary: `/tmp/helion_round0_objective_20260505_230436/guided_round0_iter12_policy`

Lower is better. `round0_best` is `min(perf_ms)` among autotune CSV rows with `generation == 0` and `status == ok`.

## By Arm

```text
arm                       round0 geo verified geo      round0 range    n
range_prompt                   0.974        0.989   0.816-  1.243   57
```

## By Workload

```text
workload                           arm                       round0 geo verified geo      round0 range
attention_1k_d64                   range_prompt                   1.016        1.013   1.000-  1.033
attention_2k_d128                  range_prompt                   1.002        1.034   0.877-  1.105
attention_4k_d128                  range_prompt                   1.005        1.005   0.816-  1.243
attention_4k_d64                   range_prompt                   1.003        0.998   1.000-  1.010
bmm_8x256x384x512                  range_prompt                   0.949        0.984   0.833-  1.196
cross_entropy_32k                  range_prompt                   0.986        1.002   0.958-  1.001
cross_entropy_4k_16k               range_prompt                   1.000        0.999   1.000-  1.000
layer_norm_2k_8k                   range_prompt                   0.956        0.990   0.911-  1.003
layer_norm_4k                      range_prompt                   1.000        0.999   0.997-  1.003
matmul_1k                          range_prompt                   1.001        1.000   1.000-  1.003
matmul_skinny_m                    range_prompt                   0.942        0.942   0.833-  1.001
rms_norm_1024x16384                range_prompt                   0.918        0.965   0.900-  0.951
rms_norm_2048x4096                 range_prompt                   0.903        0.924   0.857-  0.997
rms_norm_4k                        range_prompt                   0.998        1.000   0.998-  0.998
rms_norm_8192x2048                 range_prompt                   0.946        1.000   0.941-  0.953
softmax_1k_1k                      range_prompt                   1.000        0.999   1.000-  1.000
softmax_2k_4k                      range_prompt                   0.952        0.999   0.929-  0.989
softmax_4k                         range_prompt                   0.999        1.002   0.998-  1.000
softmax_4k_2k                      range_prompt                   0.951        0.943   0.925-  0.994
```
