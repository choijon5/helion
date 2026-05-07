# Round-0 Objective Summary: `/tmp/helion_round0_objective_20260505_230436/guided_round0_iter11_policy`

Lower is better. `round0_best` is `min(perf_ms)` among autotune CSV rows with `generation == 0` and `status == ok`.

## By Arm

```text
arm                       round0 geo verified geo      round0 range    n
heuristics                     0.953        0.958   0.709-  1.423   57
range_prompt                   0.959        0.975   0.714-  1.238   57
seeds                          0.969        0.966   0.714-  1.170   57
```

## By Workload

```text
workload                           arm                       round0 geo verified geo      round0 range
attention_1k_d64                   heuristics                     0.802        0.804   0.795-  0.807
attention_1k_d64                   range_prompt                   0.995        0.997   0.986-  1.000
attention_1k_d64                   seeds                          0.803        0.805   0.796-  0.807
attention_2k_d128                  heuristics                     0.841        0.810   0.808-  0.908
attention_2k_d128                  range_prompt                   0.841        0.809   0.820-  0.882
attention_2k_d128                  seeds                          1.039        1.031   0.995-  1.124
attention_4k_d128                  heuristics                     1.194        1.184   0.995-  1.423
attention_4k_d128                  range_prompt                   1.072        1.063   0.995-  1.238
attention_4k_d128                  seeds                          1.000        1.001   0.873-  1.146
attention_4k_d64                   heuristics                     0.722        0.720   0.709-  0.733
attention_4k_d64                   range_prompt                   0.985        0.990   0.955-  1.000
attention_4k_d64                   seeds                          0.732        0.727   0.722-  0.737
bmm_8x256x384x512                  heuristics                     0.895        0.911   0.836-  1.000
bmm_8x256x384x512                  range_prompt                   0.841        0.855   0.714-  1.000
bmm_8x256x384x512                  seeds                          0.894        0.904   0.714-  1.000
cross_entropy_32k                  heuristics                     0.999        1.001   0.990-  1.007
cross_entropy_32k                  range_prompt                   0.997        0.999   0.990-  1.000
cross_entropy_32k                  seeds                          0.997        0.998   0.990-  1.001
cross_entropy_4k_16k               heuristics                     1.000        1.001   1.000-  1.000
cross_entropy_4k_16k               range_prompt                   1.000        0.999   1.000-  1.000
cross_entropy_4k_16k               seeds                          1.000        1.000   1.000-  1.000
layer_norm_2k_8k                   heuristics                     1.032        1.005   0.956-  1.096
layer_norm_2k_8k                   range_prompt                   0.999        0.998   0.912-  1.089
layer_norm_2k_8k                   seeds                          1.061        1.001   0.994-  1.100
layer_norm_4k                      heuristics                     0.999        1.001   0.997-  1.000
layer_norm_4k                      range_prompt                   1.001        1.000   0.998-  1.003
layer_norm_4k                      seeds                          0.999        1.000   0.997-  1.000
matmul_1k                          heuristics                     0.999        0.991   0.997-  1.000
matmul_1k                          range_prompt                   1.000        0.998   0.997-  1.003
matmul_1k                          seeds                          1.000        0.989   1.000-  1.000
matmul_skinny_m                    heuristics                     0.999        0.988   0.997-  1.001
matmul_skinny_m                    range_prompt                   1.000        1.002   0.999-  1.001
matmul_skinny_m                    seeds                          0.999        0.991   0.999-  1.000
rms_norm_1024x16384                heuristics                     1.002        1.017   1.002-  1.003
rms_norm_1024x16384                range_prompt                   0.903        0.987   0.900-  0.908
rms_norm_1024x16384                seeds                          0.994        1.018   0.979-  1.003
rms_norm_2048x4096                 heuristics                     0.908        0.945   0.857-  1.003
rms_norm_2048x4096                 range_prompt                   0.904        0.930   0.857-  1.003
rms_norm_2048x4096                 seeds                          1.053        1.033   0.998-  1.170
rms_norm_4k                        heuristics                     0.999        1.007   0.998-  1.000
rms_norm_4k                        range_prompt                   0.999        1.004   0.998-  1.000
rms_norm_4k                        seeds                          1.000        1.005   0.999-  1.000
rms_norm_8192x2048                 heuristics                     0.964        1.004   0.948-  0.993
rms_norm_8192x2048                 range_prompt                   0.945        1.007   0.899-  0.996
rms_norm_8192x2048                 seeds                          1.002        1.003   0.951-  1.057
softmax_1k_1k                      heuristics                     1.000        1.003   1.000-  1.000
softmax_1k_1k                      range_prompt                   0.924        1.002   0.793-  1.000
softmax_1k_1k                      seeds                          1.000        0.999   1.000-  1.000
softmax_2k_4k                      heuristics                     0.942        0.997   0.933-  0.958
softmax_2k_4k                      range_prompt                   0.951        1.000   0.926-  0.992
softmax_2k_4k                      seeds                          0.998        1.001   0.938-  1.065
softmax_4k                         heuristics                     1.000        0.996   1.000-  1.000
softmax_4k                         range_prompt                   1.000        0.998   1.000-  1.000
softmax_4k                         seeds                          1.000        0.996   1.000-  1.000
softmax_4k_2k                      heuristics                     0.902        0.919   0.857-  0.925
softmax_4k_2k                      range_prompt                   0.904        0.918   0.859-  0.928
softmax_4k_2k                      seeds                          0.903        0.920   0.859-  0.925
```
