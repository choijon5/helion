# Data-Derived General Heuristics

These rules are selected from structural config families, not exact AOT config hashes. Lower slowdown is better; 1.0 means the selected template family contains the measured winner for that shape bucket.

| Kernel class | Rules | Shapes | Rows | Geo slowdown | P90 slowdown | Holdout geo | Holdout p90 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `matmul_fp4` | 19 | 40 | 10917 | 1.006 | 1.0279 | 1.0929 | 1.3591 |
| `matmul_int16` | 19 | 40 | 7486 | 1.0169 | 1.0716 | 1.2248 | 1.4848 |
| `matmul_int4` | 19 | 40 | 8348 | 1.0114 | 1.0253 | 1.2215 | 1.6101 |

## `matmul_fp4` `{"aspect":"balanced","dtype":"fp16_bf16","k_bin":"<=4096","m_bin":"<=4096","n_bin":"<=4096"}`

Shapes: 5; rows: 1233; oracle geo slowdown: 1.0041; p90: 1.0167; coverage: 4/5; holdout geo slowdown: 1.3591; holdout p90: 1.5149; holdout coverage: 2/5
| # | Covered | Wins | Geo | P90 | Template |
| --- | --- | --- | --- | --- | --- |
| 1 | 2 | 0 | 1.0083 | 1.0167 | `{"block_sizes":[8,128,128],"l2_groupings":[1],"num_stages":5,"num_warps":4,"pid_type":"flat"}` |
| 2 | 1 | 1 | 1.0 | 1.0 | `{"block_sizes":[16,128,256],"l2_groupings":[1],"num_stages":4,"num_warps":4,"pid_type":"flat"}` |
| 3 | 2 | 1 | 1.1042 | 1.2193 | `{"block_sizes":[8,256,128],"l2_groupings":[1],"num_stages":7,"num_warps":4,"pid_type":"flat"}` |

## `matmul_fp4` `{"aspect":"skinny_m","dtype":"fp16_bf16","k_bin":"<=4096","m_bin":"<=64","n_bin":"<=4096"}`

Shapes: 5; rows: 1596; oracle geo slowdown: 1.0147; p90: 1.0723; coverage: 5/5; holdout geo slowdown: 1.0318; holdout p90: 1.0951; holdout coverage: 3/5
| # | Covered | Wins | Geo | P90 | Template |
| --- | --- | --- | --- | --- | --- |
| 1 | 3 | 0 | 1.0554 | 1.0951 | `{"block_sizes":[128,32,16],"l2_groupings":[1],"num_stages":5,"num_warps":4,"pid_type":"flat"}` |
| 2 | 3 | 1 | 1.1138 | 1.3789 | `{"block_sizes":[128,16,16],"l2_groupings":[1],"num_stages":3,"num_warps":4,"pid_type":"flat"}` |
| 3 | 1 | 1 | 1.0 | 1.0 | `{"block_sizes":[128,32,16],"l2_groupings":[1],"num_sm_multiplier":1,"num_stages":3,"num_warps":4,"pid_type":"persistent_interleaved"}` |

## `matmul_fp4` `{"aspect":"skinny_n","dtype":"fp16_bf16","k_bin":"<=4096","m_bin":"<=4096","n_bin":"<=64"}`

Shapes: 4; rows: 1238; oracle geo slowdown: 1.0; p90: 1.0; coverage: 4/4; holdout geo slowdown: 1.0; holdout p90: 1.0; holdout coverage: 4/4
| # | Covered | Wins | Geo | P90 | Template |
| --- | --- | --- | --- | --- | --- |
| 1 | 3 | 2 | 1.0416 | 1.13 | `{"block_sizes":[128,32,16],"l2_groupings":[1],"num_sm_multiplier":1,"num_stages":3,"num_warps":4,"pid_type":"persistent_blocked"}` |
| 2 | 2 | 2 | 1.0 | 1.0 | `{"block_sizes":[128,32,16],"l2_groupings":[2],"num_stages":3,"num_warps":4,"pid_type":"flat"}` |

## `matmul_fp4` `{"aspect":"balanced","dtype":"fp16_bf16","k_bin":"<=1024","m_bin":"<=1024","n_bin":"<=1024"}`

Shapes: 2; rows: 552; oracle geo slowdown: 1.0007; p90: 1.0015; coverage: 2/2; holdout geo slowdown: 1.101; holdout p90: 1.101; holdout coverage: 1/2
| # | Covered | Wins | Geo | P90 | Template |
| --- | --- | --- | --- | --- | --- |
| 1 | 2 | 1 | 1.0493 | 1.101 | `{"block_sizes":[64,128,32],"l2_groupings":[1],"num_stages":1,"num_warps":8,"pid_type":"flat"}` |
| 2 | 1 | 0 | 1.0015 | 1.0015 | `{"block_sizes":[128,128,32],"l2_groupings":[4],"num_stages":1,"num_warps":8,"pid_type":"flat"}` |

## `matmul_fp4` `{"aspect":"balanced","dtype":"fp16_bf16","k_bin":"<=512","m_bin":"<=512","n_bin":"<=512"}`

Shapes: 2; rows: 599; oracle geo slowdown: 1.0; p90: 1.0; coverage: 2/2; holdout geo slowdown: inf; holdout p90: inf; holdout coverage: 0/2
| # | Covered | Wins | Geo | P90 | Template |
| --- | --- | --- | --- | --- | --- |
| 1 | 2 | 0 | 1.001 | 1.0021 | `{"block_sizes":[64,64,16],"l2_groupings":[1],"num_stages":1,"num_warps":8,"pid_type":"flat"}` |
| 2 | 1 | 1 | 1.0 | 1.0 | `{"block_sizes":[128,128,16],"l2_groupings":[1],"num_stages":1,"num_warps":8,"pid_type":"flat"}` |

## `matmul_fp4` `{"aspect":"skinny_k","dtype":"fp16_bf16","k_bin":"<=128","m_bin":"<=4096","n_bin":"<=4096"}`

Shapes: 2; rows: 539; oracle geo slowdown: 1.0; p90: 1.0; coverage: 2/2; holdout geo slowdown: 1.0; holdout p90: 1.0; holdout coverage: 1/2
| # | Covered | Wins | Geo | P90 | Template |
| --- | --- | --- | --- | --- | --- |
| 1 | 2 | 1 | 1.0 | 1.0 | `{"block_sizes":[8,128,128],"l2_groupings":[1],"num_stages":3,"num_warps":4,"pid_type":"flat"}` |

## `matmul_fp4` `{"aspect":"skinny_k","dtype":"fp16_bf16","k_bin":"<=256","m_bin":"<=4096","n_bin":"<=4096"}`

Shapes: 2; rows: 421; oracle geo slowdown: 1.0; p90: 1.0; coverage: 2/2; holdout geo slowdown: 1.201; holdout p90: 1.4425; holdout coverage: 2/2
| # | Covered | Wins | Geo | P90 | Template |
| --- | --- | --- | --- | --- | --- |
| 1 | 2 | 0 | 1.0 | 1.0 | `{"block_sizes":[8,128,128],"l2_groupings":[1],"num_stages":4,"num_warps":4,"pid_type":"flat"}` |

## `matmul_fp4` `{"aspect":"skinny_k","dtype":"fp16_bf16","k_bin":"<=512","m_bin":"<=4096","n_bin":"<=4096"}`

Shapes: 2; rows: 413; oracle geo slowdown: 1.0; p90: 1.0; coverage: 2/2; holdout geo slowdown: 1.0561; holdout p90: 1.0561; holdout coverage: 1/2
| # | Covered | Wins | Geo | P90 | Template |
| --- | --- | --- | --- | --- | --- |
| 1 | 2 | 1 | 1.0277 | 1.0561 | `{"block_sizes":[8,128,128],"l2_groupings":[1],"num_stages":3,"num_warps":4,"pid_type":"flat"}` |
| 2 | 1 | 1 | 1.0 | 1.0 | `{"block_sizes":[8,256,128],"l2_groupings":[8],"num_stages":6,"num_warps":4,"pid_type":"flat"}` |

## `matmul_fp4` `{"aspect":"skinny_k","dtype":"fp16_bf16","k_bin":"<=64","m_bin":"<=4096","n_bin":"<=4096"}`

Shapes: 2; rows: 409; oracle geo slowdown: 1.0603; p90: 1.1242; coverage: 2/2; holdout geo slowdown: 1.1284; holdout p90: 1.1284; holdout coverage: 1/2
| # | Covered | Wins | Geo | P90 | Template |
| --- | --- | --- | --- | --- | --- |
| 1 | 2 | 1 | 1.0613 | 1.1263 | `{"block_sizes":[16,128,256],"l2_groupings":[1],"num_stages":3,"num_warps":4,"pid_type":"flat"}` |
| 2 | 1 | 0 | 1.1242 | 1.1242 | `{"block_sizes":[8,128,128],"l2_groupings":[32],"num_stages":5,"num_warps":4,"pid_type":"flat"}` |

## `matmul_fp4` `{"aspect":"skinny_m","dtype":"fp16_bf16","k_bin":"<=4096","m_bin":"<=256","n_bin":"<=4096"}`

Shapes: 2; rows: 499; oracle geo slowdown: 1.0001; p90: 1.0003; coverage: 2/2; holdout geo slowdown: inf; holdout p90: inf; holdout coverage: 0/2
| # | Covered | Wins | Geo | P90 | Template |
| --- | --- | --- | --- | --- | --- |
| 1 | 1 | 1 | 1.0 | 1.0 | `{"block_sizes":[64,64,16],"l2_groupings":[1],"num_stages":3,"num_warps":2,"pid_type":"flat"}` |
| 2 | 1 | 1 | 1.0003 | 1.0003 | `{"block_sizes":[64,128,32],"l2_groupings":[1],"num_stages":1,"num_warps":8,"pid_type":"flat"}` |

## `matmul_fp4` `{"aspect":"skinny_n","dtype":"fp16_bf16","k_bin":"<=4096","m_bin":"<=4096","n_bin":"<=1024"}`

Shapes: 2; rows: 722; oracle geo slowdown: 1.0001; p90: 1.0002; coverage: 2/2; holdout geo slowdown: 1.0063; holdout p90: 1.012; holdout coverage: 2/2
| # | Covered | Wins | Geo | P90 | Template |
| --- | --- | --- | --- | --- | --- |
| 1 | 2 | 1 | 1.0004 | 1.0006 | `{"block_sizes":[8,128,128],"l2_groupings":[1],"num_stages":6,"num_warps":4,"pid_type":"flat"}` |
| 2 | 2 | 1 | 1.006 | 1.012 | `{"block_sizes":[8,128,128],"l2_groupings":[4],"num_stages":8,"num_warps":4,"pid_type":"flat"}` |

## `matmul_fp4` `{"aspect":"skinny_n","dtype":"fp16_bf16","k_bin":"<=4096","m_bin":"<=4096","n_bin":"<=128"}`

Shapes: 2; rows: 527; oracle geo slowdown: 1.0002; p90: 1.0004; coverage: 2/2; holdout geo slowdown: inf; holdout p90: inf; holdout coverage: 0/2
| # | Covered | Wins | Geo | P90 | Template |
| --- | --- | --- | --- | --- | --- |
| 1 | 1 | 1 | 1.0 | 1.0 | `{"block_sizes":[64,64,16],"l2_groupings":[4],"num_stages":3,"num_warps":2,"pid_type":"flat"}` |
| 2 | 1 | 1 | 1.0004 | 1.0004 | `{"block_sizes":[128,128,32],"l2_groupings":[1],"num_sm_multiplier":1,"num_stages":1,"num_warps":8,"pid_type":"persistent_blocked"}` |

## `matmul_fp4` `{"aspect":"skinny_n","dtype":"fp16_bf16","k_bin":"<=4096","m_bin":"<=4096","n_bin":"<=256"}`

Shapes: 2; rows: 607; oracle geo slowdown: 1.0279; p90: 1.0567; coverage: 2/2; holdout geo slowdown: inf; holdout p90: inf; holdout coverage: 0/2
| # | Covered | Wins | Geo | P90 | Template |
| --- | --- | --- | --- | --- | --- |
| 1 | 2 | 0 | 1.1365 | 1.2222 | `{"block_sizes":[64,128,32],"l2_groupings":[1],"num_stages":1,"num_warps":8,"pid_type":"flat"}` |
| 2 | 1 | 0 | 1.0 | 1.0 | `{"block_sizes":[128,128,32],"l2_groupings":[2],"num_sm_multiplier":1,"num_stages":1,"num_warps":8,"pid_type":"persistent_interleaved"}` |
| 3 | 1 | 0 | 1.0567 | 1.0567 | `{"block_sizes":[64,128,32],"l2_groupings":[2],"num_stages":1,"num_warps":4,"pid_type":"flat"}` |

## `matmul_fp4` `{"aspect":"balanced","dtype":"fp16_bf16","k_bin":"<=256","m_bin":"<=256","n_bin":"<=256"}`

Shapes: 1; rows: 219; oracle geo slowdown: 1.0069; p90: 1.0069; coverage: 1/1; holdout geo slowdown: inf; holdout p90: inf; holdout coverage: 0/1
| # | Covered | Wins | Geo | P90 | Template |
| --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0 | 1.0069 | 1.0069 | `{"block_sizes":[128,32,16],"l2_groupings":[1],"num_sm_multiplier":1,"num_stages":3,"num_warps":4,"pid_type":"persistent_blocked"}` |

## `matmul_fp4` `{"aspect":"skinny_k","dtype":"fp16_bf16","k_bin":"<=1024","m_bin":"<=4096","n_bin":"<=4096"}`

Shapes: 1; rows: 208; oracle geo slowdown: 1.0; p90: 1.0; coverage: 1/1; holdout geo slowdown: inf; holdout p90: inf; holdout coverage: 0/1
| # | Covered | Wins | Geo | P90 | Template |
| --- | --- | --- | --- | --- | --- |
| 1 | 1 | 1 | 1.0 | 1.0 | `{"block_sizes":[8,256,128],"l2_groupings":[16],"num_stages":6,"num_warps":4,"pid_type":"flat"}` |

## `matmul_fp4` `{"aspect":"skinny_m","dtype":"fp16_bf16","k_bin":"<=4096","m_bin":"<=1024","n_bin":"<=4096"}`

Shapes: 1; rows: 421; oracle geo slowdown: 1.0003; p90: 1.0003; coverage: 1/1; holdout geo slowdown: inf; holdout p90: inf; holdout coverage: 0/1
| # | Covered | Wins | Geo | P90 | Template |
| --- | --- | --- | --- | --- | --- |
| 1 | 1 | 1 | 1.0003 | 1.0003 | `{"block_sizes":[16,128,256],"l2_groupings":[2],"num_stages":3,"num_warps":4,"pid_type":"flat"}` |

## `matmul_fp4` `{"aspect":"skinny_m","dtype":"fp16_bf16","k_bin":"<=4096","m_bin":"<=128","n_bin":"<=4096"}`

Shapes: 1; rows: 203; oracle geo slowdown: 1.0009; p90: 1.0009; coverage: 1/1; holdout geo slowdown: inf; holdout p90: inf; holdout coverage: 0/1
| # | Covered | Wins | Geo | P90 | Template |
| --- | --- | --- | --- | --- | --- |
| 1 | 1 | 1 | 1.0009 | 1.0009 | `{"block_sizes":[64,64,16],"l2_groupings":[1],"num_stages":3,"num_warps":2,"pid_type":"flat"}` |

## `matmul_fp4` `{"aspect":"skinny_m","dtype":"fp16_bf16","k_bin":"<=4096","m_bin":"<=512","n_bin":">4096"}`

Shapes: 1; rows: 208; oracle geo slowdown: 1.0003; p90: 1.0003; coverage: 1/1; holdout geo slowdown: inf; holdout p90: inf; holdout coverage: 0/1
| # | Covered | Wins | Geo | P90 | Template |
| --- | --- | --- | --- | --- | --- |
| 1 | 1 | 1 | 1.0003 | 1.0003 | `{"block_sizes":[8,128,128],"l2_groupings":[1],"num_stages":3,"num_warps":4,"pid_type":"flat"}` |

## `matmul_fp4` `{"aspect":"skinny_n","dtype":"fp16_bf16","k_bin":"<=4096","m_bin":">4096","n_bin":"<=512"}`

Shapes: 1; rows: 303; oracle geo slowdown: 1.0; p90: 1.0; coverage: 1/1; holdout geo slowdown: inf; holdout p90: inf; holdout coverage: 0/1
| # | Covered | Wins | Geo | P90 | Template |
| --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0 | 1.0 | 1.0 | `{"block_sizes":[8,128,128],"l2_groupings":[1],"num_stages":3,"num_warps":4,"pid_type":"flat"}` |

## `matmul_int16` `{"aspect":"balanced","dtype":"fp16_bf16","k_bin":"<=4096","m_bin":"<=4096","n_bin":"<=4096"}`

Shapes: 5; rows: 770; oracle geo slowdown: 1.0664; p90: 1.1395; coverage: 5/5; holdout geo slowdown: 1.1698; holdout p90: 1.2566; holdout coverage: 5/5
| # | Covered | Wins | Geo | P90 | Template |
| --- | --- | --- | --- | --- | --- |
| 1 | 5 | 0 | 1.1972 | 1.2791 | `{"block_sizes":[256,128,64],"l2_groupings":[1],"num_stages":3,"num_warps":4,"pid_type":"flat"}` |
| 2 | 2 | 0 | 1.0694 | 1.087 | `{"block_sizes":[128,128,64],"l2_groupings":[4],"num_sm_multiplier":1,"num_stages":4,"num_warps":8,"pid_type":"persistent_blocked"}` |
| 3 | 3 | 1 | 1.2148 | 1.5714 | `{"block_sizes":[256,256,64],"l2_groupings":[1],"num_stages":3,"num_warps":4,"pid_type":"flat"}` |

## `matmul_int16` `{"aspect":"skinny_m","dtype":"fp16_bf16","k_bin":"<=4096","m_bin":"<=64","n_bin":"<=4096"}`

Shapes: 5; rows: 1104; oracle geo slowdown: 1.024; p90: 1.095; coverage: 4/5; holdout geo slowdown: inf; holdout p90: inf; holdout coverage: 0/5
| # | Covered | Wins | Geo | P90 | Template |
| --- | --- | --- | --- | --- | --- |
| 1 | 2 | 0 | 1.0486 | 1.095 | `{"block_sizes":[64,32,256],"l2_groupings":[1],"num_stages":3,"num_warps":4,"pid_type":"flat"}` |
| 2 | 1 | 0 | 1.0 | 1.0 | `{"block_sizes":[16,16,128],"l2_groupings":[1],"num_stages":7,"num_warps":2,"pid_type":"flat"}` |
| 3 | 1 | 1 | 1.0 | 1.0 | `{"block_sizes":[16,32,512],"l2_groupings":[1],"num_stages":3,"num_warps":4,"pid_type":"flat"}` |

## `matmul_int16` `{"aspect":"skinny_n","dtype":"fp16_bf16","k_bin":"<=4096","m_bin":"<=4096","n_bin":"<=64"}`

Shapes: 4; rows: 704; oracle geo slowdown: 1.0716; p90: 1.1847; coverage: 4/4; holdout geo slowdown: 1.2247; holdout p90: 1.2661; holdout coverage: 2/4
| # | Covered | Wins | Geo | P90 | Template |
| --- | --- | --- | --- | --- | --- |
| 1 | 3 | 0 | 1.1493 | 1.2661 | `{"block_sizes":[64,8,256],"l2_groupings":[1],"num_stages":3,"num_warps":1,"pid_type":"flat"}` |
| 2 | 1 | 0 | 1.0997 | 1.0997 | `{"block_sizes":[32,32,256],"l2_groupings":[1],"num_sm_multiplier":1,"num_stages":4,"num_warps":8,"pid_type":"persistent_blocked"}` |
| 3 | 1 | 0 | 1.0 | 1.0 | `{"block_sizes":[16,64,256],"l2_groupings":[1],"num_stages":3,"num_warps":4,"pid_type":"flat"}` |

## `matmul_int16` `{"aspect":"balanced","dtype":"fp16_bf16","k_bin":"<=1024","m_bin":"<=1024","n_bin":"<=1024"}`

Shapes: 2; rows: 304; oracle geo slowdown: 1.0014; p90: 1.0028; coverage: 2/2; holdout geo slowdown: inf; holdout p90: inf; holdout coverage: 0/2
| # | Covered | Wins | Geo | P90 | Template |
| --- | --- | --- | --- | --- | --- |
| 1 | 1 | 1 | 1.0 | 1.0 | `{"block_sizes":[128,64,64],"l2_groupings":[4],"num_stages":3,"num_warps":8,"pid_type":"flat"}` |
| 2 | 1 | 0 | 1.0028 | 1.0028 | `{"block_sizes":[128,32,128],"l2_groupings":[1],"num_sm_multiplier":1,"num_stages":4,"num_warps":8,"pid_type":"persistent_interleaved"}` |

## `matmul_int16` `{"aspect":"balanced","dtype":"fp16_bf16","k_bin":"<=512","m_bin":"<=512","n_bin":"<=512"}`

Shapes: 2; rows: 301; oracle geo slowdown: 1.0; p90: 1.0; coverage: 2/2; holdout geo slowdown: inf; holdout p90: inf; holdout coverage: 0/2
| # | Covered | Wins | Geo | P90 | Template |
| --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0 | 1.0 | 1.0 | `{"block_sizes":[32,32,128],"l2_groupings":[1],"num_sm_multiplier":1,"num_stages":3,"num_warps":8,"pid_type":"persistent_blocked"}` |
| 2 | 1 | 1 | 1.0 | 1.0 | `{"block_sizes":[32,64,128],"l2_groupings":[1],"num_stages":4,"num_warps":4,"pid_type":"flat"}` |

## `matmul_int16` `{"aspect":"skinny_k","dtype":"fp16_bf16","k_bin":"<=128","m_bin":"<=4096","n_bin":"<=4096"}`

Shapes: 2; rows: 566; oracle geo slowdown: 1.0; p90: 1.0; coverage: 2/2; holdout geo slowdown: inf; holdout p90: inf; holdout coverage: 0/2
| # | Covered | Wins | Geo | P90 | Template |
| --- | --- | --- | --- | --- | --- |
| 1 | 2 | 0 | 1.091 | 1.1902 | `{"block_sizes":[256,64,128],"l2_groupings":[1],"num_stages":3,"num_warps":8,"pid_type":"flat"}` |
| 2 | 1 | 1 | 1.0 | 1.0 | `{"block_sizes":[256,128,32],"l2_groupings":[1],"num_stages":3,"num_warps":4,"pid_type":"flat"}` |

## `matmul_int16` `{"aspect":"skinny_k","dtype":"fp16_bf16","k_bin":"<=256","m_bin":"<=4096","n_bin":"<=4096"}`

Shapes: 2; rows: 325; oracle geo slowdown: 1.0006; p90: 1.0012; coverage: 2/2; holdout geo slowdown: 1.0752; holdout p90: 1.0752; holdout coverage: 1/2
| # | Covered | Wins | Geo | P90 | Template |
| --- | --- | --- | --- | --- | --- |
| 1 | 2 | 1 | 1.0369 | 1.0752 | `{"block_sizes":[256,128,64],"l2_groupings":[8],"num_sm_multiplier":1,"num_stages":3,"num_warps":8,"pid_type":"persistent_blocked"}` |
| 2 | 1 | 1 | 1.0012 | 1.0012 | `{"block_sizes":[256,128,64],"l2_groupings":[2],"num_stages":2,"num_warps":4,"pid_type":"flat"}` |

## `matmul_int16` `{"aspect":"skinny_k","dtype":"fp16_bf16","k_bin":"<=512","m_bin":"<=4096","n_bin":"<=4096"}`

Shapes: 2; rows: 300; oracle geo slowdown: 1.0342; p90: 1.0549; coverage: 2/2; holdout geo slowdown: inf; holdout p90: inf; holdout coverage: 0/2
| # | Covered | Wins | Geo | P90 | Template |
| --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0 | 1.014 | 1.014 | `{"block_sizes":[256,128,32],"l2_groupings":[1],"num_stages":6,"num_warps":8,"pid_type":"flat"}` |
| 2 | 1 | 1 | 1.0549 | 1.0549 | `{"block_sizes":[256,256,64],"l2_groupings":[8],"num_stages":3,"num_warps":8,"pid_type":"flat"}` |

## `matmul_int16` `{"aspect":"skinny_k","dtype":"fp16_bf16","k_bin":"<=64","m_bin":"<=4096","n_bin":"<=4096"}`

Shapes: 2; rows: 423; oracle geo slowdown: 1.0; p90: 1.0; coverage: 2/2; holdout geo slowdown: 1.2049; holdout p90: 1.2049; holdout coverage: 1/2
| # | Covered | Wins | Geo | P90 | Template |
| --- | --- | --- | --- | --- | --- |
| 1 | 2 | 0 | 1.0035 | 1.0069 | `{"block_sizes":[128,64,32],"l2_groupings":[1],"num_stages":2,"num_warps":2,"pid_type":"flat"}` |
| 2 | 2 | 1 | 1.0977 | 1.2049 | `{"block_sizes":[64,64,32],"l2_groupings":[1],"num_stages":2,"num_warps":1,"pid_type":"flat"}` |

## `matmul_int16` `{"aspect":"skinny_m","dtype":"fp16_bf16","k_bin":"<=4096","m_bin":"<=256","n_bin":"<=4096"}`

Shapes: 2; rows: 427; oracle geo slowdown: 1.0; p90: 1.0; coverage: 2/2; holdout geo slowdown: 1.4848; holdout p90: 1.4848; holdout coverage: 1/2
| # | Covered | Wins | Geo | P90 | Template |
| --- | --- | --- | --- | --- | --- |
| 1 | 2 | 0 | 1.146 | 1.2371 | `{"block_sizes":[128,64,128],"l2_groupings":[1],"num_sm_multiplier":1,"num_stages":3,"num_warps":8,"pid_type":"persistent_blocked"}` |
| 2 | 2 | 1 | 1.2185 | 1.4848 | `{"block_sizes":[128,32,128],"l2_groupings":[1],"num_sm_multiplier":1,"num_stages":4,"num_warps":8,"pid_type":"persistent_interleaved"}` |
| 3 | 1 | 1 | 1.0 | 1.0 | `{"block_sizes":[128,64,128],"l2_groupings":[8],"num_stages":4,"num_warps":4,"pid_type":"flat"}` |

## `matmul_int16` `{"aspect":"skinny_n","dtype":"fp16_bf16","k_bin":"<=4096","m_bin":"<=4096","n_bin":"<=1024"}`

Shapes: 2; rows: 286; oracle geo slowdown: 1.001; p90: 1.0021; coverage: 2/2; holdout geo slowdown: inf; holdout p90: inf; holdout coverage: 0/2
| # | Covered | Wins | Geo | P90 | Template |
| --- | --- | --- | --- | --- | --- |
| 1 | 1 | 1 | 1.0 | 1.0 | `{"block_sizes":[256,128,64],"l2_groupings":[16],"num_sm_multiplier":1,"num_stages":3,"num_warps":8,"pid_type":"persistent_blocked"}` |
| 2 | 1 | 1 | 1.0021 | 1.0021 | `{"block_sizes":[128,64,64],"l2_groupings":[2],"num_stages":3,"num_warps":4,"pid_type":"flat"}` |

## `matmul_int16` `{"aspect":"skinny_n","dtype":"fp16_bf16","k_bin":"<=4096","m_bin":"<=4096","n_bin":"<=128"}`

Shapes: 2; rows: 556; oracle geo slowdown: 1.0; p90: 1.0; coverage: 2/2; holdout geo slowdown: inf; holdout p90: inf; holdout coverage: 0/2
| # | Covered | Wins | Geo | P90 | Template |
| --- | --- | --- | --- | --- | --- |
| 1 | 1 | 1 | 1.0 | 1.0 | `{"block_sizes":[128,32,128],"l2_groupings":[1],"num_stages":4,"num_warps":2,"pid_type":"flat"}` |
| 2 | 1 | 1 | 1.0 | 1.0 | `{"block_sizes":[64,32,256],"l2_groupings":[16],"num_stages":3,"num_warps":8,"pid_type":"flat"}` |

## `matmul_int16` `{"aspect":"skinny_n","dtype":"fp16_bf16","k_bin":"<=4096","m_bin":"<=4096","n_bin":"<=256"}`

Shapes: 2; rows: 299; oracle geo slowdown: 1.0016; p90: 1.0033; coverage: 2/2; holdout geo slowdown: inf; holdout p90: inf; holdout coverage: 0/2
| # | Covered | Wins | Geo | P90 | Template |
| --- | --- | --- | --- | --- | --- |
| 1 | 1 | 1 | 1.0 | 1.0 | `{"block_sizes":[128,64,128],"l2_groupings":[1],"num_stages":4,"num_warps":8,"pid_type":"flat"}` |
| 2 | 1 | 0 | 1.0033 | 1.0033 | `{"block_sizes":[128,32,128],"l2_groupings":[4],"num_stages":3,"num_warps":8,"pid_type":"flat"}` |

## `matmul_int16` `{"aspect":"balanced","dtype":"fp16_bf16","k_bin":"<=256","m_bin":"<=256","n_bin":"<=256"}`

Shapes: 1; rows: 206; oracle geo slowdown: 1.0; p90: 1.0; coverage: 1/1; holdout geo slowdown: inf; holdout p90: inf; holdout coverage: 0/1
| # | Covered | Wins | Geo | P90 | Template |
| --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0 | 1.0 | 1.0 | `{"block_sizes":[16,32,256],"l2_groupings":[2],"num_sm_multiplier":1,"num_stages":1,"num_warps":8,"pid_type":"persistent_blocked"}` |

## `matmul_int16` `{"aspect":"skinny_k","dtype":"fp16_bf16","k_bin":"<=1024","m_bin":"<=4096","n_bin":"<=4096"}`

Shapes: 1; rows: 137; oracle geo slowdown: 1.0; p90: 1.0; coverage: 1/1; holdout geo slowdown: inf; holdout p90: inf; holdout coverage: 0/1
| # | Covered | Wins | Geo | P90 | Template |
| --- | --- | --- | --- | --- | --- |
| 1 | 1 | 1 | 1.0 | 1.0 | `{"block_sizes":[256,128,32],"l2_groupings":[4],"num_stages":3,"num_warps":4,"pid_type":"flat"}` |

## `matmul_int16` `{"aspect":"skinny_m","dtype":"fp16_bf16","k_bin":"<=4096","m_bin":"<=1024","n_bin":"<=4096"}`

Shapes: 1; rows: 187; oracle geo slowdown: 1.0; p90: 1.0; coverage: 1/1; holdout geo slowdown: inf; holdout p90: inf; holdout coverage: 0/1
| # | Covered | Wins | Geo | P90 | Template |
| --- | --- | --- | --- | --- | --- |
| 1 | 1 | 1 | 1.0 | 1.0 | `{"block_sizes":[256,128,64],"l2_groupings":[8],"num_sm_multiplier":1,"num_stages":3,"num_warps":8,"pid_type":"persistent_blocked"}` |

## `matmul_int16` `{"aspect":"skinny_m","dtype":"fp16_bf16","k_bin":"<=4096","m_bin":"<=128","n_bin":"<=4096"}`

Shapes: 1; rows: 201; oracle geo slowdown: 1.131; p90: 1.131; coverage: 1/1; holdout geo slowdown: inf; holdout p90: inf; holdout coverage: 0/1
| # | Covered | Wins | Geo | P90 | Template |
| --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0 | 1.131 | 1.131 | `{"block_sizes":[64,32,128],"l2_groupings":[1],"num_stages":4,"num_warps":8,"pid_type":"flat"}` |

## `matmul_int16` `{"aspect":"skinny_m","dtype":"fp16_bf16","k_bin":"<=4096","m_bin":"<=512","n_bin":">4096"}`

Shapes: 1; rows: 244; oracle geo slowdown: 1.0; p90: 1.0; coverage: 1/1; holdout geo slowdown: inf; holdout p90: inf; holdout coverage: 0/1
| # | Covered | Wins | Geo | P90 | Template |
| --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0 | 1.0 | 1.0 | `{"block_sizes":[256,128,64],"l2_groupings":[8],"num_stages":3,"num_warps":8,"pid_type":"flat"}` |

## `matmul_int16` `{"aspect":"skinny_n","dtype":"fp16_bf16","k_bin":"<=4096","m_bin":">4096","n_bin":"<=512"}`

Shapes: 1; rows: 146; oracle geo slowdown: 1.0; p90: 1.0; coverage: 1/1; holdout geo slowdown: inf; holdout p90: inf; holdout coverage: 0/1
| # | Covered | Wins | Geo | P90 | Template |
| --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0 | 1.0 | 1.0 | `{"block_sizes":[128,128,64],"l2_groupings":[1],"num_stages":3,"num_warps":4,"pid_type":"flat"}` |

## `matmul_int4` `{"aspect":"balanced","dtype":"fp16_bf16","k_bin":"<=4096","m_bin":"<=4096","n_bin":"<=4096"}`

Shapes: 5; rows: 993; oracle geo slowdown: 1.0; p90: 1.0; coverage: 4/5; holdout geo slowdown: 1.2795; holdout p90: 1.5682; holdout coverage: 2/5
| # | Covered | Wins | Geo | P90 | Template |
| --- | --- | --- | --- | --- | --- |
| 1 | 2 | 2 | 1.0 | 1.0 | `{"block_sizes":[16,128,128],"l2_groupings":[2],"num_stages":6,"num_warps":4,"pid_type":"flat"}` |
| 2 | 2 | 1 | 1.0109 | 1.0218 | `{"block_sizes":[16,128,128],"l2_groupings":[2],"num_stages":3,"num_warps":4,"pid_type":"flat"}` |
| 3 | 1 | 1 | 1.0 | 1.0 | `{"block_sizes":[16,128,256],"l2_groupings":[1],"num_stages":8,"num_warps":8,"pid_type":"flat"}` |

## `matmul_int4` `{"aspect":"skinny_m","dtype":"fp16_bf16","k_bin":"<=4096","m_bin":"<=64","n_bin":"<=4096"}`

Shapes: 5; rows: 1239; oracle geo slowdown: 1.0018; p90: 1.0074; coverage: 4/5; holdout geo slowdown: 1.6101; holdout p90: 1.6101; holdout coverage: 1/5
| # | Covered | Wins | Geo | P90 | Template |
| --- | --- | --- | --- | --- | --- |
| 1 | 2 | 0 | 1.0037 | 1.0074 | `{"block_sizes":[128,32,16],"l2_groupings":[1],"num_stages":4,"num_warps":4,"pid_type":"flat"}` |
| 2 | 1 | 0 | 1.0 | 1.0 | `{"block_sizes":[128,16,16],"l2_groupings":[1],"num_stages":5,"num_warps":2,"pid_type":"xyz"}` |
| 3 | 1 | 0 | 1.0 | 1.0 | `{"block_sizes":[128,16,32],"l2_groupings":[16],"num_stages":5,"num_warps":4,"pid_type":"flat"}` |

## `matmul_int4` `{"aspect":"skinny_n","dtype":"fp16_bf16","k_bin":"<=4096","m_bin":"<=4096","n_bin":"<=64"}`

Shapes: 4; rows: 906; oracle geo slowdown: 1.0253; p90: 1.1053; coverage: 4/4; holdout geo slowdown: 1.4316; holdout p90: 1.8544; holdout coverage: 2/4
| # | Covered | Wins | Geo | P90 | Template |
| --- | --- | --- | --- | --- | --- |
| 1 | 2 | 0 | 1.0513 | 1.1053 | `{"block_sizes":[64,64,16],"l2_groupings":[1],"num_sm_multiplier":1,"num_stages":3,"num_warps":2,"pid_type":"persistent_blocked"}` |
| 2 | 1 | 1 | 1.0 | 1.0 | `{"block_sizes":[256,16,16],"l2_groupings":[1],"num_stages":3,"num_warps":4,"pid_type":"xyz"}` |
| 3 | 1 | 1 | 1.0 | 1.0 | `{"block_sizes":[64,64,16],"l2_groupings":[16],"num_stages":3,"num_warps":2,"pid_type":"flat"}` |

## `matmul_int4` `{"aspect":"balanced","dtype":"fp16_bf16","k_bin":"<=1024","m_bin":"<=1024","n_bin":"<=1024"}`

Shapes: 2; rows: 306; oracle geo slowdown: 1.0; p90: 1.0; coverage: 2/2; holdout geo slowdown: inf; holdout p90: inf; holdout coverage: 0/2
| # | Covered | Wins | Geo | P90 | Template |
| --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0 | 1.0 | 1.0 | `{"block_sizes":[16,128,64],"l2_groupings":[1],"num_stages":3,"num_warps":4,"pid_type":"flat"}` |
| 2 | 1 | 1 | 1.0 | 1.0 | `{"block_sizes":[64,64,64],"l2_groupings":[16],"num_sm_multiplier":1,"num_stages":3,"num_warps":8,"pid_type":"persistent_interleaved"}` |

## `matmul_int4` `{"aspect":"balanced","dtype":"fp16_bf16","k_bin":"<=512","m_bin":"<=512","n_bin":"<=512"}`

Shapes: 2; rows: 362; oracle geo slowdown: 1.0; p90: 1.0; coverage: 2/2; holdout geo slowdown: inf; holdout p90: inf; holdout coverage: 0/2
| # | Covered | Wins | Geo | P90 | Template |
| --- | --- | --- | --- | --- | --- |
| 1 | 2 | 1 | 1.0753 | 1.1563 | `{"block_sizes":[64,32,32],"l2_groupings":[1],"num_sm_multiplier":1,"num_stages":3,"num_warps":4,"pid_type":"persistent_interleaved"}` |
| 2 | 1 | 0 | 1.0 | 1.0 | `{"block_sizes":[32,32,32],"l2_groupings":[1],"num_stages":3,"num_warps":2,"pid_type":"flat"}` |

## `matmul_int4` `{"aspect":"skinny_k","dtype":"fp16_bf16","k_bin":"<=128","m_bin":"<=4096","n_bin":"<=4096"}`

Shapes: 2; rows: 468; oracle geo slowdown: 1.0; p90: 1.0; coverage: 2/2; holdout geo slowdown: 1.1465; holdout p90: 1.1622; holdout coverage: 2/2
| # | Covered | Wins | Geo | P90 | Template |
| --- | --- | --- | --- | --- | --- |
| 1 | 2 | 0 | 1.0535 | 1.1098 | `{"block_sizes":[16,128,256],"l2_groupings":[1],"num_stages":4,"num_warps":8,"pid_type":"flat"}` |
| 2 | 2 | 1 | 1.0635 | 1.131 | `{"block_sizes":[16,128,128],"l2_groupings":[1],"num_stages":3,"num_warps":4,"pid_type":"flat"}` |

## `matmul_int4` `{"aspect":"skinny_k","dtype":"fp16_bf16","k_bin":"<=256","m_bin":"<=4096","n_bin":"<=4096"}`

Shapes: 2; rows: 374; oracle geo slowdown: 1.0196; p90: 1.0395; coverage: 2/2; holdout geo slowdown: 1.0603; holdout p90: 1.1205; holdout coverage: 2/2
| # | Covered | Wins | Geo | P90 | Template |
| --- | --- | --- | --- | --- | --- |
| 1 | 2 | 0 | 1.0207 | 1.0402 | `{"block_sizes":[16,128,128],"l2_groupings":[1],"num_stages":4,"num_warps":4,"pid_type":"flat"}` |
| 2 | 2 | 0 | 1.0585 | 1.1205 | `{"block_sizes":[16,128,256],"l2_groupings":[1],"num_stages":3,"num_warps":4,"pid_type":"flat"}` |
| 3 | 2 | 1 | 1.0212 | 1.0395 | `{"block_sizes":[16,128,128],"l2_groupings":[1],"num_stages":3,"num_warps":4,"pid_type":"flat"}` |

## `matmul_int4` `{"aspect":"skinny_k","dtype":"fp16_bf16","k_bin":"<=512","m_bin":"<=4096","n_bin":"<=4096"}`

Shapes: 2; rows: 422; oracle geo slowdown: 1.0; p90: 1.0; coverage: 2/2; holdout geo slowdown: inf; holdout p90: inf; holdout coverage: 0/2
| # | Covered | Wins | Geo | P90 | Template |
| --- | --- | --- | --- | --- | --- |
| 1 | 1 | 1 | 1.0 | 1.0 | `{"block_sizes":[16,128,128],"l2_groupings":[1],"num_stages":3,"num_warps":4,"pid_type":"flat"}` |
| 2 | 1 | 0 | 1.0 | 1.0 | `{"block_sizes":[16,128,256],"l2_groupings":[16],"num_stages":5,"num_warps":8,"pid_type":"flat"}` |

## `matmul_int4` `{"aspect":"skinny_k","dtype":"fp16_bf16","k_bin":"<=64","m_bin":"<=4096","n_bin":"<=4096"}`

Shapes: 2; rows: 369; oracle geo slowdown: 1.0; p90: 1.0; coverage: 2/2; holdout geo slowdown: 1.0; holdout p90: 1.0; holdout coverage: 1/2
| # | Covered | Wins | Geo | P90 | Template |
| --- | --- | --- | --- | --- | --- |
| 1 | 2 | 0 | 1.0 | 1.0 | `{"block_sizes":[16,128,128],"l2_groupings":[1],"num_stages":3,"num_warps":4,"pid_type":"flat"}` |

## `matmul_int4` `{"aspect":"skinny_m","dtype":"fp16_bf16","k_bin":"<=4096","m_bin":"<=256","n_bin":"<=4096"}`

Shapes: 2; rows: 398; oracle geo slowdown: 1.0; p90: 1.0; coverage: 2/2; holdout geo slowdown: inf; holdout p90: inf; holdout coverage: 0/2
| # | Covered | Wins | Geo | P90 | Template |
| --- | --- | --- | --- | --- | --- |
| 1 | 1 | 1 | 1.0 | 1.0 | `{"block_sizes":[32,64,32],"l2_groupings":[1],"num_stages":5,"num_warps":1,"pid_type":"flat"}` |
| 2 | 1 | 1 | 1.0 | 1.0 | `{"block_sizes":[64,64,64],"l2_groupings":[1],"num_stages":3,"num_warps":8,"pid_type":"flat"}` |

## `matmul_int4` `{"aspect":"skinny_n","dtype":"fp16_bf16","k_bin":"<=4096","m_bin":"<=4096","n_bin":"<=1024"}`

Shapes: 2; rows: 471; oracle geo slowdown: 1.0004; p90: 1.0005; coverage: 2/2; holdout geo slowdown: 1.0465; holdout p90: 1.0947; holdout coverage: 2/2
| # | Covered | Wins | Geo | P90 | Template |
| --- | --- | --- | --- | --- | --- |
| 1 | 2 | 1 | 1.0004 | 1.0005 | `{"block_sizes":[16,128,128],"l2_groupings":[1],"num_stages":4,"num_warps":4,"pid_type":"flat"}` |

## `matmul_int4` `{"aspect":"skinny_n","dtype":"fp16_bf16","k_bin":"<=4096","m_bin":"<=4096","n_bin":"<=128"}`

Shapes: 2; rows: 412; oracle geo slowdown: 1.0153; p90: 1.0307; coverage: 2/2; holdout geo slowdown: 1.16; holdout p90: 1.16; holdout coverage: 1/2
| # | Covered | Wins | Geo | P90 | Template |
| --- | --- | --- | --- | --- | --- |
| 1 | 2 | 0 | 1.0465 | 1.0952 | `{"block_sizes":[64,64,16],"l2_groupings":[1],"num_stages":3,"num_warps":2,"pid_type":"flat"}` |
| 2 | 2 | 0 | 1.0935 | 1.16 | `{"block_sizes":[64,64,32],"l2_groupings":[1],"num_stages":3,"num_warps":2,"pid_type":"flat"}` |

## `matmul_int4` `{"aspect":"skinny_n","dtype":"fp16_bf16","k_bin":"<=4096","m_bin":"<=4096","n_bin":"<=256"}`

Shapes: 2; rows: 375; oracle geo slowdown: 1.0013; p90: 1.0025; coverage: 2/2; holdout geo slowdown: 1.391; holdout p90: 1.391; holdout coverage: 1/2
| # | Covered | Wins | Geo | P90 | Template |
| --- | --- | --- | --- | --- | --- |
| 1 | 2 | 1 | 1.1809 | 1.391 | `{"block_sizes":[64,64,64],"l2_groupings":[1],"num_stages":3,"num_warps":8,"pid_type":"flat"}` |
| 2 | 1 | 1 | 1.0 | 1.0 | `{"block_sizes":[64,64,128],"l2_groupings":[1],"num_sm_multiplier":1,"num_stages":3,"num_warps":4,"pid_type":"persistent_interleaved"}` |

## `matmul_int4` `{"aspect":"balanced","dtype":"fp16_bf16","k_bin":"<=256","m_bin":"<=256","n_bin":"<=256"}`

Shapes: 1; rows: 97; oracle geo slowdown: 1.0; p90: 1.0; coverage: 1/1; holdout geo slowdown: inf; holdout p90: inf; holdout coverage: 0/1
| # | Covered | Wins | Geo | P90 | Template |
| --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0 | 1.0 | 1.0 | `{"block_sizes":[128,32,16],"l2_groupings":[1],"num_sm_multiplier":1,"num_stages":2,"num_warps":4,"pid_type":"persistent_interleaved"}` |

## `matmul_int4` `{"aspect":"skinny_k","dtype":"fp16_bf16","k_bin":"<=1024","m_bin":"<=4096","n_bin":"<=4096"}`

Shapes: 1; rows: 340; oracle geo slowdown: 1.0005; p90: 1.0005; coverage: 1/1; holdout geo slowdown: inf; holdout p90: inf; holdout coverage: 0/1
| # | Covered | Wins | Geo | P90 | Template |
| --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0 | 1.0005 | 1.0005 | `{"block_sizes":[16,128,128],"l2_groupings":[1],"num_stages":7,"num_warps":4,"pid_type":"flat"}` |

## `matmul_int4` `{"aspect":"skinny_m","dtype":"fp16_bf16","k_bin":"<=4096","m_bin":"<=1024","n_bin":"<=4096"}`

Shapes: 1; rows: 100; oracle geo slowdown: 1.1649; p90: 1.1649; coverage: 1/1; holdout geo slowdown: inf; holdout p90: inf; holdout coverage: 0/1
| # | Covered | Wins | Geo | P90 | Template |
| --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0 | 1.1649 | 1.1649 | `{"block_sizes":[16,256,128],"l2_groupings":[1],"num_stages":2,"num_warps":8,"pid_type":"flat"}` |

## `matmul_int4` `{"aspect":"skinny_m","dtype":"fp16_bf16","k_bin":"<=4096","m_bin":"<=128","n_bin":"<=4096"}`

Shapes: 1; rows: 194; oracle geo slowdown: 1.0; p90: 1.0; coverage: 1/1; holdout geo slowdown: inf; holdout p90: inf; holdout coverage: 0/1
| # | Covered | Wins | Geo | P90 | Template |
| --- | --- | --- | --- | --- | --- |
| 1 | 1 | 1 | 1.0 | 1.0 | `{"block_sizes":[128,64,32],"l2_groupings":[1],"num_stages":3,"num_warps":8,"pid_type":"flat"}` |

## `matmul_int4` `{"aspect":"skinny_m","dtype":"fp16_bf16","k_bin":"<=4096","m_bin":"<=512","n_bin":">4096"}`

Shapes: 1; rows: 196; oracle geo slowdown: 1.0; p90: 1.0; coverage: 1/1; holdout geo slowdown: inf; holdout p90: inf; holdout coverage: 0/1
| # | Covered | Wins | Geo | P90 | Template |
| --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0 | 1.0 | 1.0 | `{"block_sizes":[16,128,256],"l2_groupings":[1],"num_stages":4,"num_warps":8,"pid_type":"flat"}` |

## `matmul_int4` `{"aspect":"skinny_n","dtype":"fp16_bf16","k_bin":"<=4096","m_bin":">4096","n_bin":"<=512"}`

Shapes: 1; rows: 326; oracle geo slowdown: 1.0; p90: 1.0; coverage: 1/1; holdout geo slowdown: inf; holdout p90: inf; holdout coverage: 0/1
| # | Covered | Wins | Geo | P90 | Template |
| --- | --- | --- | --- | --- | --- |
| 1 | 1 | 1 | 1.0 | 1.0 | `{"block_sizes":[16,128,128],"l2_groupings":[4],"num_stages":7,"num_warps":4,"pid_type":"flat"}` |
