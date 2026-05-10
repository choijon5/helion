# LLM Autotune Heuristics Research

## Dataset

| Kernel | Raw samples | Aggregated rows | Shapes | Configs |
| --- | --- | --- | --- | --- |
| `_bf16xint16_gemm` | 7486 | 7486 | 40 | 7045 |
| `matmul_bf16_int4` | 8348 | 8348 | 40 | 7921 |
| `nvfp4_matmul` | 10917 | 10917 | 40 | 10507 |

## `_bf16xint16_gemm`

| # | Wins | Covered shapes | Median slowdown | Config hash | Compact config |
| --- | --- | --- | --- | --- | --- |
| 1 | 2 | 3 | 1.0 | `e52e91ad4ccfe348` | `{"block_sizes":[256,128,64],"indexing":["pointer","pointer","tensor_descriptor"],"l2_groupings":[8],"load_eviction_policies":["",""],"maxnreg":256,"num_sm_multiplier":1,"num_stages":3,"num_warps":8,"pid_type":"persistent_blocked","range_flattens":[null,null],"range_multi_buffers":[null,null],"range_num_stages":[0,0],"range_unroll_factors":[0,0],"range_warp_specializes":[null,null]}` |
| 2 | 1 | 3 | 1.0 | `9a94d03bc6b50d75` | `{"block_sizes":[256,128,64],"indexing":["pointer","pointer","tensor_descriptor"],"l2_groupings":[8],"load_eviction_policies":["",""],"maxnreg":128,"num_sm_multiplier":1,"num_stages":3,"num_warps":8,"pid_type":"persistent_blocked","range_flattens":[null,null],"range_multi_buffers":[null,null],"range_num_stages":[0,0],"range_unroll_factors":[0,0],"range_warp_specializes":[null,null]}` |
| 3 | 1 | 1 | 1.0 | `d569e77fb894111c` | `{"block_sizes":[32,16,64],"indexing":["pointer","pointer","pointer"],"l2_groupings":[1],"load_eviction_policies":["",""],"maxnreg":128,"num_sm_multiplier":1,"num_stages":3,"num_warps":2,"pid_type":"persistent_interleaved","range_flattens":[null,null],"range_multi_buffers":[null,null],"range_num_stages":[0,0],"range_unroll_factors":[0,2],"range_warp_specializes":[false,false]}` |
| 4 | 1 | 1 | 1.0 | `140e155fbb9fe602` | `{"block_sizes":[32,32,128],"indexing":["pointer","pointer","pointer"],"l2_groupings":[1],"load_eviction_policies":["first",""],"maxnreg":64,"num_sm_multiplier":1,"num_stages":3,"num_warps":8,"pid_type":"persistent_interleaved","range_flattens":[null,null],"range_multi_buffers":[null,null],"range_num_stages":[0,0],"range_unroll_factors":[0,1],"range_warp_specializes":[false,null]}` |
| 5 | 1 | 1 | 1.0 | `7e87c39d2c519f0b` | `{"block_sizes":[32,64,128],"indexing":["pointer","pointer","pointer"],"l2_groupings":[1],"load_eviction_policies":["",""],"num_stages":4,"num_warps":4,"pid_type":"flat","range_flattens":[null,true],"range_multi_buffers":[null,null],"range_num_stages":[0,0],"range_unroll_factors":[0,0],"range_warp_specializes":[null,null]}` |

Top shape regimes:
- `M=1024,N=1024,K=1024,size=<=1024`: 1 shapes; 63d4d65184a90ed6 wins=1
- `M=1024,N=4096,K=2048,size=<=4096`: 1 shapes; 59e8bd0946d786c9 wins=1
- `M=128,N=2048,K=2048,size=<=4096`: 1 shapes; 2d3ee3bd540f8938 wins=1
- `M=1536,N=1536,K=1536,size=<=4096`: 1 shapes; 8923919026db76ef wins=1
- `M=1536,N=1536,K=3072,size=<=4096`: 1 shapes; b3be71e495336ca3 wins=1

## `matmul_bf16_int4`

| # | Wins | Covered shapes | Median slowdown | Config hash | Compact config |
| --- | --- | --- | --- | --- | --- |
| 1 | 2 | 2 | 1.0 | `c68e6f5a5c776737` | `{"block_sizes":[64,64,16],"indexing":["pointer","pointer","pointer"],"l2_groupings":[16],"load_eviction_policies":["first",""],"num_stages":3,"num_warps":2,"pid_type":"flat","range_flattens":[null,null],"range_multi_buffers":[null,null],"range_num_stages":[0,0],"range_unroll_factors":[0,1],"range_warp_specializes":[null,null]}` |
| 2 | 1 | 3 | 1.0 | `38ff822cf797c083` | `{"block_sizes":[16,128,256],"indexing":["tensor_descriptor","pointer","tensor_descriptor"],"l2_groupings":[1],"load_eviction_policies":["last",""],"num_stages":3,"num_warps":8,"pid_type":"flat","range_flattens":[null,null],"range_multi_buffers":[null,true],"range_num_stages":[0,0],"range_unroll_factors":[0,0],"range_warp_specializes":[null,null]}` |
| 3 | 1 | 3 | 1.0 | `7c95ac6a4f06ceba` | `{"block_sizes":[16,128,256],"indexing":["pointer","tensor_descriptor","tensor_descriptor"],"l2_groupings":[1],"load_eviction_policies":["",""],"num_stages":4,"num_warps":8,"pid_type":"flat","range_flattens":[null,null],"range_multi_buffers":[null,true],"range_num_stages":[0,0],"range_unroll_factors":[0,2],"range_warp_specializes":[null,null]}` |
| 4 | 1 | 1 | 1.0 | `ff0dfe0edae30a3e` | `{"block_sizes":[64,32,16],"indexing":["pointer","pointer","pointer"],"l2_groupings":[1],"load_eviction_policies":["",""],"num_stages":2,"num_warps":4,"pid_type":"flat","range_flattens":[null,null],"range_multi_buffers":[null,null],"range_num_stages":[0,0],"range_unroll_factors":[0,0],"range_warp_specializes":[null,null]}` |
| 5 | 1 | 1 | 1.0 | `c0662000be7cf6b6` | `{"block_sizes":[64,64,64],"indexing":["pointer","pointer","pointer"],"l2_groupings":[16],"load_eviction_policies":["first","first"],"num_sm_multiplier":1,"num_stages":3,"num_warps":8,"pid_type":"persistent_interleaved","range_flattens":[null,null],"range_multi_buffers":[null,null],"range_num_stages":[0,0],"range_unroll_factors":[0,0],"range_warp_specializes":[null,null]}` |

Top shape regimes:
- `M=1024,N=1024,K=1024,size=<=1024`: 1 shapes; 189242e4f5d0cadc wins=1
- `M=1024,N=4096,K=2048,size=<=4096`: 1 shapes; 2b033777f98d975a wins=1
- `M=128,N=2048,K=2048,size=<=4096`: 1 shapes; 0fea68f99e15e54f wins=1
- `M=1536,N=1536,K=1536,size=<=4096`: 1 shapes; 4ea8a2ad1e1ecec4 wins=1
- `M=1536,N=1536,K=3072,size=<=4096`: 1 shapes; 0c99d46d44055593 wins=1

## `nvfp4_matmul`

| # | Wins | Covered shapes | Median slowdown | Config hash | Compact config |
| --- | --- | --- | --- | --- | --- |
| 1 | 2 | 3 | 1.0 | `2a83e4f8196f7e04` | `{"block_sizes":[8,128,128],"indexing":["tensor_descriptor","tensor_descriptor","tensor_descriptor"],"l2_groupings":[1],"load_eviction_policies":["","first"],"num_stages":3,"num_warps":4,"pid_type":"flat","range_flattens":[null,null],"range_multi_buffers":[null,null],"range_num_stages":[0,0],"range_unroll_factors":[0,1],"range_warp_specializes":[null,null]}` |
| 2 | 2 | 3 | 1.0 | `5124f6df676a36b2` | `{"block_sizes":[8,128,128],"indexing":["pointer","tensor_descriptor","tensor_descriptor"],"l2_groupings":[1],"load_eviction_policies":["","first"],"num_stages":3,"num_warps":4,"pid_type":"flat","range_flattens":[null,true],"range_multi_buffers":[null,false],"range_num_stages":[0,0],"range_unroll_factors":[0,0],"range_warp_specializes":[null,null]}` |
| 3 | 1 | 1 | 1.0 | `b6ceaf8164b3b398` | `{"block_sizes":[128,32,16],"indexing":["pointer","pointer","pointer"],"l2_groupings":[1],"load_eviction_policies":["first","last"],"num_stages":4,"num_warps":4,"pid_type":"flat","range_flattens":[null,null],"range_multi_buffers":[null,null],"range_num_stages":[0,2],"range_unroll_factors":[0,0],"range_warp_specializes":[null,false]}` |
| 4 | 1 | 1 | 1.0 | `68478515ee4be2f4` | `{"block_sizes":[64,64,16],"indexing":["pointer","pointer","pointer"],"l2_groupings":[1],"load_eviction_policies":["last",""],"num_stages":3,"num_warps":2,"pid_type":"flat","range_flattens":[null,null],"range_multi_buffers":[null,null],"range_num_stages":[0,0],"range_unroll_factors":[0,0],"range_warp_specializes":[null,null]}` |
| 5 | 1 | 1 | 1.0 | `b259678ac1601a52` | `{"block_sizes":[128,128,32],"indexing":["tensor_descriptor","pointer","pointer"],"l2_groupings":[1],"load_eviction_policies":["",""],"num_stages":1,"num_warps":8,"pid_type":"flat","range_flattens":[null,null],"range_multi_buffers":[null,null],"range_num_stages":[0,0],"range_unroll_factors":[0,0],"range_warp_specializes":[null,false]}` |

Top shape regimes:
- `M=1024,N=1024,K=1024,size=<=1024`: 1 shapes; f018ac9236667440 wins=1
- `M=1024,N=4096,K=2048,size=<=4096`: 1 shapes; 534726c2eaf1bd68 wins=1
- `M=128,N=2048,K=2048,size=<=4096`: 1 shapes; b5b5c339fed18c57 wins=1
- `M=1536,N=1536,K=1536,size=<=4096`: 1 shapes; f53a6ab81c836e3c wins=1
- `M=1536,N=1536,K=3072,size=<=4096`: 1 shapes; 1607b4e424a498c3 wins=1
