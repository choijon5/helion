# LLM heuristic autoresearch: attention

- Output root: `/tmp/helion_llm_autoresearch_attention_router_h2_20260507`
- Autotuner: `LLMGuidedSearch`
- Model: `gpt-5-2`
- Repeats requested: `3`
- Workloads: `attention_1k_d64, attention_2k_d128, attention_4k_d64, attention_4k_d128`
- Arms: `baseline, heuristics`

Lower is better.

| arm | perf geo | time geo | cfg geo | perf range | time range | cfg range |
|---|---:|---:|---:|---:|---:|---:|
| heuristics | 0.804 | 0.802 | 0.993 | 0.717-1.002 | 0.332-1.632 | 0.880-1.042 |

Per-workload geomeans:

| workload | arm | perf geo | time geo | cfg geo |
|---|---|---:|---:|---:|
| attention_1k_d64 | heuristics | 0.807 | 0.719 | 0.988 |
| attention_2k_d128 | heuristics | 0.799 | 1.024 | 0.958 |
| attention_4k_d128 | heuristics | 0.891 | 0.973 | 1.028 |
| attention_4k_d64 | heuristics | 0.727 | 0.576 | 1.001 |

Attention time distribution:

| workload | arm | perf geo | time geo | time p25 | time median | time p75 | time max | compile total | compile max | bench total |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| attention_1k_d64 | heuristics | 0.807 | 0.719 | 0.597 | 0.731 | 0.914 | 1.098 | 0.680 | 0.297 | 0.653 |
| attention_2k_d128 | heuristics | 0.799 | 1.024 | 0.849 | 1.018 | 1.284 | 1.549 | 1.240 | 0.910 | 1.067 |
| attention_4k_d128 | heuristics | 0.891 | 0.973 | 0.775 | 0.968 | 1.300 | 1.632 | 0.984 | 1.008 | 0.980 |
| attention_4k_d64 | heuristics | 0.727 | 0.576 | 0.430 | 0.527 | 0.809 | 1.091 | 0.759 | 0.443 | 0.507 |
