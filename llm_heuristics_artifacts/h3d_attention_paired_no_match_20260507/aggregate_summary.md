# H3d attention paired no-match

- Output root: `/tmp/helion_llm_autoresearch_attention_h3d_paired_no_match_20260507`
- Autotuner: `LLMGuidedSearch`
- Model: `gpt-5-2`
- Repeats requested: `5`
- Workloads: `attention_512_d64, attention_1k_d64, attention_2k_d64, attention_2k_d128, attention_4k_d64, attention_4k_d128`
- Arms: `baseline, heuristics`
- Mode: `paired-no-match`

Command shape:

```bash
cd /home/jongsokchoi/helion_2_llm_priors
HELION_LLM_OBSERVED_HEURISTICS_PATH=/home/jongsokchoi/helion_2_aot_pretune_data_all/llm_heuristics_artifacts/h3b_attention_strict_family_no_r4_observed_heuristics_b200.json \
CUDA_VISIBLE_DEVICES=3 /home/jongsokchoi/.conda/envs/helion_2/bin/python \
  scripts/llm_heuristics_autoresearch.py \
  --gpu 3 \
  --suite core_rows \
  --workloads attention_512_d64,attention_1k_d64,attention_2k_d64,attention_2k_d128,attention_4k_d64,attention_4k_d128 \
  --arms baseline,heuristics \
  --autotuner LLMGuidedSearch \
  --model gpt-5-2 \
  --llm-max-rounds 1 \
  --llm-round0-mode paired-no-match \
  --repeats 5 \
  --output-root /tmp/helion_llm_autoresearch_attention_h3d_paired_no_match_20260507
```

Lower is better.

Gate metrics:

| metric | value |
|---|---:|
| overall corrected `round0_best_geo` | 0.822 |
| matched corrected `round0_best_geo` | 0.791 |
| no-match `attention_4k_d128` corrected geo | 1.001 |
| no-match median repeat ratio | 0.9996 |
| no-match max repeat ratio | 1.0048 |

Archived aggregate JSON direct summaries:

| scope | perf geo | time geo | cfg geo | note |
|---|---:|---:|---:|---|
| all rows | 0.815 | 0.966 | 0.928 | script aggregate over 30 workload/repeat rows |
| matched observed-rule rows | 0.783 | | | direct `perf_cost` split in archived JSON |
| no-match `attention_4k_d128` | 1.000 | 0.995 | 1.000 | direct `perf_cost` geo `0.999646` |

Per-workload direct `perf_cost` geomeans:

| workload | matched observed rule | perf geo | repeat range |
|---|---|---:|---|
| attention_512_d64 | true | 0.834 | 0.726-0.876 |
| attention_1k_d64 | true | 0.804 | 0.792-0.808 |
| attention_2k_d64 | true | 0.764 | 0.751-0.794 |
| attention_2k_d128 | true | 0.790 | 0.785-0.803 |
| attention_4k_d64 | true | 0.726 | 0.716-0.748 |
| attention_4k_d128 | false | 1.000 | 0.993-1.007 |

Replay validation:

- Baseline arms recorded round-0 LLM responses for all workloads.
- Matched heuristic arms did not replay the baseline response; they used
  `off_matched_heuristic`.
- The no-match `attention_4k_d128` heuristic arm replayed the paired baseline
  response in all five repeats.
- Candidate overlap was exact enough for the guardrail validation, with
  `same_best=true`.

Decision: H3d PASS. The paired no-match result removes the H3b no-match noise
blocker and unlocks H4 non-attention broadening while preserving the regression
guardrail.
