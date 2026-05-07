# H5b Attention Hybrid LFBO L2 1/8/16 Paired No-Match

- Output root:
  `/tmp/helion_llm_autoresearch_attention_h5b_l2_1_8_16_hybrid_lfbo_paired_no_match_20260507`
- Autotuner: `LLMSeededLFBOTreeSearch`
- Model: `gpt-5-2`
- Repeats: `5`
- GPU: `CUDA_VISIBLE_DEVICES=3`, `--gpu 3`
- Policy:
  `llm_heuristics_artifacts/h5b_attention_strict_family_l2_1_8_16_observed_heuristics_b200.json`
- Workloads:
  `attention_512_d64`, `attention_1k_d64`, `attention_2k_d64`,
  `attention_2k_d128`, `attention_4k_d64`, `attention_4k_d128`
- Arms: `baseline`, `heuristics`

Decision: HOLD for packaging, useful for H5c.

H5b adds only `l2_groupings=[1]` to the d64 `seq<=2048` attention family from
H3b. This fixes the H5 `attention_512_d64` final verified regression and
improves the LLM round-0 seed objective, but final hybrid/LFBO performance is
weaker than H5 overall. The next candidate should split `seq<=1024` from
`seq<=2048` so `l2=1` does not apply to the 2k d64 bucket.

## Corrected Stage Results

Lower is better. Stage 1 is corrected from `_stage1_llm.csv`; stage 2 is the
LFBO stage CSV; verified is the final benchmark median ratio from the aggregate
script.

| workload | stage1 | stage2 | verified | verified range | matched |
|---|---:|---:|---:|---:|---|
| `attention_512_d64` | 0.832 | 0.959 | 0.948 | 0.832-0.998 | yes |
| `attention_1k_d64` | 0.786 | 0.907 | 0.918 | 0.857-0.997 | yes |
| `attention_2k_d64` | 0.740 | 0.875 | 0.887 | 0.796-1.004 | yes |
| `attention_2k_d128` | 0.827 | 0.914 | 0.927 | 0.884-1.011 | yes |
| `attention_4k_d64` | 0.733 | 0.880 | 0.898 | 0.820-0.994 | yes |
| `attention_4k_d128` | 1.001 | 0.996 | 0.991 | 0.898-1.097 | no |
| overall | 0.815 | 0.921 | 0.928 | | |
| matched | 0.782 | 0.907 | 0.915 | | |
| no-match | 1.001 | 0.996 | 0.991 | | |

## Comparison to H5

| metric | H5 | H5b | interpretation |
|---|---:|---:|---|
| round0 overall | 0.830 | 0.815 | H5b improves seed objective |
| round0 matched | 0.799 | 0.782 | H5b improves matched seed objective |
| no-match round0 | 1.001 | 1.001 | guardrail remains neutral |
| final verified overall | 0.896 | 0.928 | H5b weaker final LFBO |
| final verified matched | 0.882 | 0.915 | H5b weaker final LFBO |
| `attention_512_d64` final | 1.025 | 0.948 | H5b fixes H5 regression |

## Command

```bash
cd /home/jongsokchoi/helion_2_llm_priors
CUDA_VISIBLE_DEVICES=3 \
HELION_AUTOTUNE_RANDOM_SEED=20260507 \
HELION_AUTOTUNE_BENCH_SUBPROCESS=1 \
HELION_LLM_OBSERVED_HEURISTICS_PATH=/home/jongsokchoi/helion_2_aot_pretune_data_all/llm_heuristics_artifacts/h5b_attention_strict_family_l2_1_8_16_observed_heuristics_b200.json \
/home/jongsokchoi/.conda/envs/helion_2/bin/python \
  scripts/llm_heuristics_autoresearch.py \
  --gpu 3 \
  --suite core_rows \
  --workloads attention_512_d64,attention_1k_d64,attention_2k_d64,attention_2k_d128,attention_4k_d64,attention_4k_d128 \
  --arms baseline,heuristics \
  --autotuner LLMSeededLFBOTreeSearch \
  --model gpt-5-2 \
  --llm-max-rounds 1 \
  --llm-round0-mode paired-no-match \
  --repeats 5 \
  --timeout-s 2400 \
  --output-root /tmp/helion_llm_autoresearch_attention_h5b_l2_1_8_16_hybrid_lfbo_paired_no_match_20260507
```

## Next

Run H5c with a split d64 policy:

- Add an exact `seq_bin="<=1024"` d64 rule with `l2_groupings=[1,8,16]`.
- Restore the d64 `seq_bin="<=2048"` rule to the H3b `l2_groupings=[8,16]`.
- Leave the `seq_bin="<=4096"` d64 rule and d128 short rule unchanged.
- Keep `attention_4k_d128` as no-match.

This tests whether the `l2=1` benefit is specific to the short d64 bucket
without diluting `attention_2k_d64`.
