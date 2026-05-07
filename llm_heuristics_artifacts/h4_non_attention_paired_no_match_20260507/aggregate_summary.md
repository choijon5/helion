# H4 Non-Attention Paired-No-Match

Output root:
`/tmp/helion_llm_autoresearch_h4_non_attention_paired_no_match_20260507`

Archived aggregate:
`llm_heuristics_artifacts/h4_non_attention_paired_no_match_20260507/aggregate_results.json`

The copied aggregate JSON is 256529 bytes and preserves the per-repeat rows,
script aggregate diagnostics, matched-rule metadata, and zero-error result. Logs
and per-workload CSVs were not duplicated.

## Command

The original shell argv was not embedded in the aggregate JSON; this is the
reconstructed command shape from the run metadata and H4 run contract.

```bash
cd /home/jongsokchoi/helion_2_llm_priors
HELION_LLM_OBSERVED_HEURISTICS_PATH=/home/jongsokchoi/helion_2_aot_pretune_data_all/llm_heuristics_artifacts/h4_non_attention_observed_heuristics_b200.json \
CUDA_VISIBLE_DEVICES=3 /home/jongsokchoi/.conda/envs/helion_2/bin/python \
  scripts/llm_heuristics_autoresearch.py \
  --gpu 3 \
  --suite core_rows \
  --workloads rms_norm_2048x4096,rms_norm_1024x16384,rms_norm_8192x2048,rms_norm_4k,softmax_4k_2k,softmax_2k_4k,softmax_1k_1k,softmax_4k,bmm_8x256x384x512,bmm_16x128x512x256,cross_entropy_32k,cross_entropy_1k_64k,matmul_1k,matmul_skinny_m,layer_norm_4k,attention_4k_d128 \
  --arms baseline,heuristics \
  --autotuner LLMGuidedSearch \
  --model gpt-5-2 \
  --llm-max-rounds 1 \
  --llm-round0-mode paired-no-match \
  --repeats 5 \
  --output-root /tmp/helion_llm_autoresearch_h4_non_attention_paired_no_match_20260507
```

## Result

Status: HOLD.

H4 did not clear the material-win threshold. Corrected paired-no-match results:

| scope | round0_best_geo | decision |
|---|---:|---|
| overall | 0.993 | below material win |
| matched active non-attention | 0.989 | below material win |
| no-match guardrails | 0.998 | neutral |

Per-class corrected results:

| kernel class | round0_best_geo | note |
|---|---:|---|
| row_norm_rms | 0.989 | active, not material |
| row_softmax | 0.988 | active, not material |
| guardrails | neutral | BMM, cross-entropy, matmul, layer norm, and attention did not show meaningful leakage |

Best active buckets:

| workload | round0_best_geo | note |
|---|---:|---|
| rms_norm_1024x16384 | 0.958 | best RMS bucket, still below material-win threshold |
| softmax_2k_4k | 0.955 | best softmax bucket, still below material-win threshold |

Replay validation passed:

| check | result |
|---|---:|
| baseline round-0 records | 80/80 |
| no-match replays | 40/40 |
| matched arms with `off_matched_heuristic` | 40/40 |
| fatal errors | 0 |

Next step: do not keep the broad RMS/softmax policy. Either run H4b narrow
diagnostics around `softmax_2k_4k` and `rms_norm_1024x16384`, or improve policy
derivation from AOT data before another non-attention gate. Attention remains
the only clean material win so far.
