# H5 Attention Hybrid LFBO Paired No-Match

- Output root: `/tmp/helion_llm_autoresearch_attention_h5_hybrid_lfbo_paired_no_match_20260507`
- Autotuner: `LLMSeededLFBOTreeSearch`
- Model: `gpt-5-2`
- Repeats: `5`
- GPU: `CUDA_VISIBLE_DEVICES=3`, `--gpu 3`
- Policy:
  `llm_heuristics_artifacts/h3b_attention_strict_family_no_r4_observed_heuristics_b200.json`
- Workloads:
  `attention_512_d64`, `attention_1k_d64`, `attention_2k_d64`,
  `attention_2k_d128`, `attention_4k_d64`, `attention_4k_d128`
- Arms: `baseline`, `heuristics`

Decision: HOLD.

The strict attention heuristic remains useful for the LLM round-0 seed batch,
but hybrid LFBO weakens the advantage and introduces a final verified
regression on `attention_512_d64`. Keep the attention policy active for H5b
diagnosis, but do not package it until the handoff preserves the round-0 win
without promoted-workload regressions.

## Round-0 Stage-1 Result

Lower is better. Ratios compare the heuristics arm against the paired
non-heuristics baseline arm.

| scope | round0_best_geo | decision |
|---|---:|---|
| overall attention | 0.830 | material win |
| matched attention | 0.799 | strong win |
| no-match `attention_4k_d128` | 1.001 | neutral guardrail |

Replay validation was clean: baseline round 0 was recorded, matched arms did
not replay the paired baseline, and no-match `attention_4k_d128` replayed the
paired baseline with equivalent candidate overlap and `same_best=true`.

## Final Verified Performance

These are the aggregate script ratios after the LFBO stage.

| scope or workload | final verified geo | note |
|---|---:|---|
| overall attention | 0.896 | material aggregate win |
| matched attention | 0.882 | material aggregate win |
| no-match `attention_4k_d128` | 0.971 | guardrail ok |
| `attention_512_d64` | 1.025 | regression; H5b blocker |
| `attention_1k_d64` | 0.879 | win |
| `attention_2k_d64` | 0.791 | strong win |
| `attention_2k_d128` | 0.907 | near material |
| `attention_4k_d64` | 0.826 | strong win |

Script aggregate diagnostics:

| arm | perf geo | time geo | cfg geo | perf range | time range | cfg range |
|---|---:|---:|---:|---|---|---|
| heuristics | 0.896 | 1.429 | 1.166 | 0.711-1.077 | 0.672-3.313 | 0.582-1.855 |

Per-workload script diagnostics:

| workload | perf geo | time geo | cfg geo |
|---|---:|---:|---:|
| `attention_512_d64` | 1.025 | 1.460 | 1.047 |
| `attention_1k_d64` | 0.879 | 2.137 | 1.268 |
| `attention_2k_d64` | 0.791 | 1.236 | 1.163 |
| `attention_2k_d128` | 0.907 | 1.184 | 1.095 |
| `attention_4k_d64` | 0.826 | 1.784 | 1.303 |
| `attention_4k_d128` | 0.971 | 1.044 | 1.144 |

## Command

```bash
cd /home/jongsokchoi/helion_2_llm_priors
CUDA_VISIBLE_DEVICES=3 \
HELION_AUTOTUNE_RANDOM_SEED=20260507 \
HELION_LLM_OBSERVED_HEURISTICS_PATH=/home/jongsokchoi/helion_2_aot_pretune_data_all/llm_heuristics_artifacts/h3b_attention_strict_family_no_r4_observed_heuristics_b200.json \
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
  --output-root /tmp/helion_llm_autoresearch_attention_h5_hybrid_lfbo_paired_no_match_20260507
```

## Next

Run H5b before packaging:

- Diagnose whether LFBO is failing to preserve the best round-0 candidates or
  whether the seed candidate family is weak for short d64.
- Focus first on `attention_512_d64`, because it is the only final verified
  promoted-workload regression.
- Keep `attention_4k_d128` as a paired no-match guardrail.
- Do not add RMS/softmax or other non-attention classes back into the hybrid
  handoff matrix until attention is clean.
