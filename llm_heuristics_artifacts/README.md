# LLM Heuristics Artifacts

This directory archives derived LLM heuristic artifacts on the
`choijon5/aot-pretune-data` fork/data branch so the branch is self-contained.
The live experiment workspace is `/home/jongsokchoi/helion_2_llm_priors`.

Provenance:

- `observed_heuristics_b200.json`: copied from
  `/home/jongsokchoi/helion_2_llm_priors/helion/autotuner/llm/data/observed_heuristics_b200.json`.
- `runtime_observed_heuristics_b200.json`: copied from
  `/tmp/helion_heuristics_loop/codex/range_policy_data_snapshot/runtime_observed_heuristics_b200.json`.
- `h2_attention_router_observed_heuristics_b200.json`: derived from
  `observed_heuristics_b200.json` by keeping the observed attention buckets for
  `attention_1k_d64`, `attention_2k_d128`, and `attention_4k_d64`, while
  omitting the known bad `attention_4k_d128` bucket for env-var testing with
  `HELION_LLM_OBSERVED_HEURISTICS_PATH`.
- `h3b_attention_empty_control_observed_heuristics_b200.json` and
  `h3b_attention_strict_family_no_r4_observed_heuristics_b200.json`: H3b
  policy artifacts using the H2 observed-heuristics schema. H3 found matched
  geo `0.810`, heldout matched geo `0.819`, and no-match
  `attention_4k_d128` geo `1.147`, so broadening is HOLD. Benchmark the empty
  control first, then the strict no-R4 attention family; do not test R4 until
  no-match neutrality is established.
- `h3d_attention_paired_no_match_20260507/aggregate_summary.md` and
  `h3d_attention_paired_no_match_20260507/aggregate_results.json`: archived
  H3d paired-no-match result. H3d passed with corrected overall
  `round0_best_geo=0.822`, matched geo `0.791`, no-match
  `attention_4k_d128` geo `1.001`, no-match median `0.9996`, and max repeat
  `1.0048`; paired-no-match mode should be reused for future guardrail
  experiments.
- `h4_non_attention_observed_heuristics_b200.json`: H4 candidate artifact using
  the observed-heuristics schema with only `row_norm_rms` and narrow/mid
  `row_softmax` active. It was benchmarked in paired-no-match mode with
  RMS/softmax active and BMM, cross-entropy, matmul, layer norm, and attention
  as guardrails.
- `h4_non_attention_paired_no_match_20260507/aggregate_summary.md` and
  `h4_non_attention_paired_no_match_20260507/aggregate_results.json`: archived
  H4 broad non-attention result. H4 is HOLD: overall `round0_best_geo=0.993`,
  matched active non-attention `0.989`, no-match guardrails `0.998`,
  `row_norm_rms=0.989`, and `row_softmax=0.988`. Best active buckets were
  `rms_norm_1024x16384=0.958` and `softmax_2k_4k=0.955`, still below the
  material-win threshold. Replay validation passed with baseline records
  80/80, no-match replays 40/40, matched arms using `off_matched_heuristic`
  40/40, and no fatal errors.
- `h5_attention_hybrid_lfbo_paired_no_match_20260507/aggregate_summary.md` and
  `h5_attention_hybrid_lfbo_paired_no_match_20260507/aggregate_results.json`:
  archived H5 attention hybrid/LFBO handoff result. H5 is HOLD: round-0
  stage-1 overall `0.830`, matched attention `0.799`, and no-match
  `attention_4k_d128` `1.001`; final verified overall `0.896`, matched
  attention `0.882`, no-match `attention_4k_d128` `0.971`, but
  `attention_512_d64` regressed to `1.025`. H5b should preserve round-0
  winners through LFBO and diagnose the short d64 regression before packaging.
- `h5b_attention_strict_family_l2_1_8_16_observed_heuristics_b200.json`:
  H5b candidate derived from H3b by adding only `l2_groupings=[1]` to the d64
  `seq<=2048` rule.
- `h5b_attention_hybrid_lfbo_l2_1_8_16_paired_no_match_20260507/aggregate_summary.md`
  and
  `h5b_attention_hybrid_lfbo_l2_1_8_16_paired_no_match_20260507/aggregate_results.json`:
  archived H5b result. H5b improved the corrected seed objective
  (`round0` overall `0.815`, matched `0.782`, no-match `1.001`) and fixed
  `attention_512_d64` final verified performance (`0.948` vs H5 `1.025`), but
  final hybrid/LFBO overall was weaker than H5 (`0.928` vs `0.896`). H5b is
  HOLD for packaging; next test should split `seq<=1024` from `seq<=2048`.
- `h1_round0_verification_20260507.md`: copied from
  `/tmp/helion_h1_round0_verification_20260507.md`.
- `h2_attention_router_20260507/aggregate_summary.md` and
  `h2_attention_router_20260507/aggregate_results.json`: copied from
  `/tmp/helion_llm_autoresearch_attention_router_h2_20260507/`.
- `claude_h2_policy_critique.md`: records the Claude Opus 4.7 no-tools H2
  critique supplied with the 2026-05-07 manager update.
- `policies/range_policy_iteration_12.*` and
  `policies/range_policy_iteration_13.*`: copied from
  `/tmp/helion_heuristics_loop/claude/`.
- `corrected_round0_summaries/*/round0_summary.md`: copied from
  `/tmp/helion_round0_objective_20260505_230436/*/round0_summary.md`.
- `shared_context.md`: copied from
  `/tmp/helion_heuristics_loop/input/shared_context.md`.

GPU 3 is the future-run GPU. Any GPU 2 references inside copied snapshots are
historical context, not instructions for new benchmark runs. No benchmarks or
tests were run to create this archive.
