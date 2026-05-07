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
