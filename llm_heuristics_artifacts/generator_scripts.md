# LLM Heuristics Generator Scripts

This branch archives the scripts used to derive and validate the LLM
heuristics artifacts:

- `scripts/llm_heuristics_research.py`: offline generator for
  `runtime_observed_heuristics_b200.json` and related Markdown/JSON reports
  from `aot_pretune_data/b200` measurement CSVs.
- `scripts/llm_heuristics_experiment.py`: single-suite harness for comparing
  baseline and heuristic LLM autotuning arms.
- `scripts/llm_heuristics_autoresearch.py`: repeated-run wrapper and aggregate
  reporter used for the H2-H7 experiments.

The generator entry point is:

```bash
python scripts/llm_heuristics_research.py \
  --data-root aot_pretune_data/b200 \
  --output-dir /tmp/helion_llm_heuristics_research
```

To write a runtime JSON directly:

```bash
python scripts/llm_heuristics_research.py \
  --data-root aot_pretune_data/b200 \
  --output-dir /tmp/helion_llm_heuristics_research \
  --runtime-heuristics-path llm_heuristics_artifacts/runtime_observed_heuristics_b200.json
```

The H5/H6/H7 hand-edited candidate JSONs in `llm_heuristics_artifacts/` were
derived from this generated schema and then validated with the experiment
harness.
