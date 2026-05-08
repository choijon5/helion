# Row-Reduction Heuristic Hill-Climbing

This directory tracks hill-climbing experiments that try to improve LLM-seeded
autotuning (`LLMSeededSearch` / `LLMSeededLFBOTreeSearch`) on the
row-reduction kernel family: `layer_norm`, `rms_norm`, and `softmax`. The
directory is still named `norms/` for brevity; softmax is bundled because it
shares the same row-reduction + per-element scaling structure.

The loop borrows the gate-by-gate structure from `../llm_heuristics_plan.md`
but differs in three ways:

- **Target kernels** are norms + softmax, not attention.
- **LLM autotuner model** is `claude-opus-4-7` with maximum reasoning effort
  (ambient Helion LLMSeededSearch default is `gpt-5-2`; we override via
  `HELION_LLM_MODEL` and, once plumbed, an extended-thinking flag).
- **Orchestration model** is Claude subagents only (no Codex), spawned via
  the `Agent` tool.

Contents:

- `plan.md`: gate-by-gate plan with the `round0_best_geo ≤ 0.80` terminal
  goal.
- `manager.md`: workflow for managing Claude subagents in this loop.
- `N0_baseline.json`: offline AOT heuristic reproduction (input data to the
  heuristic, not the live score).
- `iterations/N<n>_<slug>/`: per-iteration candidate policy, harness command,
  CSVs, and score report.
