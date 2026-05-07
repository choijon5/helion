# Claude H2 Policy Critique

Source: Claude Opus 4.7 max-effort no-tools critique, 2026-05-07.

Recommendation: H2 PASS with caveat.

Primary reason: the H2 attention router produced a 20% corrected round-0
geomean win over baseline across 3 repeats, with `cfg_geo` about 1.0 and script
aggregate `time_geo=0.802`.

Caveat: `attention_4k_d128` improved while `observed_rule_match=false`, so H3
should split `round0_geo` by match/no-match and run attribution plus holdouts
before broadening.

Next direction: stay on attention for H3 coverage and attribution before
broadening to RMS, softmax, and BMM.
