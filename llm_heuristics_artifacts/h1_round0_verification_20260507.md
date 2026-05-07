# H1 Round-0 Objective Verification

Recommendation: **H1 PASS**

## Method

- Loaded only the existing artifact `summary.json` files under the three requested directories.
- For each metadata row, used JSON fields `workload`, `arm`, and `autotune_log_csv`; no workload or arm was inferred from filenames.
- For every `autotune_log_csv`, used the referenced CSV as the source, except when a sibling `_stage1_llm.csv` existed; in that case the stage-1 LLM CSV was used for the hybrid/LFBO handoff.
- Computed `round0_best_ms = min(perf_ms)` over CSV rows with `generation == 0` and `status == ok`.
- Divided each candidate arm by the same run/repeat/workload baseline arm, then computed the geomean across workloads and repeats.

## Trace

- Summary metadata rows inspected: 438
- Metadata rows resolved to CSVs: 438
- Candidate ratio rows: 300
- Hybrid/LFBO effective `_stage1_llm.csv` rows used: 96
- Trace errors: 0
- Ratio errors: 0

## Aggregate

| run | arm | n | computed | rounded | expected | match | min | max |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| guided iter11 | heuristics | 57 | 0.952684747 | 0.953 | 0.953 | True | 0.709021986 | 1.422943347 |
| guided iter11 | range_prompt | 57 | 0.959271664 | 0.959 | 0.959 | True | 0.714285714 | 1.238438211 |
| guided iter11 | seeds | 57 | 0.968517931 | 0.969 | 0.969 | True | 0.714285714 | 1.169712794 |
| guided iter12 | range_prompt | 57 | 0.974486583 | 0.974 | 0.974 | True | 0.816058168 | 1.242843602 |
| hybrid LFBO | heuristics | 24 | 0.948877958 | 0.949 | 0.949 | True | 0.806451613 | 1.400000000 |
| hybrid LFBO | range_prompt | 24 | 0.999182870 | 0.999 | 0.999 | True | 0.825436180 | 1.234613265 |
| hybrid LFBO | seeds | 24 | 0.987971432 | 0.988 | 0.988 | True | 0.806955645 | 1.400000000 |

## Mismatches

- None; all expected aggregate values match at three decimals.

## Attention Buckets

| run | workload | arm | n | geo | min | max |
|---|---|---:|---:|---:|---:|---:|
| guided iter11 | attention_1k_d64 | heuristics | 3 | 0.802353823 | 0.795228628 | 0.806549118 |
| guided iter11 | attention_1k_d64 | range_prompt | 3 | 0.994838554 | 0.986083499 | 1.000000000 |
| guided iter11 | attention_1k_d64 | seeds | 3 | 0.802520945 | 0.795725646 | 0.806549118 |
| guided iter11 | attention_2k_d128 | heuristics | 3 | 0.840767678 | 0.807607901 | 0.907894737 |
| guided iter11 | attention_2k_d128 | range_prompt | 3 | 0.840651851 | 0.819678127 | 0.881578947 |
| guided iter11 | attention_2k_d128 | seeds | 3 | 1.039375183 | 0.995427944 | 1.124074836 |
| guided iter11 | attention_4k_d128 | heuristics | 3 | 1.193654939 | 0.995264255 | 1.422943347 |
| guided iter11 | attention_4k_d128 | range_prompt | 3 | 1.072265508 | 0.995264255 | 1.238438211 |
| guided iter11 | attention_4k_d128 | seeds | 3 | 1.000388179 | 0.873388931 | 1.146082049 |
| guided iter11 | attention_4k_d64 | heuristics | 3 | 0.721664878 | 0.709021986 | 0.733364284 |
| guided iter11 | attention_4k_d64 | range_prompt | 3 | 0.984912511 | 0.955269143 | 1.000309502 |
| guided iter11 | attention_4k_d64 | seeds | 3 | 0.732027739 | 0.722365428 | 0.736923553 |
| guided iter12 | attention_1k_d64 | range_prompt | 3 | 1.015547412 | 1.000000000 | 1.032795419 |
| guided iter12 | attention_2k_d128 | range_prompt | 3 | 1.001609461 | 0.876932990 | 1.104809253 |
| guided iter12 | attention_4k_d128 | range_prompt | 3 | 1.004690147 | 0.816058168 | 1.242843602 |
| guided iter12 | attention_4k_d64 | range_prompt | 3 | 1.003270550 | 0.999845321 | 1.010000000 |
| hybrid LFBO | attention_1k_d64 | heuristics | 3 | 0.824303250 | 0.806451613 | 0.833854167 |
| hybrid LFBO | attention_1k_d64 | range_prompt | 3 | 1.037293534 | 1.014112903 | 1.065070276 |
| hybrid LFBO | attention_1k_d64 | seeds | 3 | 0.824474944 | 0.806955645 | 0.833420094 |
| hybrid LFBO | attention_2k_d128 | heuristics | 3 | 0.821277226 | 0.811019284 | 0.841874643 |
| hybrid LFBO | attention_2k_d128 | range_prompt | 3 | 0.954493502 | 0.825436180 | 1.234613265 |
| hybrid LFBO | attention_2k_d128 | seeds | 3 | 1.057300970 | 0.999449036 | 1.140547492 |

## Attention Notes

- guided iter11: `attention_1k_d64` and `attention_4k_d64` strongly favor `heuristics`/`seeds` versus baseline, while `range_prompt` is near baseline.
- guided iter11: `attention_2k_d128` improves for `heuristics` and `range_prompt`; `seeds` regresses. `attention_4k_d128` is the opposite risk bucket, with `heuristics` at 1.194 geo and `range_prompt` at 1.072 geo.
- guided iter12: `range_prompt` is near baseline across attention buckets overall, with mixed repeat-level spread on `attention_2k_d128` and `attention_4k_d128`.
- hybrid LFBO: using `_stage1_llm.csv`, `attention_1k_d64` favors `heuristics`/`seeds` but `range_prompt` regresses; `attention_2k_d128` favors `heuristics`, has mixed `range_prompt`, and `seeds` regresses.
