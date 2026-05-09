"""Data-driven knob priors for LLM autotuning prompts.

Instead of dumping raw templates in the prompt, analyze the pretuned library
for a kernel class and synthesize a compressed prior: which fields are
LOCKED (single dominant value across all good templates), FAVORED (dominant
but not universal), or VARIES (LLM should tune).

This is Approach 9 — meant to replace or augment raw template injection.
Reasoning: LLMs are better at following explicit rules than inferring
patterns from 3 examples in dense JSON.

Enabled via ``HELION_LLM_PRETUNED_LIBRARY_PATH`` (same env var as the
template injection) plus ``HELION_LLM_USE_KNOB_PRIOR=1`` to switch from
raw-template mode to knob-prior mode.
"""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from typing import Any

from .pretuned_library import classify_shape, load_pretuned_library

# Quality threshold: only templates with geomean_slowdown ≤ this count toward the prior
_MAX_TEMPLATE_SLOWDOWN = 1.10
# Threshold for considering a value "dominant" (LOCK)
_LOCK_THRESHOLD = 0.90
# Threshold for "favored" (FAVOR)
_FAVOR_THRESHOLD = 0.60


def _summarize_field(values: list[str]) -> tuple[str, str, int, int]:
    """For a sorted list of JSON-serialized values, return (status, repr, top_count, total).

    status ∈ {"LOCK", "FAVOR", "VARIES"}.
    """
    if not values:
        return "VARIES", "", 0, 0
    counter = Counter(values)
    total = sum(counter.values())
    top, top_count = counter.most_common(1)[0]
    pct = top_count / total
    if pct >= _LOCK_THRESHOLD:
        return "LOCK", top, top_count, total
    if pct >= _FAVOR_THRESHOLD:
        return "FAVOR", top, top_count, total
    return "VARIES", "", 0, total


def build_knob_prior_for_class(kernel_class: str, rules: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Synthesize the knob prior for one kernel class.

    Returns None if no rules match. Otherwise returns a dict with:
        {"field_name": {"status": "LOCK|FAVOR|VARIES", "value": ..., "top_distribution": [...]}, ...}
    """
    entries: list[dict[str, Any]] = []
    for r in rules:
        if r.get("kernel_class") != kernel_class:
            continue
        for t in r.get("templates", []):
            if t.get("geomean_slowdown", 2.0) > _MAX_TEMPLATE_SLOWDOWN:
                continue
            entries.append({
                "template": t["template"],
                "bucket": r.get("shape_bucket", {}),
            })
    if not entries:
        return None

    all_fields: set[str] = set()
    for e in entries:
        all_fields.update(e["template"].keys())

    prior: dict[str, Any] = {}
    for field in sorted(all_fields):
        values = [json.dumps(e["template"].get(field)) for e in entries if field in e["template"]]
        status, value_repr, top_count, total = _summarize_field(values)
        entry: dict[str, Any] = {"status": status, "presence": f"{len(values)}/{len(entries)}"}
        if status in ("LOCK", "FAVOR"):
            entry["value"] = json.loads(value_repr)
            entry["dominance"] = f"{top_count}/{total}"
        else:
            # VARIES — list top 4 values with counts
            counter = Counter(values)
            entry["top_values"] = [
                {"value": json.loads(v), "count": c}
                for v, c in counter.most_common(4)
            ]
        prior[field] = entry

    return {
        "kernel_class": kernel_class,
        "num_templates_analyzed": len(entries),
        "num_buckets": len({tuple(sorted(e["bucket"].items())) for e in entries}),
        "fields": prior,
    }


def format_knob_prior_for_prompt(prior: dict[str, Any]) -> str:
    """Render a knob prior dict as a compact prompt section.

    Example output:
        Based on 5 empirically-measured templates across 5 buckets for
        kernel class 'attention', the following knob pattern was found:
          - LOCK num_warps = 4  (all 5 templates)
          - FAVOR block_sizes = [1, 128, 128]  (4/5 templates)
          ...
    """
    if not prior:
        return ""
    lines = [
        f"For kernel class '{prior['kernel_class']}', analysis of "
        f"{prior['num_templates_analyzed']} empirically-measured templates "
        f"across {prior['num_buckets']} shape buckets (geomean_slowdown ≤ "
        f"{_MAX_TEMPLATE_SLOWDOWN}) shows the following knob priors:",
    ]
    # Show LOCK and FAVOR first, then VARIES
    lock_fields = []
    favor_fields = []
    vary_fields = []
    for field, info in prior["fields"].items():
        if info["status"] == "LOCK":
            lock_fields.append((field, info))
        elif info["status"] == "FAVOR":
            favor_fields.append((field, info))
        else:
            vary_fields.append((field, info))
    for field, info in lock_fields:
        lines.append(
            f"  - STRONGLY PREFER {field} = {json.dumps(info['value'])}  "
            f"(dominant in {info['dominance']} templates; use this unless you have a specific reason otherwise)"
        )
    for field, info in favor_fields:
        lines.append(
            f"  - FAVOR {field} = {json.dumps(info['value'])}  "
            f"(best in {info['dominance']} templates; reasonable default)"
        )
    for field, info in vary_fields:
        top_str = ", ".join(
            f"{json.dumps(v['value'])} (in {v['count']} templates)"
            for v in info.get("top_values", [])[:4]
        )
        lines.append(
            f"  - TUNE {field}: varies across templates. Top values: {top_str}"
        )
    lines.append(
        "When proposing configs: respect STRONGLY PREFER values; optionally explore FAVOR variations; actively tune TUNE fields."
    )
    return "\n".join(lines)


def get_knob_prior_hint(kernel_name: str, args: tuple[Any, ...]) -> str:
    """Top-level entry point: return prompt text or empty string.

    Opt-in via HELION_LLM_USE_KNOB_PRIOR=1 (requires PRETUNED_LIBRARY_PATH also set).
    """
    if os.environ.get("HELION_LLM_USE_KNOB_PRIOR", "").strip() not in ("1", "true", "yes"):
        return ""
    rules = load_pretuned_library()
    if not rules:
        return ""
    classification = classify_shape(kernel_name, args)
    if classification is None:
        return ""
    kernel_class, _ = classification
    prior = build_knob_prior_for_class(kernel_class, rules)
    if prior is None:
        return ""
    return format_knob_prior_for_prompt(prior)
