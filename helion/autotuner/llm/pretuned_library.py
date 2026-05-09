"""Pretuned configs library for bootstrapping LLMGuidedSearch.

Loads a shape-bucketed library of near-optimal configs (empirically measured
on the target GPU) and retrieves the matching template(s) for the current
kernel at autotune time. Used to enrich the initial LLM prompt with concrete
starting points.

The library file is a JSON blob with schema_version=1 and a list of rules,
each keyed by (kernel_class, shape_bucket). See the observed_heuristics_b200
artifacts under llm_heuristics_artifacts/ for example data.

Enable by setting ``HELION_LLM_PRETUNED_LIBRARY_PATH`` to the JSON path.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PretunedTemplate:
    """One empirically-winning config template for a shape bucket."""

    kernel_class: str
    shape_bucket: dict[str, str]
    template: dict[str, Any]
    geomean_slowdown: float
    p90_slowdown: float


# ──────────────────────────────────────────────────────────────────────────
# Shape classification: map (kernel_name, args) → (kernel_class, shape_bucket)
# ──────────────────────────────────────────────────────────────────────────


def _bucket_seq(seq: int) -> str:
    for cap in (2048, 4096, 8192, 16384):
        if seq <= cap:
            return f"<={cap}"
    return f">{cap}"


def _bucket_head_dim(hd: int) -> str:
    for cap in (64, 128, 256):
        if hd <= cap:
            return f"<={cap}"
    return f">{cap}"


def _bucket_batch_heads(bh: int) -> str:
    for cap in (128, 512, 2048):
        if bh <= cap:
            return f"<={cap}"
    return f">{cap}"


def _bucket_cols(cols: int) -> str:
    for cap in (1024, 4096, 8192, 16384, 32768):
        if cols <= cap:
            return f"<={cap}"
    return f">{cap}"


def _bucket_rows(rows: int) -> str:
    for cap in (1024, 4096, 16384):
        if rows <= cap:
            return f"<={cap}"
    return f">{cap}"


def _dtype_family(dtype_str: str) -> str:
    s = dtype_str.lower()
    if "fp16" in s or "bf16" in s or "float16" in s or "bfloat16" in s:
        return "fp16_bf16"
    if "fp32" in s or "float32" in s:
        return "fp32"
    if "fp8" in s:
        return "fp8"
    return s


def classify_shape(kernel_name: str, args: tuple[Any, ...]) -> tuple[str, dict[str, str]] | None:
    """Map (kernel_name, args) to (kernel_class, shape_bucket).

    Returns None if the kernel isn't in the supported classification set —
    the caller should then skip pretuned injection.
    """
    import torch

    def _first_tensor(args: tuple[Any, ...]) -> "torch.Tensor | None":
        for a in args:
            if isinstance(a, torch.Tensor):
                return a
        return None

    t0 = _first_tensor(args)
    if t0 is None:
        return None
    dtype = _dtype_family(str(t0.dtype))

    name = kernel_name.lower()

    # --- attention ---
    if "attention" in name:
        # Expect (q, k, v) with shape [B, H, S, D] or [B, S, D]
        q = t0
        if q.dim() == 4:
            bh = q.shape[0] * q.shape[1]
            seq = q.shape[2]
            hd = q.shape[3]
        elif q.dim() == 3:
            bh = q.shape[0]
            seq = q.shape[1]
            hd = q.shape[2]
        else:
            return None
        return "attention", {
            "batch_heads_bin": _bucket_batch_heads(int(bh)),
            "dtype": dtype,
            "head_dim_bin": _bucket_head_dim(int(hd)),
            "seq_bin": _bucket_seq(int(seq)),
        }

    # --- matmul (basic mm, not fp8) ---
    if "matmul" in name or name == "matmul":
        # args = (x, y): x is [M,K], y is [K,N]
        if len(args) >= 2 and isinstance(args[1], torch.Tensor):
            x, y = args[0], args[1]
            if x.dim() == 2 and y.dim() == 2:
                m, k = x.shape
                _, n = y.shape
                # Aspect bucket matches the library's semantic labels:
                #   skinny_m: M is much smaller than K,N
                #   skinny_n: N is much smaller than M,K
                #   skinny_k: K is much smaller than M,N
                #   balanced: all roughly comparable
                m_, k_, n_ = int(m), int(k), int(n)
                mx = max(m_, k_, n_); mn = min(m_, k_, n_)
                ratio = mx / max(mn, 1)
                if ratio < 4:
                    aspect = "balanced"
                elif m_ == mn:
                    aspect = "skinny_m"
                elif n_ == mn:
                    aspect = "skinny_n"
                elif k_ == mn:
                    aspect = "skinny_k"
                else:
                    aspect = "balanced"
                klass = "matmul_fp8" if dtype == "fp8" else "matmul"
                return klass, {
                    "m_bin": _bucket_cols(m_),
                    "k_bin": _bucket_cols(k_),
                    "n_bin": _bucket_cols(n_),
                    "dtype": dtype,
                    "aspect": aspect,
                }
        return None

    # --- row-wise ops: softmax, layernorm, rmsnorm, cross_entropy ---
    if name in ("softmax",) or "softmax" in name:
        klass = "row_softmax"
    elif "layernorm" in name or "layer_norm" in name:
        klass = "row_norm_layer"
    elif "rms_norm" in name or "rmsnorm" in name:
        klass = "row_norm_rms"
    elif "cross_entropy" in name:
        # cross_entropy is a row-wise reduction (batch × vocab); map to row_softmax
        klass = "row_softmax"
    else:
        return None

    x = t0
    if x.dim() >= 2:
        cols = int(x.shape[-1])
        rows = 1
        for dim_size in x.shape[:-1]:
            rows *= int(dim_size)
        return klass, {
            "cols_bin": _bucket_cols(cols),
            "rows_bin": _bucket_rows(rows),
            "dtype": dtype,
        }
    return None


# ──────────────────────────────────────────────────────────────────────────
# Library loading & matching
# ──────────────────────────────────────────────────────────────────────────


def _bucket_rank(rule_bucket: dict[str, str], query_bucket: dict[str, str]) -> int:
    """Score how well a rule bucket matches the query. Lower is better.

    Missing keys in the rule are neutral (the rule is more general).
    A key that mismatches is +100. A key that's a cap string (e.g. "<=2048")
    matches when the query's value is also <= the same or smaller cap.
    """
    score = 0
    for key, rule_val in rule_bucket.items():
        if key not in query_bucket:
            score += 100
            continue
        q_val = query_bucket[key]
        if rule_val == q_val:
            continue
        # Try to reason about "<=N" style caps numerically
        if isinstance(rule_val, str) and isinstance(q_val, str) \
                and rule_val.startswith("<=") and q_val.startswith("<="):
            try:
                rv = int(rule_val[2:])
                qv = int(q_val[2:])
                # Rule cap >= query cap → rule still covers it, small penalty
                if rv >= qv:
                    score += (rv - qv) // 1024 + 1
                else:
                    score += 50  # rule is tighter than query — mismatch
                continue
            except ValueError:
                pass
        score += 10
    return score


def load_pretuned_library(path: Path | None = None) -> list[dict[str, Any]] | None:
    """Load the pretuned library from disk. Returns None if unavailable."""
    if path is None:
        env = os.environ.get("HELION_LLM_PRETUNED_LIBRARY_PATH")
        if not env:
            return None
        path = Path(env)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(data, dict) or "rules" not in data:
        return None
    return data.get("rules", [])


def find_matching_templates(
    kernel_class: str,
    shape_bucket: dict[str, str],
    rules: list[dict[str, Any]],
    max_results: int = 3,
) -> list[PretunedTemplate]:
    """Return the best-matching templates for the given shape, sorted by
    (rule bucket match score, template geomean_slowdown).

    Returns up to `max_results` templates. Empty list if no matches.
    """
    scored: list[tuple[int, float, PretunedTemplate]] = []
    for rule in rules:
        if rule.get("kernel_class") != kernel_class:
            continue
        score = _bucket_rank(rule.get("shape_bucket", {}), shape_bucket)
        for tpl in rule.get("templates", []):
            t = PretunedTemplate(
                kernel_class=kernel_class,
                shape_bucket=rule["shape_bucket"],
                template=tpl["template"],
                geomean_slowdown=float(tpl.get("geomean_slowdown", 1.0)),
                p90_slowdown=float(tpl.get("p90_slowdown", 1.0)),
            )
            scored.append((score, t.geomean_slowdown, t))
    scored.sort(key=lambda x: (x[0], x[1]))
    return [t for _, _, t in scored[:max_results]]


def format_templates_for_prompt(templates: list[PretunedTemplate]) -> str:
    """Render templates as a prompt section listing empirical winners."""
    if not templates:
        return ""
    lines = [
        "These configs were empirically measured to be near-optimal (geomean slowdown ≤ 1.02) "
        "on the same GPU for kernels in the same class and shape bucket. "
        "Use them as strong starting points for your proposals."
    ]
    for i, t in enumerate(templates, start=1):
        bucket_desc = ", ".join(f"{k}={v}" for k, v in sorted(t.shape_bucket.items()))
        lines.append(
            f"- Template {i} ({t.kernel_class}, bucket: {bucket_desc}, "
            f"geomean_slowdown={t.geomean_slowdown:.4f}): "
            f"{json.dumps(t.template, sort_keys=True)}"
        )
    return "\n".join(lines)


def get_pretuned_hint(kernel_name: str, args: tuple[Any, ...]) -> str:
    """Top-level entry point: return prompt text or empty string.

    Controlled by ``HELION_LLM_PRETUNED_LIBRARY_PATH``. When unset or the
    library can't be loaded or no rules match, returns "" so the caller
    can use ``_join_sections`` without a special branch.
    """
    rules = load_pretuned_library()
    if not rules:
        return ""
    classification = classify_shape(kernel_name, args)
    if classification is None:
        return ""
    kernel_class, shape_bucket = classification
    matches = find_matching_templates(kernel_class, shape_bucket, rules, max_results=3)
    if not matches:
        return ""
    return format_templates_for_prompt(matches)


def get_pretuned_config_dicts(
    kernel_name: str, args: tuple[Any, ...], max_results: int = 3
) -> list[dict[str, Any]]:
    """Return the template dicts (suitable for Config(**d)) for the current shape.

    Same gating as get_pretuned_hint: empty list if env var unset, library
    missing, kernel unclassifiable, or no bucket matches. Used by LLMGuidedSearch
    to inject pretuned configs as round-0 seeds in addition to the prompt hint.
    """
    rules = load_pretuned_library()
    if not rules:
        return []
    classification = classify_shape(kernel_name, args)
    if classification is None:
        return []
    kernel_class, shape_bucket = classification
    matches = find_matching_templates(kernel_class, shape_bucket, rules, max_results=max_results)
    return [m.template for m in matches]
