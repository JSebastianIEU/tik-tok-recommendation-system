from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Tuple

from src.common.schemas import compute_engagement_rate, compute_engagement_total


_NUMERIC_METRICS = ("views", "likes", "comments_count", "shares")
_WORD_RE = re.compile(r"[A-Za-z0-9']+")


def _safe_int(x: Any) -> int | None:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def _safe_str(x: Any) -> str:
    return "" if x is None else str(x)


def _quantile(sorted_vals: List[float], q: float) -> float | None:
    """Linear interpolation quantile, q in [0,1]. Requires sorted_vals."""
    n = len(sorted_vals)
    if n == 0:
        return None
    if n == 1:
        return float(sorted_vals[0])
    pos = (n - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_vals[lo])
    frac = pos - lo
    return float(sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac)


def _describe(values: List[int]) -> Dict[str, float | int | None]:
    vals = [v for v in values if v is not None]
    if not vals:
        return {
            "count": 0,
            "min": None,
            "p10": None,
            "median": None,
            "mean": None,
            "p90": None,
            "max": None,
        }
    vals_sorted = sorted(vals)
    return {
        "count": len(vals_sorted),
        "min": int(vals_sorted[0]),
        "p10": _quantile(vals_sorted, 0.10),
        "median": float(median(vals_sorted)),
        "mean": float(mean(vals_sorted)),
        "p90": _quantile(vals_sorted, 0.90),
        "max": int(vals_sorted[-1]),
    }


def _pearson(x: List[float], y: List[float]) -> float | None:
    """Pearson correlation for paired lists; returns None if undefined."""
    if len(x) != len(y):
        raise ValueError("x and y must have same length")
    n = len(x)
    if n < 2:
        return None
    mx = mean(x)
    my = mean(y)
    num = 0.0
    dx2 = 0.0
    dy2 = 0.0
    for xi, yi in zip(x, y):
        dx = xi - mx
        dy = yi - my
        num += dx * dy
        dx2 += dx * dx
        dy2 += dy * dy
    den = math.sqrt(dx2 * dy2)
    if den == 0:
        return None
    return num / den


def _tokenize_caption(caption: str) -> List[str]:
    # Basic, lightweight baseline: lowercase word tokens
    return [m.group(0).lower() for m in _WORD_RE.finditer(caption or "")]


def _normalize_hashtag(tag: str) -> str:
    # input already like "#fitness", keep leading # and lowercase
    t = _safe_str(tag).strip()
    if not t:
        return ""
    if not t.startswith("#"):
        t = "#" + t
    return t.lower()


def _engagement(post: Dict[str, Any]) -> Dict[str, float]:
    likes_i = _safe_int(post.get("likes")) or 0
    comments_i = _safe_int(post.get("comments_count")) or 0
    shares_i = _safe_int(post.get("shares")) or 0
    views_i = _safe_int(post.get("views")) or 0

    total = float(
        compute_engagement_total(
            likes=likes_i, comments_count=comments_i, shares=shares_i
        )
    )
    rate = float(
        compute_engagement_rate(
            likes=likes_i,
            comments_count=comments_i,
            shares=shares_i,
            views=views_i,
        )
    )
    return {
        "likes": float(likes_i),
        "comments_count": float(comments_i),
        "shares": float(shares_i),
        "views": float(views_i),
        "engagement_total": total,
        "engagement_rate": rate,
    }


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    yield obj
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no}: {e}") from e


def compute_stats(jsonl_path: Path) -> Dict[str, Any]:
    posts: List[Dict[str, Any]] = list(_iter_jsonl(jsonl_path))

    # Collect numeric distributions
    metric_values: Dict[str, List[int]] = {m: [] for m in _NUMERIC_METRICS}
    engagement_totals: List[float] = []
    engagement_rates: List[float] = []

    caption_lengths: List[float] = []
    hashtag_counts: List[float] = []

    # For correlations, keep aligned series per post
    series: Dict[str, List[float]] = defaultdict(list)

    # Top hashtags/keywords by engagement
    hashtag_eng: Dict[str, float] = defaultdict(float)
    keyword_eng: Dict[str, float] = defaultdict(float)

    # Keyword frequency baseline (caption tokens)
    caption_token_counts: Counter[str] = Counter()
    caption_token_eng: Dict[str, float] = defaultdict(float)

    # Also count provided "keywords" field (phrases)
    provided_keyword_counts: Counter[str] = Counter()
    provided_keyword_eng: Dict[str, float] = defaultdict(float)

    for p in posts:
        # Numeric metrics
        for m in _NUMERIC_METRICS:
            metric_values[m].append(_safe_int(p.get(m)) or 0)

        eng = _engagement(p)
        engagement_totals.append(eng["engagement_total"])
        engagement_rates.append(eng["engagement_rate"])

        # Aligned series for corr
        for m in _NUMERIC_METRICS:
            series[m].append(float(_safe_int(p.get(m)) or 0))
        series["engagement_total"].append(float(eng["engagement_total"]))
        series["engagement_rate"].append(float(eng["engagement_rate"]))

        caption = _safe_str(p.get("caption"))
        cap_len = float(len(caption))
        caption_lengths.append(cap_len)
        series["caption_length"].append(cap_len)

        hashtags = p.get("hashtags") or []
        if not isinstance(hashtags, list):
            hashtags = []
        hcount = float(len([h for h in hashtags if _normalize_hashtag(h)]))
        hashtag_counts.append(hcount)
        series["hashtag_count"].append(hcount)

        # Engagement weight for ranking content features
        weight = float(eng["engagement_total"])

        # Hashtags by engagement (sum engagement)
        for h in hashtags:
            hh = _normalize_hashtag(h)
            if hh:
                hashtag_eng[hh] += weight

        # Provided keywords field (phrases)
        kws = p.get("keywords") or []
        if not isinstance(kws, list):
            kws = []
        for kw in kws:
            k = _safe_str(kw).strip().lower()
            if not k:
                continue
            provided_keyword_counts[k] += 1
            provided_keyword_eng[k] += weight
            keyword_eng[k] += weight  # treat same bucket for "top keywords by engagement"

        # Caption token frequency
        toks = _tokenize_caption(caption)
        for t in toks:
            caption_token_counts[t] += 1
            caption_token_eng[t] += weight

    # Descriptive stats
    dist = {m: _describe(vals) for m, vals in metric_values.items()}
    dist["engagement_total"] = _describe([int(x) for x in engagement_totals])
    # engagement_rate is float; do custom describe
    er_sorted = sorted(engagement_rates)
    dist["engagement_rate"] = {
        "count": len(er_sorted),
        "min": er_sorted[0] if er_sorted else None,
        "p10": _quantile(er_sorted, 0.10) if er_sorted else None,
        "median": _quantile(er_sorted, 0.50) if er_sorted else None,
        "mean": float(mean(er_sorted)) if er_sorted else None,
        "p90": _quantile(er_sorted, 0.90) if er_sorted else None,
        "max": er_sorted[-1] if er_sorted else None,
    }

    # Correlations between metrics
    corr_pairs = [
        ("views", "likes"),
        ("views", "comments_count"),
        ("views", "shares"),
        ("likes", "comments_count"),
        ("likes", "shares"),
        ("comments_count", "shares"),
        ("caption_length", "engagement_total"),
        ("caption_length", "engagement_rate"),
        ("hashtag_count", "engagement_total"),
        ("hashtag_count", "engagement_rate"),
    ]
    corrs: Dict[str, float | None] = {}
    for a, b in corr_pairs:
        corrs[f"{a}__{b}"] = _pearson(series[a], series[b])

    # Top hashtags/keywords by engagement
    top_hashtags = sorted(hashtag_eng.items(), key=lambda kv: kv[1], reverse=True)[:20]
    top_keywords = sorted(keyword_eng.items(), key=lambda kv: kv[1], reverse=True)[:20]

    # Keyword frequency (caption tokens), exclude super-short/noisy tokens
    def keep_token(t: str) -> bool:
        return len(t) >= 3 and not t.isdigit()

    top_caption_tokens = [
        (t, c, caption_token_eng[t])
        for t, c in caption_token_counts.most_common()
        if keep_token(t)
    ][:30]

    top_provided_keywords = sorted(provided_keyword_counts.items(), key=lambda kv: kv[1], reverse=True)[:30]

    return {
        "n_posts": len(posts),
        "distributions": dist,
        "correlations": corrs,
        "top_hashtags_by_engagement": top_hashtags,
        "top_keywords_by_engagement": top_keywords,
        "caption_token_frequency": top_caption_tokens,
        "provided_keyword_frequency": top_provided_keywords,
        "notes": {
            "engagement_total_definition": "likes + comments_count + shares",
            "engagement_rate_definition": "(likes + comments_count + shares) / max(views, 1)",
            "engagement_schema_source": "src.common.schemas.compute_engagement_total / compute_engagement_rate",
            "text_baseline": "caption_length and hashtag_count correlations; simple caption token frequency",
        },
    }


def _fmt_num(x: Any, digits: int = 3) -> str:
    if x is None:
        return "NA"
    if isinstance(x, (int,)) and not isinstance(x, bool):
        return f"{x}"
    try:
        xf = float(x)
        # For rates, keep scientific-ish precision small; for large ints, no decimals
        if abs(xf) >= 1000:
            return f"{xf:,.0f}"
        return f"{xf:.{digits}f}"
    except Exception:
        return str(x)


def _md_table(headers: List[str], rows: List[List[str]]) -> str:
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def write_report(stats: Dict[str, Any], report_path: Path) -> None:
    n = stats.get("n_posts", 0)
    d = stats.get("distributions", {})
    corrs = stats.get("correlations", {})

    # Dist table
    dist_rows = []
    for metric in ("views", "likes", "comments_count", "shares", "engagement_total", "engagement_rate"):
        desc = d.get(metric, {})
        dist_rows.append(
            [
                metric,
                _fmt_num(desc.get("count"), 0),
                _fmt_num(desc.get("min")),
                _fmt_num(desc.get("p10")),
                _fmt_num(desc.get("median")),
                _fmt_num(desc.get("mean")),
                _fmt_num(desc.get("p90")),
                _fmt_num(desc.get("max")),
            ]
        )

    corr_rows = []
    # Keep a readable ordering
    for key in [
        "views__likes",
        "views__comments_count",
        "views__shares",
        "likes__comments_count",
        "likes__shares",
        "comments_count__shares",
        "caption_length__engagement_total",
        "caption_length__engagement_rate",
        "hashtag_count__engagement_total",
        "hashtag_count__engagement_rate",
    ]:
        corr_rows.append([key.replace("__", " ↔ "), _fmt_num(corrs.get(key))])

    top_hashtags = stats.get("top_hashtags_by_engagement", [])[:10]
    top_keywords = stats.get("top_keywords_by_engagement", [])[:10]
    top_tokens = stats.get("caption_token_frequency", [])[:12]
    top_kw_freq = stats.get("provided_keyword_frequency", [])[:12]

    hashtags_rows = [[h, _fmt_num(v)] for h, v in top_hashtags]
    keywords_rows = [[k, _fmt_num(v)] for k, v in top_keywords]
    token_rows = [[t, str(c), _fmt_num(eng)] for (t, c, eng) in top_tokens]
    kwfreq_rows = [[k, str(c)] for (k, c) in top_kw_freq]

    # Lightweight implication text based on signs/magnitudes (don’t overclaim)
    def interpret_corr(name: str, val: float | None) -> str:
        if val is None:
            return f"- {name}: NA (not enough variance)"
        a = abs(val)
        if a >= 0.7:
            strength = "strong"
        elif a >= 0.4:
            strength = "moderate"
        elif a >= 0.2:
            strength = "weak"
        else:
            strength = "very weak"
        direction = "positive" if val > 0 else "negative"
        return f"- {name}: {strength} {direction} (r={val:.3f})"

    insights = [
        interpret_corr("Views ↔ Likes", corrs.get("views__likes")),
        interpret_corr("Views ↔ Shares", corrs.get("views__shares")),
        interpret_corr("Caption length ↔ Engagement rate", corrs.get("caption_length__engagement_rate")),
        interpret_corr("Hashtag count ↔ Engagement rate", corrs.get("hashtag_count__engagement_rate")),
    ]

    md = []
    md.append("# Baseline Analytics Report (Sprint 1)")
    md.append("")
    md.append(f"Generated from mocked TikTok JSONL. Posts analyzed: **{n}**.")
    md.append("")
    md.append(
        "**Engagement definitions** match the mock JSONL schema in `src/common/schemas.py` "
        "(`compute_engagement_total`, `compute_engagement_rate`; same as `TikTokPost` computed fields)."
    )
    md.append("")
    md.append("## What we computed")
    md.append("- Descriptive distributions for likes/views/comments/shares and derived engagement.")
    md.append("- Pearson correlations across core engagement metrics.")
    md.append("- Top hashtags and keyword phrases ranked by total engagement (likes+comments+shares).")
    md.append("- Lightweight text baselines: caption length vs engagement, hashtag count vs engagement, basic keyword/token frequency.")
    md.append("")
    md.append("## Descriptive stats")
    md.append(_md_table(
        ["metric", "count", "min", "p10", "median", "mean", "p90", "max"],
        dist_rows
    ))
    md.append("")
    md.append("## Correlations (Pearson r)")
    md.append(_md_table(["pair", "r"], corr_rows))
    md.append("")
    md.append("### Quick read")
    md.append("\n".join(insights))
    md.append("")
    md.append("## Top hashtags by engagement")
    md.append(_md_table(["hashtag", "total_engagement"], hashtags_rows))
    md.append("")
    md.append("## Top provided keyword phrases by engagement")
    md.append(_md_table(["keyword_phrase", "total_engagement"], keywords_rows))
    md.append("")
    md.append("## Caption token frequency (simple baseline)")
    md.append(_md_table(["token", "count", "total_engagement"], token_rows))
    md.append("")
    md.append("## Provided keyword frequency (from `keywords` field)")
    md.append(_md_table(["keyword_phrase", "count"], kwfreq_rows))
    md.append("")
    md.append("## Implications for recommendation logic (baseline)")
    md.append(
        "- **Engagement-rate is more comparable than raw likes** (raw counts scale with views). "
        "Use engagement_rate or a normalized score as a primary label/sanity check.\n"
        "- **Hashtags/keywords are usable cold-start features**: even a simple ranking by engagement highlights topical clusters.\n"
        "- **Text features are weak but cheap**: caption length and hashtag count correlations can flag spammy extremes, "
        "but shouldn’t dominate ranking without stronger modeling.\n"
        "- **Next step after baseline**: compute per-topic averages (by hashtag/keyword) and test a simple content-based recommender "
        "(e.g., recommend posts with overlapping hashtags/keywords weighted by historical engagement_rate)."
    )
    md.append("")
    md.append("---")
    md.append("*Note: This report is autogenerated by `scripts/run_baseline.py` via `src/baseline/baseline_stats.py`.*")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(md) + "\n", encoding="utf-8")
