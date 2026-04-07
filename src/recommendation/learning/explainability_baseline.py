from __future__ import annotations

from typing import Any, Dict, List

from .baseline_common import round_score


def apply_explainability(
    *,
    items: List[Dict[str, Any]],
    run_counterfactuals: bool,
) -> None:
    for item in items:
        evidence_cards = {
            "feature_contribution": {
                "semantic_relevance": round_score(
                    item["score_components"]["semantic_relevance"], 6
                ),
                "intent_alignment": round_score(
                    item["score_components"]["intent_alignment"], 6
                ),
                "reference_usefulness": round_score(
                    item["score_components"]["reference_usefulness"], 6
                ),
                "support_confidence": round_score(
                    item["score_components"]["support_confidence"], 6
                ),
            },
            "neighbor_evidence": {
                "winner_count": int(
                    1 + (1 if item["score_components"]["semantic_relevance"] >= 0.7 else 0)
                ),
                "loser_count": int(
                    1 + (1 if item["score_components"]["support_confidence"] < 0.7 else 0)
                ),
                "retrieval_branches": list(item["retrieval_branches"]),
                "top_reasons": list(item["ranking_reasons"][:3]),
            },
        }
        item["evidence_cards"] = evidence_cards
        item["temporal_confidence_band"] = {
            "low": round_score(max(0.0, item["score"] - 0.08), 6),
            "mid": round_score(item["score"], 6),
            "high": round_score(min(1.0, item["score"] + 0.08), 6),
            "label": "baseline_band",
        }
        if run_counterfactuals:
            item["counterfactual_scenarios"] = [
                {
                    "scenario": name,
                    "impact_direction": direction,
                    "rationale": rationale,
                }
                for name, direction, rationale in (
                    (
                        "stronger_hook",
                        "up",
                        "Higher topical overlap tends to lift semantic retrieval coverage.",
                    ),
                    (
                        "narrower_hashtags",
                        "up",
                        "Cleaner topic tags usually improve hashtag and topic alignment.",
                    ),
                    (
                        "broader_hashtags",
                        "mixed",
                        "Broader tags may increase recall but weaken intent precision.",
                    ),
                    (
                        "cta_follow",
                        "mixed",
                        "CTA shifts can help engagement but may hurt conversion comparability.",
                    ),
                    (
                        "cta_save",
                        "up",
                        "Save-oriented framing usually helps tutorial comparables.",
                    ),
                    (
                        "locale_match",
                        "up",
                        "Locale alignment improves structured compatibility.",
                    ),
                    (
                        "locale_mismatch",
                        "down",
                        "Locale mismatch weakens final comparability.",
                    ),
                    (
                        "content_type_match",
                        "up",
                        "Closer format match improves intent alignment.",
                    ),
                    (
                        "content_type_shift",
                        "mixed",
                        "Format shifts can broaden recall but reduce execution similarity.",
                    ),
                    (
                        "audience_refine",
                        "up",
                        "Audience specificity improves structured retrieval precision.",
                    ),
                )
            ]
