from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

from .baseline_common import (
    CTA_TERMS,
    HASHTAG_BRANCH_TOP_K,
    SEMANTIC_BRANCH_TOP_K,
    SHORTLIST_TOP_K,
    STRUCTURED_BRANCH_TOP_K,
    clamp,
    jaccard,
    normalize_text,
    round_score,
    sanitize_probability,
    tokenize,
    uniq,
)


def objective_compatibility(query_objective: str, candidate_objective: str) -> float:
    if query_objective == candidate_objective:
        return 1.0
    if (
        (query_objective == "reach" and candidate_objective == "engagement")
        or (query_objective == "engagement" and candidate_objective == "reach")
    ):
        return 0.65
    if (
        (query_objective == "community" and candidate_objective == "engagement")
        or (query_objective == "engagement" and candidate_objective == "community")
    ):
        return 0.78
    if (
        (query_objective == "conversion" and candidate_objective == "engagement")
        or (query_objective == "engagement" and candidate_objective == "conversion")
    ):
        return 0.55
    return 0.35


def content_type_compatibility(query_type: str, candidate_type: str) -> float:
    if query_type == candidate_type:
        return 1.0
    if (
        (query_type == "tutorial" and candidate_type == "educational")
        or (query_type == "educational" and candidate_type == "tutorial")
    ):
        return 0.85
    if (
        (query_type == "story" and candidate_type == "behind_the_scenes")
        or (query_type == "commentary" and candidate_type == "opinion")
    ):
        return 0.70
    return 0.25


def cta_alignment(primary_cta: str, text: str) -> float:
    terms = CTA_TERMS.get(primary_cta, ())
    if not terms:
        return 0.60
    normalized = normalize_text(text)
    return 1.0 if any(normalize_text(term) in normalized for term in terms) else 0.25


def audience_compatibility(audience: Dict[str, Any], candidate_tokens: Sequence[str]) -> float:
    tokens = uniq(
        list(audience.get("segments") or [])
        + tokenize([audience.get("normalized_label") or ""])
    )
    if not tokens:
        return 0.55
    return round_score(clamp(jaccard(tokens, candidate_tokens), 0.0, 1.0), 6)


def locale_compatibility(
    query_locale: str | None,
    query_language: str | None,
    candidate_locale: str | None,
    candidate_language: str | None,
) -> float:
    if query_locale and candidate_locale and query_locale == candidate_locale:
        return 1.0
    if query_language and candidate_language and query_language == candidate_language:
        return 0.75
    if not candidate_locale and not candidate_language:
        return 0.50
    return 0.20


def retrieval_blend_score(
    scores: Dict[str, float], branch_count: int, support_score: float = 0.0
) -> float:
    agreement = 0.08 if branch_count >= 3 else 0.04 if branch_count == 2 else 0.0
    return round_score(
        clamp(
            (scores["semantic"] * 0.45)
            + (scores["hashtag_topic"] * 0.30)
            + (scores["structured_compatibility"] * 0.25)
            + agreement,
            0.0,
            1.0,
        ),
        6,
    )


def hashtag_topic_score(query_profile: Dict[str, Any], candidate: Dict[str, Any]) -> float:
    return round_score(
        clamp(
            (jaccard(query_profile["hashtags"], candidate["hashtags"]) * 0.65)
            + (0.35 if candidate["topic_key"] == query_profile["topic_key"] else 0.0),
            0.0,
            1.0,
        ),
        6,
    )


def structured_compatibility_score(
    query_profile: Dict[str, Any], candidate: Dict[str, Any]
) -> float:
    return round_score(
        clamp(
            (
                objective_compatibility(
                    query_profile["objective"], candidate["objective_guess"]
                )
                * 0.25
            )
            + (
                content_type_compatibility(
                    query_profile["content_type"], candidate["content_type"]
                )
                * 0.30
            )
            + (cta_alignment(query_profile["primary_cta"], candidate["text"]) * 0.20)
            + (
                audience_compatibility(query_profile["audience"], candidate["audience_tokens"])
                * 0.15
            )
            + (
                locale_compatibility(
                    query_profile["locale"],
                    query_profile["language"],
                    candidate["locale"],
                    candidate["language"],
                )
                * 0.10
            ),
            0.0,
            1.0,
        ),
        6,
    )


def retrieve_shortlist(
    *,
    usable_candidates: Sequence[Dict[str, Any]],
    query_profile: Dict[str, Any],
    retrieve_k: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    semantic_branch = sorted(
        (
            {
                "candidate": item,
                "score": round_score(
                    clamp(
                        (jaccard(query_profile["tokens"], item["tokens"]) * 0.70)
                        + (jaccard(tokenize([query_profile["topic_key"]]), item["tokens"]) * 0.30),
                        0.0,
                        1.0,
                    ),
                    6,
                ),
            }
            for item in usable_candidates
        ),
        key=lambda entry: float(entry["score"]),
        reverse=True,
    )[: min(SEMANTIC_BRANCH_TOP_K, max(1, int(retrieve_k)), len(usable_candidates))]

    hashtag_branch = sorted(
        (
            {
                "candidate": item,
                "score": hashtag_topic_score(query_profile, item),
            }
            for item in usable_candidates
        ),
        key=lambda entry: float(entry["score"]),
        reverse=True,
    )[: min(HASHTAG_BRANCH_TOP_K, max(1, int(retrieve_k)), len(usable_candidates))]

    structured_branch = sorted(
        (
            {
                "candidate": item,
                "score": structured_compatibility_score(query_profile, item),
            }
            for item in usable_candidates
        ),
        key=lambda entry: float(entry["score"]),
        reverse=True,
    )[: min(STRUCTURED_BRANCH_TOP_K, max(1, int(retrieve_k)), len(usable_candidates))]

    merged: Dict[str, Dict[str, Any]] = {}

    def register_branch(branch_name: str, rows: Sequence[Dict[str, Any]]) -> None:
        for row in rows:
            candidate = row["candidate"]
            existing = merged.get(candidate["candidate_id"])
            if existing is None:
                existing = candidate
                merged[candidate["candidate_id"]] = existing
            existing["retrieval_branch_scores"][branch_name] = float(row["score"])
            if branch_name not in existing["retrieval_branches"]:
                existing["retrieval_branches"].append(branch_name)

    register_branch("semantic", semantic_branch)
    register_branch("hashtag_topic", hashtag_branch)
    register_branch("structured_compatibility", structured_branch)

    shortlist = sorted(
        (
            {
                **item,
                "retrieval_branch_scores": {
                    **item["retrieval_branch_scores"],
                    "fused_retrieval": retrieval_blend_score(
                        item["retrieval_branch_scores"],
                        len(item["retrieval_branches"]),
                        item["support_score"],
                    ),
                },
            }
            for item in merged.values()
        ),
        key=lambda item: float(item["retrieval_branch_scores"]["fused_retrieval"]),
        reverse=True,
    )[: min(SHORTLIST_TOP_K, max(1, int(retrieve_k)), len(merged))]

    return shortlist, {
        "semantic": len(semantic_branch),
        "hashtag_topic": len(hashtag_branch),
        "structured_compatibility": len(structured_branch),
        "merged": len(merged),
        "shortlist": len(shortlist),
    }
