import type {
  ComparableItem,
  ReportOutput,
  ReportPolarity
} from "../../src/features/report/types";

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isString(value: unknown): value is string {
  return typeof value === "string";
}

function isNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function isStringArray(value: unknown): value is string[] {
  return Array.isArray(value) && value.every((item) => typeof item === "string");
}

function isPolarity(value: unknown): value is ReportPolarity {
  return value === "Positive" || value === "Negative" || value === "Question";
}

function validateComparableItem(value: unknown): value is ComparableItem {
  if (!isObject(value)) {
    return false;
  }

  if (
    !isString(value.id) ||
    !isString(value.candidate_id) ||
    !isString(value.caption) ||
    !isString(value.author) ||
    !isString(value.video_url) ||
    !isString(value.thumbnail_url) ||
    !isStringArray(value.hashtags) ||
    !isNumber(value.similarity) ||
    !isString(value.support_level) ||
    !isString(value.confidence_label) ||
    !isStringArray(value.matched_keywords) ||
    !isStringArray(value.observations) ||
    !isString(value.why_this_was_chosen) ||
    !isStringArray(value.ranking_reasons) ||
    !isObject(value.score_components) ||
    !isStringArray(value.retrieval_branches)
  ) {
    return false;
  }

  if (!isObject(value.metrics)) {
    return false;
  }

  return (
    isNumber(value.metrics.views) &&
    isNumber(value.metrics.likes) &&
    isNumber(value.metrics.comments_count) &&
    isNumber(value.metrics.shares) &&
    isString(value.metrics.engagement_rate) &&
    isNumber(value.score_components.semantic_relevance) &&
    isNumber(value.score_components.intent_alignment) &&
    isNumber(value.score_components.performance_quality) &&
    isNumber(value.score_components.reference_usefulness) &&
    isNumber(value.score_components.support_confidence)
  );
}

export function validateReportOutput(value: unknown): value is ReportOutput {
  if (!isObject(value)) {
    return false;
  }

  const {
    meta,
    header,
    executive_summary,
    comparables,
    direct_comparison,
    relevant_comments,
    recommendations,
    reasoning,
    explainability
  } = value;

  if (!isObject(meta)) {
    return false;
  }

  if (
    !isString(meta.request_id) ||
    !isString(meta.objective) ||
    !isString(meta.objective_effective) ||
    !isString(meta.generated_at) ||
    !isString(meta.recommender_source) ||
    typeof meta.fallback_mode !== "boolean" ||
    !isString(meta.evidence_label) ||
    !isString(meta.confidence_label) ||
    (meta.fallback_reason !== undefined && meta.fallback_reason !== null && !isString(meta.fallback_reason))
  ) {
    return false;
  }

  if (!isObject(header) || !isObject(header.badges)) {
    return false;
  }

  if (
    !isString(header.title) ||
    !isString(header.subtitle) ||
    !isString(header.disclaimer) ||
    !isNumber(header.badges.candidates_k) ||
    !isString(header.badges.model) ||
    !isString(header.badges.mode)
  ) {
    return false;
  }

  if (!isObject(executive_summary) || !Array.isArray(executive_summary.metrics)) {
    return false;
  }

  if (
    !executive_summary.metrics.every((metric) => {
      if (!isObject(metric)) {
        return false;
      }
      return isString(metric.id) && isString(metric.label) && isString(metric.value);
    }) ||
    !isStringArray(executive_summary.extracted_keywords) ||
    !isStringArray(executive_summary.meaning_points) ||
    !isString(executive_summary.summary_text)
  ) {
    return false;
  }

  if (!Array.isArray(comparables) || !comparables.every(validateComparableItem)) {
    return false;
  }

  if (!isObject(direct_comparison) || !Array.isArray(direct_comparison.rows)) {
    return false;
  }

  if (
    !direct_comparison.rows.every((row) => {
      if (!isObject(row)) {
        return false;
      }
      return (
        isString(row.id) &&
        isString(row.label) &&
        isString(row.your_value_label) &&
        isString(row.comparable_value_label) &&
        isNumber(row.your_value_pct) &&
        isNumber(row.comparable_value_pct)
      );
    }) ||
    !isString(direct_comparison.note)
  ) {
    return false;
  }

  if (!isObject(relevant_comments) || !Array.isArray(relevant_comments.items)) {
    return false;
  }

  if (
    !relevant_comments.items.every((item) => {
      if (!isObject(item)) {
        return false;
      }
      return (
        isString(item.id) &&
        isString(item.text) &&
        isString(item.topic) &&
        isPolarity(item.polarity) &&
        isString(item.relevance_note)
      );
    }) ||
    !isString(relevant_comments.disclaimer)
  ) {
    return false;
  }

  if (!isObject(recommendations) || !Array.isArray(recommendations.items)) {
    return false;
  }

  const recommendationsOk = recommendations.items.every((item) => {
    if (!isObject(item)) {
      return false;
    }
    return (
      isString(item.id) &&
      isString(item.title) &&
      (item.priority === "High" || item.priority === "Medium" || item.priority === "Low") &&
      (item.effort === "Low" || item.effort === "Medium" || item.effort === "High") &&
      isString(item.evidence) &&
      isString(item.rationale) &&
      isString(item.confidence_label) &&
      isString(item.effect_area) &&
      Array.isArray(item.caveats) &&
      isStringArray(item.caveats) &&
      Array.isArray(item.evidence_refs) &&
      isStringArray(item.evidence_refs)
    );
  });
  if (!recommendationsOk) {
    return false;
  }

  if (!isObject(reasoning)) {
    return false;
  }
  if (
    !isObject(reasoning.evidence_pack) ||
    !Array.isArray(reasoning.explanation_units) ||
    !Array.isArray(reasoning.recommendation_units) ||
    !isObject(reasoning.reasoning_metadata)
  ) {
    return false;
  }

  if (explainability === undefined) {
    return true;
  }
  if (!isObject(explainability)) {
    return false;
  }
  if (
    !Array.isArray(explainability.evidence_cards) ||
    !Array.isArray(explainability.counterfactual_actions) ||
    !isString(explainability.disclaimer) ||
    !isObject(explainability.trace_metadata)
  ) {
    return false;
  }
  const cardsOk = explainability.evidence_cards.every((card) => {
    if (!isObject(card)) {
      return false;
    }
    return (
      isString(card.candidate_id) &&
      isNumber(card.rank) &&
      isObject(card.feature_contributions) &&
      isObject(card.neighbor_evidence) &&
      isObject(card.temporal_confidence_band)
    );
  });
  if (!cardsOk) {
    return false;
  }
  return explainability.counterfactual_actions.every((action) => {
    if (!isObject(action) || !Array.isArray(action.scenarios)) {
      return false;
    }
    if (!isString(action.candidate_id) || !isNumber(action.rank)) {
      return false;
    }
    return action.scenarios.every((scenario) => {
      if (!isObject(scenario)) {
        return false;
      }
      return (
        isString(scenario.scenario_id) &&
        isObject(scenario.expected_rank_delta_band) &&
        isString(scenario.feasibility) &&
        (scenario.reason === undefined || isString(scenario.reason)) &&
        (scenario.trace === undefined || isObject(scenario.trace))
      );
    });
  });
}
