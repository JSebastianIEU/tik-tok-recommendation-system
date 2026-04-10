import type {
  ComparableItem,
  ExplanationUnit,
  RecommendationUnit,
  ReportConfidenceLabel,
  ReportEvidenceLabel,
  ReportMeta,
  ReportReasoning,
  SupportLevel
} from "../../src/features/report/types";
import type { DemoVideoRecord } from "../../src/services/data/types";
import type { CandidateSignalProfile } from "../fallback/signalProfile";
import type { RecommenderItem, RecommenderResponsePayload } from "../recommender/client";

interface AudienceSummaryInput {
  label?: string;
  segments?: string[];
  expertise_level?: string;
}

export interface ReportReasoningContext {
  requestId: string;
  objective: string;
  objectiveEffective: string;
  generatedAt: string;
  recommenderSource: ReportMeta["recommender_source"];
  fallbackMode: boolean;
  fallbackReason?: string | null;
  experimentId?: string | null;
  variant?: "control" | "treatment" | null;
  description: string;
  hashtags: string[];
  mentions: string[];
  contentType?: string;
  primaryCta?: string;
  audience?: AudienceSummaryInput;
  locale?: string;
  language?: string;
  candidateSignals?: CandidateSignalProfile;
  candidates: DemoVideoRecord[];
  recommenderPayload: RecommenderResponsePayload;
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function round(value: number, decimals = 4): number {
  const factor = 10 ** decimals;
  return Math.round(value * factor) / factor;
}

function unique<T>(values: T[]): T[] {
  return Array.from(new Set(values));
}

function supportLevelOf(item: RecommenderItem | undefined): SupportLevel {
  const raw = String(item?.support_level ?? "").toLowerCase();
  if (raw === "full" || raw === "partial" || raw === "low") {
    return raw;
  }
  return "unknown";
}

function countBy<T>(values: T[]): Map<T, number> {
  const out = new Map<T, number>();
  for (const value of values) {
    out.set(value, (out.get(value) ?? 0) + 1);
  }
  return out;
}

function topPairs(map: Map<string, number>, key: string): Array<Record<string, string | number>> {
  return [...map.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, 6)
    .map(([name, supportCount]) => ({ [key]: name, support_count: supportCount }));
}

function classifyEvidenceLabel(params: {
  fallbackMode: boolean;
  sufficient: boolean;
  confidence: number;
}): ReportEvidenceLabel {
  if (params.fallbackMode || !params.sufficient || params.confidence < 0.45) {
    return "Limited evidence";
  }
  if (params.confidence < 0.75) {
    return "Moderate evidence";
  }
  return "Strong evidence";
}

function classifyConfidenceLabel(confidence: number): ReportConfidenceLabel {
  if (confidence < 0.45) {
    return "Low confidence";
  }
  if (confidence < 0.75) {
    return "Medium confidence";
  }
  return "High confidence";
}

function summarizeBranchMix(items: RecommenderItem[]): Record<string, number> {
  const out: Record<string, number> = {};
  for (const item of items) {
    const scores = item.retrieval_branch_scores ?? {};
    for (const [branch, score] of Object.entries(scores)) {
      if (typeof score !== "number" || !Number.isFinite(score) || score <= 0) {
        continue;
      }
      out[branch] = (out[branch] ?? 0) + 1;
    }
  }
  return out;
}

function buildTopVsRestSignals(items: RecommenderItem[]): ReportReasoning["evidence_pack"]["contrast_signals"]["top_vs_rest"] {
  if (items.length <= 1) {
    return [];
  }
  const focus = items.slice(0, Math.min(3, items.length));
  const rest = items.slice(Math.min(3, items.length));
  const features: Array<keyof NonNullable<RecommenderItem["score_components"]>> = [
    "semantic_relevance",
    "intent_alignment",
    "reference_usefulness",
    "support_confidence"
  ];
  return features.map((feature) => {
    const topAvg =
      focus.reduce((sum, item) => sum + Number(item.score_components?.[feature] ?? 0), 0) /
      Math.max(1, focus.length);
    const restAvg =
      rest.reduce((sum, item) => sum + Number(item.score_components?.[feature] ?? 0), 0) /
      Math.max(1, rest.length);
    const delta = round(topAvg - restAvg, 4);
    return {
      feature,
      direction: delta > 0.03 ? "higher" : delta < -0.03 ? "lower" : "mixed",
      magnitude: Math.abs(delta)
    };
  });
}

function buildMismatches(context: ReportReasoningContext, items: RecommenderItem[]): string[] {
  const out: string[] = [];
  const averageIntent =
    items.reduce((sum, item) => sum + Number(item.score_components?.intent_alignment ?? 0), 0) /
    Math.max(1, items.length);
  const averageSemantic =
    items.reduce((sum, item) => sum + Number(item.score_components?.semantic_relevance ?? 0), 0) /
    Math.max(1, items.length);
  if (averageSemantic - averageIntent > 0.15) {
    out.push("Topic match is stronger than execution-style match.");
  }
  if ((context.candidateSignals?.transcript_ocr.clarity_score ?? 0.6) < 0.45) {
    out.push("Uploaded draft clarity is weaker than the strongest comparables.");
  }
  if (
    context.primaryCta &&
    context.primaryCta !== "none" &&
    (context.candidateSignals?.transcript_ocr.cta_keywords_detected.length ?? 0) === 0
  ) {
    out.push("CTA intent is present in the query but not strongly expressed in draft signals.");
  }
  return out;
}

function buildConflicts(context: ReportReasoningContext): string[] {
  const out: string[] = [];
  if (context.fallbackMode) {
    out.push("Fallback mode reduced evidence richness.");
  }
  if ((context.candidateSignals?.quality_flags ?? []).length > 0) {
    out.push("Signal extraction used fallback heuristics for some draft features.");
  }
  return out;
}

function buildExplanationUnits(
  context: ReportReasoningContext,
  evidencePack: ReportReasoning["evidence_pack"]
): ExplanationUnit[] {
  const topReasons = evidencePack.aggregate_patterns.repeated_ranking_reasons
    .slice(0, 2)
    .map((item) => item.reason);
  const explanations: ExplanationUnit[] = [];
  if (topReasons.length > 0) {
    explanations.push({
      explanation_id: "exp-selection",
      claim_type: "selection_reason",
      statement: `Top comparables are consistently being selected for ${topReasons.join(" and ")}.`,
      evidence_refs: evidencePack.top_candidates.slice(0, 3).map((item) => item.candidate_id),
      confidence: evidencePack.evidence_quality.confidence,
      status: evidencePack.evidence_quality.sufficient ? "strong" : "moderate",
      caveats: context.fallbackMode ? ["Fallback mode reduced candidate evidence richness."] : []
    });
  }
  const repeatedTags = evidencePack.aggregate_patterns.repeated_hashtags
    .slice(0, 3)
    .map((item) => item.tag);
  if (repeatedTags.length > 0) {
    explanations.push({
      explanation_id: "exp-pattern",
      claim_type: "pattern_summary",
      statement: `Recurring comparable patterns cluster around ${repeatedTags.join(", ")}.`,
      evidence_refs: evidencePack.top_candidates.slice(0, 4).map((item) => item.candidate_id),
      confidence: Math.max(0.45, evidencePack.evidence_quality.confidence - 0.05),
      status: "moderate",
      caveats: []
    });
  }
  if (evidencePack.contrast_signals.mismatches.length > 0) {
    explanations.push({
      explanation_id: "exp-gap",
      claim_type: "draft_gap",
      statement: evidencePack.contrast_signals.mismatches[0] ?? "Draft execution differs from the best comparable set.",
      evidence_refs: evidencePack.top_candidates.slice(0, 2).map((item) => item.candidate_id),
      confidence: Math.max(0.4, evidencePack.evidence_quality.confidence - 0.1),
      status: evidencePack.evidence_quality.sufficient ? "moderate" : "weak",
      caveats: evidencePack.contrast_signals.conflicts
    });
  }
  explanations.push({
    explanation_id: "exp-strength",
    claim_type: evidencePack.evidence_quality.sufficient ? "strength" : "risk",
    statement: evidencePack.evidence_quality.sufficient
      ? "The comparable set has enough support richness to drive grounded recommendations."
      : "Evidence support is thinner than ideal, so recommendations should be treated as directional.",
    evidence_refs: [],
    confidence: evidencePack.evidence_quality.confidence,
    status: evidencePack.evidence_quality.sufficient ? "strong" : "fallback",
    caveats: evidencePack.evidence_quality.missing_flags
  });
  return explanations;
}

function buildRecommendationUnits(
  context: ReportReasoningContext,
  evidencePack: ReportReasoning["evidence_pack"],
  explanationUnits: ExplanationUnit[]
): RecommendationUnit[] {
  const recommendations: RecommendationUnit[] = [];
  const gap = evidencePack.contrast_signals.mismatches[0] ?? "";
  const clarityScore = context.candidateSignals?.transcript_ocr.clarity_score ?? 0.5;
  const pacingScore = context.candidateSignals?.structure.pacing_score ?? 0.5;
  recommendations.push({
    recommendation_id: "rec-hook",
    action: "Tighten the opening promise so the draft communicates a single explicit outcome immediately.",
    rationale:
      gap || "Top comparables are stronger on intent alignment than the current draft framing.",
    priority: "High",
    effort: "Low",
    confidence: clamp(evidencePack.evidence_quality.confidence, 0, 1),
    evidence_refs: explanationUnits
      .filter((item) => item.claim_type === "selection_reason" || item.claim_type === "draft_gap")
      .map((item) => item.explanation_id),
    expected_effect_area: "hook",
    caveats: []
  });
  recommendations.push({
    recommendation_id: "rec-format",
    action: pacingScore < 0.5
      ? "Increase pacing density with faster visual progress markers and clearer step transitions."
      : "Keep pacing tight, but make structure more explicit with stronger mid-video scaffolding.",
    rationale: `Estimated pacing is ${Math.round(pacingScore * 100)}/100, while top comparables maintain stronger structural momentum.`,
    priority: context.objective === "reach" ? "Medium" : "High",
    effort: "Medium",
    confidence: clamp(evidencePack.evidence_quality.confidence - 0.05, 0, 1),
    evidence_refs: ["exp-pattern"],
    expected_effect_area: "pacing",
    caveats: context.fallbackMode ? ["Draft pacing estimate comes from fallback signals."] : []
  });
  recommendations.push({
    recommendation_id: "rec-cta",
    action:
      context.primaryCta && context.primaryCta !== "none"
        ? `Make the ${context.primaryCta.replace(/_/g, " ")} CTA more explicit and easier to act on.`
        : "Clarify the next step the viewer should take at the end of the draft.",
    rationale:
      clarityScore < 0.5
        ? "The draft reads as less explicit than the top comparables in transcript and OCR clarity."
        : "Top-ranked references make the viewer action easier to understand and act on.",
    priority: context.objective === "conversion" ? "High" : "Medium",
    effort: "Low",
    confidence: clamp(evidencePack.evidence_quality.confidence - 0.08, 0, 1),
    evidence_refs: ["exp-gap", "exp-strength"].filter((id) =>
      explanationUnits.some((item) => item.explanation_id === id)
    ),
    expected_effect_area: "cta",
    caveats: []
  });
  return recommendations;
}

export function buildReportReasoning(context: ReportReasoningContext): {
  meta: ReportMeta;
  reasoning: ReportReasoning;
} {
  const items = context.recommenderPayload.items ?? [];
  const supportValues = items.map((item) => supportLevelOf(item));
  const supportCounts = countBy(supportValues);
  const hashtagCounts = countBy(
    items.flatMap((item) =>
      (context.candidates.find((candidate) => candidate.video_id === item.candidate_id)?.hashtags ?? []).slice(0, 5)
    )
  );
  const contentTypeCounts = countBy(
    items
      .map((item) => String((item as Record<string, unknown>).content_type ?? context.contentType ?? ""))
      .filter(Boolean)
  );
  const rankingReasonCounts = countBy(items.flatMap((item) => item.ranking_reasons ?? []));
  const avgComponents: ComparableItem["score_components"] = {
    semantic_relevance: round(
      items.reduce((sum, item) => sum + Number(item.score_components?.semantic_relevance ?? 0), 0) /
        Math.max(1, items.length),
      4
    ),
    intent_alignment: round(
      items.reduce((sum, item) => sum + Number(item.score_components?.intent_alignment ?? 0), 0) /
        Math.max(1, items.length),
      4
    ),
    reference_usefulness: round(
      items.reduce((sum, item) => sum + Number(item.score_components?.reference_usefulness ?? 0), 0) /
        Math.max(1, items.length),
      4
    ),
    support_confidence: round(
      items.reduce((sum, item) => sum + Number(item.score_components?.support_confidence ?? 0), 0) /
        Math.max(1, items.length),
      4
    )
  };
  const missingFlags: string[] = [];
  if (context.fallbackMode) {
    missingFlags.push("fallback_mode");
  }
  if (!context.language) {
    missingFlags.push("language_unspecified");
  }
  if ((supportCounts.get("full") ?? 0) < Math.min(4, Math.max(1, items.length))) {
    missingFlags.push("limited_full_support");
  }
  if (!context.candidateSignals) {
    missingFlags.push("draft_signal_profile_missing");
  }
  const evidenceConfidence = clamp(
    0.25 +
      ((supportCounts.get("full") ?? 0) / Math.max(1, items.length || 1)) * 0.45 +
      avgComponents.support_confidence * 0.2 +
      (context.fallbackMode ? -0.15 : 0.1),
    0.15,
    0.95
  );
  const evidencePack: ReportReasoning["evidence_pack"] = {
    version: "reasoning.v1",
    request: {
      request_id: context.requestId,
      objective: context.objective,
      objective_effective: context.objectiveEffective,
      fallback_mode: context.fallbackMode
    },
    query_summary: {
      description: context.description,
      hashtags: context.hashtags,
      mentions: context.mentions,
      content_type: context.contentType,
      primary_cta: context.primaryCta,
      audience: context.audience,
      locale: context.locale,
      language: context.language
    },
    candidate_summary: {
      final_count: items.length,
      top_k_considered: items.length,
      support_mix: {
        full: supportCounts.get("full") ?? 0,
        partial: supportCounts.get("partial") ?? 0,
        low: supportCounts.get("low") ?? 0
      },
      branch_mix: summarizeBranchMix(items)
    },
    top_candidates: items.slice(0, 8).map((item) => ({
      candidate_id: item.candidate_id,
      rank: item.rank,
      score: item.score,
      support_level: supportLevelOf(item),
      score_components: {
        semantic_relevance: round(Number(item.score_components?.semantic_relevance ?? 0), 4),
        intent_alignment: round(Number(item.score_components?.intent_alignment ?? 0), 4),
        reference_usefulness: round(Number(item.score_components?.reference_usefulness ?? 0), 4),
        support_confidence: round(Number(item.score_components?.support_confidence ?? 0), 4)
      },
      ranking_reasons: item.ranking_reasons ?? [],
      hashtags:
        context.candidates.find((candidate) => candidate.video_id === item.candidate_id)?.hashtags ?? [],
      content_type: context.contentType,
      language: context.language,
      locale: context.locale
    })),
    aggregate_patterns: {
      repeated_hashtags: topPairs(hashtagCounts as Map<string, number>, "tag") as Array<{
        tag: string;
        support_count: number;
      }>,
      repeated_content_types: topPairs(contentTypeCounts as Map<string, number>, "content_type") as Array<{
        content_type: string;
        support_count: number;
      }>,
      repeated_ranking_reasons: topPairs(rankingReasonCounts as Map<string, number>, "reason") as Array<{
        reason: string;
        support_count: number;
      }>,
      score_component_averages: avgComponents
    },
    contrast_signals: {
      top_vs_rest: buildTopVsRestSignals(items),
      mismatches: buildMismatches(context, items),
      conflicts: buildConflicts(context)
    },
    evidence_quality: {
      sufficient: (supportCounts.get("full") ?? 0) >= Math.min(3, Math.max(1, items.length)),
      confidence: round(evidenceConfidence, 4),
      missing_flags: unique(missingFlags)
    }
  };
  const explanationUnits = buildExplanationUnits(context, evidencePack);
  const recommendationUnits = buildRecommendationUnits(context, evidencePack, explanationUnits);
  const meta: ReportMeta = {
    request_id: context.requestId,
    objective: context.objective,
    objective_effective: context.objectiveEffective,
    generated_at: context.generatedAt,
    recommender_source: context.recommenderSource,
    fallback_mode: context.fallbackMode,
    fallback_reason: context.fallbackReason ?? null,
    evidence_label: classifyEvidenceLabel({
      fallbackMode: context.fallbackMode,
      sufficient: evidencePack.evidence_quality.sufficient,
      confidence: evidencePack.evidence_quality.confidence
    }),
    confidence_label: classifyConfidenceLabel(evidencePack.evidence_quality.confidence),
    experiment_id: context.experimentId ?? null,
    variant: context.variant ?? null
  };
  return {
    meta,
    reasoning: {
      evidence_pack: evidencePack,
      explanation_units: explanationUnits,
      recommendation_units: recommendationUnits,
      reasoning_metadata: {
        version: "reasoning.v1",
        fallback_mode: context.fallbackMode,
        evidence_sufficiency: evidencePack.evidence_quality.sufficient,
        reasoning_confidence: evidencePack.evidence_quality.confidence,
        missing_evidence_flags: evidencePack.evidence_quality.missing_flags
      }
    }
  };
}
