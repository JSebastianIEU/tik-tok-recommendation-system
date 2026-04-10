import type {
  ComparableItem,
  DirectComparisonRow,
  ReportOutput,
  ReportPolarity,
  ReportReasoning,
  ReportMeta
} from "../../src/features/report/types";
import type { DemoVideoRecord } from "../../src/services/data/types";
import type { RecommenderItem } from "../recommender/client";
import type { CandidateSignalProfile } from "./signalProfile";
import { HARD_CODED_EXTRACTED_KEYWORDS } from "../prompts/seedVideoContext";

interface BuildLocalBaselineReportParams {
  candidates: DemoVideoRecord[];
  mentions: string[];
  hashtags: string[];
  description: string;
  candidatesK: number;
  objective: string;
  recommenderItems: RecommenderItem[];
  candidateSignals?: CandidateSignalProfile;
  meta: ReportMeta;
  reasoning: ReportReasoning;
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function average(values: number[]): number {
  if (values.length === 0) {
    return 0;
  }
  return values.reduce((sum, current) => sum + current, 0) / values.length;
}

function normalizeText(value: string): string {
  return value
    .toLowerCase()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[^\w\s#]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function unique(values: string[]): string[] {
  return Array.from(new Set(values));
}

function toTokens(values: string[]): string[] {
  return unique(
    values
      .map((value) => normalizeText(value))
      .flatMap((value) => value.split(" "))
      .map((token) => token.replace(/^#/, "").trim())
      .filter((token) => token.length >= 2)
  );
}

function toDisplayAuthor(record: DemoVideoRecord): string {
  if (typeof record.author === "string") {
    return record.author;
  }
  if (record.author && typeof record.author === "object") {
    const authorRecord = record.author as Record<string, unknown>;
    if (typeof authorRecord.username === "string") {
      return `@${authorRecord.username.replace(/^@/, "")}`;
    }
  }
  return "@creator";
}

function extractVideoUrl(record: DemoVideoRecord): string {
  return typeof record.video_url === "string" ? record.video_url : "";
}

function toComparableMetrics(record: DemoVideoRecord): ComparableItem["metrics"] {
  const views = Math.max(1, record.metrics.views);
  const engagementRate =
    ((record.metrics.likes + record.metrics.comments_count + record.metrics.shares) / views) * 100;
  return {
    views: record.metrics.views,
    likes: record.metrics.likes,
    comments_count: record.metrics.comments_count,
    shares: record.metrics.shares,
    engagement_rate: `${engagementRate.toFixed(2)}%`
  };
}

function buildMatchedKeywords(record: DemoVideoRecord, queryTokens: string[]): string[] {
  const candidateTokens = toTokens([record.caption, ...record.keywords, ...record.hashtags]);
  const keywordTokens = toTokens(HARD_CODED_EXTRACTED_KEYWORDS);
  const matched = unique(
    candidateTokens.filter((token) => queryTokens.includes(token) || keywordTokens.includes(token))
  ).slice(0, 6);
  return matched.length > 0 ? matched : record.keywords.slice(0, 4);
}

function buildObservationLines(item: RecommenderItem | undefined): string[] {
  if (!item) {
    return ["Returned as a fallback comparable from the local candidate pool."];
  }
  const scoreParts = item.score_components ?? {};
  const reasons = (item.ranking_reasons ?? []).join(", ");
  return [
    `Baseline rank score ${(item.score * 100).toFixed(1)}/100.`,
    reasons ? `Ranking reasons: ${reasons}.` : "Ranking reasons were not returned.",
    `Semantic ${(Number(scoreParts.semantic_relevance ?? 0) * 100).toFixed(0)}/100, intent ${(Number(scoreParts.intent_alignment ?? 0) * 100).toFixed(0)}/100, usefulness ${(Number(scoreParts.reference_usefulness ?? 0) * 100).toFixed(0)}/100, confidence ${(Number(scoreParts.support_confidence ?? 0) * 100).toFixed(0)}/100.`
  ];
}

function toConfidenceLabel(value: number): ComparableItem["confidence_label"] {
  if (value < 0.45) {
    return "Low confidence";
  }
  if (value < 0.75) {
    return "Medium confidence";
  }
  return "High confidence";
}

function estimateUploadedVideoStats(
  description: string,
  hashtags: string[],
  mentions: string[],
  candidateSignals?: CandidateSignalProfile
): {
  retention: number;
  hook: number;
  clarity: number;
  competitivenessPercentile: number;
  engagementRate: number;
  likesPerView: number;
  hashtagDensity: number;
  ctaClarity: number;
} {
  const descWords = description.split(/\s+/).filter(Boolean).length;
  const keywordBoost = HARD_CODED_EXTRACTED_KEYWORDS.length;
  const pacingBoost = candidateSignals ? candidateSignals.structure.pacing_score * 10 : 0;
  const clarityBoost = candidateSignals ? candidateSignals.transcript_ocr.clarity_score * 8 : 0;
  const hookBoost = candidateSignals
    ? Math.max(0, 4 - candidateSignals.structure.hook_timing_seconds) * 3
    : 0;

  const hook = clamp(
    50 + keywordBoost * 4 + (descWords > 20 ? 6 : 0) + hookBoost + pacingBoost * 0.35,
    40,
    93
  );
  const clarity = clamp(56 + Math.min(descWords, 60) * 0.5 + mentions.length * 2 + clarityBoost, 45, 95);
  const retention = clamp(48 + hook * 0.28 + clarity * 0.2, 45, 96);
  const hashtagDensity = clamp(hashtags.length / Math.max(descWords, 8), 0.02, 0.45);
  const engagementRate = clamp(
    3.5 + hook * 0.05 + clarity * 0.03 + (candidateSignals?.audio.music_presence_score ?? 0.4),
    2.5,
    12.5
  );
  const likesPerView = clamp(0.03 + hook * 0.00065 + hashtags.length * 0.0025, 0.01, 0.13);
  const ctaClarity = clamp(52 + mentions.length * 5 + hashtags.length * 2.5, 45, 94);
  const competitivenessPercentile = clamp(52 + keywordBoost * 4, 35, 96);

  return {
    retention: Math.round(retention),
    hook: Math.round(hook),
    clarity: Math.round(clarity),
    competitivenessPercentile: Math.round(competitivenessPercentile),
    engagementRate,
    likesPerView,
    hashtagDensity,
    ctaClarity
  };
}

function buildDirectComparison(
  comparables: ComparableItem[],
  yourEstimate: ReturnType<typeof estimateUploadedVideoStats>
): DirectComparisonRow[] {
  const avgEngagementRate = average(
    comparables.map((item) => Number(item.metrics.engagement_rate.replace("%", "")))
  );
  const avgLikesPerView = average(
    comparables.map((item) => item.metrics.likes / Math.max(1, item.metrics.views))
  );
  const avgHashtagDensity = average(
    comparables.map((item) => item.hashtags.length / Math.max(1, item.caption.split(" ").length))
  );
  const avgCtaClarity = average(
    comparables.map((item) => Math.min(100, 50 + item.matched_keywords.length * 8))
  );
  const toPercentScale = (value: number, maxExpected: number): number =>
    clamp((value / maxExpected) * 100, 4, 100);

  return [
    {
      id: "engagement-rate",
      label: "Engagement rate",
      your_value_label: `${yourEstimate.engagementRate.toFixed(2)}%`,
      comparable_value_label: `${avgEngagementRate.toFixed(2)}%`,
      your_value_pct: Number(toPercentScale(yourEstimate.engagementRate, 12).toFixed(0)),
      comparable_value_pct: Number(toPercentScale(avgEngagementRate, 12).toFixed(0))
    },
    {
      id: "likes-per-view",
      label: "Likes per view",
      your_value_label: yourEstimate.likesPerView.toFixed(3),
      comparable_value_label: avgLikesPerView.toFixed(3),
      your_value_pct: Number(toPercentScale(yourEstimate.likesPerView, 0.12).toFixed(0)),
      comparable_value_pct: Number(toPercentScale(avgLikesPerView, 0.12).toFixed(0))
    },
    {
      id: "hashtag-density",
      label: "Hashtag density",
      your_value_label: yourEstimate.hashtagDensity.toFixed(2),
      comparable_value_label: avgHashtagDensity.toFixed(2),
      your_value_pct: Number(toPercentScale(yourEstimate.hashtagDensity, 0.4).toFixed(0)),
      comparable_value_pct: Number(toPercentScale(avgHashtagDensity, 0.4).toFixed(0))
    },
    {
      id: "cta-clarity",
      label: "CTA clarity",
      your_value_label: `${Math.round(yourEstimate.ctaClarity)}/100`,
      comparable_value_label: `${Math.round(avgCtaClarity)}/100`,
      your_value_pct: Math.round(yourEstimate.ctaClarity),
      comparable_value_pct: Math.round(avgCtaClarity)
    }
  ];
}

function inferTopic(text: string): string {
  const normalized = normalizeText(text);
  if (/(hook|start|first|opening)/.test(normalized)) {
    return "hook";
  }
  if (/(cta|follow|comment|action)/.test(normalized)) {
    return "cta";
  }
  if (/(edit|cut|subtitle|pace)/.test(normalized)) {
    return "editing";
  }
  if (/(clear|clarity|understand|confus)/.test(normalized)) {
    return "clarity";
  }
  return "engagement";
}

function inferPolarity(text: string): ReportPolarity {
  const normalized = normalizeText(text);
  if (/[?]/.test(text) || /^(how|what|where|why)\b/.test(normalized)) {
    return "Question";
  }
  if (/(not|dont|hard|boring|bad|pointless|difficult)/.test(normalized)) {
    return "Negative";
  }
  return "Positive";
}

function buildRelevantComments(candidates: DemoVideoRecord[]): ReportOutput["relevant_comments"]["items"] {
  const items: ReportOutput["relevant_comments"]["items"] = [];
  for (const candidate of candidates.slice(0, 8)) {
    for (const comment of candidate.comments.slice(0, 2)) {
      if (items.length >= 10) {
        return items;
      }
      const topic = inferTopic(comment);
      const polarity = inferPolarity(comment);
      items.push({
        id: `comment-${items.length + 1}`,
        text: comment,
        topic,
        polarity,
        relevance_note:
          polarity === "Question"
            ? "This question highlights missing context. Clarify the promise earlier and add a concrete visual example."
            : polarity === "Negative"
              ? "This signals friction. Tighten the opening copy and make the outcome easier to understand."
              : "This response suggests a pattern worth reinforcing in the opening and CTA."
      });
    }
  }
  return items;
}

function buildRecommendations(
  objective: string,
  candidateSignals: CandidateSignalProfile | undefined,
  recommenderItems: RecommenderItem[],
  reasoning: ReportReasoning
): ReportOutput["recommendations"]["items"] {
  const units = reasoning.recommendation_units;
  if (units.length > 0) {
    return units.map((unit) => ({
      id: unit.recommendation_id,
      title: unit.action,
      priority: unit.priority,
      effort: unit.effort,
      evidence: unit.rationale,
      rationale: unit.rationale,
      confidence_label: toConfidenceLabel(unit.confidence),
      effect_area: unit.expected_effect_area,
      caveats: unit.caveats,
      evidence_refs: unit.evidence_refs
    }));
  }
  const topItem = recommenderItems[0];
  const topReasons = (topItem?.ranking_reasons ?? []).slice(0, 2).join(", ");
  const clarityScore = candidateSignals
    ? Math.round(candidateSignals.transcript_ocr.clarity_score * 100)
    : null;
  const pacingScore = candidateSignals
    ? Math.round(candidateSignals.structure.pacing_score * 100)
    : null;
  return [
    {
      id: "rec-1",
      title:
        "Strengthen the opening promise so the comparable match is obvious within the first seconds.",
      priority: "High",
      effort: "Low",
      evidence:
        pacingScore === null
          ? "Top-ranked references are winning on relevance and opening clarity."
          : `Signal pacing is ${pacingScore}/100. The highest-ranked comparables concentrate their value proposition earlier.`,
      rationale:
        "The best comparable set wins by making the outcome clearer earlier in the video.",
      confidence_label: "Medium confidence",
      effect_area: "hook",
      caveats: [],
      evidence_refs: []
    },
    {
      id: "rec-2",
      title: "Sharpen the framing to match the intent cues the best comparables share.",
      priority: objective === "conversion" ? "High" : "Medium",
      effort: "Medium",
      evidence:
        topReasons
          ? `Top Python ranker reasons: ${topReasons}.`
          : "The recommender is rewarding stronger intent alignment than the current draft shows.",
      rationale: "Intent alignment is lagging behind broad topic similarity.",
      confidence_label: "Medium confidence",
      effect_area: "format",
      caveats: [],
      evidence_refs: []
    },
    {
      id: "rec-3",
      title: "Make the CTA more explicit and easier to act on.",
      priority: "Medium",
      effort: "Low",
      evidence:
        clarityScore === null
          ? "High-ranked references make the next step clearer."
          : `Transcript/OCR clarity is ${clarityScore}/100. Clearer end-state language should improve comparability and response quality.`,
      rationale: "Clearer viewer action should improve downstream response quality.",
      confidence_label: "Medium confidence",
      effect_area: "cta",
      caveats: [],
      evidence_refs: []
    }
  ];
}

export function buildLocalBaselineReport(
  params: BuildLocalBaselineReportParams
): ReportOutput {
  const {
    candidates,
    mentions,
    hashtags,
    description,
    candidatesK,
    objective,
    recommenderItems,
    candidateSignals,
    meta,
    reasoning
  } = params;
  const queryTokens = toTokens([description, ...mentions, ...hashtags, ...HARD_CODED_EXTRACTED_KEYWORDS]);
  const itemByCandidateId = new Map(recommenderItems.map((item) => [item.candidate_id, item]));
  const selectedCandidates = candidates.slice(0, 8);
  const comparables = selectedCandidates.map((record, index) => ({
    id: `${record.video_id}-${index + 1}`,
    candidate_id: record.video_id,
    caption: record.caption,
    author: toDisplayAuthor(record),
    video_url: extractVideoUrl(record),
    thumbnail_url: "",
    hashtags: record.hashtags.slice(0, 5),
    similarity: Number((itemByCandidateId.get(record.video_id)?.score ?? 0.5).toFixed(2)),
    support_level: (itemByCandidateId.get(record.video_id)?.support_level as ComparableItem["support_level"]) ?? "unknown",
    confidence_label: toConfidenceLabel(
      Number(itemByCandidateId.get(record.video_id)?.score_components?.support_confidence ?? 0.5)
    ),
    metrics: toComparableMetrics(record),
    matched_keywords: buildMatchedKeywords(record, queryTokens),
    observations: buildObservationLines(itemByCandidateId.get(record.video_id)),
    why_this_was_chosen:
      (itemByCandidateId.get(record.video_id)?.ranking_reasons ?? []).length > 0
        ? `Chosen for ${(itemByCandidateId.get(record.video_id)?.ranking_reasons ?? []).slice(0, 2).join(" and ")}.`
        : "Chosen as one of the strongest comparable references in the ranked set.",
    ranking_reasons: (itemByCandidateId.get(record.video_id)?.ranking_reasons ?? []).slice(0, 4),
    score_components: {
      semantic_relevance: Number(itemByCandidateId.get(record.video_id)?.score_components?.semantic_relevance ?? 0),
      intent_alignment: Number(itemByCandidateId.get(record.video_id)?.score_components?.intent_alignment ?? 0),
      reference_usefulness: Number(itemByCandidateId.get(record.video_id)?.score_components?.reference_usefulness ?? 0),
      support_confidence: Number(itemByCandidateId.get(record.video_id)?.score_components?.support_confidence ?? 0)
    },
    retrieval_branches: Object.entries(itemByCandidateId.get(record.video_id)?.retrieval_branch_scores ?? {})
      .filter(([, score]) => typeof score === "number" && score > 0)
      .map(([branch]) => branch)
  }));

  const yourEstimate = estimateUploadedVideoStats(description, hashtags, mentions, candidateSignals);

  return {
    meta,
    header: {
      title: "Report",
      subtitle: "Python-ranked comparison based on local dataset candidates",
      badges: {
        candidates_k: candidatesK,
        model: "Python Baseline Ranker",
        mode: "Guided Demo"
      },
      disclaimer:
        "Estimated report generated from Python recommender outputs, local dataset metadata, and uploaded-video signal context."
    },
    executive_summary: {
      metrics: [
        { id: "retention-estimated", label: "Estimated retention", value: `${yourEstimate.retention}%` },
        { id: "hook-strength", label: "Hook strength", value: `${yourEstimate.hook}/100` },
        { id: "message-clarity", label: "Message clarity", value: `${yourEstimate.clarity}/100` },
        {
          id: "competitiveness",
          label: "Competitiveness vs comparables (percentile)",
          value: `P${yourEstimate.competitivenessPercentile}`
        }
      ],
      extracted_keywords: [...HARD_CODED_EXTRACTED_KEYWORDS],
      meaning_points: [
        "The report is grounded in the Python recommender and deterministic reasoning layer.",
        "Top comparables are selected by semantic relevance, intent alignment, reference usefulness, and support confidence.",
        candidateSignals
          ? `Signal extractors estimate pacing score ${Math.round(candidateSignals.structure.pacing_score * 100)}/100 and transcript clarity ${Math.round(candidateSignals.transcript_ocr.clarity_score * 100)}/100.`
          : "Signal extractors were unavailable, so only metadata and ranked comparables were used.",
        ...(reasoning.explanation_units.slice(0, 2).map((item) => item.statement)),
        recommenderItems[0]
          ? `Highest-ranked comparable score ${(recommenderItems[0].score * 100).toFixed(1)}/100 with reasons ${(recommenderItems[0].ranking_reasons ?? []).slice(0, 2).join(", ")}.`
          : "No ranked comparable evidence was returned."
      ],
      summary_text:
        "The strongest next improvements are a clearer opening promise, tighter intent framing, and a more explicit CTA."
    },
    comparables,
    direct_comparison: {
      rows: buildDirectComparison(comparables, yourEstimate),
      note: "Comparison rows are estimated from the displayed Python-ranked comparables."
    },
    relevant_comments: {
      items: buildRelevantComments(selectedCandidates),
      disclaimer: "Comments come from the local test dataset."
    },
    recommendations: {
      items: buildRecommendations(objective, candidateSignals, recommenderItems, reasoning)
    },
    reasoning
  };
}
