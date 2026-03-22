import type {
  ComparableItem,
  DirectComparisonRow,
  ReportOutput,
  ReportPolarity
} from "../../src/features/report/types";
import type { DemoVideoRecord } from "../../src/services/data/types";
import {
  buildComparableNeighborhood,
  buildNeighborhoodContrast,
  type CandidateProfileCore,
  type CandidateSignalProfile,
  type ContrastClaim,
  type ComparableNeighborhood,
  type NeighborhoodCandidate,
  type NeighborhoodContrast
} from "../modeling";
import { HARD_CODED_EXTRACTED_KEYWORDS } from "../prompts/seedVideoContext";

interface BuildLocalBaselineReportParams {
  seed: DemoVideoRecord;
  candidates: DemoVideoRecord[];
  mentions: string[];
  hashtags: string[];
  description: string;
  candidatesK: number;
  candidateProfile?: CandidateProfileCore;
  candidateSignals?: CandidateSignalProfile;
  comparableNeighborhood?: ComparableNeighborhood;
  neighborhoodContrast?: NeighborhoodContrast;
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

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function average(values: number[]): number {
  if (values.length === 0) {
    return 0;
  }

  return values.reduce((sum, current) => sum + current, 0) / values.length;
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
  const value = record.video_url;
  return typeof value === "string" ? value : "";
}

function toComparableMetrics(record: DemoVideoRecord): ComparableItem["metrics"] {
  const views = Math.max(1, record.metrics.views);
  const engagementRate =
    ((record.metrics.likes + record.metrics.comments_count + record.metrics.shares) / views) *
    100;

  return {
    views: record.metrics.views,
    likes: record.metrics.likes,
    comments_count: record.metrics.comments_count,
    shares: record.metrics.shares,
    engagement_rate: `${engagementRate.toFixed(2)}%`
  };
}

function buildMatchedKeywords(
  record: DemoVideoRecord,
  queryTokens: string[]
): string[] {
  const candidateTokens = toTokens([
    record.caption,
    ...record.keywords,
    ...record.hashtags
  ]);

  const keywordTokens = toTokens(HARD_CODED_EXTRACTED_KEYWORDS);
  const matched = unique(
    candidateTokens.filter((token) => queryTokens.includes(token) || keywordTokens.includes(token))
  ).slice(0, 6);
  return matched.length > 0 ? matched : record.keywords.slice(0, 4);
}

function toComparableItem(
  neighborhoodCandidate: NeighborhoodCandidate,
  index: number,
  queryTokens: string[]
): ComparableItem {
  const record = neighborhoodCandidate.record;
  return {
    id: `${record.video_id}-${index + 1}`,
    caption: record.caption,
    author: toDisplayAuthor(record),
    video_url: extractVideoUrl(record),
    thumbnail_url: "",
    hashtags: record.hashtags.slice(0, 5),
    similarity: Number(neighborhoodCandidate.similarity.toFixed(2)),
    metrics: toComparableMetrics(record),
    matched_keywords: buildMatchedKeywords(record, queryTokens),
    observations: [
      `Composite score ${(neighborhoodCandidate.composite_score * 100).toFixed(1)}/100 with residual ${neighborhoodCandidate.residual_log_views.toFixed(2)}.`,
      `Neighborhood reasons: ${neighborhoodCandidate.ranking_reasons.join(", ")}.`,
      "Signals combine text, hashtags, intent/format alignment, and candidate profile similarity."
    ]
  };
}

function pickDisplayNeighborhood(
  neighborhood: ComparableNeighborhood,
  maxItems = 8
): NeighborhoodCandidate[] {
  const out: NeighborhoodCandidate[] = [];
  const seen = new Set<string>();
  const ordered = [
    ...neighborhood.content_twins,
    ...neighborhood.similar_overperformers,
    ...neighborhood.similar_underperformers
  ];

  for (const item of ordered) {
    if (out.length >= maxItems) {
      break;
    }
    if (seen.has(item.candidate_key)) {
      continue;
    }
    seen.add(item.candidate_key);
    out.push(item);
  }

  return out;
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
  const hookBoost = candidateSignals ? Math.max(0, 4 - candidateSignals.structure.hook_timing_seconds) * 3 : 0;

  const hook = clamp(50 + keywordBoost * 4 + (descWords > 20 ? 6 : 0) + hookBoost + pacingBoost * 0.35, 40, 93);
  const clarity = clamp(
    56 + Math.min(descWords, 60) * 0.5 + mentions.length * 2 + clarityBoost,
    45,
    95
  );
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

function buildRelevanceNote(topic: string, polarity: ReportPolarity): string {
  if (polarity === "Question") {
    return "This question suggests viewers still need context. Add a clearer setup in the first 2 seconds and reinforce intent with on-screen text.";
  }

  if (polarity === "Negative") {
    return "This signals friction. Tighten your opening copy and show a stronger concrete outcome earlier in the video.";
  }

  if (topic === "cta") {
    return "This is a positive CTA signal. Keep your closing ask specific and easy to answer in one comment.";
  }

  return "This confirms an element that already resonates. Keep this pattern and make it more explicit in your next iteration.";
}

function buildRelevantComments(
  rankedComparables: NeighborhoodCandidate[]
): ReportOutput["relevant_comments"]["items"] {
  const items: ReportOutput["relevant_comments"]["items"] = [];

  for (const comparable of rankedComparables.slice(0, 8)) {
    for (const comment of comparable.record.comments.slice(0, 2)) {
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
        relevance_note: buildRelevanceNote(topic, polarity)
      });
    }
  }

  return items;
}

function buildRecommendations(
  candidateProfile?: CandidateProfileCore,
  candidateSignals?: CandidateSignalProfile,
  comparableNeighborhood?: ComparableNeighborhood,
  neighborhoodContrast?: NeighborhoodContrast
): ReportOutput["recommendations"]["items"] {
  if (neighborhoodContrast && neighborhoodContrast.claims.length > 0) {
    const toPriority = (claim: ContrastClaim): "High" | "Medium" | "Low" => {
      if (claim.action_confidence >= 0.72) {
        return "High";
      }
      if (claim.action_confidence >= 0.45) {
        return "Medium";
      }
      return "Low";
    };

    const toEffort = (claim: ContrastClaim): "Low" | "Medium" | "High" => {
      if (claim.domain === "focus" || claim.domain === "clarity") {
        return "Low";
      }
      if (claim.domain === "engagement" || claim.domain === "shareability") {
        return "Medium";
      }
      return "High";
    };

    return neighborhoodContrast.claims.slice(0, 3).map((claim, index) => ({
      id: claim.claim_id || `rec-${index + 1}`,
      title: claim.recommended_action,
      priority: toPriority(claim),
      effort: toEffort(claim),
      evidence: `${claim.statement} Pattern confidence ${(claim.pattern_confidence * 100).toFixed(
        0
      )}/100. Features: ${claim.supporting_feature_keys.join(", ")}.`
    }));
  }

  const pacingScore = candidateSignals ? Math.round(candidateSignals.structure.pacing_score * 100) : null;
  const clarityScore = candidateSignals ? Math.round(candidateSignals.transcript_ocr.clarity_score * 100) : null;
  const topicText = candidateProfile ? candidateProfile.tags.topic_tags.slice(0, 2).join(", ") : "your niche";
  const ctaLabel = candidateProfile ? candidateProfile.intent.primary_cta : "comment";
  const neighborhoodConfidence = comparableNeighborhood
    ? Math.round(comparableNeighborhood.confidence.overall * 100)
    : null;

  return [
    {
      id: "rec-1",
      title: "Open with a one-line promise and an immediate visual proof of that outcome.",
      priority: "High",
      effort: "Low",
      evidence:
        pacingScore === null
          ? "Based on recurring patterns in top comparable videos from the local dataset."
          : `Pacing signal is ${pacingScore}/100. Neighborhood confidence is ${neighborhoodConfidence ?? 0}/100. Higher-performing comparables in ${topicText} establish value in the opening seconds.`
    },
    {
      id: "rec-2",
      title: "Reduce generic tags and keep 3-5 focused hashtags aligned with intent and topic.",
      priority: "Medium",
      effort: "Low",
      evidence:
        candidateProfile
          ? `Current objective is '${candidateProfile.intent.objective}' with topic tags ${topicText}. Hashtag specificity is a top differentiator in the comparable set.`
          : "Hashtag density and topic alignment are stronger in higher-performing comparables."
    },
    {
      id: "rec-3",
      title:
        ctaLabel === "none"
          ? "Add a specific CTA in the final seconds to increase meaningful comments."
          : `End with a specific '${ctaLabel}' CTA to increase meaningful responses.`,
      priority: "High",
      effort: "Medium",
      evidence:
        clarityScore === null
          ? "Comment patterns show clearer CTA phrasing drives more engaged responses."
          : `Transcript/OCR clarity is ${clarityScore}/100. Overperformer vs underperformer contrast indicates clearer CTA phrasing correlates with stronger comment quality.`
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
    candidateProfile,
    candidateSignals,
    comparableNeighborhood,
    neighborhoodContrast
  } = params;

  const queryTokens = candidateProfile
    ? unique([
        ...candidateProfile.tokens.description_tokens,
        ...candidateProfile.tokens.hashtag_tokens,
        ...candidateProfile.tokens.keyphrases.flatMap((phrase) => phrase.split(" ")),
        ...candidateProfile.tags.topic_tags.flatMap((tag) => tag.split("_")),
        ...candidateProfile.tags.format_tags.flatMap((tag) => tag.split("_")),
        ...toTokens(HARD_CODED_EXTRACTED_KEYWORDS)
      ])
    : toTokens([
        description,
        ...mentions,
        ...hashtags,
        ...HARD_CODED_EXTRACTED_KEYWORDS
      ]);

  const neighborhood =
    comparableNeighborhood ??
    (candidateProfile
      ? buildComparableNeighborhood({
          candidateProfile,
          candidateSignals,
          records: candidates
        })
      : null);
  const contrast =
    neighborhoodContrast ??
    (candidateProfile && neighborhood
      ? buildNeighborhoodContrast({
          candidateProfile,
          candidateSignals,
          neighborhood
        })
      : null);

  const selectedNeighborhoodComparables = neighborhood
    ? pickDisplayNeighborhood(neighborhood, 8)
    : [];
  const comparables = selectedNeighborhoodComparables.map((entry, index) =>
    toComparableItem(entry, index, queryTokens)
  );

  const yourEstimate = estimateUploadedVideoStats(description, hashtags, mentions, candidateSignals);
  const directRows = buildDirectComparison(comparables, yourEstimate);

  return {
    header: {
      title: "Report",
      subtitle: "Comparison based on local dataset candidates",
      badges: {
        candidates_k: candidatesK,
        model: "DeepSeek Reasoner",
        mode: "Guided Demo"
      },
      disclaimer:
        "Estimated report generated from local dataset signals and uploaded-video analysis context."
    },
    executive_summary: {
      metrics: [
        {
          id: "retention-estimated",
          label: "Estimated retention",
          value: `${yourEstimate.retention}%`
        },
        {
          id: "hook-strength",
          label: "Hook strength",
          value: `${yourEstimate.hook}/100`
        },
        {
          id: "message-clarity",
          label: "Message clarity",
          value: `${yourEstimate.clarity}/100`
        },
        {
          id: "competitiveness",
          label: "Competitiveness vs comparables (percentile)",
          value: `P${yourEstimate.competitivenessPercentile}`
        }
      ],
      extracted_keywords: [...HARD_CODED_EXTRACTED_KEYWORDS],
      meaning_points: [
        "Your current concept aligns well with proven creator/business content themes in this local dataset.",
        "Top comparables consistently establish value in the first seconds and close with a direct CTA.",
        candidateSignals
          ? `Signal extractors estimate pacing score ${Math.round(candidateSignals.structure.pacing_score * 100)}/100 and transcript clarity ${Math.round(candidateSignals.transcript_ocr.clarity_score * 100)}/100.`
          : "Keyword alignment is strong; the main opportunity is sharper opening clarity and pacing.",
        contrast && contrast.claims.length > 0
          ? `Neighborhood contrast suggests: ${contrast.claims[0].title} (confidence ${Math.round(
              contrast.claims[0].pattern_confidence * 100
            )}/100).`
          : "Contrast evidence is limited; recommendations are based on broader neighborhood patterns."
      ],
      summary_text:
        "Your uploaded video concept is competitive in topic relevance. The strongest improvements are in the first two seconds and CTA specificity."
    },
    comparables,
    direct_comparison: {
      rows: directRows,
      note: neighborhood
        ? `Estimated from local dataset candidates. Neighborhood confidence: ${Math.round(
            neighborhood.confidence.overall * 100
          )}/100${
            contrast
              ? `; contrast confidence: ${Math.round(contrast.neighborhood_confidence * 100)}/100`
              : ""
          }.`
        : "Estimated from local dataset candidates."
    },
    relevant_comments: {
      items: buildRelevantComments(selectedNeighborhoodComparables),
      disclaimer: "Comments come from the local test dataset."
    },
    recommendations: {
      items: buildRecommendations(
        candidateProfile,
        candidateSignals,
        neighborhood ?? undefined,
        contrast ?? undefined
      )
    }
  };
}
