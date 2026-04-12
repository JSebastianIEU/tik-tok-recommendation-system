import type {
  ComparableItem,
  DirectComparisonRow,
  ReportOutput,
  ReportPolarity
} from "../../features/report/types";
import { loadDemoDataset } from "../data/loadDemoDataset";
import type { DemoVideoRecord } from "../data/types";

interface GenerateMockReportParams {
  seedVideoId: string;
  mentions: string[];
  hashtags: string[];
  description: string;
}

const EXTRACTED_KEYWORDS = [
  "coding",
  "app development",
  "marketing",
  "startup",
  "personal brand"
];

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

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function average(values: number[]): number {
  if (values.length === 0) {
    return 0;
  }

  return values.reduce((sum, current) => sum + current, 0) / values.length;
}

function tokenize(values: string[]): string[] {
  return unique(
    values
      .map((value) => normalizeText(value))
      .flatMap((value) => value.split(" "))
      .map((token) => token.replace(/^#/, "").trim())
      .filter((token) => token.length >= 2)
  );
}

function toAuthorLabel(record: DemoVideoRecord): string {
  if (typeof record.author === "string") {
    return record.author.startsWith("@") ? record.author : `@${record.author}`;
  }

  if (record.author && typeof record.author === "object") {
    const username = (record.author as Record<string, unknown>).username;
    if (typeof username === "string" && username.trim()) {
      return `@${username.replace(/^@/, "")}`;
    }
  }

  return "@creator";
}

function toVideoUrl(record: DemoVideoRecord): string {
  const value = record.video_url;
  return typeof value === "string" ? value : "";
}

function estimateSeedMetrics(
  description: string,
  mentions: string[],
  hashtags: string[]
): {
  retention: number;
  hookStrength: number;
  clarity: number;
  competitivenessPercentile: number;
  engagementRate: number;
  likesPerView: number;
  hashtagDensity: number;
  ctaClarity: number;
} {
  const descriptionWords = description.split(/\s+/).filter(Boolean).length;
  const mentionBoost = mentions.length * 2.2;
  const hashtagBoost = hashtags.length * 2.8;

  const hookStrength = clamp(58 + descriptionWords * 0.32 + hashtagBoost, 45, 95);
  const clarity = clamp(62 + descriptionWords * 0.25 + mentionBoost, 48, 96);
  const retention = clamp(52 + hookStrength * 0.24 + clarity * 0.18, 46, 97);
  const competitivenessPercentile = clamp(55 + EXTRACTED_KEYWORDS.length * 4.5, 42, 98);

  const engagementRate = clamp(3.2 + hookStrength * 0.052 + clarity * 0.028, 2.4, 13.2);
  const likesPerView = clamp(0.03 + hookStrength * 0.00062 + hashtags.length * 0.0022, 0.01, 0.14);
  const hashtagDensity = clamp(
    hashtags.length / Math.max(descriptionWords, 8),
    0.02,
    0.44
  );
  const ctaClarity = clamp(54 + mentions.length * 5 + hashtags.length * 2.4, 42, 95);

  return {
    retention: Math.round(retention),
    hookStrength: Math.round(hookStrength),
    clarity: Math.round(clarity),
    competitivenessPercentile: Math.round(competitivenessPercentile),
    engagementRate,
    likesPerView,
    hashtagDensity,
    ctaClarity
  };
}

function toComparableMetrics(record: DemoVideoRecord): ComparableItem["metrics"] {
  const views = Math.max(1, record.metrics.views);
  const engagementRate =
    ((record.metrics.likes + record.metrics.comments_count + record.metrics.shares) /
      views) *
    100;

  return {
    views: record.metrics.views,
    likes: record.metrics.likes,
    comments_count: record.metrics.comments_count,
    shares: record.metrics.shares,
    engagement_rate: `${engagementRate.toFixed(2)}%`
  };
}

function computeSimilarity(
  record: DemoVideoRecord,
  queryTokens: string[]
): { score: number; matchedKeywords: string[] } {
  const candidateTokens = tokenize([
    record.caption,
    ...record.keywords,
    ...record.hashtags
  ]);

  const extractedTokens = tokenize(EXTRACTED_KEYWORDS);
  const matchedKeywords = unique(
    candidateTokens.filter(
      (token) => queryTokens.includes(token) || extractedTokens.includes(token)
    )
  ).slice(0, 6);

  const overlap = matchedKeywords.length;
  const union = unique([...candidateTokens, ...queryTokens, ...extractedTokens]).length || 1;
  const score = clamp(overlap / union + 0.56, 0.34, 0.99);

  return {
    score: Number(score.toFixed(2)),
    matchedKeywords
  };
}

function inferTopic(text: string): string {
  const normalized = normalizeText(text);

  if (/(hook|opening|start|first)/.test(normalized)) {
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

  if (text.includes("?") || /^(how|what|where|why)\b/.test(normalized)) {
    return "Question";
  }

  if (/(not|dont|hard|boring|bad|weak|confus)/.test(normalized)) {
    return "Negative";
  }

  return "Positive";
}

function buildRelevanceNote(topic: string, polarity: ReportPolarity): string {
  if (polarity === "Question") {
    return "This indicates missing context. Clarify intent in the first 2 seconds and add one guiding subtitle.";
  }

  if (polarity === "Negative") {
    return "This signals friction. Tighten your opening statement and surface a clearer outcome earlier.";
  }

  if (topic === "cta") {
    return "Positive CTA response. Keep the closing ask specific and easy to answer in one comment.";
  }

  return "This validates a pattern that already works. Keep it, but make it more explicit in your next version.";
}

function buildDirectComparisonRows(
  comparables: ComparableItem[],
  estimate: ReturnType<typeof estimateSeedMetrics>
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
    comparables.map((item) => Math.min(100, 52 + item.matched_keywords.length * 7))
  );

  const scale = (value: number, maxExpected: number): number =>
    clamp((value / maxExpected) * 100, 4, 100);

  return [
    {
      id: "engagement-rate",
      label: "Engagement rate",
      your_value_label: `${estimate.engagementRate.toFixed(2)}%`,
      comparable_value_label: `${avgEngagementRate.toFixed(2)}%`,
      your_value_pct: Number(scale(estimate.engagementRate, 12).toFixed(0)),
      comparable_value_pct: Number(scale(avgEngagementRate, 12).toFixed(0))
    },
    {
      id: "likes-per-view",
      label: "Likes per view",
      your_value_label: estimate.likesPerView.toFixed(3),
      comparable_value_label: avgLikesPerView.toFixed(3),
      your_value_pct: Number(scale(estimate.likesPerView, 0.12).toFixed(0)),
      comparable_value_pct: Number(scale(avgLikesPerView, 0.12).toFixed(0))
    },
    {
      id: "hashtag-density",
      label: "Hashtag density",
      your_value_label: estimate.hashtagDensity.toFixed(2),
      comparable_value_label: avgHashtagDensity.toFixed(2),
      your_value_pct: Number(scale(estimate.hashtagDensity, 0.4).toFixed(0)),
      comparable_value_pct: Number(scale(avgHashtagDensity, 0.4).toFixed(0))
    },
    {
      id: "cta-clarity",
      label: "CTA clarity",
      your_value_label: `${Math.round(estimate.ctaClarity)}/100`,
      comparable_value_label: `${Math.round(avgCtaClarity)}/100`,
      your_value_pct: Math.round(estimate.ctaClarity),
      comparable_value_pct: Math.round(avgCtaClarity)
    }
  ];
}

function buildComparableObservations(matchedKeywords: string[]): string[] {
  const firstKeyword = matchedKeywords[0] ?? "core topic";

  return [
    `Strong early relevance around ${firstKeyword}.`,
    "The pacing supports retention through short visual beats.",
    "The call-to-action is specific and easy to follow."
  ];
}

function normalizeHashtags(values: string[]): string[] {
  return values
    .map((value) => value.trim().replace(/^#/, ""))
    .filter(Boolean)
    .map((value) => `#${value}`);
}

function fallbackComments(records: DemoVideoRecord[]): string[] {
  const comments = records.flatMap((record) => record.comments);

  if (comments.length > 0) {
    return comments;
  }

  return [
    "Strong concept, but I needed the value proposition earlier.",
    "The pacing is good and keeps attention.",
    "Can you make the CTA more specific?"
  ];
}

export function generateMockReport(
  params: GenerateMockReportParams
): ReportOutput {
  const dataset = loadDemoDataset();
  const seed =
    dataset.find((record) => record.video_id === params.seedVideoId) ??
    dataset.find((record) => record.video_id === "s001") ??
    dataset[0];

  const candidates = dataset.filter((record) => record.video_id !== seed?.video_id);
  const queryTokens = tokenize([
    params.description,
    ...params.mentions,
    ...normalizeHashtags(params.hashtags),
    ...EXTRACTED_KEYWORDS
  ]);

  const ranked = candidates
    .map((candidate, index) => {
      const similarity = computeSimilarity(candidate, queryTokens);

      const comparable: ComparableItem = {
        id: candidate.video_id.toUpperCase() || `C-${index + 1}`,
        candidate_id: candidate.video_id,
        caption: candidate.caption,
        author: toAuthorLabel(candidate),
        video_url: toVideoUrl(candidate),
        thumbnail_url: "",
        hashtags: candidate.hashtags.slice(0, 5),
        similarity: similarity.score,
        support_level: "full",
        confidence_label: "Medium confidence",
        metrics: toComparableMetrics(candidate),
        matched_keywords:
          similarity.matchedKeywords.length > 0
            ? similarity.matchedKeywords
            : candidate.keywords.slice(0, 4),
        observations: buildComparableObservations(similarity.matchedKeywords),
        why_this_was_chosen: "Selected by the local mock baseline for lexical topic overlap.",
        ranking_reasons: ["topic overlap", "hashtag overlap"],
        score_components: {
          semantic_relevance: similarity.score,
          intent_alignment: clamp(similarity.score - 0.08, 0, 1),
          performance_quality: clamp(similarity.score - 0.06, 0, 1),
          reference_usefulness: clamp(similarity.score - 0.04, 0, 1),
          support_confidence: 0.72
        },
        retrieval_branches: ["lexical", "topic"]
      };

      return { comparable, raw: candidate };
    })
    .sort((left, right) => right.comparable.similarity - left.comparable.similarity);

  const topComparables = ranked.slice(0, 8).map((entry) => entry.comparable);
  const estimate = estimateSeedMetrics(
    params.description || seed?.caption || "",
    params.mentions,
    params.hashtags
  );

  const commentPool = fallbackComments(ranked.slice(0, 8).map((entry) => entry.raw));
  const relevantComments = commentPool.slice(0, 10).map((comment, index) => {
    const topic = inferTopic(comment);
    const polarity = inferPolarity(comment);

    return {
      id: `comment-${index + 1}`,
      text: comment,
      topic,
      polarity,
      relevance_note: buildRelevanceNote(topic, polarity)
    };
  });

  const topComparableIds = topComparables.slice(0, 3).map((item) => item.id);
  const evidenceText =
    topComparableIds.length > 0
      ? `Based on comparables: ${topComparableIds.join(", ")}`
      : "Based on local baseline comparables.";

  return {
    meta: {
      request_id: "mock-report",
      objective: "engagement",
      objective_effective: "engagement",
      generated_at: new Date().toISOString(),
      recommender_source: "deterministic-local",
      fallback_mode: true,
      fallback_reason: "mock_only_mode",
      evidence_label: "Limited evidence",
      confidence_label: "Medium confidence",
      experiment_id: null,
      variant: null
    },
    header: {
      title: "Report",
      subtitle: "Comparison based on similar videos (baseline TF-IDF)",
      badges: {
        candidates_k: candidates.length,
        model: "TF-IDF baseline",
        mode: "Local dataset"
      },
      disclaimer:
        "Baseline lexical retrieval from a local dataset. This is an estimation, not a real TikTok ranking model."
    },
    executive_summary: {
      metrics: [
        {
          id: "retention-estimated",
          label: "Estimated retention",
          value: `${estimate.retention}%`
        },
        {
          id: "hook-strength",
          label: "Hook strength",
          value: `${estimate.hookStrength}/100`
        },
        {
          id: "message-clarity",
          label: "Message clarity",
          value: `${estimate.clarity}/100`
        },
        {
          id: "competitiveness",
          label: "Competitiveness vs comparables (percentile)",
          value: `P${estimate.competitivenessPercentile}`
        }
      ],
      extracted_keywords: [...EXTRACTED_KEYWORDS],
      meaning_points: [
        "Your topic alignment is strong against similar videos in the local dataset.",
        "Top comparables front-load value and keep transitions tight.",
        "The highest-impact improvement is a sharper opening line."
      ],
      summary_text:
        "Your concept is competitive in relevance. Prioritize the first two seconds and a clearer CTA to improve overall performance."
    },
    comparables: topComparables,
    direct_comparison: {
      rows: buildDirectComparisonRows(topComparables, estimate),
      note: "Estimation based on local dataset (baseline)."
    },
    relevant_comments: {
      items: relevantComments,
      disclaimer: "Simulated comments from a local test dataset."
    },
    recommendations: {
      items: [
        {
          id: "rec-1",
          title: "Start with a one-line promise and immediate proof of outcome.",
          priority: "High",
          effort: "Low",
          evidence: evidenceText,
          rationale: "Top lexical comparables lead with a clearer promise.",
          confidence_label: "Medium confidence",
          effect_area: "hook",
          caveats: [],
          evidence_refs: []
        },
        {
          id: "rec-2",
          title: "Use 3 to 5 tightly aligned hashtags instead of broad generic tags.",
          priority: "Medium",
          effort: "Low",
          evidence: evidenceText,
          rationale: "Hashtag overlap is one of the strongest recurring signals in the mock baseline.",
          confidence_label: "Medium confidence",
          effect_area: "topic_alignment",
          caveats: [],
          evidence_refs: []
        },
        {
          id: "rec-3",
          title: "Close with a specific question to increase high-intent comments.",
          priority: "High",
          effort: "Medium",
          evidence: evidenceText,
          rationale: "Comparables that end with a clearer response cue tend to look stronger in comments.",
          confidence_label: "Medium confidence",
          effect_area: "cta",
          caveats: [],
          evidence_refs: []
        }
      ]
    },
    reasoning: {
      evidence_pack: {
        version: "mock.reasoning.v1",
        request: {
          request_id: "mock-report",
          objective: "engagement",
          objective_effective: "engagement",
          fallback_mode: true
        },
        query_summary: {
          description: params.description,
          hashtags: params.hashtags,
          mentions: params.mentions
        },
        candidate_summary: {
          final_count: topComparables.length,
          top_k_considered: topComparables.length,
          support_mix: { full: topComparables.length, partial: 0, low: 0 },
          branch_mix: { lexical: topComparables.length }
        },
        top_candidates: topComparables.map((item, index) => ({
          candidate_id: item.candidate_id,
          rank: index + 1,
          score: item.similarity,
          support_level: item.support_level,
          score_components: item.score_components,
          ranking_reasons: item.ranking_reasons,
          hashtags: item.hashtags
        })),
        aggregate_patterns: {
          repeated_hashtags: [],
          repeated_content_types: [],
          repeated_ranking_reasons: [
            { reason: "topic overlap", support_count: topComparables.length }
          ],
          score_component_averages: {
            semantic_relevance: 0.7,
            intent_alignment: 0.62,
            performance_quality: 0.55,
            reference_usefulness: 0.66,
            support_confidence: 0.72
          }
        },
        contrast_signals: {
          top_vs_rest: [],
          mismatches: [],
          conflicts: ["Mock report uses local-only baseline reasoning."]
        },
        evidence_quality: {
          sufficient: false,
          confidence: 0.55,
          missing_flags: ["mock_only_mode"]
        }
      },
      explanation_units: [
        {
          explanation_id: "mock-exp-1",
          claim_type: "selection_reason",
          statement: "Comparables were selected for lexical topic and hashtag overlap.",
          evidence_refs: topComparables.slice(0, 3).map((item) => item.candidate_id),
          confidence: 0.55,
          status: "fallback",
          caveats: ["Mock mode is not production evidence."]
        }
      ],
      recommendation_units: [],
      reasoning_metadata: {
        version: "mock.reasoning.v1",
        fallback_mode: true,
        evidence_sufficiency: false,
        reasoning_confidence: 0.55,
        missing_evidence_flags: ["mock_only_mode"]
      }
    }
  };
}
