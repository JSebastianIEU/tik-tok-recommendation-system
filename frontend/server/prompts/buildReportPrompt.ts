import type { DemoVideoRecord } from "../../src/services/data/types";
import type {
  CandidateProfileCore,
  CandidateSignalProfile,
  ComparableNeighborhood,
  NeighborhoodContrast
} from "../modeling";
import {
  HARD_CODED_EXTRACTED_KEYWORDS,
  SEED_VIDEO_ALGORITHM_DESCRIPTION
} from "./seedVideoContext";

function summarizeCandidate(record: DemoVideoRecord): Record<string, unknown> {
  return {
    candidate_key: record.video_id,
    caption: record.caption,
    hashtags: record.hashtags,
    keywords: record.keywords,
    author: record.author,
    metrics: record.metrics,
    comments: record.comments
  };
}

export interface BuildReportPromptParams {
  seed: DemoVideoRecord;
  candidates: DemoVideoRecord[];
  mentions: string[];
  hashtags: string[];
  description: string;
  candidatesK: number;
  candidateProfile: CandidateProfileCore;
  candidateSignals: CandidateSignalProfile;
  comparableNeighborhood: ComparableNeighborhood;
  neighborhoodContrast: NeighborhoodContrast;
}

export function buildReportPrompt(params: BuildReportPromptParams): string {
  const {
    seed,
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

  const promptPayload = {
    role: "Senior content strategist and growth analyst",
    objective:
      "Produce an advanced, practical report with concrete recommendations that do not feel generic.",
    language: "English",
    constraints: [
      "No emojis.",
      "Do not claim official TikTok ranking access.",
      "Treat this as local-dataset analysis plus uploaded-video context.",
      "Do not reference internal video IDs in natural-language fields."
    ],
    uploaded_video_context: {
      status: "not published yet",
      source: "uploaded by user in app",
      user_input: {
        mentions,
        hashtags,
        description
      },
      candidate_profile_core: {
        version: candidateProfile.profile_version,
        locale: candidateProfile.locale,
        normalized_description: candidateProfile.normalized.description,
        normalized_hashtags: candidateProfile.normalized.hashtags,
        normalized_mentions: candidateProfile.normalized.mentions,
        keyphrases: candidateProfile.tokens.keyphrases,
        topic_tags: candidateProfile.tags.topic_tags,
        format_tags: candidateProfile.tags.format_tags,
        intent: candidateProfile.intent,
        quality: candidateProfile.quality,
        retrieval_text: candidateProfile.retrieval.text
      },
      candidate_signal_profile: candidateSignals,
      extracted_keywords_source: "hardcoded extraction for demo",
      extracted_keywords: HARD_CODED_EXTRACTED_KEYWORDS,
      transcript_analysis: SEED_VIDEO_ALGORITHM_DESCRIPTION,
      uploaded_video_metadata: {
        caption: seed.caption,
        hashtags: seed.hashtags,
        keywords: seed.keywords
      }
    },
    candidate_pool: {
      size_k: candidatesK,
      retrieval_note:
        "Candidates come from local dataset rows excluding the uploaded seed row.",
      candidates: candidates.map(summarizeCandidate),
      neighborhood_evidence: {
        version: comparableNeighborhood.version,
        confidence: comparableNeighborhood.confidence,
        content_twins: comparableNeighborhood.content_twins.slice(0, 8).map((item) => ({
          candidate_key: item.candidate_key,
          similarity: item.similarity,
          residual_log_views: item.residual_log_views,
          ranking_reasons: item.ranking_reasons
        })),
        similar_overperformers: comparableNeighborhood.similar_overperformers
          .slice(0, 6)
          .map((item) => ({
            candidate_key: item.candidate_key,
            similarity: item.similarity,
            residual_log_views: item.residual_log_views,
            ranking_reasons: item.ranking_reasons
          })),
        similar_underperformers: comparableNeighborhood.similar_underperformers
          .slice(0, 6)
          .map((item) => ({
            candidate_key: item.candidate_key,
            similarity: item.similarity,
            residual_log_views: item.residual_log_views,
            ranking_reasons: item.ranking_reasons
          })),
        ranking_traces: comparableNeighborhood.ranking_traces.slice(0, 12),
        contrast_evidence: {
          version: neighborhoodContrast.version,
          fallback_mode: neighborhoodContrast.fallback_mode,
          confidence: neighborhoodContrast.neighborhood_confidence,
          conflicts: neighborhoodContrast.conflicts,
          top_claims: neighborhoodContrast.claims.slice(0, 4),
          top_deltas: neighborhoodContrast.normalized_deltas.slice(0, 10),
          summary: neighborhoodContrast.summary
        }
      }
    },
    output_quality_rules: [
      "Recommendations must include priority, effort, and evidence.",
      "Evidence should describe recurring patterns, not internal IDs.",
      "Comment insights must include why each comment matters for improving the uploaded video.",
      "Use concise, specific language with actionable guidance."
    ],
    required_schema: {
      header: {
        title: "string",
        subtitle: "string",
        badges: {
          candidates_k: "number",
          model: "string",
          mode: "string"
        },
        disclaimer: "string"
      },
      executive_summary: {
        metrics: [
          {
            id: "retention-estimated|hook-strength|message-clarity|competitiveness",
            label: "string",
            value: "string"
          }
        ],
        extracted_keywords: ["string"],
        meaning_points: ["string"],
        summary_text: "string"
      },
      comparables: [
        {
          id: "internal comparable key used for media matching",
          caption: "string",
          author: "string",
          video_url: "string",
          thumbnail_url: "string",
          hashtags: ["string"],
          similarity: "number_0_to_1",
          metrics: {
            views: "number",
            likes: "number",
            comments_count: "number",
            shares: "number",
            engagement_rate: "string"
          },
          matched_keywords: ["string"],
          observations: ["string"]
        }
      ],
      direct_comparison: {
        rows: [
          {
            id: "string",
            label: "string",
            your_value_label: "string",
            comparable_value_label: "string",
            your_value_pct: "number_0_to_100",
            comparable_value_pct: "number_0_to_100"
          }
        ],
        note: "string"
      },
      relevant_comments: {
        items: [
          {
            id: "string",
            text: "string",
            topic: "string",
            polarity: "Positive|Negative|Question",
            relevance_note: "string"
          }
        ],
        disclaimer: "string"
      },
      recommendations: {
        items: [
          {
            id: "string",
            title: "string",
            priority: "High|Medium|Low",
            effort: "Low|Medium|High",
            evidence: "string"
          }
        ]
      }
    },
    output_rules: {
      strict_json_only: true,
      no_markdown: true,
      no_text_outside_json: true,
      preserve_candidates_k: candidatesK
    }
  };

  return JSON.stringify(promptPayload, null, 2);
}
