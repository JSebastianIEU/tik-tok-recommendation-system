import type { DemoVideoRecord } from "../../src/services/data/types";
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
}

export function buildReportPrompt(params: BuildReportPromptParams): string {
  const { seed, candidates, mentions, hashtags, description, candidatesK } = params;

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
      candidates: candidates.map(summarizeCandidate)
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
