import assert from "node:assert/strict";
import test from "node:test";

import type { ReportOutput } from "../../src/features/report/types";
import { validateReportOutput } from "./validateReportOutput";

function buildValidReport(): ReportOutput {
  return {
    meta: {
      request_id: "018f0f57-21cb-7f81-8d17-6efec2b5f2be",
      objective: "engagement",
      objective_effective: "engagement",
      generated_at: "2026-04-04T12:00:00.000Z",
      recommender_source: "python-service",
      fallback_mode: false,
      fallback_reason: null,
      evidence_label: "Strong evidence",
      confidence_label: "High confidence",
      experiment_id: "rec_v2",
      variant: "control"
    },
    header: {
      title: "Draft report",
      subtitle: "Structured report",
      badges: {
        candidates_k: 8,
        model: "baseline",
        mode: "Guided demo"
      },
      disclaimer: "Deterministic report."
    },
    executive_summary: {
      metrics: [{ id: "clarity", label: "Clarity", value: "84/100" }],
      extracted_keywords: ["marketing", "launch"],
      meaning_points: ["Top comparables are tutorial-led."],
      summary_text: "Tutorial-style comparables lead the neighborhood."
    },
    comparables: [
      {
        id: "comp-1",
        candidate_id: "cand-1",
        caption: "App marketing tutorial",
        author: "@creator",
        video_url: "https://www.tiktok.com/@creator/video/1",
        thumbnail_url: "https://example.com/thumb.jpg",
        hashtags: ["#appmarketing"],
        similarity: 0.82,
        support_level: "full",
        confidence_label: "High confidence",
        metrics: {
          views: 1000,
          likes: 100,
          comments_count: 10,
          shares: 5,
          engagement_rate: "11.50%"
        },
        matched_keywords: ["marketing"],
        observations: ["Clear CTA"],
        why_this_was_chosen: "Strong intent alignment.",
        ranking_reasons: ["strong_intent_alignment"],
        score_components: {
          semantic_relevance: 0.7,
          intent_alignment: 0.9,
          reference_usefulness: 0.8,
          support_confidence: 0.85
        },
        retrieval_branches: ["semantic", "structured_compatibility"]
      }
    ],
    direct_comparison: {
      rows: [
        {
          id: "engagement-rate",
          label: "Engagement rate",
          your_value_label: "7.00%",
          comparable_value_label: "11.50%",
          your_value_pct: 58,
          comparable_value_pct: 96
        }
      ],
      note: "Estimated from dataset."
    },
    relevant_comments: {
      items: [
        {
          id: "comment-1",
          text: "This is useful.",
          topic: "cta",
          polarity: "Positive",
          relevance_note: "Positive response to clear call to action."
        }
      ],
      disclaimer: "Comments are illustrative."
    },
    recommendations: {
      items: [
        {
          id: "rec-1",
          title: "Clarify the CTA.",
          priority: "High",
          effort: "Low",
          evidence: "Top comparables use more explicit CTA language.",
          rationale: "The strongest items make the next action obvious.",
          confidence_label: "High confidence",
          effect_area: "cta",
          caveats: [],
          evidence_refs: ["exp-1"]
        }
      ]
    },
    reasoning: {
      evidence_pack: {
        version: "v1",
        request: {
          request_id: "018f0f57-21cb-7f81-8d17-6efec2b5f2be",
          objective: "engagement",
          objective_effective: "engagement",
          fallback_mode: false
        },
        query_summary: {
          description: "App marketing tutorial",
          hashtags: ["#appmarketing"],
          mentions: [],
          content_type: "tutorial",
          primary_cta: "save",
          locale: "en-US",
          language: "en"
        },
        candidate_summary: {
          final_count: 1,
          top_k_considered: 1,
          support_mix: { full: 1, partial: 0, low: 0 },
          branch_mix: { semantic: 1 }
        },
        top_candidates: [
          {
            candidate_id: "cand-1",
            rank: 1,
            score: 0.82,
            support_level: "full",
            score_components: {
              semantic_relevance: 0.7,
              intent_alignment: 0.9,
              reference_usefulness: 0.8,
              support_confidence: 0.85
            },
            ranking_reasons: ["strong_intent_alignment"],
            hashtags: ["#appmarketing"],
            content_type: "tutorial",
            language: "en",
            locale: "en-US"
          }
        ],
        aggregate_patterns: {
          repeated_hashtags: [{ tag: "#appmarketing", support_count: 1 }],
          repeated_content_types: [{ content_type: "tutorial", support_count: 1 }],
          repeated_ranking_reasons: [{ reason: "strong_intent_alignment", support_count: 1 }],
          score_component_averages: {
            semantic_relevance: 0.7,
            intent_alignment: 0.9,
            reference_usefulness: 0.8,
            support_confidence: 0.85
          }
        },
        contrast_signals: {
          top_vs_rest: [],
          mismatches: [],
          conflicts: []
        },
        evidence_quality: {
          sufficient: true,
          confidence: 0.82,
          missing_flags: []
        }
      },
      explanation_units: [],
      recommendation_units: [],
      reasoning_metadata: {
        version: "v1",
        fallback_mode: false,
        evidence_sufficiency: true,
        reasoning_confidence: 0.82,
        missing_evidence_flags: []
      }
    },
    explainability: {
      evidence_cards: [
        {
          candidate_id: "cand-1",
          rank: 1,
          feature_contributions: { semantic_relevance: 0.7 },
          neighbor_evidence: { retrieval_branches: ["semantic"] },
          temporal_confidence_band: { low: 0.7, mid: 0.8, high: 0.9 }
        }
      ],
      counterfactual_actions: [
        {
          candidate_id: "cand-1",
          rank: 1,
          scenarios: [
            {
              scenario_id: "scenario-1",
              expected_rank_delta_band: {},
              feasibility: "medium"
            }
          ]
        }
      ],
      disclaimer: "Directional only.",
      trace_metadata: {}
    }
  };
}

test("validateReportOutput accepts a fully structured report payload", () => {
  const report = buildValidReport();
  assert.equal(validateReportOutput(report), true);
});

test("validateReportOutput rejects counterfactual scenarios without scenario ids", () => {
  const report = buildValidReport();
  delete (report.explainability?.counterfactual_actions[0]?.scenarios[0] as { scenario_id?: string })
    .scenario_id;
  assert.equal(validateReportOutput(report), false);
});
