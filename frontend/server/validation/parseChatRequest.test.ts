import assert from "node:assert/strict";
import test from "node:test";

import { parseChatRequest } from "./parseChatRequest";

function buildValidReport() {
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
      extracted_keywords: ["marketing"],
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
          performance_quality: 0.6,
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
          mentions: []
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
              performance_quality: 0.6,
              reference_usefulness: 0.8,
              support_confidence: 0.85
            },
            ranking_reasons: ["strong_intent_alignment"]
          }
        ],
        aggregate_patterns: {
          repeated_hashtags: [],
          repeated_content_types: [],
          repeated_ranking_reasons: [],
          score_component_averages: {
            semantic_relevance: 0.7,
            intent_alignment: 0.9,
            performance_quality: 0.6,
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
    }
  };
}

test("parseChatRequest accepts question/history/report", () => {
  const result = parseChatRequest({
    question: "How do I improve retention?",
    history: [
      { role: "assistant", content: "Ask me anything." },
      { role: "user", content: "How can I improve retention?" }
    ],
    objective_effective: "engagement",
    metric_focus: "retention",
    report: buildValidReport(),
    videoAnalysis: { transcript: "Hello world" }
  });

  assert.equal(result.ok, true);
  if (!result.ok) {
    return;
  }
  assert.equal(result.value.question, "How do I improve retention?");
  assert.equal(result.value.history.length, 2);
  assert.equal(result.value.objectiveEffective, "engagement");
  assert.equal(result.value.metricFocus, "retention");
  assert.equal(result.value.report?.meta.objective_effective, "engagement");
  assert.equal(result.value.videoAnalysis?.transcript, "Hello world");
});

test("parseChatRequest rejects missing question", () => {
  const result = parseChatRequest({ history: [] });
  assert.equal(result.ok, false);
});

test("parseChatRequest rejects malformed history", () => {
  const result = parseChatRequest({
    question: "hello",
    history: "invalid"
  });
  assert.equal(result.ok, false);
});

test("parseChatRequest gracefully nulls invalid report", () => {
  const result = parseChatRequest({
    question: "hello",
    report: { foo: "bar" }
  });
  assert.equal(result.ok, true);
  if (!result.ok) {
    return;
  }
  assert.equal(result.value.report, null);
});
