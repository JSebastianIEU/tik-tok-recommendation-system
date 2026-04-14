import assert from "node:assert/strict";
import test from "node:test";

import type { ReportOutput } from "../../src/features/report/types";
import type { KnowledgeBaseEntry } from "../knowledgeBase/knowledgeBase";
import { buildLocalChatAnswer } from "./buildLocalChatAnswer";

function buildReport(): ReportOutput {
  return {
    meta: {
      request_id: "chat-format-test",
      objective: "engagement",
      objective_effective: "engagement",
      generated_at: "2026-04-14T10:00:00.000Z",
      recommender_source: "python-service",
      fallback_mode: false,
      fallback_reason: null,
      evidence_label: "Moderate evidence",
      confidence_label: "Medium confidence",
      experiment_id: "rec_v2_default",
      variant: "control"
    },
    header: {
      title: "Test Report",
      subtitle: "Chat format test",
      badges: {
        candidates_k: 8,
        model: "baseline",
        mode: "test"
      },
      disclaimer: "For testing."
    },
    executive_summary: {
      metrics: [
        { id: "hook_strength", label: "Hook strength", value: "66/100" },
        { id: "retention_estimated", label: "Estimated retention", value: "91/100" }
      ],
      extracted_keywords: ["vlog", "lifestyle"],
      meaning_points: ["Hook can be tighter."],
      summary_text: "Strong retention, but hook and CTA can be sharper."
    },
    comparables: [
      {
        id: "comp-1",
        candidate_id: "cand-1",
        caption: "Daily vlog with stronger opening",
        author: "@creator1",
        video_url: "https://example.com/video/1",
        thumbnail_url: "https://example.com/thumb/1.jpg",
        hashtags: ["#dailyvlog", "#lifestyle", "#creator"],
        similarity: 0.71,
        support_level: "full",
        confidence_label: "Medium confidence",
        metrics: {
          views: 1000,
          likes: 100,
          comments_count: 20,
          shares: 10,
          engagement_rate: "13.00%"
        },
        matched_keywords: ["vlog"],
        observations: ["Clearer hook"],
        why_this_was_chosen: "Strong early pacing.",
        ranking_reasons: ["hook_clarity"],
        score_components: {
          semantic_relevance: 0.7,
          intent_alignment: 0.6,
          performance_quality: 0.8,
          reference_usefulness: 0.7,
          support_confidence: 0.75
        },
        retrieval_branches: ["semantic"]
      }
    ],
    direct_comparison: {
      rows: [
        {
          id: "engagement-rate",
          label: "Engagement rate",
          your_value_label: "8.10%",
          comparable_value_label: "11.20%",
          your_value_pct: 60,
          comparable_value_pct: 80
        }
      ],
      note: "Test-only"
    },
    relevant_comments: {
      items: [],
      disclaimer: "Test-only"
    },
    recommendations: {
      items: [
        {
          id: "rec-1",
          title: "Tighten the first two seconds with a single clear payoff",
          priority: "High",
          effort: "Low",
          evidence: "Hook underperforms peers",
          rationale: "Faster context should improve early hold.",
          confidence_label: "Medium confidence",
          effect_area: "hook",
          caveats: [],
          evidence_refs: ["exp-1"]
        },
        {
          id: "rec-2",
          title: "Trim one low-information mid-video beat",
          priority: "Medium",
          effort: "Low",
          evidence: "Mid-video pacing dips",
          rationale: "Smoother pacing should keep viewers through the middle.",
          confidence_label: "Medium confidence",
          effect_area: "pacing",
          caveats: [],
          evidence_refs: ["exp-2"]
        },
        {
          id: "rec-3",
          title: "End with a specific comment CTA",
          priority: "Medium",
          effort: "Low",
          evidence: "CTA is currently broad",
          rationale: "A specific ask tends to drive more comments.",
          confidence_label: "Medium confidence",
          effect_area: "cta",
          caveats: [],
          evidence_refs: ["exp-3"]
        }
      ]
    },
    reasoning: {
      evidence_pack: {
        version: "v1",
        request: {
          request_id: "chat-format-test",
          objective: "engagement",
          objective_effective: "engagement",
          fallback_mode: false
        },
        query_summary: {
          description: "test",
          hashtags: [],
          mentions: []
        },
        candidate_summary: {
          final_count: 1,
          top_k_considered: 1,
          support_mix: {
            full: 1,
            partial: 0,
            low: 0
          },
          branch_mix: { semantic: 1 }
        },
        top_candidates: [],
        aggregate_patterns: {
          repeated_hashtags: [],
          repeated_content_types: [],
          repeated_ranking_reasons: [],
          score_component_averages: {
            semantic_relevance: 0.7,
            intent_alignment: 0.6,
            performance_quality: 0.8,
            reference_usefulness: 0.7,
            support_confidence: 0.75
          }
        },
        contrast_signals: {
          top_vs_rest: [],
          mismatches: [],
          conflicts: []
        },
        evidence_quality: {
          sufficient: true,
          confidence: 0.7,
          missing_flags: []
        }
      },
      explanation_units: [],
      recommendation_units: [],
      reasoning_metadata: {
        version: "v1",
        fallback_mode: false,
        evidence_sufficiency: true,
        reasoning_confidence: 0.7,
        missing_evidence_flags: []
      }
    },
    explainability: {
      evidence_cards: [],
      counterfactual_actions: [],
      disclaimer: "test-only",
      trace_metadata: {}
    }
  };
}

function buildKnowledgeEntries(): KnowledgeBaseEntry[] {
  return [
    {
      id: "kb-1",
      title: "Hook clarity improves early hold",
      category: "hooks",
      content: ["Specific hook lines usually improve first-second hold."],
      action_hint: "Open with one specific promise in the first second.",
      impact_area: "hook_strength",
      keywords: ["hook", "opening", "first second"],
      objective_tags: ["engagement"],
      updated_at: "2026-04-14T10:00:00.000Z",
      confidence: "high",
      active: true
    },
    {
      id: "kb-2",
      title: "CTA specificity matters",
      category: "cta",
      content: ["Specific comment prompts tend to improve interaction quality."],
      action_hint: "Use a direct, single-step comment CTA.",
      impact_area: "comment_rate",
      keywords: ["cta", "comment"],
      objective_tags: ["engagement", "conversion"],
      updated_at: "2026-04-14T10:00:00.000Z",
      confidence: "medium",
      active: true
    }
  ];
}

test("buildLocalChatAnswer returns the conversational coaching format", () => {
  const report = buildReport();
  const answer = buildLocalChatAnswer({
    report,
    question: "How can I improve engagement?",
    knowledgeBaseEntries: buildKnowledgeEntries()
  });

  assert.match(answer, /Quick diagnosis:/);
  assert.match(answer, /Top 3 actions:/);
  assert.match(answer, /1\./);
  assert.match(answer, /2\./);
  assert.match(answer, /3\./);
  assert.match(answer, /Expected impact:/);
  assert.match(answer, /\?$/);
  assert.doesNotMatch(answer, /week(?:ly)? test plan/i);
});

test("buildLocalChatAnswer returns hashtag-specific guidance for hashtag queries", () => {
  const report = buildReport();
  const answer = buildLocalChatAnswer({
    report,
    question: "which hashtags should I use?",
    knowledgeBaseEntries: buildKnowledgeEntries()
  });

  assert.match(answer, /Top 3 actions:/);
  assert.match(answer, /#dailyvlog/i);
  assert.match(answer, /Want me to suggest 8 hashtags ranked from safest to boldest\?/);
});

test("buildLocalChatAnswer blends KB action hints into report coaching", () => {
  const report = buildReport();
  const answer = buildLocalChatAnswer({
    report,
    question: "How do I improve engagement and algorithm performance?",
    knowledgeBaseEntries: buildKnowledgeEntries()
  });

  assert.match(answer, /Open with one specific promise in the first second/i);
});

test("buildLocalChatAnswer returns structured KB guidance when report is missing", () => {
  const answer = buildLocalChatAnswer({
    report: null,
    question: "what performs well on tiktok right now?",
    knowledgeBaseEntries: buildKnowledgeEntries()
  });

  assert.match(answer, /Quick diagnosis:/);
  assert.match(answer, /Top 3 actions:/);
  assert.match(answer, /Expected impact:/);
});
