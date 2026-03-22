import assert from "node:assert/strict";
import test from "node:test";

import type { DemoVideoRecord } from "../../../src/services/data/types";
import { buildCandidateProfileCore } from "../step1/buildCandidateProfileCore";
import type { ComparableNeighborhood, NeighborhoodCandidate } from "../step2/buildComparableNeighborhood";
import { buildNeighborhoodContrast } from "./buildNeighborhoodContrast";

interface CandidateParams {
  id: string;
  author: string;
  topic: string;
  caption: string;
  hashtags: string[];
  keywords: string[];
  comments: string[];
  views: number;
  likes: number;
  commentsCount: number;
  shares: number;
  residual: number;
  similarity: number;
  createdAt: string;
}

function makeCandidate(params: CandidateParams): NeighborhoodCandidate {
  const record: DemoVideoRecord = {
    video_id: params.id,
    caption: params.caption,
    hashtags: params.hashtags,
    keywords: params.keywords,
    metrics: {
      views: params.views,
      likes: params.likes,
      comments_count: params.commentsCount,
      shares: params.shares
    },
    author: {
      author_id: params.author,
      username: params.author,
      followers: 120000
    },
    comments: params.comments,
    created_at: params.createdAt
  };
  const expectedLogViews = Math.log1p(params.views) - params.residual;
  const engagementRate =
    (params.likes + params.commentsCount + params.shares) / Math.max(1, params.views);

  return {
    candidate_key: params.id,
    record,
    author_key: params.author,
    topic_key: params.topic,
    stage1_score: params.similarity,
    composite_score: params.similarity,
    similarity: params.similarity,
    residual_log_views: params.residual,
    expected_log_views: expectedLogViews,
    engagement_rate: engagementRate,
    score_components: {
      text_similarity: params.similarity,
      hashtag_similarity: params.similarity * 0.9,
      intent_match: 0.8,
      format_match: 0.7,
      signal_match: 0.8
    },
    ranking_reasons: ["test candidate"]
  };
}

function makeNeighborhood(
  over: NeighborhoodCandidate[],
  under: NeighborhoodCandidate[],
  confidenceOverall: number
): ComparableNeighborhood {
  const merged = [...over, ...under];

  return {
    version: "step2.v1",
    generated_at: "2026-03-21T12:00:00.000Z",
    config: {
      stage1_pool_size: 40,
      max_per_author: 2,
      max_per_topic: 4,
      min_similarity: 0.28,
      weights: {
        text_similarity: 0.45,
        hashtag_similarity: 0.2,
        intent_match: 0.15,
        format_match: 0.1,
        signal_match: 0.1
      }
    },
    content_twins: merged.slice(0, 6),
    similar_overperformers: over,
    similar_underperformers: under,
    ranking_traces: merged.map((item) => ({
      candidate_key: item.candidate_key,
      stage1_score: item.stage1_score,
      composite_score: item.composite_score,
      residual_log_views: item.residual_log_views,
      expected_log_views: item.expected_log_views,
      topic_key: item.topic_key,
      author_key: item.author_key,
      score_components: item.score_components,
      ranking_reasons: item.ranking_reasons
    })),
    confidence: {
      overall: confidenceOverall,
      content_twins: Math.max(0.2, confidenceOverall - 0.05),
      overperformers: Math.max(0.2, confidenceOverall - 0.1),
      underperformers: Math.max(0.2, confidenceOverall - 0.08),
      reasons: ["test neighborhood"]
    }
  };
}

function makeContrastFixture(confidenceOverall: number): ComparableNeighborhood {
  const over = [
    makeCandidate({
      id: "over-1",
      author: "a-over-1",
      topic: "food",
      caption: "Meal prep workflow in 3 clear steps",
      hashtags: ["#mealprep", "#tutorial", "#food"],
      keywords: ["meal prep", "step by step"],
      comments: [
        "Super clear breakdown and easy to follow",
        "Saved this, great structure",
        "Love the sequence and quantities"
      ],
      views: 28000,
      likes: 4100,
      commentsCount: 650,
      shares: 430,
      residual: 1.2,
      similarity: 0.86,
      createdAt: "2026-03-20T12:00:00.000Z"
    }),
    makeCandidate({
      id: "over-2",
      author: "a-over-2",
      topic: "food",
      caption: "Fast lunch prep with exact quantities",
      hashtags: ["#mealprep", "#lunch", "#food"],
      keywords: ["quantities", "fast recipe"],
      comments: [
        "Exact amounts helped a lot",
        "Very practical and clean tutorial",
        "Saving this for next week"
      ],
      views: 24000,
      likes: 3300,
      commentsCount: 520,
      shares: 360,
      residual: 1.05,
      similarity: 0.83,
      createdAt: "2026-03-19T12:00:00.000Z"
    }),
    makeCandidate({
      id: "over-3",
      author: "a-over-3",
      topic: "food",
      caption: "Step sequence for high-protein batch prep",
      hashtags: ["#protein", "#tutorial", "#prep"],
      keywords: ["batch prep", "protein"],
      comments: [
        "Excellent flow and clarity",
        "Saved and shared with my friend",
        "Straight to the point"
      ],
      views: 30000,
      likes: 4300,
      commentsCount: 700,
      shares: 510,
      residual: 1.3,
      similarity: 0.84,
      createdAt: "2026-03-18T12:00:00.000Z"
    }),
    makeCandidate({
      id: "over-4",
      author: "a-over-4",
      topic: "food",
      caption: "Quick prep guide that people can reuse",
      hashtags: ["#guide", "#mealprep", "#food"],
      keywords: ["guide", "reuse"],
      comments: [
        "Great pacing and instructions",
        "Clear recipe flow from start to finish",
        "Bookmarking this one"
      ],
      views: 26000,
      likes: 3600,
      commentsCount: 540,
      shares: 370,
      residual: 1.1,
      similarity: 0.81,
      createdAt: "2026-03-17T12:00:00.000Z"
    })
  ];

  const under = [
    makeCandidate({
      id: "under-1",
      author: "a-under-1",
      topic: "food",
      caption: "Meal prep idea maybe try this",
      hashtags: ["#food", "#mealprep", "#viral", "#fyp", "#tips", "#life"],
      keywords: ["meal prep", "idea"],
      comments: ["What are the ingredients?", "When do we add sauce?", "Why this order?"],
      views: 28000,
      likes: 1500,
      commentsCount: 110,
      shares: 75,
      residual: -0.9,
      similarity: 0.79,
      createdAt: "2026-03-16T12:00:00.000Z"
    }),
    makeCandidate({
      id: "under-2",
      author: "a-under-2",
      topic: "food",
      caption: "Quick food post with no exact steps",
      hashtags: ["#food", "#quick", "#viral", "#trend", "#fyp", "#easy"],
      keywords: ["quick food", "no steps"],
      comments: ["How much rice?", "Can you explain step 2?", "What temperature?"],
      views: 24000,
      likes: 1200,
      commentsCount: 95,
      shares: 64,
      residual: -1.05,
      similarity: 0.78,
      createdAt: "2026-03-15T12:00:00.000Z"
    }),
    makeCandidate({
      id: "under-3",
      author: "a-under-3",
      topic: "food",
      caption: "Trying a prep format without details",
      hashtags: ["#food", "#prep", "#viral", "#foryou", "#hack", "#now"],
      keywords: ["prep", "format"],
      comments: ["Why no quantities?", "Where are the steps?", "How long to cook?"],
      views: 26000,
      likes: 1300,
      commentsCount: 102,
      shares: 67,
      residual: -0.95,
      similarity: 0.77,
      createdAt: "2026-03-14T12:00:00.000Z"
    }),
    makeCandidate({
      id: "under-4",
      author: "a-under-4",
      topic: "food",
      caption: "Fast prep clip with broad hashtags",
      hashtags: ["#food", "#prep", "#viral", "#fyp", "#trend", "#daily"],
      keywords: ["fast prep", "broad tags"],
      comments: ["What are the measurements?", "Can you repeat the order?", "Why skip ingredients?"],
      views: 25000,
      likes: 1180,
      commentsCount: 96,
      shares: 62,
      residual: -0.88,
      similarity: 0.76,
      createdAt: "2026-03-13T12:00:00.000Z"
    })
  ];

  return makeNeighborhood(over, under, confidenceOverall);
}

test("buildNeighborhoodContrast generates active claims from strong neighborhood evidence", () => {
  const profile = buildCandidateProfileCore({
    description: "How to make meal prep steps explicit and practical",
    hashtags: ["mealprep", "tutorial", "food"],
    mentions: ["coach"],
    content_type: "tutorial",
    primary_cta: "save"
  });

  const contrast = buildNeighborhoodContrast({
    candidateProfile: profile,
    neighborhood: makeContrastFixture(0.82)
  });

  assert.equal(contrast.version, "step3.v1");
  assert.equal(contrast.fallback_mode, false);
  assert.equal(contrast.normalized_deltas.length > 0, true);

  const clarityClaim = contrast.claims.find((claim) => claim.claim_id === "claim-clarity");
  assert.ok(clarityClaim);
  assert.equal(clarityClaim.status, "active");
  assert.equal(clarityClaim.pattern_confidence > 0.45, true);
  assert.equal(clarityClaim.supporting_feature_keys.includes("question_comment_rate"), true);
  assert.equal(clarityClaim.supporting_feature_keys.includes("comments_per_1k_views"), true);
});

test("buildNeighborhoodContrast switches to fallback mode and downgrades claim status when confidence is low", () => {
  const profile = buildCandidateProfileCore({
    description: "Meal prep tutorial test",
    hashtags: ["mealprep", "tutorial"],
    mentions: []
  });

  const contrast = buildNeighborhoodContrast({
    candidateProfile: profile,
    neighborhood: makeContrastFixture(0.22)
  });

  assert.equal(contrast.fallback_mode, true);
  assert.equal(contrast.claims.some((claim) => claim.status === "mixed"), true);
  assert.equal(contrast.claims.some((claim) => claim.status === "active"), false);
});

test("buildNeighborhoodContrast emits conflict flags for contradictory clarity deltas", () => {
  const profile = buildCandidateProfileCore({
    description: "Educational food tutorial",
    hashtags: ["education", "food", "tutorial"],
    mentions: []
  });

  const over = [
    makeCandidate({
      id: "conf-over-1",
      author: "conf-over-1",
      topic: "food",
      caption: "Clear long comments over group",
      hashtags: ["#mealprep", "#tutorial", "#food"],
      keywords: ["clear", "steps"],
      comments: [
        "This explanation is very detailed and easy to repeat at home",
        "The sequence and ingredient notes are very explicit and practical",
        "Saved because the process was clear and ordered"
      ],
      views: 20000,
      likes: 2200,
      commentsCount: 360,
      shares: 60,
      residual: 0.9,
      similarity: 0.8,
      createdAt: "2026-03-20T09:00:00.000Z"
    }),
    makeCandidate({
      id: "conf-over-2",
      author: "conf-over-2",
      topic: "food",
      caption: "Over sample two",
      hashtags: ["#mealprep", "#tutorial"],
      keywords: ["clarity", "tutorial"],
      comments: [
        "Extremely detailed explanation with exact sequence and quantities",
        "Very clear and complete tutorial for beginners",
        "Good detail level and helpful flow"
      ],
      views: 21000,
      likes: 2100,
      commentsCount: 330,
      shares: 58,
      residual: 0.85,
      similarity: 0.79,
      createdAt: "2026-03-19T09:00:00.000Z"
    })
  ];

  const under = [
    makeCandidate({
      id: "conf-under-1",
      author: "conf-under-1",
      topic: "food",
      caption: "Under sample one",
      hashtags: ["#food", "#viral", "#fyp", "#trend", "#tips", "#quick"],
      keywords: ["unclear", "missing"],
      comments: ["Why?", "How?", "What?"],
      views: 21000,
      likes: 1100,
      commentsCount: 95,
      shares: 180,
      residual: -0.8,
      similarity: 0.78,
      createdAt: "2026-03-18T09:00:00.000Z"
    }),
    makeCandidate({
      id: "conf-under-2",
      author: "conf-under-2",
      topic: "food",
      caption: "Under sample two",
      hashtags: ["#food", "#viral", "#foryou", "#trend", "#hack", "#easy"],
      keywords: ["confusing", "question"],
      comments: ["Where?", "When?", "Why?"],
      views: 22000,
      likes: 1040,
      commentsCount: 90,
      shares: 170,
      residual: -0.75,
      similarity: 0.77,
      createdAt: "2026-03-17T09:00:00.000Z"
    })
  ];

  const contrast = buildNeighborhoodContrast({
    candidateProfile: profile,
    neighborhood: makeNeighborhood(over, under, 0.76)
  });

  assert.equal(contrast.conflicts.includes("clarity_signal_conflict"), true);
});
