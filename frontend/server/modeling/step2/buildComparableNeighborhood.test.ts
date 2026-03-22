import assert from "node:assert/strict";
import test from "node:test";

import type { DemoVideoRecord } from "../../../src/services/data/types";
import { extractCandidateSignals } from "../part2/extractCandidateSignals";
import { buildCandidateProfileCore } from "../step1/buildCandidateProfileCore";
import { buildComparableNeighborhood } from "./buildComparableNeighborhood";

function makeRecord(
  id: string,
  author: string,
  views: number,
  likes: number,
  comments: number,
  shares: number,
  caption: string,
  hashtags: string[],
  keywords: string[]
): DemoVideoRecord {
  return {
    video_id: id,
    caption,
    hashtags,
    keywords,
    metrics: {
      views,
      likes,
      comments_count: comments,
      shares
    },
    author: {
      author_id: author,
      username: author,
      followers: 120000
    },
    comments: [
      "this is great",
      "can you share the steps?",
      "where is the link?"
    ]
  };
}

test("buildComparableNeighborhood returns neighborhood groups and ranking traces", () => {
  const profile = buildCandidateProfileCore({
    description: "How to build a quick meal prep tutorial for fitness students",
    hashtags: ["mealprep", "fitness", "tutorial"],
    mentions: ["coach"],
    objective: "engagement",
    content_type: "tutorial",
    primary_cta: "comment",
    audience: "busy students",
    locale: "en"
  });
  const signals = extractCandidateSignals(profile, {
    duration_seconds: 42,
    transcript_text: "step one prep ingredients step two cook quickly"
  });

  const records: DemoVideoRecord[] = [
    makeRecord("v1", "a1", 220000, 23000, 1800, 1200, "Meal prep tutorial in 3 steps", ["mealprep", "fitness"], ["tutorial", "meal prep"]),
    makeRecord("v2", "a2", 180000, 16000, 1200, 900, "Quick fitness recipe guide", ["recipe", "fitness"], ["guide", "quick"]),
    makeRecord("v3", "a3", 52000, 3800, 400, 120, "Budget meal prep for students", ["mealprep", "student"], ["budget", "students"]),
    makeRecord("v4", "a4", 640000, 42000, 3200, 2600, "Top 5 fitness meal ideas", ["fitness", "meal"], ["top 5", "ideas"]),
    makeRecord("v5", "a5", 26000, 1400, 220, 60, "What I eat in a day tutorial", ["tutorial", "food"], ["what i eat", "tutorial"]),
    makeRecord("v6", "a6", 120000, 9000, 600, 350, "How to meal prep for gym", ["gym", "mealprep"], ["how to", "gym"]),
    makeRecord("v7", "a7", 700000, 18000, 600, 240, "Viral dance trend", ["dance", "viral"], ["trend", "music"]),
    makeRecord("v8", "a8", 45000, 1200, 95, 40, "Opinion on nutrition myths", ["opinion", "nutrition"], ["myths", "opinion"])
  ];

  const neighborhood = buildComparableNeighborhood({
    candidateProfile: profile,
    candidateSignals: signals,
    records
  });

  assert.equal(neighborhood.version, "step2.v1");
  assert.equal(neighborhood.ranking_traces.length > 0, true);
  assert.equal(neighborhood.content_twins.length > 0, true);
  assert.equal(neighborhood.confidence.overall >= 0.2, true);
});

test("buildComparableNeighborhood enforces author caps in selected neighborhoods", () => {
  const profile = buildCandidateProfileCore({
    description: "Quick tutorial for creators",
    hashtags: ["tutorial", "creator"],
    mentions: [],
    content_type: "tutorial"
  });

  const records: DemoVideoRecord[] = [];
  for (let i = 0; i < 10; i += 1) {
    records.push(
      makeRecord(
        `same-author-${i}`,
        "dominant-author",
        120000 + i * 4000,
        10000 + i * 300,
        800 + i * 20,
        500 + i * 10,
        "Creator tutorial quick guide",
        ["tutorial", "creator"],
        ["guide", "creator"]
      )
    );
  }
  records.push(
    makeRecord("other-author", "other-author", 150000, 12000, 900, 600, "Creator guide for growth", ["creator"], ["growth"])
  );

  const neighborhood = buildComparableNeighborhood({
    candidateProfile: profile,
    records
  });

  const authorCounts = new Map<string, number>();
  for (const item of neighborhood.content_twins) {
    authorCounts.set(item.author_key, (authorCounts.get(item.author_key) ?? 0) + 1);
  }
  for (const count of authorCounts.values()) {
    assert.equal(count <= 2, true);
  }
});

