import type { DemoVideoRecord } from "../../../src/services/data/types";
import type { CandidateProfileCore } from "../step1/types";
import type { CandidateSignalProfile } from "../part2/extractCandidateSignals";

interface ScoreComponents {
  text_similarity: number;
  hashtag_similarity: number;
  intent_match: number;
  format_match: number;
  signal_match: number;
}

export interface NeighborhoodCandidate {
  candidate_key: string;
  record: DemoVideoRecord;
  author_key: string;
  topic_key: string;
  stage1_score: number;
  composite_score: number;
  similarity: number;
  residual_log_views: number;
  expected_log_views: number;
  engagement_rate: number;
  score_components: ScoreComponents;
  ranking_reasons: string[];
}

export interface RankingTrace {
  candidate_key: string;
  stage1_score: number;
  composite_score: number;
  residual_log_views: number;
  expected_log_views: number;
  topic_key: string;
  author_key: string;
  score_components: ScoreComponents;
  ranking_reasons: string[];
}

export interface NeighborhoodConfidence {
  overall: number;
  content_twins: number;
  overperformers: number;
  underperformers: number;
  reasons: string[];
}

export interface ComparableNeighborhood {
  version: "step2.v1";
  generated_at: string;
  config: {
    stage1_pool_size: number;
    max_per_author: number;
    max_per_topic: number;
    min_similarity: number;
    weights: ScoreComponents;
  };
  content_twins: NeighborhoodCandidate[];
  similar_overperformers: NeighborhoodCandidate[];
  similar_underperformers: NeighborhoodCandidate[];
  ranking_traces: RankingTrace[];
  confidence: NeighborhoodConfidence;
}

export interface BuildComparableNeighborhoodInput {
  candidateProfile: CandidateProfileCore;
  candidateSignals?: CandidateSignalProfile;
  records: DemoVideoRecord[];
}

const WEIGHTS: ScoreComponents = {
  text_similarity: 0.45,
  hashtag_similarity: 0.2,
  intent_match: 0.15,
  format_match: 0.1,
  signal_match: 0.1
};

const STAGE1_POOL_SIZE = 40;
const MAX_PER_AUTHOR = 2;
const MAX_PER_TOPIC = 4;
const MIN_SIMILARITY = 0.28;

const CTA_TERMS = ["follow", "comment", "save", "share", "link", "bio", "join", "subscribe", "shop"];
const FORMAT_RULES: Record<string, RegExp> = {
  how_to: /\b(how to|tutorial|guide|step|tips|hack)\b/,
  listicle: /\b(top \d+|\d+ ways|\d+ tips|\d+ steps)\b/,
  story: /\b(my story|journey|when i|what happened)\b/,
  reaction: /\b(react|reaction|duet|stitch)\b/,
  opinionated: /\b(unpopular opinion|hot take|i think)\b/,
  showcase: /\b(results|before and after|review|demo|showcase)\b/
};
const CONVERSION_TERMS = ["buy", "shop", "deal", "product", "price", "discount", "link"];
const COMMUNITY_TERMS = ["comment", "community", "family", "join", "we", "you all"];
const ENGAGEMENT_TERMS = ["viral", "engage", "save", "share", "follow", "comment"];
const TOPIC_LEXICON: Record<string, string[]> = {
  food: ["food", "recipe", "cook", "meal", "kitchen", "comida", "receta"],
  fitness: ["fitness", "workout", "gym", "exercise", "health", "salud"],
  finance: ["finance", "invest", "money", "crypto", "budget", "stock", "trading"],
  beauty: ["beauty", "makeup", "skincare", "hair", "cosmetic"],
  fashion: ["fashion", "outfit", "style", "lookbook", "streetwear"],
  tech: ["tech", "ai", "software", "app", "coding", "gadget"],
  travel: ["travel", "trip", "vacation", "city", "journey", "viaje"],
  education: ["learn", "lesson", "tutorial", "guide", "tips", "teach", "study"],
  entertainment: ["funny", "meme", "dance", "music", "viral", "trend", "reaction"],
  lifestyle: ["lifestyle", "daily", "routine", "mindset", "productivity", "selfcare"]
};

type ObjectiveGuess = "reach" | "engagement" | "conversion" | "community";

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function round(value: number, decimals = 6): number {
  const factor = 10 ** decimals;
  return Math.round(value * factor) / factor;
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
  return [...new Set(values)];
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

function safeNumber(value: unknown, fallback = 0): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function authorKey(record: DemoVideoRecord): string {
  if (typeof record.author === "string") {
    return record.author.trim().toLowerCase() || "unknown";
  }
  const fromId = typeof record.author?.author_id === "string" ? record.author.author_id.trim().toLowerCase() : "";
  if (fromId) {
    return fromId;
  }
  const fromUsername = typeof record.author?.username === "string" ? record.author.username.trim().toLowerCase() : "";
  return fromUsername || "unknown";
}

function followers(record: DemoVideoRecord): number {
  if (typeof record.author === "object" && record.author && typeof record.author.followers === "number") {
    return Math.max(0, record.author.followers);
  }
  return 0;
}

function engagementRate(record: DemoVideoRecord): number {
  const views = Math.max(1, safeNumber(record.metrics?.views, 0));
  const likes = safeNumber(record.metrics?.likes, 0);
  const comments = safeNumber(record.metrics?.comments_count, 0);
  const shares = safeNumber(record.metrics?.shares, 0);
  return (likes + comments + shares) / views;
}

function jaccard(a: string[], b: string[]): number {
  if (a.length === 0 || b.length === 0) {
    return 0;
  }
  const setA = new Set(a);
  const setB = new Set(b);
  let intersection = 0;
  for (const token of setA) {
    if (setB.has(token)) {
      intersection += 1;
    }
  }
  const union = new Set([...setA, ...setB]).size;
  return union === 0 ? 0 : intersection / union;
}

function inferCandidateFormatTags(record: DemoVideoRecord): string[] {
  const text = normalizeText([record.caption, ...record.keywords, ...record.hashtags].join(" "));
  const out: string[] = [];
  for (const [tag, rule] of Object.entries(FORMAT_RULES)) {
    if (rule.test(text)) {
      out.push(tag);
    }
  }
  return out.length > 0 ? out : ["general"];
}

function inferCandidateObjective(record: DemoVideoRecord): ObjectiveGuess {
  const text = normalizeText([record.caption, ...record.keywords, ...record.hashtags, ...record.comments].join(" "));
  const er = engagementRate(record);
  const commentsPerView = safeNumber(record.metrics?.comments_count, 0) / Math.max(1, safeNumber(record.metrics?.views, 0));

  if (CONVERSION_TERMS.some((term) => text.includes(term))) {
    return "conversion";
  }
  if (COMMUNITY_TERMS.some((term) => text.includes(term)) || commentsPerView > 0.012) {
    return "community";
  }
  if (ENGAGEMENT_TERMS.some((term) => text.includes(term)) || er > 0.085) {
    return "engagement";
  }
  return "reach";
}

function inferTopicKey(record: DemoVideoRecord): string {
  const tokens = tokenize([record.caption, ...record.keywords, ...record.hashtags]);
  for (const [topic, terms] of Object.entries(TOPIC_LEXICON)) {
    if (terms.some((term) => tokens.includes(term))) {
      return topic;
    }
  }
  if (record.hashtags.length > 0) {
    return normalizeText(record.hashtags[0]).replace(/^#/, "") || "general";
  }
  return "general";
}

function inferDurationBucket(record: DemoVideoRecord): "short" | "medium" | "long" {
  const fromDuration = safeNumber((record as { duration_sec?: number }).duration_sec, NaN);
  const fromVideoMeta = safeNumber(
    (record as { video_meta?: { duration_seconds?: number } }).video_meta?.duration_seconds,
    NaN
  );
  const duration = Number.isFinite(fromDuration)
    ? fromDuration
    : Number.isFinite(fromVideoMeta)
      ? fromVideoMeta
      : clamp(Math.round(normalizeText(record.caption).split(" ").filter(Boolean).length * 1.5), 12, 180);

  if (duration < 30) {
    return "short";
  }
  if (duration < 60) {
    return "medium";
  }
  return "long";
}

function inferAuthorBucket(record: DemoVideoRecord): "nano" | "small" | "mid" | "large" {
  const count = followers(record);
  if (count < 50000) {
    return "nano";
  }
  if (count < 250000) {
    return "small";
  }
  if (count < 1000000) {
    return "mid";
  }
  return "large";
}

function percentile(sortedValues: number[], p: number): number {
  if (sortedValues.length === 0) {
    return 0;
  }
  const index = clamp(p, 0, 1) * (sortedValues.length - 1);
  const lower = Math.floor(index);
  const upper = Math.ceil(index);
  if (lower === upper) {
    return sortedValues[lower];
  }
  const weight = index - lower;
  return sortedValues[lower] * (1 - weight) + sortedValues[upper] * weight;
}

function selectWithCaps(
  ranked: NeighborhoodCandidate[],
  maxItems: number,
  excluded: Set<string>
): NeighborhoodCandidate[] {
  const out: NeighborhoodCandidate[] = [];
  const authorCounts = new Map<string, number>();
  const topicCounts = new Map<string, number>();

  for (const item of ranked) {
    if (out.length >= maxItems) {
      break;
    }
    if (excluded.has(item.candidate_key)) {
      continue;
    }
    const authorCount = authorCounts.get(item.author_key) ?? 0;
    if (authorCount >= MAX_PER_AUTHOR) {
      continue;
    }
    const topicCount = topicCounts.get(item.topic_key) ?? 0;
    if (topicCount >= MAX_PER_TOPIC) {
      continue;
    }
    out.push(item);
    authorCounts.set(item.author_key, authorCount + 1);
    topicCounts.set(item.topic_key, topicCount + 1);
    excluded.add(item.candidate_key);
  }

  return out;
}

function weightedScore(components: ScoreComponents): number {
  return (
    components.text_similarity * WEIGHTS.text_similarity +
    components.hashtag_similarity * WEIGHTS.hashtag_similarity +
    components.intent_match * WEIGHTS.intent_match +
    components.format_match * WEIGHTS.format_match +
    components.signal_match * WEIGHTS.signal_match
  );
}

function makeReasons(components: ScoreComponents, residual: number): string[] {
  const entries = Object.entries(components).sort((a, b) => b[1] - a[1]);
  const reasons = entries.slice(0, 2).map(([name]) => `high_${name}`);
  if (residual > 0.2) {
    reasons.push("positive_residual_performance");
  } else if (residual < -0.2) {
    reasons.push("negative_residual_performance");
  }
  return reasons;
}

export function buildComparableNeighborhood(
  input: BuildComparableNeighborhoodInput
): ComparableNeighborhood {
  const { candidateProfile, candidateSignals, records } = input;
  const queryTokens = unique([
    ...candidateProfile.tokens.description_tokens,
    ...candidateProfile.tokens.hashtag_tokens,
    ...candidateProfile.tokens.keyphrases.flatMap((phrase) => phrase.split(" ")),
    ...candidateProfile.tags.topic_tags.flatMap((tag) => tag.split("_")),
    ...candidateProfile.tags.format_tags.flatMap((tag) => tag.split("_"))
  ]);
  const queryHashtags = candidateProfile.normalized.hashtags.map((tag) => normalizeText(tag));
  const queryFormatTags = candidateProfile.tags.format_tags.map((tag) => normalizeText(tag.replace("creator_selected:", "")));
  const expectedPacing = candidateSignals?.structure.pacing_score ?? 0.5;
  const expectedClarity = candidateSignals?.transcript_ocr.clarity_score ?? 0.5;

  const stage1Ranked = records
    .map((record) => {
      const candidateTokens = tokenize([record.caption, ...record.keywords, ...record.hashtags]);
      const candidateHashtags = record.hashtags.map((tag) => normalizeText(tag).replace(/^#/, ""));
      const textSimilarity = jaccard(queryTokens, candidateTokens);
      const hashtagSimilarity = jaccard(queryHashtags, candidateHashtags);
      const stage1Score = round(textSimilarity * 0.7 + hashtagSimilarity * 0.3, 6);
      return {
        record,
        stage1Score,
        textSimilarity,
        hashtagSimilarity
      };
    })
    .sort((a, b) => b.stage1Score - a.stage1Score)
    .slice(0, Math.min(STAGE1_POOL_SIZE, records.length));

  const allLogViews = records
    .map((record) => Math.log1p(Math.max(0, safeNumber(record.metrics?.views, 0))))
    .sort((a, b) => a - b);
  const globalBaseline = allLogViews.length > 0 ? allLogViews.reduce((sum, value) => sum + value, 0) / allLogViews.length : 0;

  const baselineByCompositeBucket = new Map<string, number[]>();
  const baselineByTopic = new Map<string, number[]>();
  for (const record of records) {
    const topic = inferTopicKey(record);
    const bucket = `${topic}|${inferDurationBucket(record)}|${inferAuthorBucket(record)}`;
    const logViews = Math.log1p(Math.max(0, safeNumber(record.metrics?.views, 0)));
    baselineByCompositeBucket.set(bucket, [...(baselineByCompositeBucket.get(bucket) ?? []), logViews]);
    baselineByTopic.set(topic, [...(baselineByTopic.get(topic) ?? []), logViews]);
  }

  const rankedCandidates = stage1Ranked.map((row) => {
    const record = row.record;
    const topicKey = inferTopicKey(record);
    const durationBucket = inferDurationBucket(record);
    const authorBucket = inferAuthorBucket(record);
    const baselineKey = `${topicKey}|${durationBucket}|${authorBucket}`;
    const bucketValues = baselineByCompositeBucket.get(baselineKey) ?? [];
    const topicValues = baselineByTopic.get(topicKey) ?? [];
    const expectedLogViews =
      bucketValues.length >= 3
        ? bucketValues.reduce((sum, value) => sum + value, 0) / bucketValues.length
        : topicValues.length >= 3
          ? topicValues.reduce((sum, value) => sum + value, 0) / topicValues.length
          : globalBaseline;

    const candidateObjective = inferCandidateObjective(record);
    const intentMatch =
      candidateObjective === candidateProfile.intent.objective
        ? 1
        : candidateObjective === "engagement" && candidateProfile.intent.objective === "reach"
          ? 0.62
          : candidateObjective === "reach" && candidateProfile.intent.objective === "engagement"
            ? 0.62
            : 0.35;

    const candidateFormats = inferCandidateFormatTags(record);
    const formatMatch = jaccard(queryFormatTags, candidateFormats.map((tag) => normalizeText(tag)));

    const er = engagementRate(record);
    const questionRate =
      record.comments.length === 0
        ? 0
        : record.comments.filter((comment) => comment.includes("?")).length / record.comments.length;
    const candidatePacingProxy = clamp(0.2 + record.hashtags.length * 0.04 + er * 3.2, 0, 1);
    const candidateClarityProxy = clamp(0.3 + (1 - questionRate) * 0.35 + er * 2.1, 0, 1);
    const signalDistance =
      (Math.abs(expectedPacing - candidatePacingProxy) + Math.abs(expectedClarity - candidateClarityProxy)) / 2;
    const signalMatch = clamp(1 - signalDistance, 0, 1);

    const scoreComponents: ScoreComponents = {
      text_similarity: row.textSimilarity,
      hashtag_similarity: row.hashtagSimilarity,
      intent_match: intentMatch,
      format_match: formatMatch,
      signal_match: signalMatch
    };
    const compositeScore = round(weightedScore(scoreComponents), 6);
    const logViews = Math.log1p(Math.max(0, safeNumber(record.metrics?.views, 0)));
    const residual = round(logViews - expectedLogViews, 6);

    const comparable: NeighborhoodCandidate = {
      candidate_key: record.video_id,
      record,
      author_key: authorKey(record),
      topic_key: topicKey,
      stage1_score: row.stage1Score,
      composite_score: compositeScore,
      similarity: compositeScore,
      residual_log_views: residual,
      expected_log_views: round(expectedLogViews, 6),
      engagement_rate: round(er, 6),
      score_components: scoreComponents,
      ranking_reasons: makeReasons(scoreComponents, residual)
    };

    return comparable;
  });

  rankedCandidates.sort((a, b) => b.composite_score - a.composite_score);

  const residualsSorted = rankedCandidates.map((item) => item.residual_log_views).sort((a, b) => a - b);
  const lowResidualCut = percentile(residualsSorted, 0.25);
  const highResidualCut = percentile(residualsSorted, 0.75);

  const excluded = new Set<string>();
  const contentTwins = selectWithCaps(
    rankedCandidates.filter((item) => item.similarity >= MIN_SIMILARITY),
    8,
    excluded
  );
  const overperformers = selectWithCaps(
    rankedCandidates.filter(
      (item) => item.similarity >= MIN_SIMILARITY * 0.85 && item.residual_log_views >= highResidualCut
    ),
    6,
    excluded
  );
  const underperformers = selectWithCaps(
    rankedCandidates.filter(
      (item) => item.similarity >= MIN_SIMILARITY * 0.85 && item.residual_log_views <= lowResidualCut
    ),
    6,
    excluded
  );

  const confidenceReasons: string[] = [];
  const avgTwinScore =
    contentTwins.length > 0
      ? contentTwins.reduce((sum, item) => sum + item.composite_score, 0) / contentTwins.length
      : 0;
  const avgOverResidual =
    overperformers.length > 0
      ? overperformers.reduce((sum, item) => sum + item.residual_log_views, 0) / overperformers.length
      : 0;
  const avgUnderResidual =
    underperformers.length > 0
      ? underperformers.reduce((sum, item) => sum + item.residual_log_views, 0) / underperformers.length
      : 0;

  if (contentTwins.length < 4) {
    confidenceReasons.push("limited_content_twin_coverage");
  }
  if (overperformers.length < 2) {
    confidenceReasons.push("limited_overperformer_evidence");
  }
  if (underperformers.length < 2) {
    confidenceReasons.push("limited_underperformer_evidence");
  }

  const overallConfidence = round(
    clamp(
      avgTwinScore * 0.55 +
        clamp(avgOverResidual, 0, 1) * 0.25 +
        clamp(Math.abs(avgUnderResidual), 0, 1) * 0.2 -
        confidenceReasons.length * 0.06,
      0.2,
      0.98
    ),
    4
  );

  return {
    version: "step2.v1",
    generated_at: new Date().toISOString(),
    config: {
      stage1_pool_size: STAGE1_POOL_SIZE,
      max_per_author: MAX_PER_AUTHOR,
      max_per_topic: MAX_PER_TOPIC,
      min_similarity: MIN_SIMILARITY,
      weights: WEIGHTS
    },
    content_twins: contentTwins,
    similar_overperformers: overperformers,
    similar_underperformers: underperformers,
    ranking_traces: rankedCandidates.slice(0, 30).map((item) => ({
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
      overall: overallConfidence,
      content_twins: round(clamp(avgTwinScore, 0, 1), 4),
      overperformers: round(clamp(avgOverResidual, -1, 1), 4),
      underperformers: round(clamp(avgUnderResidual, -1, 1), 4),
      reasons: confidenceReasons
    }
  };
}

