import type { CandidateProfileCore } from "../step1/types";
import type { CandidateSignalProfile } from "../part2/extractCandidateSignals";
import type { ComparableNeighborhood, NeighborhoodCandidate } from "../step2/buildComparableNeighborhood";

type Direction = "over_higher" | "under_higher" | "neutral";
type ClaimStatus = "active" | "mixed" | "insufficient";

interface WeightedStats {
  mean: number;
  variance: number;
  count: number;
}

interface FeatureRow {
  feature_key: string;
  over_mean: number;
  under_mean: number;
  delta: number;
  z_delta: number;
  direction: Direction;
  support_over: number;
  support_under: number;
  reliability: number;
}

interface ClaimEvidenceTrace {
  candidate_keys_over: string[];
  candidate_keys_under: string[];
  top_feature_keys: string[];
}

export interface ContrastClaim {
  claim_id: string;
  domain: "clarity" | "engagement" | "shareability" | "focus" | "hook" | "mixed";
  status: ClaimStatus;
  title: string;
  statement: string;
  recommended_action: string;
  pattern_confidence: number;
  action_confidence: number;
  support_count: number;
  supporting_feature_keys: string[];
  evidence_trace: ClaimEvidenceTrace;
}

export interface NeighborhoodContrast {
  version: "step3.v1";
  generated_at: string;
  fallback_mode: boolean;
  neighborhood_confidence: number;
  normalized_deltas: FeatureRow[];
  claims: ContrastClaim[];
  conflicts: string[];
  summary: {
    top_strengths: string[];
    top_risks: string[];
  };
}

export interface BuildNeighborhoodContrastInput {
  candidateProfile: CandidateProfileCore;
  candidateSignals?: CandidateSignalProfile;
  neighborhood: ComparableNeighborhood;
}

interface FeatureStatsInput {
  over: number[];
  under: number[];
  overWeights: number[];
  underWeights: number[];
  neighborhoodConfidence: number;
}

const EPSILON = 1e-8;
const MIN_SUPPORT_PER_GROUP = 2;
const MIN_ABS_Z_DELTA = 0.35;
const MIN_RELIABILITY = 0.45;

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function round(value: number, decimals = 6): number {
  const factor = 10 ** decimals;
  return Math.round(value * factor) / factor;
}

function parseMaybeDate(value: unknown): Date | null {
  if (typeof value !== "string") {
    return null;
  }
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? null : parsed;
}

function getRecordDate(candidate: NeighborhoodCandidate): Date | null {
  const record = candidate.record as Record<string, unknown>;
  return (
    parseMaybeDate(record.created_at) ??
    parseMaybeDate(record.posted_at) ??
    parseMaybeDate(record.video_created_at) ??
    null
  );
}

function computeFreshnessWeights(values: Array<Date | null>): number[] {
  const dates = values.filter((value): value is Date => value !== null);
  if (dates.length === 0) {
    return values.map(() => 1);
  }

  const maxTs = Math.max(...dates.map((date) => date.getTime()));
  const halfLifeDays = 45;
  return values.map((date) => {
    if (!date) {
      return 0.7;
    }
    const ageDays = Math.max(0, (maxTs - date.getTime()) / (1000 * 60 * 60 * 24));
    return Math.exp((-Math.log(2) * ageDays) / halfLifeDays);
  });
}

function weightedStats(values: number[], weights: number[]): WeightedStats {
  if (values.length === 0) {
    return { mean: 0, variance: 0, count: 0 };
  }
  const normalizedWeights = values.map((_, index) => Math.max(0, weights[index] ?? 1));
  const weightSum = normalizedWeights.reduce((sum, weight) => sum + weight, 0);
  if (weightSum <= EPSILON) {
    const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
    const variance = values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / Math.max(1, values.length - 1);
    return { mean, variance, count: values.length };
  }

  const mean =
    values.reduce((sum, value, index) => sum + value * normalizedWeights[index], 0) / weightSum;

  const variance =
    values.reduce(
      (sum, value, index) => sum + normalizedWeights[index] * (value - mean) ** 2,
      0
    ) / weightSum;

  return { mean, variance, count: values.length };
}

function featureStats(input: FeatureStatsInput): Omit<FeatureRow, "feature_key"> {
  const overStats = weightedStats(input.over, input.overWeights);
  const underStats = weightedStats(input.under, input.underWeights);
  const delta = overStats.mean - underStats.mean;
  const pooledStd = Math.sqrt((overStats.variance + underStats.variance) / 2 + EPSILON);
  const zDelta = delta / pooledStd;
  const direction: Direction =
    Math.abs(zDelta) < 0.1 ? "neutral" : zDelta > 0 ? "over_higher" : "under_higher";

  const sampleScore = clamp(Math.min(overStats.count, underStats.count) / 6, 0, 1);
  const effectScore = clamp(Math.abs(zDelta) / 2.2, 0, 1);
  const reliability = clamp(
    0.45 * sampleScore + 0.35 * effectScore + 0.2 * input.neighborhoodConfidence,
    0,
    1
  );

  return {
    over_mean: round(overStats.mean, 6),
    under_mean: round(underStats.mean, 6),
    delta: round(delta, 6),
    z_delta: round(zDelta, 6),
    direction,
    support_over: overStats.count,
    support_under: underStats.count,
    reliability: round(reliability, 6)
  };
}

function safeNumber(value: unknown, fallback = 0): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function normalizedText(value: string): string {
  return value
    .toLowerCase()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[^\w\s#]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function questionRate(candidate: NeighborhoodCandidate): number {
  const comments = Array.isArray(candidate.record.comments) ? candidate.record.comments : [];
  if (comments.length === 0) {
    return 0;
  }
  const questions = comments.filter((comment) => comment.includes("?")).length;
  return questions / comments.length;
}

function avgCommentLength(candidate: NeighborhoodCandidate): number {
  const comments = Array.isArray(candidate.record.comments) ? candidate.record.comments : [];
  if (comments.length === 0) {
    return 0;
  }
  return comments.reduce((sum, comment) => sum + normalizedText(comment).split(" ").filter(Boolean).length, 0) / comments.length;
}

function likesPerKViews(candidate: NeighborhoodCandidate): number {
  const views = Math.max(1, safeNumber(candidate.record.metrics?.views, 0));
  const likes = safeNumber(candidate.record.metrics?.likes, 0);
  return (likes * 1000) / views;
}

function commentsPerKViews(candidate: NeighborhoodCandidate): number {
  const views = Math.max(1, safeNumber(candidate.record.metrics?.views, 0));
  const comments = safeNumber(candidate.record.metrics?.comments_count, 0);
  return (comments * 1000) / views;
}

function sharesPerKViews(candidate: NeighborhoodCandidate): number {
  const views = Math.max(1, safeNumber(candidate.record.metrics?.views, 0));
  const shares = safeNumber(candidate.record.metrics?.shares, 0);
  return (shares * 1000) / views;
}

function captionWordCount(candidate: NeighborhoodCandidate): number {
  return normalizedText(candidate.record.caption).split(" ").filter(Boolean).length;
}

function hashtagCount(candidate: NeighborhoodCandidate): number {
  return candidate.record.hashtags.length;
}

function buildFeatureRows(
  over: NeighborhoodCandidate[],
  under: NeighborhoodCandidate[],
  neighborhoodConfidence: number
): FeatureRow[] {
  const overDates = over.map(getRecordDate);
  const underDates = under.map(getRecordDate);
  const overWeights = computeFreshnessWeights(overDates);
  const underWeights = computeFreshnessWeights(underDates);

  const rows: FeatureRow[] = [
    {
      feature_key: "engagement_rate",
      ...featureStats({
        over: over.map((item) => item.engagement_rate),
        under: under.map((item) => item.engagement_rate),
        overWeights,
        underWeights,
        neighborhoodConfidence
      })
    },
    {
      feature_key: "residual_log_views",
      ...featureStats({
        over: over.map((item) => item.residual_log_views),
        under: under.map((item) => item.residual_log_views),
        overWeights,
        underWeights,
        neighborhoodConfidence
      })
    },
    {
      feature_key: "comments_per_1k_views",
      ...featureStats({
        over: over.map(commentsPerKViews),
        under: under.map(commentsPerKViews),
        overWeights,
        underWeights,
        neighborhoodConfidence
      })
    },
    {
      feature_key: "shares_per_1k_views",
      ...featureStats({
        over: over.map(sharesPerKViews),
        under: under.map(sharesPerKViews),
        overWeights,
        underWeights,
        neighborhoodConfidence
      })
    },
    {
      feature_key: "likes_per_1k_views",
      ...featureStats({
        over: over.map(likesPerKViews),
        under: under.map(likesPerKViews),
        overWeights,
        underWeights,
        neighborhoodConfidence
      })
    },
    {
      feature_key: "question_comment_rate",
      ...featureStats({
        over: over.map(questionRate),
        under: under.map(questionRate),
        overWeights,
        underWeights,
        neighborhoodConfidence
      })
    },
    {
      feature_key: "avg_comment_length",
      ...featureStats({
        over: over.map(avgCommentLength),
        under: under.map(avgCommentLength),
        overWeights,
        underWeights,
        neighborhoodConfidence
      })
    },
    {
      feature_key: "hashtag_count",
      ...featureStats({
        over: over.map(hashtagCount),
        under: under.map(hashtagCount),
        overWeights,
        underWeights,
        neighborhoodConfidence
      })
    },
    {
      feature_key: "caption_word_count",
      ...featureStats({
        over: over.map(captionWordCount),
        under: under.map(captionWordCount),
        overWeights,
        underWeights,
        neighborhoodConfidence
      })
    }
  ];

  return rows.sort((a, b) => Math.abs(b.z_delta) - Math.abs(a.z_delta));
}

function buildConflictFlags(rows: FeatureRow[]): string[] {
  const conflicts: string[] = [];
  const byKey = Object.fromEntries(rows.map((row) => [row.feature_key, row]));

  const clarityQuestion = byKey.question_comment_rate;
  const clarityLength = byKey.avg_comment_length;
  if (
    clarityQuestion &&
    clarityLength &&
    Math.abs(clarityQuestion.z_delta) > 0.5 &&
    Math.abs(clarityLength.z_delta) > 0.5 &&
    Math.sign(clarityQuestion.z_delta) !== Math.sign(clarityLength.z_delta)
  ) {
    conflicts.push("clarity_signal_conflict");
  }

  const comments = byKey.comments_per_1k_views;
  const shares = byKey.shares_per_1k_views;
  if (
    comments &&
    shares &&
    Math.abs(comments.z_delta) > 0.5 &&
    Math.abs(shares.z_delta) > 0.5 &&
    Math.sign(comments.z_delta) !== Math.sign(shares.z_delta)
  ) {
    conflicts.push("engagement_signal_conflict");
  }

  return conflicts;
}

function resolveFeature(rows: FeatureRow[], key: string): FeatureRow | undefined {
  return rows.find((row) => row.feature_key === key);
}

function canUseForClaim(row: FeatureRow | undefined): boolean {
  if (!row) {
    return false;
  }
  return (
    row.support_over >= MIN_SUPPORT_PER_GROUP &&
    row.support_under >= MIN_SUPPORT_PER_GROUP &&
    Math.abs(row.z_delta) >= MIN_ABS_Z_DELTA &&
    row.reliability >= MIN_RELIABILITY
  );
}

function buildClaim(
  params: {
    claim_id: string;
    domain: ContrastClaim["domain"];
    title: string;
    statement: string;
    action: string;
    features: FeatureRow[];
    over: NeighborhoodCandidate[];
    under: NeighborhoodCandidate[];
    status?: ClaimStatus;
  }
): ContrastClaim {
  const featureReliability =
    params.features.length === 0
      ? 0
      : params.features.reduce((sum, feature) => sum + feature.reliability, 0) / params.features.length;
  const supportCount = params.features.length === 0 ? 0 : Math.min(...params.features.map((feature) => Math.min(feature.support_over, feature.support_under)));
  const patternConfidence = round(clamp(featureReliability * (supportCount / 6), 0, 1), 4);
  const actionConfidence = round(clamp(patternConfidence * 0.92, 0, 1), 4);

  return {
    claim_id: params.claim_id,
    domain: params.domain,
    status: params.status ?? "active",
    title: params.title,
    statement: params.statement,
    recommended_action: params.action,
    pattern_confidence: patternConfidence,
    action_confidence: actionConfidence,
    support_count: supportCount,
    supporting_feature_keys: params.features.map((feature) => feature.feature_key),
    evidence_trace: {
      candidate_keys_over: params.over.slice(0, 6).map((item) => item.candidate_key),
      candidate_keys_under: params.under.slice(0, 6).map((item) => item.candidate_key),
      top_feature_keys: params.features.map((feature) => feature.feature_key)
    }
  };
}

function buildClaims(
  rows: FeatureRow[],
  input: BuildNeighborhoodContrastInput
): ContrastClaim[] {
  const over = input.neighborhood.similar_overperformers;
  const under = input.neighborhood.similar_underperformers;
  const claims: ContrastClaim[] = [];

  const questionRate = resolveFeature(rows, "question_comment_rate");
  const commentsPerView = resolveFeature(rows, "comments_per_1k_views");
  if (canUseForClaim(questionRate) && canUseForClaim(commentsPerView)) {
    const clarityImproves = (questionRate?.z_delta ?? 0) < 0 && (commentsPerView?.z_delta ?? 0) > 0;
    claims.push(
      buildClaim({
        claim_id: "claim-clarity",
        domain: "clarity",
        title: clarityImproves
          ? "Overperformers trigger fewer confusion questions."
          : "Comment friction may be suppressing performance.",
        statement: clarityImproves
          ? "Comparable overperformers receive more comments per view while having fewer question-heavy comment threads."
          : "Comparable underperformers show elevated question-heavy comments relative to their performance.",
        action:
          "Clarify the value proposition in the first 2 seconds and add explicit context text before the main action starts.",
        features: [questionRate!, commentsPerView!],
        over,
        under
      })
    );
  }

  const sharesPerView = resolveFeature(rows, "shares_per_1k_views");
  const likesPerView = resolveFeature(rows, "likes_per_1k_views");
  if (canUseForClaim(sharesPerView) && canUseForClaim(likesPerView)) {
    claims.push(
      buildClaim({
        claim_id: "claim-shareability",
        domain: "shareability",
        title: "Shareability differentiates top comparables.",
        statement:
          "Overperformers exhibit higher shares per 1k views and stronger likes per 1k views than underperformers in the same neighborhood.",
        action:
          "Add a copy-ready takeaway and an explicit share/save cue in the closing beat.",
        features: [sharesPerView!, likesPerView!],
        over,
        under
      })
    );
  }

  const hashtagDelta = resolveFeature(rows, "hashtag_count");
  const captionDelta = resolveFeature(rows, "caption_word_count");
  if (canUseForClaim(hashtagDelta) && canUseForClaim(captionDelta)) {
    claims.push(
      buildClaim({
        claim_id: "claim-focus",
        domain: "focus",
        title: "Message focus is a measurable separator.",
        statement:
          "The strongest comparables cluster around tighter hashtag usage and more consistent caption length than weaker peers.",
        action:
          "Keep 3-5 targeted hashtags and trim non-essential caption fragments to reduce ambiguity.",
        features: [hashtagDelta!, captionDelta!],
        over,
        under
      })
    );
  }

  if (claims.length === 0) {
    claims.push(
      buildClaim({
        claim_id: "claim-fallback",
        domain: "mixed",
        status: "insufficient",
        title: "Evidence is limited for high-confidence contrast claims.",
        statement:
          "Current neighborhood does not provide enough consistent contrast signals for strong recommendations.",
        action:
          "Increase candidate coverage or provide richer transcript/OCR hints before applying aggressive content changes.",
        features: rows.slice(0, 1),
        over,
        under
      })
    );
  }

  return claims
    .sort((a, b) => b.pattern_confidence - a.pattern_confidence)
    .slice(0, 4);
}

function buildSummary(claims: ContrastClaim[]): NeighborhoodContrast["summary"] {
  const strengths = claims
    .filter((claim) => claim.status === "active")
    .slice(0, 2)
    .map((claim) => claim.title);
  const risks = claims
    .filter((claim) => claim.status !== "active")
    .slice(0, 2)
    .map((claim) => claim.title);
  return {
    top_strengths: strengths,
    top_risks: risks
  };
}

export function buildNeighborhoodContrast(
  input: BuildNeighborhoodContrastInput
): NeighborhoodContrast {
  const neighborhood = input.neighborhood;
  const over = neighborhood.similar_overperformers;
  const under = neighborhood.similar_underperformers;
  const fallbackMode =
    over.length < MIN_SUPPORT_PER_GROUP ||
    under.length < MIN_SUPPORT_PER_GROUP ||
    neighborhood.confidence.overall < 0.35;

  const rows =
    over.length > 0 && under.length > 0
      ? buildFeatureRows(over, under, neighborhood.confidence.overall)
      : [];

  const conflicts = buildConflictFlags(rows);
  const claims = buildClaims(rows, input).map((claim) => {
    if (fallbackMode && claim.status === "active") {
      return { ...claim, status: "mixed" as ClaimStatus };
    }
    return claim;
  });

  return {
    version: "step3.v1",
    generated_at: new Date().toISOString(),
    fallback_mode: fallbackMode,
    neighborhood_confidence: round(neighborhood.confidence.overall, 4),
    normalized_deltas: rows,
    claims,
    conflicts,
    summary: buildSummary(claims)
  };
}

