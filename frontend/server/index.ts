import cors from "cors";
import express from "express";
import OpenAI from "openai";
import {
  DEEPSEEK_API_KEY,
  DEEPSEEK_BASE_URL,
  DEEPSEEK_ENABLED,
  DEEPSEEK_MODEL,
  RECOMMENDER_BUDGET_BUFFER_MS,
  RECOMMENDER_BUDGET_EXPLAINABILITY_MS,
  RECOMMENDER_BUDGET_NETWORK_MS,
  RECOMMENDER_BUDGET_PARSE_MS,
  RECOMMENDER_BUDGET_RANKING_MS,
  RECOMMENDER_BUDGET_RETRIEVAL_MS,
  RECOMMENDER_BUDGET_SERIALIZE_MS,
  RECOMMENDER_BREAKER_CONSECUTIVE_FAILURES,
  RECOMMENDER_BREAKER_ERROR_RATE,
  RECOMMENDER_BREAKER_HALF_OPEN_PROBES,
  RECOMMENDER_BREAKER_HALF_OPEN_SUCCESS,
  RECOMMENDER_BREAKER_MIN_REQUESTS,
  RECOMMENDER_BREAKER_OPEN_MS,
  RECOMMENDER_BREAKER_WINDOW_MS,
  RECOMMENDER_COMPAT_CACHE_TTL_MS,
  RECOMMENDER_COMPAT_CHECK_INTERVAL_MS,
  RECOMMENDER_ENABLED,
  RECOMMENDER_EXPERIMENT_CONTROL_BUNDLE_ID,
  RECOMMENDER_EXPERIMENT_DEFAULT_ID,
  RECOMMENDER_EXPERIMENT_SALT,
  RECOMMENDER_EXPERIMENT_TREATMENT_BUNDLE_ID,
  RECOMMENDER_EXPERIMENT_TREATMENT_RATIO,
  RECOMMENDER_FEEDBACK_DB_URL,
  RECOMMENDER_FEEDBACK_ENABLED,
  RECOMMENDER_FALLBACK_BUNDLE_DIR,
  RECOMMENDER_FALLBACK_CACHE_TTL_MS,
  SERVER_PORT
} from "./config";
import { getSeedVideo } from "../src/services/data/selectDemoSlices";
import type { DemoVideoRecord } from "../src/services/data/types";
import type { ReportOutput } from "../src/features/report/types";
import { loadDatasetFromFile } from "./dataset/loadDatasetFromFile";
import { buildLocalBaselineReport } from "./fallback/buildLocalBaselineReport";
import { buildLocalChatAnswer } from "./fallback/buildLocalChatAnswer";
import {
  buildSignalProfileFallback,
  type CandidateSignalProfile
} from "./fallback/signalProfile";
import { enrichComparableMedia } from "./formatters/enrichComparableMedia";
import { normalizeReportOutput } from "./formatters/normalizeReportOutput";
import { buildChatPrompt } from "./prompts/buildChatPrompt";
import { buildReportPrompt } from "./prompts/buildReportPrompt";
import { applyNarrativePolish, validateNarrativePolish } from "./report/polish";
import { buildReportReasoning } from "./report/reasoning";
import { HARD_CODED_EXTRACTED_KEYWORDS } from "./prompts/seedVideoContext";
import { parseGenerateReportRequest } from "./validation/parseGenerateReportRequest";
import { parseReportFeedbackRequest } from "./validation/parseReportFeedbackRequest";
import { parseRecommendationsRequest } from "./validation/parseRecommendationsRequest";
import { validateReportOutput } from "./validation/validateReportOutput";
import {
  requestFabricSignals,
  requestCompatibility,
  requestRecommendations,
  type FabricExtractResponsePayload,
  type RecommenderCandidatePayload,
  type RecommenderRequestPayload,
  type RecommenderResult
} from "./recommender/client";
import { FallbackBundleStore } from "./recommender/fallbackBundle";
import {
  buildExperimentAssignment,
  buildExperimentRoutePolicy
} from "./recommender/experimentation";
import { createUuidV7 } from "./recommender/requestId";
import {
  RecommenderCircuitBreakers,
  createStageLatencyTracker
} from "./recommender/resilience";
import { classifyRecommenderFailure } from "./recommender/fallbackReason";
import { buildRoutingEnvelope, type RoutingRequestInput } from "./recommender/routing";
import {
  FeedbackEventStore,
  buildRequestHash,
  type CandidateEventRecord,
  type ServedOutputRecord
} from "./feedback/store";
import {
  LabelingSessionStore,
  type LabelingReviewLabel
} from "./labeling/store";

interface ChatRequestBody {
  report?: unknown;
  question?: string;
}

const TIKTOK_OEMBED_URL = "https://www.tiktok.com/oembed?url=";
const THUMBNAIL_FETCH_TIMEOUT_MS = 7000;
const ALLOWED_TIKTOK_VIDEO_HOSTS = new Set(["tiktok.com", "www.tiktok.com"]);

const app = express();

const deepSeekClient = DEEPSEEK_ENABLED
  ? new OpenAI({
      apiKey: DEEPSEEK_API_KEY,
      baseURL: DEEPSEEK_BASE_URL
    })
  : null;

app.use(
  cors({
    origin: [
      "http://localhost:5173",
      "http://127.0.0.1:5173",
      "http://localhost:4173",
      "http://127.0.0.1:4173"
    ]
  })
);
app.use(express.json({ limit: "1mb" }));

const recommenderBreakers = new RecommenderCircuitBreakers({
  minRequests: RECOMMENDER_BREAKER_MIN_REQUESTS,
  errorRateThreshold: RECOMMENDER_BREAKER_ERROR_RATE,
  consecutiveFailureThreshold: RECOMMENDER_BREAKER_CONSECUTIVE_FAILURES,
  windowMs: RECOMMENDER_BREAKER_WINDOW_MS,
  openMs: RECOMMENDER_BREAKER_OPEN_MS,
  halfOpenMaxProbes: RECOMMENDER_BREAKER_HALF_OPEN_PROBES,
  halfOpenSuccessToClose: RECOMMENDER_BREAKER_HALF_OPEN_SUCCESS
});
const fallbackBundleStore = new FallbackBundleStore(
  RECOMMENDER_FALLBACK_BUNDLE_DIR,
  RECOMMENDER_FALLBACK_CACHE_TTL_MS
);
const feedbackStore = new FeedbackEventStore({
  enabled: RECOMMENDER_FEEDBACK_ENABLED,
  dbUrl: RECOMMENDER_FEEDBACK_DB_URL
});
const labelingSessionStore = new LabelingSessionStore();

interface CompatibilityCacheRecord {
  checked_at: number;
  ok: boolean;
  payload?: Record<string, unknown>;
  reason?: string;
}

const compatibilityCache = new Map<string, CompatibilityCacheRecord>();
const gatewayMetrics: {
  stageLatencyMs: Record<string, number[]>;
  breakerTransitions: Record<string, number>;
  fallbackByReason: Record<string, number>;
  compatibilityMismatchCount: number;
  fallbackBundleHitCount: number;
} = {
  stageLatencyMs: {},
  breakerTransitions: {},
  fallbackByReason: {},
  compatibilityMismatchCount: 0,
  fallbackBundleHitCount: 0
};

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function round(value: number, decimals = 4): number {
  const factor = 10 ** decimals;
  return Math.round(value * factor) / factor;
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function normalizeLabelingReviewLabel(value: unknown): LabelingReviewLabel | null {
  if (value === null) {
    return null;
  }
  if (typeof value !== "string") {
    return null;
  }
  const normalized = value.trim().toLowerCase();
  if (
    normalized === "saved" ||
    normalized === "relevant" ||
    normalized === "not_relevant"
  ) {
    return normalized;
  }
  return null;
}

function recordStageLatency(stage: string, valueMs: number): void {
  const bucket = gatewayMetrics.stageLatencyMs[stage] ?? [];
  bucket.push(valueMs);
  if (bucket.length > 1024) {
    bucket.splice(0, bucket.length - 1024);
  }
  gatewayMetrics.stageLatencyMs[stage] = bucket;
}

function trackFallbackReason(reason: string): void {
  const key = reason.trim() || "unknown";
  gatewayMetrics.fallbackByReason[key] = (gatewayMetrics.fallbackByReason[key] ?? 0) + 1;
}

function routeKey(endpoint: string, objectiveEffective: string): string {
  return `${endpoint}::${objectiveEffective.trim().toLowerCase() || "engagement"}`;
}

function buildGatewayBudgets() {
  return {
    parse: RECOMMENDER_BUDGET_PARSE_MS,
    network: RECOMMENDER_BUDGET_NETWORK_MS,
    retrieval: RECOMMENDER_BUDGET_RETRIEVAL_MS,
    ranking: RECOMMENDER_BUDGET_RANKING_MS,
    explainability: RECOMMENDER_BUDGET_EXPLAINABILITY_MS,
    serialize: RECOMMENDER_BUDGET_SERIALIZE_MS,
    buffer: RECOMMENDER_BUDGET_BUFFER_MS
  };
}

function summarizeLatency(values: number[]): { p50_ms: number; p95_ms: number; count: number } {
  if (values.length === 0) {
    return { p50_ms: 0, p95_ms: 0, count: 0 };
  }
  const ordered = [...values].sort((a, b) => a - b);
  const p50 = ordered[Math.floor((ordered.length - 1) * 0.5)] ?? 0;
  const p95 = ordered[Math.floor((ordered.length - 1) * 0.95)] ?? 0;
  return { p50_ms: round(p50, 4), p95_ms: round(p95, 4), count: ordered.length };
}

function removeEmoji(value: string): string {
  return value.replace(/\p{Extended_Pictographic}/gu, "").trim();
}

function sanitizeUnknownStrings<T>(value: T): T {
  if (typeof value === "string") {
    return removeEmoji(value) as T;
  }

  if (Array.isArray(value)) {
    return value.map((item) => sanitizeUnknownStrings(item)) as T;
  }

  if (value && typeof value === "object") {
    const source = value as Record<string, unknown>;
    const entries = Object.entries(source).map(([key, innerValue]) => [
      key,
      sanitizeUnknownStrings(innerValue)
    ]);
    return Object.fromEntries(entries) as T;
  }

  return value;
}

function dedupeByVideoId(records: DemoVideoRecord[]): DemoVideoRecord[] {
  const seen = new Set<string>();
  const uniqueRecords: DemoVideoRecord[] = [];

  for (const record of records) {
    const uniqueKey =
      typeof record.video_url === "string" && record.video_url.trim()
        ? record.video_url.trim()
        : `${record.video_id}-${record.caption.slice(0, 32)}`;

    if (seen.has(uniqueKey)) {
      continue;
    }
    seen.add(uniqueKey);
    uniqueRecords.push(record);
  }

  return uniqueRecords;
}

function getCombinedCandidates(
  dataset: DemoVideoRecord[],
  seedVideoId: string
): DemoVideoRecord[] {
  const filtered = dataset.filter((record) => record.video_id !== seedVideoId);
  return dedupeByVideoId(filtered);
}

function mapObjectiveForRecommender(value: string | undefined): string {
  const normalized = typeof value === "string" ? value.trim().toLowerCase() : "";
  if (!normalized) {
    return "engagement";
  }
  if (normalized === "community") {
    return "community";
  }
  if (normalized === "reach" || normalized === "engagement" || normalized === "conversion") {
    return normalized;
  }
  return "engagement";
}

function resolveTrafficMeta(
  traffic:
    | {
        class?: "production" | "synthetic";
        injected_failure?: boolean;
      }
    | undefined
): {
  trafficClass: "production" | "synthetic";
  isSynthetic: boolean;
  injectedFailure: boolean;
} {
  const trafficClass = traffic?.class === "synthetic" ? "synthetic" : "production";
  const injectedFailure = Boolean(traffic?.injected_failure);
  return {
    trafficClass,
    isSynthetic: trafficClass === "synthetic",
    injectedFailure
  };
}

function buildExperimentUnitKey(params: {
  seedVideoId?: string;
  description: string;
  hashtags: string[];
  mentions: string[];
  objective?: string;
  locale?: string;
  contentType?: string;
}): string {
  const seed = (params.seedVideoId || "").trim();
  if (seed) {
    return `seed_video_id::${seed.toLowerCase()}`;
  }
  return JSON.stringify(
    {
      description: params.description.trim().toLowerCase().slice(0, 1000),
      hashtags: params.hashtags.map((item) => item.trim().toLowerCase()).sort(),
      mentions: params.mentions.map((item) => item.trim().toLowerCase()).sort(),
      objective: mapObjectiveForRecommender(params.objective),
      locale: (params.locale || "").trim().toLowerCase(),
      content_type: (params.contentType || "").trim().toLowerCase()
    },
    ["description", "hashtags", "mentions", "objective", "locale", "content_type"]
  );
}

function toCandidateEventsFromResponse(params: {
  requestId: string;
  payload: Record<string, unknown>;
}): CandidateEventRecord[] {
  const out: CandidateEventRecord[] = [];
  const debug = params.payload.debug as Record<string, unknown> | undefined;
  const retrieved = Array.isArray(debug?.retrieved_universe)
    ? (debug?.retrieved_universe as Array<Record<string, unknown>>)
    : [];
  for (const item of retrieved) {
    const candidateId = String(item.candidate_id || "").trim();
    if (!candidateId) {
      continue;
    }
    out.push({
      requestId: params.requestId,
      candidateId,
      stage: "retrieved_universe",
      retrievedRank:
        typeof item.retrieved_rank === "number" ? Math.round(item.retrieved_rank) : null,
      finalRank: null,
      selected: false,
      retrievalBranchScores:
        (item.retrieval_branch_scores as Record<string, number> | undefined) || {},
      similarity: (item.similarity as Record<string, number> | undefined) || {},
      scoreRaw: null,
      scoreCalibrated: null,
      policyAdjustedScore: null,
      policyTrace: null,
      calibrationTrace: null,
      explainabilityAvailable: false
    });
  }

  const rankingUniverse = Array.isArray(debug?.ranking_universe)
    ? (debug?.ranking_universe as Array<Record<string, unknown>>)
    : [];
  for (const item of rankingUniverse) {
    const candidateId = String(item.candidate_id || "").trim();
    if (!candidateId) {
      continue;
    }
    const policyTrace =
      (item.policy_trace as Record<string, unknown> | undefined) || {};
    const portfolioTrace =
      (item.portfolio_trace as Record<string, unknown> | undefined) || null;
    out.push({
      requestId: params.requestId,
      candidateId,
      stage: "ranked_universe",
      retrievedRank:
        typeof item.retrieved_rank === "number" ? Math.round(item.retrieved_rank) : null,
      finalRank: typeof item.final_rank === "number" ? Math.round(item.final_rank) : null,
      selected: Boolean(item.selected),
      retrievalBranchScores:
        (item.retrieval_branch_scores as Record<string, number> | undefined) || {},
      similarity: (item.similarity as Record<string, number> | undefined) || {},
      scoreRaw: typeof item.score_raw === "number" ? item.score_raw : null,
      scoreCalibrated: typeof item.score_calibrated === "number" ? item.score_calibrated : null,
      policyAdjustedScore:
        typeof item.policy_adjusted_score === "number" ? item.policy_adjusted_score : null,
      policyTrace:
        portfolioTrace && Object.keys(portfolioTrace).length > 0
          ? { ...policyTrace, portfolio_trace: portfolioTrace }
          : policyTrace,
      calibrationTrace: (item.calibration_trace as Record<string, unknown> | undefined) || null,
      explainabilityAvailable: Boolean(item.explainability_available)
    });
  }
  return out;
}

function toServedOutputEvents(params: {
  requestId: string;
  payload: Record<string, unknown>;
}): ServedOutputRecord[] {
  const items = Array.isArray(params.payload.items)
    ? (params.payload.items as Array<Record<string, unknown>>)
    : [];
  const out: ServedOutputRecord[] = [];
  for (const item of items) {
    const candidateId = String(item.candidate_id || "").trim();
    const rank = Number(item.rank);
    const score = Number(item.score);
    if (!candidateId || !Number.isFinite(rank) || !Number.isFinite(score)) {
      continue;
    }
    out.push({
      requestId: params.requestId,
      candidateId,
      rank: Math.round(rank),
      score,
      metadata: {
        served_rank: Math.round(rank),
        visible_position: Math.round(rank),
        was_exposed: true,
        author_id:
          typeof item.author_id === "string" && item.author_id.trim()
            ? item.author_id.trim()
            : null,
        topic_key:
          typeof item.topic_key === "string" && item.topic_key.trim()
            ? item.topic_key.trim()
            : null,
        content_type:
          typeof item.content_type === "string" && item.content_type.trim()
            ? item.content_type.trim()
            : null,
        language:
          typeof item.language === "string" && item.language.trim()
            ? item.language.trim()
            : null,
        locale:
          typeof item.locale === "string" && item.locale.trim()
            ? item.locale.trim()
            : null,
        hashtags: Array.isArray(item.hashtags) ? item.hashtags : [],
        keywords: Array.isArray(item.keywords) ? item.keywords : [],
        retrieval_branch_scores:
          (item.retrieval_branch_scores as Record<string, unknown> | undefined) || {},
        similarity: (item.similarity as Record<string, unknown> | undefined) || {},
        selected_ranker_id: item.selected_ranker_id,
        confidence: item.confidence,
        user_affinity_score: item.user_affinity_score,
        creator_retrieval_score: item.creator_retrieval_score,
        user_affinity_trace:
          (item.user_affinity_trace as Record<string, unknown> | undefined) || {},
        creator_retrieval_trace:
          (item.creator_retrieval_trace as Record<string, unknown> | undefined) || {},
        portfolio_trace: (item.portfolio_trace as Record<string, unknown> | undefined) || {},
        portfolio_mode:
          typeof params.payload.portfolio_mode === "boolean"
            ? params.payload.portfolio_mode
            : false,
        portfolio_metadata:
          (params.payload.portfolio_metadata as Record<string, unknown> | undefined) || {},
        retrieval_personalization_metadata:
          (params.payload.retrieval_personalization_metadata as
            | Record<string, unknown>
            | undefined) || {},
        objective_model: (item.trace as Record<string, unknown> | undefined)?.objective_model ?? null,
        score_components:
          (item.score_components as Record<string, unknown> | undefined) || {},
        support_level: item.support_level ?? null,
        support_score: item.support_score ?? null,
        ranking_reasons: Array.isArray(item.ranking_reasons) ? item.ranking_reasons : []
      }
    });
  }
  return out;
}

async function buildFallbackItemsFromBundle(params: {
  objective: string | undefined;
  candidates: DemoVideoRecord[];
  topK: number;
}): Promise<
  | {
      bundleVersion: string;
      generatedAt: string;
      items: Array<{
        candidate_id: string;
        rank: number;
        score: number;
        similarity: { sparse: number; dense: number; fused: number };
        trace: { objective_model: string; ranker_backend: string };
        comment_trace: Record<string, unknown>;
      }>;
    }
  | null
> {
  const objective = mapObjectiveForRecommender(params.objective);
  const objectiveEffective = objective === "community" ? "engagement" : objective;
  const bundle = await fallbackBundleStore.get(objectiveEffective);
  if (!bundle || bundle.items.length === 0) {
    return null;
  }
  const candidateById = new Map(params.candidates.map((item) => [item.video_id, item]));
  const selected = bundle.items
    .filter((item) => candidateById.has(item.candidate_id))
    .slice(0, Math.max(1, params.topK));
  if (selected.length === 0) {
    return null;
  }
  gatewayMetrics.fallbackBundleHitCount += 1;
  return {
    bundleVersion: bundle.version ?? "fallback_bundle.v1",
    generatedAt: bundle.generated_at,
    items: selected.map((item, index) => ({
      candidate_id: item.candidate_id,
      rank: index + 1,
      score: Number(item.score.toFixed(6)),
      similarity: { sparse: 0, dense: 0, fused: 0 },
      trace: {
        objective_model: objectiveEffective,
        ranker_backend: "fallback-bundle"
      },
      comment_trace: buildCommentIntelligenceHint(candidateById.get(item.candidate_id)?.comments, {
        caption: candidateById.get(item.candidate_id)?.caption,
        hashtags: candidateById.get(item.candidate_id)?.hashtags,
        keywords: candidateById.get(item.candidate_id)?.keywords,
        searchQuery: candidateById.get(item.candidate_id)?.search_query
      })
    }))
  };
}

function inferCandidateAsOf(record: DemoVideoRecord, fallbackIso: string): string {
  const postedAt =
    typeof record.posted_at === "string" && record.posted_at.trim()
      ? new Date(record.posted_at)
      : null;
  if (postedAt && !Number.isNaN(postedAt.getTime())) {
    return postedAt.toISOString();
  }
  return fallbackIso;
}

function tokenizeComment(value: string): string[] {
  return value
    .toLowerCase()
    .replace(/[^\w\s?]/g, " ")
    .split(/\s+/)
    .filter((token) => token.length >= 2);
}

function tokenizeContent(value: string): string[] {
  const stopwords = new Set([
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "with",
    "you",
    "your"
  ]);
  return value
    .toLowerCase()
    .replace(/[^\w\s#]/g, " ")
    .split(/\s+/)
    .map((token) => token.replace(/^#/, ""))
    .filter((token) => token.length >= 3 && !stopwords.has(token));
}

function buildCommentIntelligenceHint(
  comments: string[] | undefined,
  context?: {
    caption?: string;
    hashtags?: string[];
    keywords?: string[];
    searchQuery?: string;
  }
): Record<string, unknown> {
  const pool = Array.isArray(comments) ? comments.filter((value) => typeof value === "string") : [];
  const contentSignals = [
    context?.caption ?? "",
    ...(Array.isArray(context?.hashtags) ? context?.hashtags : []),
    ...(Array.isArray(context?.keywords) ? context?.keywords : []),
    context?.searchQuery ?? ""
  ];
  const valueProps = new Set(tokenizeContent(contentSignals.join(" ")));
  const artifactTokens = new Set([
    "algorithm",
    "camera",
    "editing",
    "fyp",
    "lighting",
    "music",
    "sound",
    "thumbnail"
  ]);
  if (pool.length === 0) {
    return {
      source: "node_dataset_hint",
      available: false,
      taxonomy_version: "comment_taxonomy.v2.0.0",
      dominant_intents: [],
      confusion_index: 0,
      help_seeking_index: 0,
      sentiment_volatility: 0,
      sentiment_shift_early_late: 0,
      reply_depth_max: 0,
      reply_branch_factor: 0,
      reply_ratio: 0,
      root_thread_concentration: 0,
      alignment_score: 0,
      value_prop_coverage: 0,
      on_topic_ratio: 0,
      artifact_drift_ratio: 0,
      alignment_shift_early_late: 0,
      alignment_confidence: 0,
      alignment_method_version: "alignment.v2.hybrid.lexical+embedding.node_hint",
      confidence: 0,
      missingness_flags: ["not_available"]
    };
  }

  const questionCount = pool.filter((value) => value.includes("?")).length;
  const confusionCount = pool.filter((value) => /(confus|which|what do you mean|unclear|dont understand|don't understand)/i.test(value)).length;
  const helpCount = pool.filter((value) => /(how|can you|please|steps|help)/i.test(value)).length;
  const praiseCount = pool.filter((value) => /(great|love|amazing|awesome|helpful|fire)/i.test(value)).length;
  const complaintCount = pool.filter((value) => /(bad|broken|doesn't work|doesnt work|issue|problem)/i.test(value)).length;
  const skepticismCount = pool.filter((value) => /(fake|cap|really\?|sure\?)/i.test(value)).length;
  const purchaseCount = pool.filter((value) => /(price|cost|buy|link|where to get|worth it)/i.test(value)).length;
  const saveCount = pool.filter((value) => /(save|saved|bookmark)/i.test(value)).length;

  const polarity = pool.map((value) => {
    const tokens = tokenizeComment(value);
    const positive = tokens.filter((token) => ["great", "love", "awesome", "amazing", "helpful", "perfect", "good"].includes(token)).length;
    const negative = tokens.filter((token) => ["bad", "broken", "issue", "problem", "hate", "wrong", "terrible"].includes(token)).length;
    return (positive - negative) / Math.max(1, tokens.length);
  });
  const mean = polarity.reduce((sum, value) => sum + value, 0) / Math.max(1, polarity.length);
  const variance =
    polarity.reduce((sum, value) => sum + (value - mean) ** 2, 0) / Math.max(1, polarity.length - 1);
  const dominant = [
    ["confusion", confusionCount],
    ["help_seeking", helpCount],
    ["purchase_intent", purchaseCount],
    ["save_intent", saveCount],
    ["praise", praiseCount],
    ["complaint", complaintCount],
    ["skepticism", skepticismCount],
    ["off_topic", Math.max(0, pool.length - (confusionCount + helpCount + purchaseCount + saveCount + praiseCount + complaintCount + skepticismCount))]
  ]
    .sort((a, b) => b[1] - a[1])
    .filter((entry) => entry[1] > 0)
    .slice(0, 3)
    .map((entry) => entry[0]);
  const commentTokenSets = pool.map((value) => new Set(tokenizeContent(value)));
  const coveredProps = new Set<string>();
  let onTopic = 0;
  let artifactDrift = 0;
  for (const tokens of commentTokenSets) {
    const overlap = [...tokens].filter((token) => valueProps.has(token));
    for (const token of overlap) {
      coveredProps.add(token);
    }
    if (overlap.length > 0) {
      onTopic += 1;
    }
    const artifactCount = [...tokens].filter((token) => artifactTokens.has(token)).length;
    if (artifactCount > 0 && overlap.length === 0) {
      artifactDrift += 1;
    }
  }
  const coverage = valueProps.size > 0 ? coveredProps.size / valueProps.size : 0;
  const onTopicRatio = onTopic / Math.max(1, pool.length);
  const artifactDriftRatio = artifactDrift / Math.max(1, pool.length);
  const alignmentScore = clamp(0.65 * ((0.6 * coverage) + (0.4 * onTopicRatio)) - 0.25 * artifactDriftRatio, 0, 1);
  const alignmentConfidence = clamp(
    0.2 + Math.min(0.5, pool.length * 0.06) + (valueProps.size > 0 ? 0.15 : 0),
    0.2,
    0.95
  );
  const alignmentMissingness: string[] = [];
  if (valueProps.size === 0) {
    alignmentMissingness.push("alignment_no_content_signals");
  }

  return {
    source: "node_dataset_hint",
    available: true,
    taxonomy_version: "comment_taxonomy.v2.0.0",
    dominant_intents: dominant.length > 0 ? dominant : ["off_topic"],
    confusion_index: round(clamp(confusionCount / Math.max(1, pool.length), 0, 1), 6),
    help_seeking_index: round(clamp(helpCount / Math.max(1, pool.length), 0, 1), 6),
    sentiment_volatility: round(Math.sqrt(Math.max(0, variance)), 6),
    sentiment_shift_early_late: 0,
    reply_depth_max: 0,
    reply_branch_factor: 0,
    reply_ratio: round(clamp(questionCount / Math.max(1, pool.length), 0, 1), 6),
    root_thread_concentration: 1,
    alignment_score: round(alignmentScore, 6),
    value_prop_coverage: round(clamp(coverage, 0, 1), 6),
    on_topic_ratio: round(clamp(onTopicRatio, 0, 1), 6),
    artifact_drift_ratio: round(clamp(artifactDriftRatio, 0, 1), 6),
    alignment_shift_early_late: 0,
    alignment_confidence: round(alignmentConfidence, 6),
    alignment_method_version: "alignment.v2.hybrid.lexical+embedding.node_hint",
    confidence: round(clamp(0.2 + Math.min(0.7, pool.length * 0.08), 0.2, 0.95), 6),
    missingness_flags:
      pool.length < 3 ? ["low_quality", ...alignmentMissingness] : alignmentMissingness
  };
}

function buildRecommenderCandidates(
  records: DemoVideoRecord[],
  asOfTimeIso: string
): RecommenderCandidatePayload[] {
  return records.map((record) => {
    const authorId =
      typeof record.author === "object" && record.author
        ? typeof record.author.author_id === "string"
          ? record.author.author_id
          : typeof record.author.username === "string"
            ? record.author.username
            : "unknown"
        : typeof record.author === "string"
          ? record.author
          : "unknown";

    return {
      candidate_id: record.video_id,
      caption: record.caption,
      text: [record.caption, ...record.hashtags, ...record.keywords].join(" "),
      hashtags: record.hashtags,
      keywords: record.keywords,
      topic_key: typeof record.search_query === "string" ? record.search_query : undefined,
      author_id: authorId,
      as_of_time: inferCandidateAsOf(record, asOfTimeIso),
      posted_at:
        typeof record.posted_at === "string" && record.posted_at.trim()
          ? new Date(record.posted_at).toISOString()
          : undefined,
      language:
        typeof record.language === "string" && record.language.trim()
          ? record.language.trim().toLowerCase()
          : undefined,
      locale:
        typeof record.locale === "string" && record.locale.trim()
          ? record.locale.trim().toLowerCase()
          : undefined,
      content_type:
        typeof record.content_type === "string" && record.content_type.trim()
          ? record.content_type.trim().toLowerCase()
          : undefined,
      signal_hints: {
        comment_intelligence: buildCommentIntelligenceHint(record.comments, {
          caption: record.caption,
          hashtags: record.hashtags,
          keywords: record.keywords,
          searchQuery: typeof record.search_query === "string" ? record.search_query : undefined
        })
      }
    };
  });
}

function applyRecommenderOrder(
  candidates: DemoVideoRecord[],
  recommenderResult: RecommenderResult
): DemoVideoRecord[] {
  if (!recommenderResult.ok || recommenderResult.payload.items.length === 0) {
    return candidates;
  }
  const rankById = new Map(
    recommenderResult.payload.items.map((item, index) => [item.candidate_id, index + 1])
  );
  return [...candidates].sort((a, b) => {
    const rankA = rankById.get(a.video_id) ?? Number.MAX_SAFE_INTEGER;
    const rankB = rankById.get(b.video_id) ?? Number.MAX_SAFE_INTEGER;
    if (rankA === rankB) {
      return b.metrics.views - a.metrics.views;
    }
    return rankA - rankB;
  });
}

function summarizeCommentTrace(recommenderResult: RecommenderResult): string | undefined {
  if (!recommenderResult.ok) {
    return undefined;
  }
  const traces = recommenderResult.payload.items
    .map((item) => item.comment_trace)
    .filter((trace): trace is NonNullable<typeof trace> => Boolean(trace));
  if (traces.length === 0) {
    return undefined;
  }
  const intents = new Map<string, number>();
  let confusionSum = 0;
  let volatilitySum = 0;
  let confidenceSum = 0;
  let alignmentSum = 0;
  let coverageSum = 0;
  let driftSum = 0;
  let shiftSum = 0;
  for (const trace of traces) {
    const topIntent = trace.dominant_intents?.[0];
    if (topIntent) {
      intents.set(topIntent, (intents.get(topIntent) ?? 0) + 1);
    }
    confusionSum += typeof trace.confusion_index === "number" ? trace.confusion_index : 0;
    volatilitySum +=
      typeof trace.sentiment_volatility === "number" ? trace.sentiment_volatility : 0;
    confidenceSum += typeof trace.confidence === "number" ? trace.confidence : 0;
    alignmentSum += typeof trace.alignment_score === "number" ? trace.alignment_score : 0;
    coverageSum += typeof trace.value_prop_coverage === "number" ? trace.value_prop_coverage : 0;
    driftSum += typeof trace.artifact_drift_ratio === "number" ? trace.artifact_drift_ratio : 0;
    shiftSum +=
      typeof trace.alignment_shift_early_late === "number"
        ? trace.alignment_shift_early_late
        : 0;
  }
  const dominant = [...intents.entries()].sort((a, b) => b[1] - a[1])[0]?.[0] ?? "off_topic";
  return `dominant_intent=${dominant}, confusion=${round(confusionSum / traces.length, 3)}, sentiment_volatility=${round(volatilitySum / traces.length, 3)}, alignment=${round(alignmentSum / traces.length, 3)}, value_prop_coverage=${round(coverageSum / traces.length, 3)}, artifact_drift=${round(driftSum / traces.length, 3)}, alignment_shift=${round(shiftSum / traces.length, 3)}, confidence=${round(confidenceSum / traces.length, 3)}`;
}

function buildAlignmentStrategyHints(recommenderResult: RecommenderResult): string[] {
  if (!recommenderResult.ok) {
    return [];
  }
  const traces = recommenderResult.payload.items
    .map((item) => item.comment_trace)
    .filter((trace): trace is NonNullable<typeof trace> => Boolean(trace));
  if (traces.length === 0) {
    return [];
  }
  const avgAlignment =
    traces.reduce((sum, trace) => sum + (typeof trace.alignment_score === "number" ? trace.alignment_score : 0), 0) /
    Math.max(1, traces.length);
  const avgCoverage =
    traces.reduce(
      (sum, trace) => sum + (typeof trace.value_prop_coverage === "number" ? trace.value_prop_coverage : 0),
      0
    ) / Math.max(1, traces.length);
  const avgDrift =
    traces.reduce(
      (sum, trace) => sum + (typeof trace.artifact_drift_ratio === "number" ? trace.artifact_drift_ratio : 0),
      0
    ) / Math.max(1, traces.length);
  const hints: string[] = [];
  if (avgCoverage < 0.45) {
    hints.push("Audience comments mention too few intended value props; surface the core promise earlier and more explicitly.");
  }
  if (avgDrift > 0.35) {
    hints.push("Discussion is drifting to unrelated artifacts; reduce distracting elements and reinforce the intended outcome in overlays/caption.");
  }
  if (avgAlignment >= 0.6 && avgDrift <= 0.25) {
    hints.push("Comment discussion aligns well with intended value props; preserve current framing and iterate on execution quality.");
  }
  return hints;
}

function buildExplainabilitySection(
  recommenderResult: RecommenderResult
): NonNullable<ReportOutput["explainability"]> {
  if (!recommenderResult.ok) {
    return {
      evidence_cards: [],
      counterfactual_actions: [],
      disclaimer:
        "Explainability is unavailable because recommender evidence was not returned. Deterministic fallback remains active.",
      trace_metadata: {
        fallback_mode: true,
        reason: recommenderResult.error,
        source: "deterministic-fallback"
      }
    };
  }
  const payload = recommenderResult.payload;
  const evidenceCards = payload.items
    .map((item) => ({
      candidate_id: item.candidate_id,
      rank: item.rank,
      feature_contributions:
        (item.evidence_cards?.feature_contributions as Record<string, unknown>) ??
        (item.evidence_cards?.feature_contribution as Record<string, unknown>) ??
        {},
      neighbor_evidence: (item.evidence_cards?.neighbor_evidence as Record<string, unknown>) ?? {},
      temporal_confidence_band:
        (item.temporal_confidence_band as Record<string, unknown>) ??
        (item.evidence_cards?.temporal_confidence_band as Record<string, unknown>) ??
        {},
      trajectory_trace: (item.trajectory_trace as Record<string, unknown>) ?? {},
      comment_alignment: {
        alignment_score: item.comment_trace?.alignment_score ?? null,
        value_prop_coverage: item.comment_trace?.value_prop_coverage ?? null,
        on_topic_ratio: item.comment_trace?.on_topic_ratio ?? null,
        artifact_drift_ratio: item.comment_trace?.artifact_drift_ratio ?? null,
        alignment_shift_early_late: item.comment_trace?.alignment_shift_early_late ?? null,
        alignment_confidence: item.comment_trace?.alignment_confidence ?? null
      }
    }))
    .filter(
      (card) =>
        Object.keys(card.feature_contributions).length > 0 ||
        Object.keys(card.trajectory_trace).length > 0 ||
        card.comment_alignment.alignment_score !== null
    );
  const counterfactualActions = payload.items
    .filter((item) => Array.isArray(item.counterfactual_scenarios) && item.counterfactual_scenarios.length > 0)
    .map((item) => ({
      candidate_id: item.candidate_id,
      rank: item.rank,
      scenarios: (item.counterfactual_scenarios ?? []).map((scenario, index) => ({
        scenario_id:
          (typeof scenario.scenario_id === "string" && scenario.scenario_id.trim()) ||
          `scenario-${item.rank}-${index + 1}`,
        expected_rank_delta_band: scenario.expected_rank_delta_band ?? {},
        feasibility: scenario.feasibility ?? "unknown",
        reason: scenario.reason,
        trace: scenario.trace
      }))
    }));
  const methods =
    (payload.explainability_metadata?.methods as string[] | undefined)?.filter(
      (value) => typeof value === "string"
    ) ?? [];
  return {
    evidence_cards: evidenceCards,
    counterfactual_actions: counterfactualActions,
    disclaimer:
      "Evidence cards and scenario deltas are deterministic post-score analyses; they are directional and not causal guarantees.",
    trace_metadata: {
      fallback_mode: Boolean(payload.fallback_mode),
      objective: payload.objective_effective,
      explainability_metadata: payload.explainability_metadata ?? null,
      methods,
      comment_alignment_summary: summarizeCommentTrace(recommenderResult) ?? null,
      comment_alignment_hints: buildAlignmentStrategyHints(recommenderResult),
      trajectory: {
        manifest_id: payload.trajectory_manifest_id ?? null,
        version: payload.trajectory_version ?? null,
        mode: payload.trajectory_mode ?? null,
        prediction: payload.trajectory_prediction ?? null
      },
      portfolio: {
        mode: Boolean(payload.portfolio_mode),
        metadata: payload.portfolio_metadata ?? null
      }
    }
  };
}

function normalizeTagValues(values: string[]): string[] {
  return values
    .map((value) => value.trim().replace(/^#/, ""))
    .filter(Boolean)
    .map((value) => `#${value}`);
}

function buildUploadedSeedRecord(
  sourceSeed: DemoVideoRecord,
  description: string,
  hashtags: string[]
): DemoVideoRecord {
  const normalizedHashtags = normalizeTagValues(hashtags);
  const fallbackCaption = sourceSeed.caption.trim();
  const nextCaption = description.trim() || fallbackCaption;

  return {
    ...sourceSeed,
    caption: nextCaption,
    hashtags: normalizedHashtags.length > 0 ? normalizedHashtags : sourceSeed.hashtags.slice(0, 4),
    keywords: [...HARD_CODED_EXTRACTED_KEYWORDS],
    comments: [],
    video_url: "",
    metrics: {
      views: 0,
      likes: 0,
      comments_count: 0,
      shares: 0
    }
  };
}

async function normalizeAndEnrichReport(
  report: ReportOutput,
  candidates: DemoVideoRecord[],
  recommenderMeta?: {
    source: "recommender" | "fallback";
    note?: string;
    commentSummary?: string;
  },
  explainabilitySection?: ReportOutput["explainability"]
): Promise<ReportOutput> {
  const normalized = normalizeReportOutput(report, {
    candidatesK: candidates.length,
    extractedKeywords: [...HARD_CODED_EXTRACTED_KEYWORDS],
    modelLabel: DEEPSEEK_MODEL
  });

  const enriched = await enrichComparableMedia(normalized, candidates);
  if (!recommenderMeta) {
    return enriched;
  }
  const suffix =
    recommenderMeta.source === "recommender"
      ? "Recommendation engine: hybrid retrieval + objective ranker."
      : `Recommendation engine fallback: deterministic local neighborhood scoring.${
          recommenderMeta.note ? ` (${recommenderMeta.note})` : ""
        }`;
  const commentSuffix = recommenderMeta.commentSummary
    ? ` Comment intelligence: ${recommenderMeta.commentSummary}`
    : "";
  return {
    ...enriched,
    ...(explainabilitySection ? { explainability: explainabilitySection } : {}),
    header: {
      ...enriched.header,
      subtitle: `${enriched.header.subtitle} | ${suffix}${commentSuffix}`.slice(0, 220),
      disclaimer: `${enriched.header.disclaimer} ${suffix}${commentSuffix}`.trim()
    }
  };
}

function ensureDeepSeekClient(): OpenAI {
  if (!deepSeekClient) {
    throw new Error("DEEPSEEK_API_KEY is not configured in .env.local.");
  }
  return deepSeekClient;
}

interface GatewayMeta {
  request_id: string;
  experiment_id?: string;
  variant?: "control" | "treatment";
  assignment_unit_hash?: string;
  served_by: string;
  routing_decision: Record<string, unknown>;
  compatibility_status: Record<string, unknown>;
  fallback_mode: boolean;
  fallback_reason?: string;
  latency_breakdown_ms: Record<string, number>;
  circuit_state: Record<string, unknown>;
}

interface FetchRecommenderResult {
  result: RecommenderResult;
  gatewayMeta: GatewayMeta;
}

function extractCreatorId(record: DemoVideoRecord | undefined): string | undefined {
  if (!record) {
    return undefined;
  }
  const rawAuthor = record.author;
  if (rawAuthor && typeof rawAuthor === "object" && !Array.isArray(rawAuthor)) {
    const authorId =
      typeof rawAuthor.author_id === "string" && rawAuthor.author_id.trim()
        ? rawAuthor.author_id.trim()
        : undefined;
    const username =
      typeof rawAuthor.username === "string" && rawAuthor.username.trim()
        ? rawAuthor.username.trim().replace(/^@/, "")
        : undefined;
    return (authorId || username)?.toLowerCase();
  }
  if (typeof rawAuthor === "string" && rawAuthor.trim()) {
    return rawAuthor.trim().replace(/^@/, "").toLowerCase();
  }
  return undefined;
}

async function buildCreatorUserContext(params: {
  creatorId?: string;
  objective?: string;
}): Promise<Record<string, unknown> | undefined> {
  const creatorId =
    typeof params.creatorId === "string" && params.creatorId.trim()
      ? params.creatorId.trim().toLowerCase()
      : "";
  if (!creatorId || !feedbackStore.isReady()) {
    return undefined;
  }
  try {
    return (
      (await feedbackStore.loadCreatorPreferenceProfile({
        creatorId,
        objectiveEffective: mapObjectiveForRecommender(params.objective),
        historyDays: 180,
        maxFeedbackRows: 750
      })) as Record<string, unknown> | null
    ) ?? undefined;
  } catch (error) {
    console.error("creator_preference_profile_failed", error);
    return undefined;
  }
}

async function refreshCompatibility(
  routeIdentifier: string,
  requiredCompat: Record<string, string>,
  force = false
): Promise<CompatibilityCacheRecord> {
  const now = Date.now();
  const cached = compatibilityCache.get(routeIdentifier);
  if (!force && cached && now - cached.checked_at <= RECOMMENDER_COMPAT_CACHE_TTL_MS) {
    return cached;
  }
  const compatibility = await requestCompatibility({
    timeoutMs: Math.max(200, RECOMMENDER_BUDGET_NETWORK_MS)
  });
  if (!compatibility.ok) {
    const record: CompatibilityCacheRecord = {
      checked_at: now,
      ok: false,
      reason: compatibility.error
    };
    compatibilityCache.set(routeIdentifier, record);
    return record;
  }
  const payload = compatibility.payload;
  const mismatches: Array<Record<string, unknown>> = [];
  const fingerprints = (payload.fingerprints ?? {}) as Record<string, unknown>;
  for (const [key, expected] of Object.entries(requiredCompat)) {
    const actual = String(fingerprints[key] ?? "");
    if (actual !== expected) {
      mismatches.push({ key, expected, actual });
    }
  }
  if (mismatches.length > 0) {
    gatewayMetrics.compatibilityMismatchCount += 1;
  }
  const record: CompatibilityCacheRecord = {
    checked_at: now,
    ok: Boolean(payload.ok) && mismatches.length === 0,
    payload: {
      ...(payload as Record<string, unknown>),
      required_compat: requiredCompat,
      mismatches
    },
    reason: mismatches.length > 0 ? "required_compat_mismatch" : undefined
  };
  compatibilityCache.set(routeIdentifier, record);
  return record;
}

function mergeGatewayMeta(
  payload: Record<string, unknown>,
  gatewayMeta: GatewayMeta
): Record<string, unknown> {
  return {
    ...payload,
    request_id:
      (typeof payload.request_id === "string" && payload.request_id) || gatewayMeta.request_id,
    experiment_id:
      (typeof payload.experiment_id === "string" && payload.experiment_id) ||
      gatewayMeta.experiment_id ||
      null,
    variant:
      ((payload.variant === "control" || payload.variant === "treatment")
        ? payload.variant
        : gatewayMeta.variant) ?? null,
    served_by: gatewayMeta.served_by,
    routing_decision: {
      ...(payload.routing_decision && typeof payload.routing_decision === "object"
        ? (payload.routing_decision as Record<string, unknown>)
        : {}),
      ...gatewayMeta.routing_decision
    },
    compatibility_status: {
      ...(payload.compatibility_status && typeof payload.compatibility_status === "object"
        ? (payload.compatibility_status as Record<string, unknown>)
        : {}),
      ...gatewayMeta.compatibility_status
    },
    fallback_mode: Boolean(payload.fallback_mode) || gatewayMeta.fallback_mode,
    fallback_reason:
      (typeof payload.fallback_reason === "string" && payload.fallback_reason) ||
      gatewayMeta.fallback_reason ||
      null,
    latency_breakdown_ms: {
      ...(payload.latency_breakdown_ms && typeof payload.latency_breakdown_ms === "object"
        ? (payload.latency_breakdown_ms as Record<string, number>)
        : {}),
      ...gatewayMeta.latency_breakdown_ms
    },
    circuit_state: gatewayMeta.circuit_state
  };
}

async function fetchRecommenderResult(params: {
  requestId: string;
  objective: string | undefined;
  description: string;
  hashtags: string[];
  mentions: string[];
  candidates: DemoVideoRecord[];
  asOfTimeIso: string;
  language?: string;
  locale?: string;
  contentType?: string;
  candidateIds?: string[];
  policyOverrides?: {
    strict_language?: boolean;
    strict_locale?: boolean;
    max_items_per_author?: number;
  };
  portfolio?: {
    enabled?: boolean;
    weights?: {
      reach?: number;
      conversion?: number;
      durability?: number;
    };
    risk_aversion?: number;
    candidate_pool_cap?: number;
  };
  graphControls?: {
    enable_graph_branch?: boolean;
  };
  trajectoryControls?: {
    enabled?: boolean;
  };
  explainability?: {
    enabled?: boolean;
    top_features?: number;
    neighbor_k?: number;
    run_counterfactuals?: boolean;
  };
  creatorId?: string;
  userContext?: Record<string, unknown>;
  experiment: {
    experiment_id: string;
    variant: "control" | "treatment";
    unit_hash: string;
    model_family_suffix: string;
    required_compat: Record<string, string>;
  };
  routing?: RoutingRequestInput;
  topK?: number;
  retrieveK?: number;
  debug?: boolean;
}): Promise<FetchRecommenderResult> {
  const tracker = createStageLatencyTracker();
  const budgets = buildGatewayBudgets();
  if (!RECOMMENDER_ENABLED) {
    tracker.mark("gateway_disabled");
    trackFallbackReason("recommender_disabled");
    return {
      result: { ok: false, error: "Recommender disabled by config." },
      gatewayMeta: {
        request_id: params.requestId,
        experiment_id: params.experiment.experiment_id,
        variant: params.experiment.variant,
        assignment_unit_hash: params.experiment.unit_hash,
        served_by: "node-gateway",
        routing_decision: {},
        compatibility_status: { ok: false, reason: "recommender_disabled" },
        fallback_mode: true,
        fallback_reason: "recommender_disabled",
        latency_breakdown_ms: tracker.end(),
        circuit_state: { state: "closed" }
      }
    };
  }
  const requestedObjective = mapObjectiveForRecommender(params.objective);
  const routingEnvelope = buildRoutingEnvelope({
    objectiveRequested: requestedObjective,
    track: params.routing?.track,
    allowFallback: params.routing?.allow_fallback,
    requiredCompat: {
      ...(params.routing?.required_compat || {}),
      ...params.experiment.required_compat
    },
    requestId: params.requestId,
    modelFamilySuffix: params.experiment.model_family_suffix,
    experiment: {
      id: params.experiment.experiment_id,
      variant: params.experiment.variant,
      unit_hash: params.experiment.unit_hash
    },
    stageBudgetsMs: {
      retrieval: budgets.retrieval,
      ranking: budgets.ranking,
      explainability: budgets.explainability
    }
  });
  const objectiveEffective = routingEnvelope.objective_effective;
  const breakerKey = routeKey("/v1/recommendations", objectiveEffective);
  const compatibility = await refreshCompatibility(breakerKey, routingEnvelope.required_compat);
  tracker.mark("compatibility");
  if (!compatibility.ok) {
    trackFallbackReason(compatibility.reason ?? "compatibility_unhealthy");
    return {
      result: { ok: false, error: compatibility.reason ?? "Compatibility check failed." },
      gatewayMeta: {
        request_id: params.requestId,
        experiment_id: params.experiment.experiment_id,
        variant: params.experiment.variant,
        assignment_unit_hash: params.experiment.unit_hash,
        served_by: "node-gateway",
        routing_decision: routingEnvelope,
        compatibility_status: {
          ok: false,
          reason: compatibility.reason ?? "compatibility_unhealthy",
          ...(compatibility.payload ?? {})
        },
        fallback_mode: true,
        fallback_reason: compatibility.reason ?? "compatibility_unhealthy",
        latency_breakdown_ms: tracker.end(),
        circuit_state: recommenderBreakers.snapshot(breakerKey)
      }
    };
  }
  const breakerPermission = recommenderBreakers.shouldAllow(breakerKey);
  if (!breakerPermission.allow) {
    trackFallbackReason("circuit_open");
    gatewayMetrics.breakerTransitions["open"] = (gatewayMetrics.breakerTransitions["open"] ?? 0) + 1;
    return {
      result: { ok: false, error: "Circuit breaker open for recommender route." },
      gatewayMeta: {
        request_id: params.requestId,
        experiment_id: params.experiment.experiment_id,
        variant: params.experiment.variant,
        assignment_unit_hash: params.experiment.unit_hash,
        served_by: "node-gateway",
        routing_decision: routingEnvelope,
        compatibility_status: { ok: true, ...(compatibility.payload ?? {}) },
        fallback_mode: true,
        fallback_reason: "circuit_open",
        latency_breakdown_ms: tracker.end(),
        circuit_state: recommenderBreakers.snapshot(breakerKey)
      }
    };
  }
  const payload: RecommenderRequestPayload = {
    objective: requestedObjective,
    as_of_time: params.asOfTimeIso,
    query: {
      description: params.description,
      hashtags: params.hashtags,
      mentions: params.mentions,
      text: [params.description, ...params.hashtags, ...params.mentions].join(" ").trim(),
      topic_key: params.hashtags[0]?.replace(/^#/, "") || "general",
      author_id: params.creatorId,
      audience: params.audience,
      primary_cta: params.primaryCta,
      language: params.language,
      locale: params.locale,
      content_type: params.contentType,
      as_of_time: params.asOfTimeIso
    },
    candidates: buildRecommenderCandidates(params.candidates, params.asOfTimeIso),
    user_context: params.userContext,
    language: params.language,
    locale: params.locale,
    content_type: params.contentType,
    candidate_ids: params.candidateIds,
    policy_overrides: params.policyOverrides,
    portfolio: params.portfolio,
    graph_controls: params.graphControls,
    trajectory_controls: params.trajectoryControls,
    explainability: params.explainability,
    routing: routingEnvelope,
    top_k: Math.max(1, params.topK ?? 20),
    retrieve_k: Math.max(1, params.retrieveK ?? 200),
    debug: true
  };
  const requestTimeoutMs =
    budgets.network + budgets.retrieval + budgets.ranking + budgets.explainability + budgets.buffer;
  const result = await requestRecommendations(payload, { timeoutMs: requestTimeoutMs });
  tracker.mark("python_roundtrip");
  const latency = tracker.end();
  for (const [stage, value] of Object.entries(latency)) {
    recordStageLatency(stage, value);
  }
  if (result.ok) {
    const nextState = recommenderBreakers.recordSuccess(breakerKey);
    gatewayMetrics.breakerTransitions[nextState] =
      (gatewayMetrics.breakerTransitions[nextState] ?? 0) + 1;
    return {
      result,
      gatewayMeta: {
        request_id: params.requestId,
        experiment_id: params.experiment.experiment_id,
        variant: params.experiment.variant,
        assignment_unit_hash: params.experiment.unit_hash,
        served_by: "node-gateway",
        routing_decision: routingEnvelope,
        compatibility_status: { ok: true, ...(compatibility.payload ?? {}) },
        fallback_mode: false,
        latency_breakdown_ms: latency,
        circuit_state: recommenderBreakers.snapshot(breakerKey)
      }
    };
  }
  const nextState = recommenderBreakers.recordFailure(breakerKey);
  gatewayMetrics.breakerTransitions[nextState] =
    (gatewayMetrics.breakerTransitions[nextState] ?? 0) + 1;
  const failureClass = classifyRecommenderFailure(result.error);
  const fallbackReason = failureClass.reason;
  if (failureClass.compatibilityMismatch) {
    gatewayMetrics.compatibilityMismatchCount += 1;
  }
  trackFallbackReason(fallbackReason);
  return {
    result,
    gatewayMeta: {
      request_id: params.requestId,
      experiment_id: params.experiment.experiment_id,
      variant: params.experiment.variant,
      assignment_unit_hash: params.experiment.unit_hash,
      served_by: "node-gateway",
      routing_decision: routingEnvelope,
      compatibility_status: { ok: true, ...(compatibility.payload ?? {}) },
      fallback_mode: true,
      fallback_reason: fallbackReason,
      latency_breakdown_ms: latency,
      circuit_state: recommenderBreakers.snapshot(breakerKey)
    }
  };
}

function mapFabricResponseToSignalProfile(
  profileDescription: string,
  hints: Record<string, unknown> | undefined,
  payload: FabricExtractResponsePayload
): CandidateSignalProfile {
  const transcript = typeof hints?.transcript_text === "string" ? hints.transcript_text : "";
  const ocr = typeof hints?.ocr_text === "string" ? hints.ocr_text : "";
  const combinedText = [profileDescription, transcript, ocr].filter(Boolean).join(" ").trim();
  const tokenCount = payload.text.token_count;
  const uniqueTokenCount = payload.text.unique_token_count;
  const overallConfidence = round(
    clamp(
      (payload.text.confidence.calibrated +
        payload.structure.confidence.calibrated +
        payload.audio.confidence.calibrated +
        payload.visual.confidence.calibrated) /
        4,
      0.25,
      1
    ),
    2
  );
  const qualityFlags = [
    ...Object.entries(payload.text.missing).map(([key, value]) => `${key}:${value.reason}`),
    ...Object.entries(payload.structure.missing).map(([key, value]) => `${key}:${value.reason}`),
    ...Object.entries(payload.audio.missing).map(([key, value]) => `${key}:${value.reason}`),
    ...Object.entries(payload.visual.missing).map(([key, value]) => `${key}:${value.reason}`)
  ];

  return {
    pipeline_version: "extractors.v1",
    generated_at: payload.generated_at,
    duration_seconds: Math.max(
      3,
      Math.round(payload.structure.payoff_window_end_sec ?? payload.structure.payoff_timing_seconds ?? 30)
    ),
    visual: {
      confidence: round(clamp(payload.visual.confidence.calibrated, 0.25, 1), 2),
      shot_change_rate: round(payload.visual.shot_change_rate ?? 0, 4),
      visual_motion_score: round(payload.visual.visual_motion_score ?? 0, 4),
      visual_style_tags: payload.visual.style_tags ?? [],
      semantic_embedding_proxy: [
        round(clamp(payload.text.clarity_score, 0, 1), 6),
        round(clamp(payload.structure.pacing_score, 0, 1), 6),
        round(clamp(payload.audio.music_presence_score ?? 0, 0, 1), 6),
        round(clamp(payload.visual.visual_motion_score ?? 0, 0, 1), 6)
      ]
    },
    audio: {
      confidence: round(clamp(payload.audio.confidence.calibrated, 0.25, 1), 2),
      speech_ratio: round(clamp(payload.audio.speech_ratio ?? 0, 0, 1), 4),
      tempo_bpm_estimate: round(payload.audio.tempo_bpm ?? 0, 2),
      audio_energy: round(clamp(payload.audio.energy ?? 0, 0, 1), 4),
      music_presence_score: round(clamp(payload.audio.music_presence_score ?? 0, 0, 1), 4),
      audio_style_tags: [
        payload.audio.speech_ratio != null && payload.audio.speech_ratio >= 0.62
          ? "voice_forward"
          : "music_forward"
      ]
    },
    transcript_ocr: {
      confidence: round(clamp(payload.text.confidence.calibrated, 0.25, 1), 2),
      transcript_text: transcript,
      ocr_text: ocr,
      combined_text: combinedText,
      token_count: tokenCount,
      unique_token_count: uniqueTokenCount,
      clarity_score: round(clamp(payload.text.clarity_score, 0, 1), 4),
      cta_keywords_detected: []
    },
    structure: {
      confidence: round(clamp(payload.structure.confidence.calibrated, 0.25, 1), 2),
      hook_timing_seconds: round(payload.structure.hook_timing_seconds, 2),
      payoff_timing_seconds: round(payload.structure.payoff_timing_seconds, 2),
      step_density: round(payload.structure.step_density, 4),
      pacing_score: round(clamp(payload.structure.pacing_score, 0, 1), 4)
    },
    overall_confidence: overallConfidence,
    quality_flags: qualityFlags
  };
}

async function resolveCandidateSignals(params: {
  videoId: string;
  asOfTime: string;
  description: string;
  hashtags: string[];
  keywords: string[];
  contentType?: string;
  signalHints?: Record<string, unknown>;
}): Promise<{ signals: CandidateSignalProfile; source: "python-fabric" | "ts-shim" }> {
  const hints = params.signalHints;
  const fabricResult = await requestFabricSignals({
    video_id: params.videoId,
    as_of_time: params.asOfTime,
    caption: params.description,
    hashtags: params.hashtags,
    keywords: params.keywords,
    transcript_text: typeof hints?.transcript_text === "string" ? hints.transcript_text : undefined,
    ocr_text: typeof hints?.ocr_text === "string" ? hints.ocr_text : undefined,
    duration_seconds: typeof hints?.duration_seconds === "number" ? hints.duration_seconds : undefined,
    content_type: params.contentType,
    hints
  });
  if (fabricResult.ok) {
    return {
      signals: mapFabricResponseToSignalProfile(params.description, hints, fabricResult.payload),
      source: "python-fabric"
    };
  }
  return {
    signals: buildSignalProfileFallback({
      description: params.description,
      contentType: params.contentType,
      signalHints: hints
    }),
    source: "ts-shim"
  };
}

function extractTextContent(content: unknown): string {
  if (typeof content === "string") {
    return content;
  }

  if (Array.isArray(content)) {
    return content
      .map((part) => {
        if (part && typeof part === "object" && "text" in part) {
          const textValue = (part as { text?: unknown }).text;
          return typeof textValue === "string" ? textValue : "";
        }
        return "";
      })
      .join("")
      .trim();
  }

  return "";
}

function extractFirstJsonObject(rawContent: string): unknown {
  const trimmed = rawContent.trim();

  try {
    return JSON.parse(trimmed);
  } catch {
    // Continue with fence and object extraction fallback.
  }

  const fencedMatch = trimmed.match(/```(?:json)?\s*([\s\S]*?)\s*```/i);
  if (fencedMatch?.[1]) {
    return JSON.parse(fencedMatch[1]);
  }

  let startIndex = -1;
  let depth = 0;
  let inString = false;
  let isEscaped = false;

  for (let index = 0; index < trimmed.length; index += 1) {
    const character = trimmed[index];

    if (isEscaped) {
      isEscaped = false;
      continue;
    }

    if (character === "\\") {
      isEscaped = true;
      continue;
    }

    if (character === "\"") {
      inString = !inString;
      continue;
    }

    if (inString) {
      continue;
    }

    if (character === "{") {
      if (depth === 0) {
        startIndex = index;
      }
      depth += 1;
      continue;
    }

    if (character === "}" && depth > 0) {
      depth -= 1;
      if (depth === 0 && startIndex >= 0) {
        const candidate = trimmed.slice(startIndex, index + 1);
        return JSON.parse(candidate);
      }
    }
  }

  throw new Error("No valid JSON was found in the provider response.");
}

function normalizeQueryParam(value: unknown): string {
  return typeof value === "string" ? value.trim() : "";
}

function parseHttpUrl(value: string): URL | null {
  if (!value) {
    return null;
  }

  try {
    const parsed = new URL(value);
    if (parsed.protocol !== "https:" && parsed.protocol !== "http:") {
      return null;
    }
    return parsed;
  } catch {
    return null;
  }
}

function isAllowedThumbnailHost(hostname: string): boolean {
  const normalized = hostname.toLowerCase();
  return (
    normalized === "tiktokcdn.com" ||
    normalized.endsWith(".tiktokcdn.com") ||
    normalized === "tiktokcdn-us.com" ||
    normalized.endsWith(".tiktokcdn-us.com") ||
    normalized.includes("tiktokcdn")
  );
}

function isAllowedTikTokVideoHost(hostname: string): boolean {
  const normalized = hostname.toLowerCase();
  return (
    ALLOWED_TIKTOK_VIDEO_HOSTS.has(normalized) ||
    normalized.endsWith(".tiktok.com")
  );
}

async function fetchWithTimeout(
  url: string,
  options: RequestInit = {},
  timeoutMs = THUMBNAIL_FETCH_TIMEOUT_MS
): Promise<Response> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  try {
    return await fetch(url, {
      ...options,
      signal: controller.signal
    });
  } finally {
    clearTimeout(timeoutId);
  }
}

async function fetchThumbnailFromVideoUrl(videoUrl: string): Promise<string> {
  const parsedVideoUrl = parseHttpUrl(videoUrl);
  if (!parsedVideoUrl || !isAllowedTikTokVideoHost(parsedVideoUrl.hostname)) {
    return "";
  }

  try {
    const response = await fetchWithTimeout(
      `${TIKTOK_OEMBED_URL}${encodeURIComponent(parsedVideoUrl.toString())}`,
      {
        headers: {
          Accept: "application/json"
        }
      },
      4500
    );

    if (!response.ok) {
      return "";
    }

    const parsed = (await response.json()) as { thumbnail_url?: unknown };
    const thumbnail =
      typeof parsed.thumbnail_url === "string" ? parsed.thumbnail_url.trim() : "";

    const parsedThumbnailUrl = parseHttpUrl(thumbnail);
    if (!parsedThumbnailUrl) {
      return "";
    }

    if (!isAllowedThumbnailHost(parsedThumbnailUrl.hostname)) {
      return "";
    }

    return parsedThumbnailUrl.toString();
  } catch {
    return "";
  }
}

async function fetchThumbnailImage(
  thumbnailUrl: string
): Promise<{ body: Buffer; contentType: string } | null> {
  const parsedThumbnailUrl = parseHttpUrl(thumbnailUrl);
  if (!parsedThumbnailUrl || !isAllowedThumbnailHost(parsedThumbnailUrl.hostname)) {
    return null;
  }

  try {
    const response = await fetchWithTimeout(parsedThumbnailUrl.toString(), {
      headers: {
        Accept: "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        "User-Agent":
          "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
      }
    });

    if (!response.ok) {
      return null;
    }

    const contentType = response.headers.get("content-type") ?? "";
    if (!contentType.toLowerCase().startsWith("image/")) {
      return null;
    }

    const arrayBuffer = await response.arrayBuffer();
    return {
      body: Buffer.from(arrayBuffer),
      contentType
    };
  } catch {
    return null;
  }
}

app.get("/thumbnail", async (request, response) => {
  try {
    const directThumbnailUrl = normalizeQueryParam(request.query.url);
    const videoUrl = normalizeQueryParam(request.query.video);
    const candidateUrls: string[] = [];

    if (directThumbnailUrl) {
      const parsed = parseHttpUrl(directThumbnailUrl);
      if (parsed && isAllowedThumbnailHost(parsed.hostname)) {
        candidateUrls.push(parsed.toString());
      }
    }

    if (videoUrl) {
      const refreshedThumbnail = await fetchThumbnailFromVideoUrl(videoUrl);
      if (refreshedThumbnail) {
        candidateUrls.push(refreshedThumbnail);
      }
    }

    const uniqueCandidates = [...new Set(candidateUrls)];
    if (uniqueCandidates.length === 0) {
      response.status(400).json({ error: "No valid thumbnail source was provided." });
      return;
    }

    for (const candidateUrl of uniqueCandidates) {
      const image = await fetchThumbnailImage(candidateUrl);
      if (!image) {
        continue;
      }

      response.setHeader("Content-Type", image.contentType);
      response.setHeader("Cache-Control", "public, max-age=21600");
      response.send(image.body);
      return;
    }

    response.status(404).json({ error: "Thumbnail is unavailable." });
  } catch (error) {
    console.error(error);
    response.status(500).json({ error: "Could not load thumbnail." });
  }
});

app.post("/recommendations", async (request, response) => {
  try {
    const gatewayTracker = createStageLatencyTracker();
    const parsedRequest = parseRecommendationsRequest(request.body);
    if (!parsedRequest.ok) {
      response.status(400).json({ error: parsedRequest.error });
      return;
    }
    gatewayTracker.mark("parse_validate");
    if ((gatewayTracker.end().parse_validate ?? 0) > RECOMMENDER_BUDGET_PARSE_MS) {
      response.status(408).json({
        error: "request_parse_timeout",
        fallback_mode: true,
        fallback_reason: "parse_budget_exceeded"
      });
      return;
    }
    const body = parsedRequest.value;
    const requestId = createUuidV7();
    const requestReceivedAt = new Date().toISOString();
    const traceId = requestId;
    const trafficMeta = resolveTrafficMeta(body.traffic);
    const experimentAssignment = buildExperimentAssignment({
      objectiveRequested: body.objective,
      assignmentUnit: buildExperimentUnitKey({
        seedVideoId: body.seed_video_id,
        description: body.description,
        hashtags: body.hashtags,
        mentions: body.mentions,
        objective: body.objective,
        locale: body.locale,
        contentType: body.content_type
      }),
      experiment: body.experiment,
      config: {
        defaultExperimentId: RECOMMENDER_EXPERIMENT_DEFAULT_ID,
        treatmentRatio: RECOMMENDER_EXPERIMENT_TREATMENT_RATIO,
        salt: RECOMMENDER_EXPERIMENT_SALT,
        controlBundleId: RECOMMENDER_EXPERIMENT_CONTROL_BUNDLE_ID,
        treatmentBundleId: RECOMMENDER_EXPERIMENT_TREATMENT_BUNDLE_ID
      }
    });
    const experimentRoutePolicy = buildExperimentRoutePolicy({
      assignment: experimentAssignment,
      config: {
        defaultExperimentId: RECOMMENDER_EXPERIMENT_DEFAULT_ID,
        treatmentRatio: RECOMMENDER_EXPERIMENT_TREATMENT_RATIO,
        salt: RECOMMENDER_EXPERIMENT_SALT,
        controlBundleId: RECOMMENDER_EXPERIMENT_CONTROL_BUNDLE_ID,
        treatmentBundleId: RECOMMENDER_EXPERIMENT_TREATMENT_BUNDLE_ID
      }
    });
    const resolvedSignals = await resolveCandidateSignals({
      videoId: body.seed_video_id || "uploaded-seed",
      asOfTime: body.as_of_time,
      description: body.description,
      hashtags: body.hashtags,
      keywords: body.hashtags,
      contentType: body.content_type,
      signalHints:
        body.signal_hints && typeof body.signal_hints === "object"
          ? (body.signal_hints as Record<string, unknown>)
          : undefined
    });
    const candidateSignals = resolvedSignals.signals;
    response.setHeader("x-signal-source", resolvedSignals.source);

    const dataset = await loadDatasetFromFile();
    if (dataset.length === 0) {
      response.status(400).json({ error: "Local dataset is empty or invalid." });
      return;
    }
    const seedVideoId = body.seed_video_id;
    const seedRecord =
      dataset.find((record) => record.video_id === seedVideoId) ?? getSeedVideo(dataset);
    const creatorId = extractCreatorId(seedRecord);
    const userContext = await buildCreatorUserContext({
      creatorId,
      objective: body.objective
    });
    const candidates = getCombinedCandidates(dataset, seedVideoId);
    if (candidates.length === 0) {
      response.status(400).json({ error: "No comparable candidates were found." });
      return;
    }

    const recommenderAttempt = await fetchRecommenderResult({
      requestId,
      objective: body.objective,
      description: body.description,
      hashtags: body.hashtags,
      mentions: body.mentions,
      candidates,
      asOfTimeIso: body.as_of_time,
      language:
        body.language ??
        (typeof body.locale === "string" ? body.locale.split("-")[0]?.toLowerCase() : undefined),
      locale: body.locale,
      contentType: body.content_type,
      candidateIds:
        body.candidate_ids.length > 0
          ? body.candidate_ids
          : candidates.map((candidate) => candidate.video_id),
      policyOverrides: body.policy_overrides,
      portfolio: body.portfolio,
      graphControls: body.graph_controls,
      trajectoryControls: body.trajectory_controls,
      explainability: body.explainability,
      creatorId,
      userContext,
      experiment: {
        experiment_id: experimentAssignment.experiment_id,
        variant: experimentAssignment.variant,
        unit_hash: experimentAssignment.unit_hash,
        model_family_suffix: experimentRoutePolicy.modelFamilySuffix,
        required_compat: experimentRoutePolicy.requiredCompat
      },
      routing: body.routing,
      topK: body.top_k,
      retrieveK: body.retrieve_k,
      debug: body.debug
    });
    const recommenderResult = recommenderAttempt.result;
    const gatewayMeta = recommenderAttempt.gatewayMeta;
    console.log(
      JSON.stringify({
        event: "recommendation_request",
        trace_id: traceId,
        request_id: requestId,
        experiment_id: experimentAssignment.experiment_id,
        variant: experimentAssignment.variant,
        route_key: routeKey(
          "/recommendations",
          String((gatewayMeta.routing_decision.objective_effective as string) || "engagement")
        ),
        fallback_mode: gatewayMeta.fallback_mode,
        fallback_reason: gatewayMeta.fallback_reason ?? null
      })
    );

    if (recommenderResult.ok) {
      response.setHeader("x-recommender-source", "python-service");
      const payloadForLogging = mergeGatewayMeta(
        recommenderResult.payload as Record<string, unknown>,
        gatewayMeta
      );
      const merged = { ...payloadForLogging };
      if (!body.debug && "debug" in merged) {
        delete merged.debug;
      }
      try {
        await feedbackStore.writeRecommendationTrace({
          request: {
            requestId,
            endpoint: "/recommendations",
            receivedAt: requestReceivedAt,
            servedAt: new Date().toISOString(),
            objectiveRequested: mapObjectiveForRecommender(body.objective),
            objectiveEffective:
              String((merged.objective_effective as string) || "engagement"),
            experimentId: experimentAssignment.experiment_id,
            variant: experimentAssignment.variant,
            assignmentUnitHash: experimentAssignment.unit_hash,
            requestHash: buildRequestHash({
              objectiveRequested: mapObjectiveForRecommender(body.objective),
              objectiveEffective: String((merged.objective_effective as string) || "engagement"),
              requestId,
              assignmentUnitHash: experimentAssignment.unit_hash
            }),
            routingDecision:
              (merged.routing_decision as Record<string, unknown> | undefined) || {},
            compatibilityStatus:
              (merged.compatibility_status as Record<string, unknown> | undefined) || {},
            fallbackMode: Boolean(merged.fallback_mode),
            fallbackReason:
              typeof merged.fallback_reason === "string" ? merged.fallback_reason : null,
            circuitState: (merged.circuit_state as Record<string, unknown> | undefined) || {},
            latencyBreakdownMs:
              (merged.latency_breakdown_ms as Record<string, number> | undefined) || {},
            policyVersion:
              typeof merged.policy_version === "string" ? merged.policy_version : undefined,
            calibrationVersion:
              typeof merged.calibration_version === "string"
                ? merged.calibration_version
                : undefined,
            policyMetadata:
              (merged.policy_metadata as Record<string, unknown> | undefined) || {},
            requestContext: {
              creator_id: creatorId ?? null,
              seed_video_id: seedVideoId || null,
              endpoint_mode: "recommendations"
            },
            trafficClass: trafficMeta.trafficClass,
            isSynthetic: trafficMeta.isSynthetic,
            injectedFailure: trafficMeta.injectedFailure,
            bundleFingerprint:
              (merged.compatibility_status as Record<string, unknown> | undefined)?.fingerprints &&
              typeof (merged.compatibility_status as Record<string, unknown>).fingerprints ===
                "object"
                ? ((merged.compatibility_status as Record<string, unknown>)
                    .fingerprints as Record<string, unknown>)
                : {}
          },
          assignment: {
            experimentId: experimentAssignment.experiment_id,
            objectiveEffective:
              String((merged.objective_effective as string) || "engagement"),
            unitHash: experimentAssignment.unit_hash,
            variant: experimentAssignment.variant,
            requestId
          },
          candidates: toCandidateEventsFromResponse({
            requestId,
            payload: payloadForLogging
          }),
          served: toServedOutputEvents({
            requestId,
            payload: payloadForLogging
          })
        });
      } catch (feedbackError) {
        console.error("feedback_store_write_failed", feedbackError);
      }
      response.json(merged);
      return;
    }

    const bundleFallback = await buildFallbackItemsFromBundle({
      objective: body.objective,
      candidates,
      topK: body.top_k
    });
    if (bundleFallback && bundleFallback.items.length > 0) {
      response.setHeader("x-recommender-source", "fallback-bundle");
      const bundlePayload = {
        request_id: requestId,
        experiment_id: experimentAssignment.experiment_id,
        variant: experimentAssignment.variant,
        objective: mapObjectiveForRecommender(body.objective),
        objective_effective:
          mapObjectiveForRecommender(body.objective) === "community"
            ? "engagement"
            : mapObjectiveForRecommender(body.objective),
        generated_at: new Date().toISOString(),
        fallback_mode: true,
        fallback_reason: gatewayMeta.fallback_reason ?? recommenderResult.error,
        routing_decision: gatewayMeta.routing_decision,
        compatibility_status: gatewayMeta.compatibility_status,
        latency_breakdown_ms: gatewayMeta.latency_breakdown_ms,
        circuit_state: gatewayMeta.circuit_state,
        served_by: gatewayMeta.served_by,
        comment_feature_manifest_id: null,
        comment_intelligence_version: "comment_intelligence.v2",
        trajectory_manifest_id: null,
        trajectory_version: "trajectory.v2",
        trajectory_mode: "fallback_bundle",
        portfolio_mode: false,
        portfolio_metadata: {
          enabled_requested: Boolean(body.portfolio.enabled),
          enabled: false,
          fallback_reason: gatewayMeta.fallback_reason ?? recommenderResult.error
        },
        fallback_bundle_version: bundleFallback.bundleVersion,
        fallback_bundle_generated_at: bundleFallback.generatedAt,
        items: bundleFallback.items
      };
      response.status(200).json(bundlePayload);
      try {
        await feedbackStore.writeRecommendationTrace({
          request: {
            requestId,
            endpoint: "/recommendations",
            receivedAt: requestReceivedAt,
            servedAt: new Date().toISOString(),
            objectiveRequested: mapObjectiveForRecommender(body.objective),
            objectiveEffective:
              String(bundlePayload.objective_effective || "engagement"),
            experimentId: experimentAssignment.experiment_id,
            variant: experimentAssignment.variant,
            assignmentUnitHash: experimentAssignment.unit_hash,
            requestHash: buildRequestHash({
              objectiveRequested: mapObjectiveForRecommender(body.objective),
              objectiveEffective: String(bundlePayload.objective_effective || "engagement"),
              requestId,
              assignmentUnitHash: experimentAssignment.unit_hash
            }),
            routingDecision: bundlePayload.routing_decision || {},
            compatibilityStatus: bundlePayload.compatibility_status || {},
            fallbackMode: true,
            fallbackReason:
              typeof bundlePayload.fallback_reason === "string"
                ? bundlePayload.fallback_reason
                : null,
            circuitState: bundlePayload.circuit_state || {},
            latencyBreakdownMs:
              (bundlePayload.latency_breakdown_ms as Record<string, number> | undefined) || {},
            policyVersion: undefined,
            calibrationVersion: undefined,
            policyMetadata:
              (bundlePayload.policy_metadata as Record<string, unknown> | undefined) || {},
            requestContext: {
              creator_id: creatorId ?? null,
              seed_video_id: seedVideoId || null,
              endpoint_mode: "recommendations"
            },
            trafficClass: trafficMeta.trafficClass,
            isSynthetic: trafficMeta.isSynthetic,
            injectedFailure: trafficMeta.injectedFailure,
            bundleFingerprint: {}
          },
          assignment: {
            experimentId: experimentAssignment.experiment_id,
            objectiveEffective: String(bundlePayload.objective_effective || "engagement"),
            unitHash: experimentAssignment.unit_hash,
            variant: experimentAssignment.variant,
            requestId
          },
          candidates: [],
          served: toServedOutputEvents({
            requestId,
            payload: bundlePayload as Record<string, unknown>
          })
        });
      } catch (feedbackError) {
        console.error("feedback_store_write_failed", feedbackError);
      }
      return;
    }

    response.status(503).json({
      error: "Recommendations are unavailable right now.",
      fallback_mode: true,
      fallback_reason: gatewayMeta.fallback_reason ?? recommenderResult.error
    });
    return;
  } catch (error) {
    console.error(error);
    response.status(500).json({
      error: "Recommendations are unavailable right now.",
      fallback_mode: true
    });
  }
});

app.get("/recommender-gateway-metrics", (_request, response) => {
  const stageLatencySummary = Object.fromEntries(
    Object.entries(gatewayMetrics.stageLatencyMs).map(([stage, values]) => [
      stage,
      summarizeLatency(values)
    ])
  );
  response.json({
    stage_latency: stageLatencySummary,
    breaker_transitions: gatewayMetrics.breakerTransitions,
    fallback_by_reason: gatewayMetrics.fallbackByReason,
    compatibility_mismatch_count: gatewayMetrics.compatibilityMismatchCount,
    fallback_bundle_hit_count: gatewayMetrics.fallbackBundleHitCount,
    feedback_store: feedbackStore.status()
  });
});

app.get("/labeling/sources", async (_request, response) => {
  try {
    const sources = await labelingSessionStore.listSources();
    response.json({ sources });
  } catch (error) {
    console.error("labeling_sources_failed", error);
    response.status(500).json({ error: "Failed to load labeling sources." });
  }
});

app.get("/labeling/sessions", async (_request, response) => {
  try {
    const sessions = await labelingSessionStore.listSessions();
    response.json({ sessions });
  } catch (error) {
    console.error("labeling_sessions_failed", error);
    response.status(500).json({ error: "Failed to load labeling sessions." });
  }
});

app.post("/labeling/sessions", async (request, response) => {
  try {
    const body = isPlainObject(request.body) ? request.body : {};
    const sourceId =
      typeof body.source_id === "string" && body.source_id.trim().length > 0
        ? body.source_id.trim()
        : undefined;
    const sessionName =
      typeof body.session_name === "string" && body.session_name.trim().length > 0
        ? body.session_name.trim()
        : undefined;
    const session = await labelingSessionStore.createSession({
      sourceId,
      sessionName
    });
    response.status(201).json({ session });
  } catch (error) {
    console.error("labeling_session_create_failed", error);
    const message =
      error instanceof Error && error.message === "no_labeling_sources_available"
        ? "No benchmark sources are available for labeling."
        : "Failed to create labeling session.";
    response.status(500).json({ error: message });
  }
});

app.get("/labeling/sessions/:sessionId", async (request, response) => {
  try {
    const session = await labelingSessionStore.loadSession(request.params.sessionId);
    response.json({ session });
  } catch (error) {
    console.error("labeling_session_load_failed", error);
    response.status(404).json({ error: "Labeling session not found." });
  }
});

app.put(
  "/labeling/sessions/:sessionId/cases/:caseId/candidates/:candidateId",
  async (request, response) => {
    try {
      const body = isPlainObject(request.body) ? request.body : {};
      if (!("label" in body)) {
        response.status(400).json({ error: "label is required." });
        return;
      }
      const label = normalizeLabelingReviewLabel(body.label);
      if (body.label !== null && label === null) {
        response
          .status(400)
          .json({ error: "label must be saved, relevant, not_relevant, or null." });
        return;
      }
      const note =
        typeof body.note === "string" ? body.note.slice(0, 500) : undefined;
      const session = await labelingSessionStore.updateCandidateReview({
        sessionId: request.params.sessionId,
        caseId: request.params.caseId,
        candidateId: request.params.candidateId,
        label,
        note
      });
      response.json({ session });
    } catch (error) {
      console.error("labeling_session_update_failed", error);
      const message =
        error instanceof Error &&
        (error.message === "labeling_case_not_found" ||
          error.message === "labeling_candidate_not_found")
          ? "Case or candidate not found."
          : "Failed to update labeling session.";
      response.status(400).json({ error: message });
    }
  }
);

app.post("/generate-report", async (request, response) => {
  try {
    const requestReceivedAt = new Date().toISOString();
    const parsedRequest = parseGenerateReportRequest(request.body);
    if (!parsedRequest.ok) {
      response.status(400).json({ error: parsedRequest.error });
      return;
    }
    const body = parsedRequest.value;
    const resolvedSignals = await resolveCandidateSignals({
      videoId: body.seed_video_id || "uploaded-seed",
      asOfTime: new Date().toISOString(),
      description: body.description,
      hashtags: body.hashtags,
      keywords: body.hashtags,
      contentType: body.content_type,
      signalHints:
        body.signal_hints && typeof body.signal_hints === "object"
          ? (body.signal_hints as Record<string, unknown>)
          : undefined
    });
    const candidateSignals = resolvedSignals.signals;
    response.setHeader("x-signal-source", resolvedSignals.source);
    const description = body.description.trim();
    const mentions = body.mentions.map((mention) =>
      mention.trim().startsWith("@") ? mention.trim() : `@${mention.trim()}`
    );
    const hashtags = body.hashtags.map((tag) =>
      tag.trim().startsWith("#") ? tag.trim() : `#${tag.trim()}`
    );
    const seedVideoId = body.seed_video_id;

    const dataset = await loadDatasetFromFile();
    if (dataset.length === 0) {
      response.status(400).json({ error: "Local dataset is empty or invalid." });
      return;
    }

    const seed = dataset.find((record) => record.video_id === seedVideoId) ?? getSeedVideo(dataset);
    if (!seed) {
      response.status(404).json({ error: "Seed video was not found in local dataset." });
      return;
    }
    const creatorId = extractCreatorId(seed);
    const userContext = await buildCreatorUserContext({
      creatorId,
      objective: body.objective
    });

    const uploadedSeed = buildUploadedSeedRecord(seed, description, hashtags);
    const candidates = getCombinedCandidates(dataset, seedVideoId);
    if (candidates.length === 0) {
      response.status(400).json({ error: "No comparable candidates were found." });
      return;
    }
    const reportRequestId = createUuidV7();
    const reportExperimentAssignment = buildExperimentAssignment({
      objectiveRequested: body.objective,
      assignmentUnit: buildExperimentUnitKey({
        seedVideoId: body.seed_video_id,
        description,
        hashtags,
        mentions,
        objective: body.objective,
        locale: body.locale,
        contentType: body.content_type
      }),
      config: {
        defaultExperimentId: RECOMMENDER_EXPERIMENT_DEFAULT_ID,
        treatmentRatio: RECOMMENDER_EXPERIMENT_TREATMENT_RATIO,
        salt: RECOMMENDER_EXPERIMENT_SALT,
        controlBundleId: RECOMMENDER_EXPERIMENT_CONTROL_BUNDLE_ID,
        treatmentBundleId: RECOMMENDER_EXPERIMENT_TREATMENT_BUNDLE_ID
      }
    });
    const reportExperimentRoutePolicy = buildExperimentRoutePolicy({
      assignment: reportExperimentAssignment,
      config: {
        defaultExperimentId: RECOMMENDER_EXPERIMENT_DEFAULT_ID,
        treatmentRatio: RECOMMENDER_EXPERIMENT_TREATMENT_RATIO,
        salt: RECOMMENDER_EXPERIMENT_SALT,
        controlBundleId: RECOMMENDER_EXPERIMENT_CONTROL_BUNDLE_ID,
        treatmentBundleId: RECOMMENDER_EXPERIMENT_TREATMENT_BUNDLE_ID
      }
    });

    const recommenderAttempt = await fetchRecommenderResult({
      requestId: reportRequestId,
      objective: body.objective,
      description,
      hashtags,
      mentions,
      candidates,
      asOfTimeIso: new Date().toISOString(),
      language:
        body.language ??
        (typeof body.locale === "string" ? body.locale.split("-")[0]?.toLowerCase() : undefined),
      locale: body.locale,
      contentType: body.content_type,
      candidateIds: candidates.map((candidate) => candidate.video_id),
      explainability: {
        enabled: true,
        top_features: 5,
        neighbor_k: 3,
        run_counterfactuals: true
      },
      trajectoryControls: {
        enabled: true
      },
      creatorId,
      userContext,
      experiment: {
        experiment_id: reportExperimentAssignment.experiment_id,
        variant: reportExperimentAssignment.variant,
        unit_hash: reportExperimentAssignment.unit_hash,
        model_family_suffix: reportExperimentRoutePolicy.modelFamilySuffix,
        required_compat: reportExperimentRoutePolicy.requiredCompat
      },
      routing: {
        track: "post_publication",
        allow_fallback: true
      },
      topK: 20,
      retrieveK: 200
    });
    let recommenderResult = recommenderAttempt.result;
    let recommenderSource = recommenderResult.ok ? "python-service" : "fallback-bundle";
    if (!recommenderResult.ok) {
      const bundleFallback = await buildFallbackItemsFromBundle({
        objective: body.objective,
        candidates,
        topK: 20
      });
      if (bundleFallback) {
        recommenderSource = "fallback-bundle";
        recommenderResult = {
          ok: true,
          payload: {
            objective: mapObjectiveForRecommender(body.objective),
            objective_effective:
              mapObjectiveForRecommender(body.objective) === "community"
                ? "engagement"
                : mapObjectiveForRecommender(body.objective),
            generated_at: new Date().toISOString(),
            fallback_mode: true,
            fallback_reason:
              recommenderAttempt.gatewayMeta.fallback_reason ?? "fallback_bundle",
            items: bundleFallback.items
          }
        };
      } else {
        response.status(503).json({
          error: "Report generation is unavailable because ranked comparable evidence is unavailable.",
          fallback_mode: true,
          fallback_reason:
            recommenderAttempt.gatewayMeta.fallback_reason ?? recommenderResult.error
        });
        return;
      }
    }
    const commentTraceSummary = summarizeCommentTrace(recommenderResult);
    const explainabilitySection = buildExplainabilitySection(recommenderResult);
    const rankedCandidates = applyRecommenderOrder(candidates, recommenderResult);
    response.setHeader("x-recommender-source", recommenderSource);
    const recommenderFallbackNote = recommenderResult.ok
      ? undefined
      : recommenderAttempt.gatewayMeta.fallback_reason ?? recommenderResult.error;
    const recommenderPayload = recommenderResult.ok
      ? recommenderResult.payload
      : {
          request_id: recommenderAttempt.gatewayMeta.request_id,
          objective: mapObjectiveForRecommender(body.objective),
          objective_effective:
            mapObjectiveForRecommender(body.objective) === "community"
              ? "engagement"
              : mapObjectiveForRecommender(body.objective),
          generated_at: new Date().toISOString(),
          fallback_mode: true,
          fallback_reason: recommenderAttempt.gatewayMeta.fallback_reason ?? recommenderResult.error,
          items: []
        };
    const reasoningArtifacts = buildReportReasoning({
      requestId:
        (typeof recommenderPayload.request_id === "string" && recommenderPayload.request_id) ||
        recommenderAttempt.gatewayMeta.request_id,
      objective: body.objective,
      objectiveEffective: recommenderPayload.objective_effective,
      generatedAt: recommenderPayload.generated_at,
      recommenderSource:
        recommenderSource === "python-service" ? "python-service" : "fallback-bundle",
      fallbackMode: Boolean(recommenderPayload.fallback_mode),
      fallbackReason: recommenderPayload.fallback_reason ?? null,
      experimentId:
        (typeof recommenderPayload.experiment_id === "string" && recommenderPayload.experiment_id) ||
        recommenderAttempt.gatewayMeta.experiment_id ||
        null,
      variant:
        recommenderPayload.variant === "control" || recommenderPayload.variant === "treatment"
          ? recommenderPayload.variant
          : recommenderAttempt.gatewayMeta.variant ?? null,
      description,
      hashtags,
      mentions,
      contentType: body.content_type,
      primaryCta: body.primary_cta,
      audience: body.audience ? { label: body.audience } : undefined,
      locale: body.locale,
      language: body.language,
      candidateSignals,
      candidates: rankedCandidates,
      recommenderPayload
    });

    const deterministicReport = buildLocalBaselineReport({
      candidates: rankedCandidates,
      mentions,
      hashtags,
      description,
      candidatesK: rankedCandidates.length,
      objective: body.objective,
      recommenderItems: recommenderPayload.items ?? [],
      candidateSignals,
      meta: reasoningArtifacts.meta,
      reasoning: reasoningArtifacts.reasoning
    });

    const persistedPayloadForLogging = mergeGatewayMeta(
      recommenderPayload as Record<string, unknown>,
      recommenderAttempt.gatewayMeta
    );
    const persistReportTrace = async (): Promise<void> => {
      try {
        await feedbackStore.writeRecommendationTrace({
          request: {
            requestId:
              (typeof persistedPayloadForLogging.request_id === "string" &&
                persistedPayloadForLogging.request_id) ||
              reportRequestId,
            endpoint: "/generate-report",
            receivedAt: requestReceivedAt,
            servedAt: new Date().toISOString(),
            objectiveRequested: mapObjectiveForRecommender(body.objective),
            objectiveEffective:
              String((persistedPayloadForLogging.objective_effective as string) || "engagement"),
            experimentId: reportExperimentAssignment.experiment_id,
            variant: reportExperimentAssignment.variant,
            assignmentUnitHash: reportExperimentAssignment.unit_hash,
            requestHash: buildRequestHash({
              objectiveRequested: mapObjectiveForRecommender(body.objective),
              objectiveEffective:
                String((persistedPayloadForLogging.objective_effective as string) || "engagement"),
              requestId:
                (typeof persistedPayloadForLogging.request_id === "string" &&
                  persistedPayloadForLogging.request_id) ||
                reportRequestId,
              assignmentUnitHash: reportExperimentAssignment.unit_hash
            }),
            routingDecision:
              (persistedPayloadForLogging.routing_decision as Record<string, unknown> | undefined) ||
              {},
            compatibilityStatus:
              (persistedPayloadForLogging.compatibility_status as
                | Record<string, unknown>
                | undefined) || {},
            fallbackMode: Boolean(persistedPayloadForLogging.fallback_mode),
            fallbackReason:
              typeof persistedPayloadForLogging.fallback_reason === "string"
                ? persistedPayloadForLogging.fallback_reason
                : null,
            circuitState:
              (persistedPayloadForLogging.circuit_state as Record<string, unknown> | undefined) ||
              {},
            latencyBreakdownMs:
              (persistedPayloadForLogging.latency_breakdown_ms as
                | Record<string, number>
                | undefined) || {},
            policyVersion:
              typeof persistedPayloadForLogging.policy_version === "string"
                ? persistedPayloadForLogging.policy_version
                : undefined,
            calibrationVersion:
              typeof persistedPayloadForLogging.calibration_version === "string"
                ? persistedPayloadForLogging.calibration_version
                : undefined,
            policyMetadata:
              (persistedPayloadForLogging.policy_metadata as Record<string, unknown> | undefined) ||
              {},
            requestContext: {
              creator_id: creatorId ?? null,
              seed_video_id: body.seed_video_id || null,
              endpoint_mode: "generate_report"
            },
            trafficClass: "production",
            isSynthetic: false,
            injectedFailure: false,
            bundleFingerprint:
              (persistedPayloadForLogging.compatibility_status as
                | Record<string, unknown>
                | undefined)?.fingerprints &&
              typeof (persistedPayloadForLogging.compatibility_status as Record<string, unknown>)
                .fingerprints === "object"
                ? ((persistedPayloadForLogging.compatibility_status as Record<string, unknown>)
                    .fingerprints as Record<string, unknown>)
                : {}
          },
          assignment: {
            experimentId: reportExperimentAssignment.experiment_id,
            objectiveEffective:
              String((persistedPayloadForLogging.objective_effective as string) || "engagement"),
            unitHash: reportExperimentAssignment.unit_hash,
            variant: reportExperimentAssignment.variant,
            requestId:
              (typeof persistedPayloadForLogging.request_id === "string" &&
                persistedPayloadForLogging.request_id) ||
              reportRequestId
          },
          candidates: toCandidateEventsFromResponse({
            requestId:
              (typeof persistedPayloadForLogging.request_id === "string" &&
                persistedPayloadForLogging.request_id) ||
              reportRequestId,
            payload: persistedPayloadForLogging
          }),
          served: toServedOutputEvents({
            requestId:
              (typeof persistedPayloadForLogging.request_id === "string" &&
                persistedPayloadForLogging.request_id) ||
              reportRequestId,
            payload: persistedPayloadForLogging
          })
        });
      } catch (feedbackError) {
        console.error("feedback_store_report_trace_failed", feedbackError);
      }
    };

    const reportPrompt = buildReportPrompt({
      report: deterministicReport
    });

    if (!DEEPSEEK_ENABLED) {
      response.setHeader("x-report-source", "baseline-local-no-key");
      const report = await normalizeAndEnrichReport(
        sanitizeUnknownStrings(deterministicReport) as ReportOutput,
        rankedCandidates,
        {
          source: recommenderResult.ok ? "recommender" : "fallback",
          note: recommenderFallbackNote,
          commentSummary: commentTraceSummary
        },
        explainabilitySection
      );
      await persistReportTrace();
      response.json({
        report
      });
      return;
    }

    const client = ensureDeepSeekClient();

    try {
      const completion = await client.chat.completions.create({
        model: DEEPSEEK_MODEL,
        temperature: 0.2,
        messages: [
          {
            role: "system",
            content:
              "You are a senior growth and content analyst. Return valid JSON only, in English, no markdown, no extra text, no emojis."
          },
          {
            role: "user",
            content: reportPrompt
          }
        ]
      });

      const rawContent = extractTextContent(completion.choices[0]?.message?.content ?? "");
      if (!rawContent) {
        response.setHeader("x-report-source", "baseline-local-empty-provider-response");
        const report = await normalizeAndEnrichReport(
          sanitizeUnknownStrings(deterministicReport) as ReportOutput,
          rankedCandidates,
          {
            source: recommenderResult.ok ? "recommender" : "fallback",
            note: recommenderFallbackNote,
            commentSummary: commentTraceSummary
          },
          explainabilitySection
        );
        await persistReportTrace();
        response.json({
          report
        });
        return;
      }

      const parsed = extractFirstJsonObject(rawContent);
      const sanitizedPolish = sanitizeUnknownStrings(parsed);

      if (!validateNarrativePolish(sanitizedPolish)) {
        response.setHeader("x-report-source", "baseline-local-invalid-provider-schema");
        const report = await normalizeAndEnrichReport(
          sanitizeUnknownStrings(deterministicReport) as ReportOutput,
          rankedCandidates,
          {
            source: recommenderResult.ok ? "recommender" : "fallback",
            note: recommenderFallbackNote,
            commentSummary: commentTraceSummary
          },
          explainabilitySection
        );
        response.json({
          report
        });
        return;
      }

      response.setHeader("x-report-source", "deepseek");
      const report = await normalizeAndEnrichReport(
        applyNarrativePolish(
          sanitizeUnknownStrings(deterministicReport) as ReportOutput,
          sanitizedPolish
        ),
        rankedCandidates,
        {
          source: recommenderResult.ok ? "recommender" : "fallback",
          note: recommenderFallbackNote,
          commentSummary: commentTraceSummary
        },
        explainabilitySection
      );
      await persistReportTrace();
      response.json({
        report
      });
    } catch (providerError) {
      console.error(providerError);
      response.setHeader("x-report-source", "baseline-local-provider-error");
      const report = await normalizeAndEnrichReport(
        sanitizeUnknownStrings(deterministicReport) as ReportOutput,
        rankedCandidates,
        {
          source: recommenderResult.ok ? "recommender" : "fallback",
          note: recommenderFallbackNote,
          commentSummary: commentTraceSummary
        },
        explainabilitySection
      );
      await persistReportTrace();
      response.json({
        report
      });
    }
  } catch (error) {
    console.error(error);
    response.status(500).json({
      error: "The report could not be generated right now."
    });
  }
});

app.post("/report-feedback", async (request, response) => {
  const parsed = parseReportFeedbackRequest(request.body);
  if (!parsed.ok) {
    response.status(400).json({ error: parsed.error });
    return;
  }

  try {
    await feedbackStore.writeUiFeedbackEvent({
      requestId: parsed.value.request_id,
      eventName: parsed.value.event_name,
      entityType: parsed.value.entity_type,
      entityId: parsed.value.entity_id ?? null,
      section: parsed.value.section,
      rank: parsed.value.rank ?? null,
      objectiveEffective: parsed.value.objective_effective,
      experimentId: parsed.value.experiment_id ?? null,
      variant: parsed.value.variant ?? null,
      signalStrength: parsed.value.signal_strength,
      labelDirection: parsed.value.label_direction,
      metadata: parsed.value.metadata,
      createdAt: new Date().toISOString()
    });

    response.status(202).json({
      ok: true,
      ready: feedbackStore.isReady()
    });
  } catch (error) {
    console.error("feedback_store_ui_write_failed", error);
    response.status(500).json({
      error: "Feedback could not be recorded right now."
    });
  }
});

app.post("/chat", async (request, response) => {
  try {
    const body = request.body as ChatRequestBody;
    const question = typeof body.question === "string" ? body.question.trim() : "";

    if (!question) {
      response.status(400).json({ error: "A question is required." });
      return;
    }

    if (!validateReportOutput(body.report)) {
      response.status(400).json({ error: "The provided report payload is invalid." });
      return;
    }

    if (!DEEPSEEK_ENABLED) {
      response.setHeader("x-chat-source", "baseline-local-no-key");
      response.json({
        answer: removeEmoji(buildLocalChatAnswer(body.report, question))
      });
      return;
    }

    const client = ensureDeepSeekClient();
    const chatPrompt = buildChatPrompt({
      report: body.report,
      question
    });

    try {
      const completion = await client.chat.completions.create({
        model: DEEPSEEK_MODEL,
        temperature: 0.4,
        messages: [
          {
            role: "system",
            content:
              "You are a strategic content assistant. Reply in English plain text with concrete recommendations, no emojis, and no generic filler."
          },
          {
            role: "user",
            content: chatPrompt
          }
        ]
      });

      const rawContent = extractTextContent(completion.choices[0]?.message?.content ?? "");
      const answer = removeEmoji(rawContent || "I do not have an answer right now.");

      response.setHeader("x-chat-source", "deepseek");
      response.json({ answer });
    } catch (providerError) {
      console.error(providerError);
      response.setHeader("x-chat-source", "baseline-local-provider-error");
      response.json({
        answer: removeEmoji(buildLocalChatAnswer(body.report, question))
      });
    }
  } catch (error) {
    console.error(error);
    response.status(500).json({
      error: "The chat request could not be completed."
    });
  }
});

if (RECOMMENDER_ENABLED) {
  const warmObjectives = ["reach", "engagement", "conversion"];
  for (const objective of warmObjectives) {
    const key = routeKey("/v1/recommendations", objective);
    void refreshCompatibility(key, { component: "recommender-learning-v1" }, true);
  }
  setInterval(() => {
    for (const objective of warmObjectives) {
      const key = routeKey("/v1/recommendations", objective);
      void refreshCompatibility(key, { component: "recommender-learning-v1" }, true);
    }
  }, Math.max(5000, RECOMMENDER_COMPAT_CHECK_INTERVAL_MS));
}
void feedbackStore.init();

app.listen(SERVER_PORT, () => {
  console.log(`Local API running on http://localhost:${SERVER_PORT}`);
  console.log(
    `DeepSeek enabled: ${DEEPSEEK_ENABLED ? "yes" : "no"} | model: ${DEEPSEEK_MODEL}`
  );
  console.log(`Recommender enabled: ${RECOMMENDER_ENABLED ? "yes" : "no"}`);
  const feedbackStatus = feedbackStore.status();
  console.log(`Feedback store ready: ${feedbackStatus.ready ? "yes" : "no"}`);
});
