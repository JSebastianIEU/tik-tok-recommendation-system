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
  KNOWLEDGE_BASE_PATH,
  RECOMMENDER_BASE_URL,
  RECOMMENDER_FALLBACK_BUNDLE_DIR,
  RECOMMENDER_FALLBACK_CACHE_TTL_MS,
  SERVER_PORT,
  UPLOADED_ASSET_DIR,
  UPLOAD_MAX_BYTES
} from "./config";
import { getSeedVideo } from "../src/services/data/selectDemoSlices";
import type { DemoVideoRecord } from "../src/services/data/types";
import type { ReportOutput } from "../src/features/report/types";
import { loadCandidateCorpus, type CandidateCorpus } from "./corpus/loadCandidateCorpus";
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
import {
  parseChatRequest,
  type ChatHistoryMessage
} from "./validation/parseChatRequest";
import { parseReportFeedbackRequest } from "./validation/parseReportFeedbackRequest";
import { parseRecommendationsRequest } from "./validation/parseRecommendationsRequest";
import { validateReportOutput } from "./validation/validateReportOutput";
import {
  requestFabricSignals,
  requestCompatibility,
  requestRecommendations,
  requestHashtagSuggestions,
  type CorpusScopePayload,
  type FabricExtractResponsePayload,
  type HashtagSuggestion,
  type RecommenderCandidatePayload,
  type RecommenderQueryPayload,
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
  ingestUploadedVideo,
  mergeCandidateSignalHints,
  readUploadedAssetRecord,
  type UploadedAssetRecord
} from "./uploads/assetStore";
import { createUploadAnalysisProvider } from "./uploads/providerFactory";
import { FeedbackGateway } from "./feedback/gateway";
import {
  LabelingSessionStore,
  type LabelingReviewLabel
} from "./labeling/store";
import {
  loadKnowledgeBaseStore,
  searchKnowledgeBase,
  type KnowledgeBaseEntry,
  type KnowledgeBaseStore
} from "./knowledgeBase/knowledgeBase";

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
const uploadAnalysisProvider = createUploadAnalysisProvider();
let knowledgeBaseStore: KnowledgeBaseStore;
try {
  knowledgeBaseStore = await loadKnowledgeBaseStore(KNOWLEDGE_BASE_PATH);
  console.info(
    `Knowledge base loaded: version=${knowledgeBaseStore.version} entries=${knowledgeBaseStore.entries.length}`
  );
} catch (error) {
  const reason = error instanceof Error ? error.message : String(error);
  console.error(`Knowledge base startup validation failed (${KNOWLEDGE_BASE_PATH}): ${reason}`);
  throw error;
}

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

app.post(
  "/upload-video",
  express.raw({
    type: "application/octet-stream",
    limit: UPLOAD_MAX_BYTES
  }),
  async (request, response) => {
    try {
      if (!Buffer.isBuffer(request.body) || request.body.byteLength === 0) {
        response.status(400).json({ error: "Upload body must contain video bytes." });
        return;
      }

      const fileName = request.header("x-file-name")?.trim() || "upload.mp4";
      const mimeType =
        request.header("x-file-type")?.trim() ||
        request.header("content-type")?.trim() ||
        "application/octet-stream";

      if (!mimeType.toLowerCase().startsWith("video/")) {
        response.status(400).json({ error: "Uploaded file must declare a video mime type." });
        return;
      }

      const analysis = await ingestUploadedVideo({
        fileBuffer: request.body,
        fileName,
        mimeType,
        uploadsDir: UPLOADED_ASSET_DIR,
        analysisProvider: uploadAnalysisProvider
      });

      // Generate an LLM-powered caption if no caption was produced by the
      // analyzer (e.g. when running baseline-only without VLM/BLIP) and
      // DeepSeek is available.
      if (!analysis.video_caption && deepSeekClient) {
        try {
          const captionContext: string[] = [];
          if (analysis.transcript) captionContext.push(`Transcript: ${analysis.transcript}`);
          if (analysis.ocr_text) captionContext.push(`On-screen text: ${analysis.ocr_text}`);
          const dur = analysis.duration_seconds ?? analysis.asset?.duration_seconds;
          if (dur) captionContext.push(`Duration: ${dur}s`);
          if (analysis.visual_features) {
            const vf = analysis.visual_features;
            if (vf.resolution) captionContext.push(`Resolution: ${vf.resolution}`);
            if (vf.dominant_colors?.length) captionContext.push(`Colors: ${vf.dominant_colors.join(", ")}`);
          }
          if (analysis.timeline?.length) {
            const sceneCuts = analysis.timeline.filter((f) => f.is_scene_change).length;
            captionContext.push(`Scene cuts: ${sceneCuts}`);
            const avgRelevance =
              analysis.timeline.reduce((s, f) => s + f.relevance_score, 0) / analysis.timeline.length;
            captionContext.push(`Average visual relevance: ${avgRelevance.toFixed(2)}`);
          }
          captionContext.push(`File name: ${fileName}`);

          const captionCompletion = await deepSeekClient.chat.completions.create({
            model: DEEPSEEK_MODEL,
            temperature: 0.5,
            max_tokens: 200,
            messages: [
              {
                role: "system",
                content:
                  "You are a TikTok content strategist. Given video analysis data, write a short, " +
                  "engaging TikTok caption (1-2 sentences) that describes what the video is about. " +
                  "Be specific to the content, not generic. Include a hook. " +
                  "Do NOT include hashtags — those are handled separately. No emojis. Plain text only."
              },
              {
                role: "user",
                content: captionContext.join("\n")
              }
            ]
          });

          const generatedCaption =
            captionCompletion.choices[0]?.message?.content?.trim() ?? "";
          if (generatedCaption) {
            analysis.video_caption = generatedCaption;
          }
        } catch (captionErr) {
          console.warn("LLM caption generation failed:", captionErr instanceof Error ? captionErr.message : String(captionErr));
        }
      }

      response.status(201).json(analysis);
    } catch (error) {
      console.error("Upload analysis failed:", error);
      response.status(500).json({ error: "Failed to analyze uploaded video." });
    }
  }
);

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
const feedbackGateway = new FeedbackGateway({
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

function resolveSeedRecord(
  corpus: CandidateCorpus,
  seedVideoId: string
): DemoVideoRecord | undefined {
  const normalizedSeedVideoId = seedVideoId.trim();
  if (normalizedSeedVideoId) {
    return corpus.byVideoId.get(normalizedSeedVideoId);
  }
  if (corpus.provider === "demo") {
    return getSeedVideo(corpus.records) ?? undefined;
  }
  return undefined;
}

function selectCandidatesForReport(params: {
  corpus: CandidateCorpus;
  candidates: DemoVideoRecord[];
  recommenderResult: RecommenderResult;
  limit?: number;
}): DemoVideoRecord[] {
  const limit = Math.max(1, params.limit ?? 24);
  const selected: DemoVideoRecord[] = [];
  const seen = new Set<string>();

  if (params.recommenderResult.ok) {
    for (const item of params.recommenderResult.payload.items) {
      const candidate = params.corpus.byVideoId.get(item.candidate_id);
      if (!candidate || seen.has(candidate.video_id)) {
        continue;
      }
      selected.push(candidate);
      seen.add(candidate.video_id);
      if (selected.length >= limit) {
        return selected;
      }
    }
  }

  for (const candidate of params.candidates) {
    if (seen.has(candidate.video_id)) {
      continue;
    }
    selected.push(candidate);
    seen.add(candidate.video_id);
    if (selected.length >= limit) {
      break;
    }
  }

  return selected;
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
  assetId?: string;
  seedVideoId?: string;
  description: string;
  hashtags: string[];
  mentions: string[];
  objective?: string;
  locale?: string;
  contentType?: string;
}): string {
  const assetId = (params.assetId || "").trim();
  if (assetId) {
    return `asset_id::${assetId.toLowerCase()}`;
  }
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

function normalizeAudienceForQuery(
  value: unknown
): string | Record<string, unknown> | undefined {
  if (typeof value === "string") {
    const normalized = value.trim();
    return normalized || undefined;
  }
  if (value && typeof value === "object" && !Array.isArray(value)) {
    return value as Record<string, unknown>;
  }
  return undefined;
}

function buildRecommenderQueryPayload(params: {
  queryId?: string;
  description: string;
  hashtags: string[];
  mentions: string[];
  creatorId?: string;
  audience?: unknown;
  primaryCta?: string;
  language?: string;
  locale?: string;
  contentType?: string;
  asOfTimeIso: string;
  signalHints?: Record<string, unknown>;
  additionalTextFragments?: string[];
  topicKey?: string;
}): RecommenderQueryPayload {
  const signalHints = params.signalHints;
  const transcriptText =
    typeof signalHints?.transcript_text === "string" ? signalHints.transcript_text.trim() : "";
  const ocrText =
    typeof signalHints?.ocr_text === "string" ? signalHints.ocr_text.trim() : "";
  const text = [
    params.description,
    ...params.hashtags,
    ...params.mentions,
    transcriptText,
    ocrText,
    ...(params.additionalTextFragments ?? [])
  ]
    .map((value) => value.trim())
    .filter(Boolean)
    .join(" ")
    .trim();

  return {
    ...(params.queryId ? { query_id: params.queryId } : {}),
    description: params.description,
    hashtags: params.hashtags,
    mentions: params.mentions,
    ...(text ? { text } : {}),
    topic_key:
      params.topicKey?.trim() ||
      params.hashtags[0]?.replace(/^#/, "") ||
      params.contentType?.trim().toLowerCase() ||
      "general",
    author_id: params.creatorId,
    audience: normalizeAudienceForQuery(params.audience),
    primary_cta: params.primaryCta,
    language: params.language,
    locale: params.locale,
    content_type: params.contentType,
    as_of_time: params.asOfTimeIso,
    ...(signalHints ? { signal_hints: signalHints } : {})
  };
}

function buildUploadedAssetQueryPayload(params: {
  assetRecord: UploadedAssetRecord;
  description: string;
  hashtags: string[];
  mentions: string[];
  audience?: unknown;
  primaryCta?: string;
  language?: string;
  locale?: string;
  contentType?: string;
  asOfTimeIso: string;
  signalHints?: Record<string, unknown>;
}): RecommenderQueryPayload {
  const asset = params.assetRecord.asset;
  const additionalTextFragments = [
    asset.orientation !== "unknown" ? `${asset.orientation} video` : "",
    asset.has_audio ? "audio track present" : "silent video",
    typeof asset.duration_seconds === "number"
      ? `${Math.round(asset.duration_seconds)} second clip`
      : ""
  ].filter(Boolean);

  return buildRecommenderQueryPayload({
    queryId: params.assetRecord.asset_id,
    description: params.description,
    hashtags: params.hashtags,
    mentions: params.mentions,
    audience: params.audience,
    primaryCta: params.primaryCta,
    language: params.language,
    locale: params.locale,
    contentType: params.contentType,
    asOfTimeIso: params.asOfTimeIso,
    signalHints: params.signalHints,
    additionalTextFragments,
    topicKey:
      params.hashtags[0]?.replace(/^#/, "") ||
      params.contentType?.trim().toLowerCase() ||
      "uploaded_asset"
  });
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
        score_components?: Record<string, number>;
        ranking_reasons?: string[];
        retrieval_branch_scores?: Record<string, number>;
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
      }),
      score_components: item.score_components
        ? Object.fromEntries(
            Object.entries(item.score_components).filter(([, v]) => v !== undefined).map(([k, v]) => [k, Number(v)])
          )
        : undefined,
      ranking_reasons: item.ranking_reasons,
      retrieval_branch_scores: item.retrieval_branch_scores
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

const STOP_WORDS = new Set([
  "a", "an", "the", "and", "or", "but", "is", "are", "was", "were", "be",
  "been", "being", "have", "has", "had", "do", "does", "did", "will", "would",
  "shall", "should", "may", "might", "must", "can", "could", "to", "of", "in",
  "for", "on", "with", "at", "by", "from", "as", "into", "through", "during",
  "before", "after", "about", "between", "under", "above", "up", "down", "out",
  "off", "over", "then", "than", "so", "no", "not", "only", "very", "just",
  "also", "it", "its", "i", "me", "my", "we", "our", "you", "your", "he",
  "she", "they", "them", "this", "that", "these", "those", "am", "if", "how",
  "what", "when", "where", "who", "which", "all", "each", "every", "both",
  "few", "more", "most", "other", "some", "such", "too", "here", "there",
]);

function deriveExtractedKeywords(hashtags: string[], description: string): string[] {
  const fromHashtags = hashtags
    .map((tag) => tag.replace(/^#/, "").trim().toLowerCase())
    .filter((tag) => tag.length >= 2);

  const fromDescription = description
    .toLowerCase()
    .replace(/[^\w\s]/g, " ")
    .split(/\s+/)
    .filter((word) => word.length >= 3 && !STOP_WORDS.has(word));

  const seen = new Set<string>();
  const merged: string[] = [];
  for (const kw of [...fromHashtags, ...fromDescription]) {
    if (!seen.has(kw)) {
      seen.add(kw);
      merged.push(kw);
    }
    if (merged.length >= 8) break;
  }
  return merged;
}

async function normalizeAndEnrichReport(
  report: ReportOutput,
  candidates: DemoVideoRecord[],
  recommenderMeta?: {
    source: "recommender" | "fallback";
    note?: string;
    commentSummary?: string;
  },
  explainabilitySection?: ReportOutput["explainability"],
  extractedKeywords?: string[]
): Promise<ReportOutput> {
  const keywords = extractedKeywords && extractedKeywords.length > 0
    ? extractedKeywords
    : [...HARD_CODED_EXTRACTED_KEYWORDS];
  const normalized = normalizeReportOutput(report, {
    candidatesK: candidates.length,
    extractedKeywords: keywords,
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
  supplyCandidatesToPython?: boolean;
  corpusScope?: CorpusScopePayload;
  asOfTimeIso: string;
  query?: RecommenderQueryPayload;
  audience?: unknown;
  primaryCta?: string;
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
    query:
      params.query ??
      buildRecommenderQueryPayload({
        description: params.description,
        hashtags: params.hashtags,
        mentions: params.mentions,
        creatorId: params.creatorId,
        audience: params.audience,
        primaryCta: params.primaryCta,
        language: params.language,
        locale: params.locale,
        contentType: params.contentType,
        asOfTimeIso: params.asOfTimeIso
      }),
    user_context: params.userContext,
    language: params.language,
    locale: params.locale,
    content_type: params.contentType,
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
  if (params.supplyCandidatesToPython === true) {
    payload.candidates = buildRecommenderCandidates(params.candidates, params.asOfTimeIso);
  } else if (params.corpusScope) {
    payload.corpus_scope = params.corpusScope;
  } else {
    payload.candidates = buildRecommenderCandidates(params.candidates, params.asOfTimeIso);
  }
  if (params.candidateIds && params.candidateIds.length > 0) {
    payload.candidate_ids = params.candidateIds;
  }
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
    const assetRecord = body.asset_id
      ? await readUploadedAssetRecord(UPLOADED_ASSET_DIR, body.asset_id)
      : null;
    if (body.asset_id && !assetRecord) {
      response.status(404).json({ error: "Uploaded asset was not found." });
      return;
    }
    const mergedSignalHints = mergeCandidateSignalHints(
      assetRecord?.signal_hints,
      body.signal_hints
    );
    const isUploadedQuery = Boolean(body.asset_id && assetRecord);
    const uploadedAssetRecord = isUploadedQuery
      ? (assetRecord as UploadedAssetRecord)
      : undefined;
    const effectiveReferenceId = body.asset_id ?? body.seed_video_id;
    const queryPayload = isUploadedQuery
      ? buildUploadedAssetQueryPayload({
          assetRecord: uploadedAssetRecord,
          description: body.description,
          hashtags: body.hashtags,
          mentions: body.mentions,
          audience: body.audience,
          primaryCta: body.primary_cta,
          language:
            body.language ??
            (typeof body.locale === "string"
              ? body.locale.split("-")[0]?.toLowerCase()
              : undefined),
          locale: body.locale,
          contentType: body.content_type,
          asOfTimeIso: body.as_of_time,
          signalHints:
            mergedSignalHints && typeof mergedSignalHints === "object"
              ? (mergedSignalHints as Record<string, unknown>)
              : undefined
        })
      : undefined;
    const experimentAssignment = buildExperimentAssignment({
      objectiveRequested: body.objective,
      assignmentUnit: buildExperimentUnitKey({
        assetId: body.asset_id,
        seedVideoId: isUploadedQuery ? undefined : body.seed_video_id,
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
      videoId: effectiveReferenceId || "uploaded-seed",
      asOfTime: body.as_of_time,
      description: body.description,
      hashtags: body.hashtags,
      keywords: body.hashtags,
      contentType: body.content_type,
      signalHints:
        mergedSignalHints && typeof mergedSignalHints === "object"
          ? (mergedSignalHints as Record<string, unknown>)
          : undefined
    });
    const candidateSignals = resolvedSignals.signals;
    response.setHeader("x-signal-source", resolvedSignals.source);
    if (body.asset_id) {
      response.setHeader("x-upload-asset-id", body.asset_id);
    }

    const corpus = await loadCandidateCorpus();
    if (corpus.records.length === 0) {
      response.status(400).json({ error: "Candidate corpus is empty or invalid." });
      return;
    }
    response.setHeader("x-corpus-provider", corpus.provider);
    const seedVideoId = isUploadedQuery ? "" : body.seed_video_id;
    const seedRecord = isUploadedQuery
      ? undefined
      : resolveSeedRecord(corpus, seedVideoId);
    if (!isUploadedQuery && !seedRecord) {
      response.status(404).json({ error: "Seed video was not found in the candidate corpus." });
      return;
    }
    const creatorId = extractCreatorId(seedRecord);
    const rawBody = request.body as Record<string, unknown>;
    const reqUserId =
      typeof rawBody.user_id === "string" && rawBody.user_id.trim()
        ? rawBody.user_id.trim()
        : typeof request.headers["x-user-id"] === "string" && request.headers["x-user-id"].trim()
          ? request.headers["x-user-id"].trim()
          : undefined;
    const userContext = isUploadedQuery
      ? undefined
      : await feedbackGateway.getCreatorContext({
          creatorId: creatorId ?? reqUserId,
          userId: reqUserId,
          objective: body.objective,
          mapObjective: mapObjectiveForRecommender
        });
    const candidates = getCombinedCandidates(corpus.records, seedVideoId);
    if (candidates.length === 0) {
      response.status(400).json({ error: "No comparable candidates were found." });
      return;
    }

    const effectiveLanguage =
      body.language ??
      (typeof body.locale === "string" ? body.locale.split("-")[0]?.toLowerCase() : undefined);
    const usePythonCorpus = corpus.retrievalMode === "python_bundle";
    const corpusScope: CorpusScopePayload | undefined = usePythonCorpus
      ? {
          language: effectiveLanguage,
          locale: body.locale,
          content_type: body.content_type,
          topic_key:
            body.hashtags[0]?.replace(/^#/, "") ||
            body.content_type ||
            undefined,
          max_candidates: Math.max(1, body.retrieve_k ?? 200),
          exclude_video_ids: isUploadedQuery ? [] : seedVideoId ? [seedVideoId] : []
        }
      : undefined;

    const recommenderAttempt = await fetchRecommenderResult({
      requestId,
      objective: body.objective,
      description: body.description,
      hashtags: body.hashtags,
      mentions: body.mentions,
      candidates,
      supplyCandidatesToPython: corpus.retrievalMode === "supplied_candidates",
      corpusScope,
      asOfTimeIso: body.as_of_time,
      query: queryPayload,
      audience: body.audience,
      primaryCta: body.primary_cta,
      language: effectiveLanguage,
      locale: body.locale,
      contentType: body.content_type,
      candidateIds:
        body.candidate_ids.length > 0
          ? body.candidate_ids
          : corpus.retrievalMode === "supplied_candidates"
            ? candidates.map((candidate) => candidate.video_id)
            : undefined,
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
      feedbackGateway.traceRecommendation({
        requestId,
        endpoint: "/recommendations",
        requestReceivedAt,
        objectiveRequested: mapObjectiveForRecommender(body.objective),
        objectiveEffective: String((payloadForLogging.objective_effective as string) || "engagement"),
        experimentAssignment,
        creatorId,
        seedVideoId,
        trafficMeta,
        payload: payloadForLogging
      });
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
      feedbackGateway.traceRecommendation({
        requestId,
        endpoint: "/recommendations",
        requestReceivedAt,
        objectiveRequested: mapObjectiveForRecommender(body.objective),
        objectiveEffective: String(bundlePayload.objective_effective || "engagement"),
        experimentAssignment,
        creatorId,
        seedVideoId,
        trafficMeta,
        payload: bundlePayload as Record<string, unknown>
      });
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
    feedback_store: feedbackGateway.status()
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
    const assetRecord = body.asset_id
      ? await readUploadedAssetRecord(UPLOADED_ASSET_DIR, body.asset_id)
      : null;
    if (body.asset_id && !assetRecord) {
      response.status(404).json({ error: "Uploaded asset was not found." });
      return;
    }
    const mergedSignalHints = mergeCandidateSignalHints(
      assetRecord?.signal_hints,
      body.signal_hints
    );
    const isUploadedQuery = Boolean(body.asset_id && assetRecord);
    const uploadedAssetRecord = isUploadedQuery
      ? (assetRecord as UploadedAssetRecord)
      : undefined;
    const effectiveReferenceId = body.asset_id ?? body.seed_video_id;
    const queryPayload = isUploadedQuery
      ? buildUploadedAssetQueryPayload({
          assetRecord: uploadedAssetRecord,
          description: body.description,
          hashtags: body.hashtags,
          mentions: body.mentions,
          audience: body.audience,
          primaryCta: body.primary_cta,
          language:
            body.language ??
            (typeof body.locale === "string"
              ? body.locale.split("-")[0]?.toLowerCase()
              : undefined),
          locale: body.locale,
          contentType: body.content_type,
          asOfTimeIso: new Date().toISOString(),
          signalHints:
            mergedSignalHints && typeof mergedSignalHints === "object"
              ? (mergedSignalHints as Record<string, unknown>)
              : undefined
        })
      : undefined;
    const resolvedSignals = await resolveCandidateSignals({
      videoId: effectiveReferenceId || "uploaded-seed",
      asOfTime: new Date().toISOString(),
      description: body.description,
      hashtags: body.hashtags,
      keywords: body.hashtags,
      contentType: body.content_type,
      signalHints:
        mergedSignalHints && typeof mergedSignalHints === "object"
          ? (mergedSignalHints as Record<string, unknown>)
          : undefined
    });
    const candidateSignals = resolvedSignals.signals;
    response.setHeader("x-signal-source", resolvedSignals.source);
    if (body.asset_id) {
      response.setHeader("x-upload-asset-id", body.asset_id);
    }
    const description = body.description.trim();
    const mentions = body.mentions.map((mention) =>
      mention.trim().startsWith("@") ? mention.trim() : `@${mention.trim()}`
    );
    const hashtags = body.hashtags.map((tag) =>
      tag.trim().startsWith("#") ? tag.trim() : `#${tag.trim()}`
    );
    const seedVideoId = isUploadedQuery ? "" : body.seed_video_id;

    const corpus = await loadCandidateCorpus();
    if (corpus.records.length === 0) {
      response.status(400).json({ error: "Candidate corpus is empty or invalid." });
      return;
    }
    response.setHeader("x-corpus-provider", corpus.provider);

    const seed = isUploadedQuery
      ? undefined
      : resolveSeedRecord(corpus, seedVideoId);
    if (!isUploadedQuery && !seed) {
      response.status(404).json({ error: "Seed video was not found in the candidate corpus." });
      return;
    }
    const creatorId = extractCreatorId(seed);
    const reportRawBody = request.body as Record<string, unknown>;
    const reportUserId =
      typeof reportRawBody.user_id === "string" && reportRawBody.user_id.trim()
        ? reportRawBody.user_id.trim()
        : typeof request.headers["x-user-id"] === "string" && request.headers["x-user-id"].trim()
          ? request.headers["x-user-id"].trim()
          : undefined;
    const userContext = isUploadedQuery
      ? undefined
      : await feedbackGateway.getCreatorContext({
          creatorId: creatorId ?? reportUserId,
          userId: reportUserId,
          objective: body.objective,
          mapObjective: mapObjectiveForRecommender
        });

    const candidates = getCombinedCandidates(corpus.records, seedVideoId);
    if (candidates.length === 0) {
      response.status(400).json({ error: "No comparable candidates were found." });
      return;
    }
    const reportRequestId = createUuidV7();
    const reportExperimentAssignment = buildExperimentAssignment({
      objectiveRequested: body.objective,
      assignmentUnit: buildExperimentUnitKey({
        assetId: body.asset_id,
        seedVideoId: isUploadedQuery ? undefined : body.seed_video_id,
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

    const reportEffectiveLanguage =
      body.language ??
      (typeof body.locale === "string" ? body.locale.split("-")[0]?.toLowerCase() : undefined);
    const reportUsePythonCorpus = corpus.retrievalMode === "python_bundle";
    const reportCorpusScope: CorpusScopePayload | undefined = reportUsePythonCorpus
      ? {
          language: reportEffectiveLanguage,
          locale: body.locale,
          content_type: body.content_type,
          topic_key:
            hashtags[0]?.replace(/^#/, "") ||
            body.content_type ||
            undefined,
          max_candidates: 200,
          exclude_video_ids: seedVideoId ? [seedVideoId] : []
        }
      : undefined;

    const recommenderAttempt = await fetchRecommenderResult({
      requestId: reportRequestId,
      objective: body.objective,
      description,
      hashtags,
      mentions,
      candidates,
      supplyCandidatesToPython: corpus.retrievalMode === "supplied_candidates",
      corpusScope: reportCorpusScope,
      asOfTimeIso: new Date().toISOString(),
      query: queryPayload,
      audience: body.audience,
      primaryCta: body.primary_cta,
      language: reportEffectiveLanguage,
      locale: body.locale,
      contentType: body.content_type,
      candidateIds:
        corpus.retrievalMode === "supplied_candidates"
          ? candidates.map((candidate) => candidate.video_id)
          : undefined,
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
    const rankedCandidates = selectCandidatesForReport({
      corpus,
      candidates,
      recommenderResult,
      limit: 24
    });
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

    const extractedKeywords = deriveExtractedKeywords(hashtags, description);

    let suggestedHashtags: HashtagSuggestion[] = [];
    try {
      const captionForHashtags = [description, ...hashtags.map(h => `#${h}`)].join(" ").trim();
      if (captionForHashtags) {
        const hashtagResult = await requestHashtagSuggestions({
          caption: captionForHashtags,
          top_n: 10,
          exclude_tags: hashtags.map(h => h.startsWith("#") ? h : `#${h}`),
          include_neighbours: false
        });
        if (hashtagResult.ok) {
          suggestedHashtags = hashtagResult.payload.suggestions;
        }
      }
    } catch {
      /* hashtag suggestions are best-effort */
    }

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
      reasoning: reasoningArtifacts.reasoning,
      extractedKeywords
    });

    const persistedPayloadForLogging = mergeGatewayMeta(
      recommenderPayload as Record<string, unknown>,
      recommenderAttempt.gatewayMeta
    );
    feedbackGateway.traceReport({
      requestId: reportRequestId,
      requestReceivedAt,
      objectiveRequested: mapObjectiveForRecommender(body.objective),
      objectiveEffective:
        String((persistedPayloadForLogging.objective_effective as string) || "engagement"),
      experimentAssignment: reportExperimentAssignment,
      creatorId,
      seedVideoId: body.seed_video_id,
      payload: persistedPayloadForLogging
    });

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
        explainabilitySection,
        extractedKeywords
      );

      response.json({
        report,
        suggested_hashtags: suggestedHashtags
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
          explainabilitySection,
          extractedKeywords
        );
  
        response.json({
          report,
          suggested_hashtags: suggestedHashtags
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
          explainabilitySection,
          extractedKeywords
        );
        response.json({
          report,
          suggested_hashtags: suggestedHashtags
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
        explainabilitySection,
        extractedKeywords
      );

      response.json({
        report,
        suggested_hashtags: suggestedHashtags
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
        explainabilitySection,
        extractedKeywords
      );

      response.json({
        report,
        suggested_hashtags: suggestedHashtags
      });
    }
  } catch (error) {
    console.error(error);
    response.status(500).json({
      error: "The report could not be generated right now."
    });
  }
});

app.post("/hashtags/suggest", async (request, response) => {
  try {
    const { caption, top_n, exclude_tags, include_neighbours } = request.body as {
      caption?: string;
      top_n?: number;
      exclude_tags?: string[];
      include_neighbours?: boolean;
    };

    if (!caption || typeof caption !== "string" || !caption.trim()) {
      response.status(400).json({ error: "A caption is required." });
      return;
    }

    const result = await requestHashtagSuggestions({
      caption: caption.trim(),
      top_n: top_n ?? 15,
      exclude_tags: exclude_tags ?? [],
      include_neighbours: include_neighbours ?? false
    });

    if (!result.ok) {
      response.status(502).json({ error: result.error, suggestions: [] });
      return;
    }

    // Python API returns { hashtags: [...] }, frontend expects { suggestions: [...] }
    const raw = result.payload as Record<string, unknown>;
    const suggestions = raw.suggestions ?? raw.hashtags ?? [];
    response.json({ suggestions });
  } catch (error) {
    console.error("hashtag suggest error:", error);
    response.status(500).json({ error: "Could not fetch hashtag suggestions.", suggestions: [] });
  }
});

app.post("/report-feedback", async (request, response) => {
  const parsed = parseReportFeedbackRequest(request.body);
  if (!parsed.ok) {
    response.status(400).json({ error: parsed.error });
    return;
  }

  const result = await feedbackGateway.recordUiFeedback({
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
    createdAt: new Date().toISOString(),
    userId: parsed.value.user_id ?? null
  });
  response.status(202).json(result);
});

// ---------------------------------------------------------------------------
// Agentic chatbot tools — RAG retrieval + hashtag suggestion
// ---------------------------------------------------------------------------

interface RAGVideo {
  video_id: string;
  caption: string;
  hashtags: string[];
  keywords: string[];
  author_id: string;
  content_type: string;
  language?: string;
  fused_score: number;
  branch_scores?: Record<string, number>;
}

interface ToolResult {
  source: string;
  videos?: RAGVideo[];
  hashtags?: string[];
  knowledgeBaseEntries?: KnowledgeBaseEntry[];
  knowledgeBaseMeta?: Record<string, unknown>;
  retrievalMeta?: Record<string, unknown>;
  error?: string;
}

function truncateForPrompt(value: string, maxChars: number): string {
  const trimmed = value.trim();
  if (!trimmed) {
    return "";
  }
  if (trimmed.length <= maxChars) {
    return trimmed;
  }
  return `${trimmed.slice(0, maxChars)}...`;
}

function sanitizeChatCaption(value: string, maxChars: number): string {
  return truncateForPrompt(value.replace(/\s+/g, " ").trim(), maxChars);
}

function formatCorpusExamples(videos: RAGVideo[]): string {
  const exampleLines = videos.slice(0, 2).map((video, index) => {
    const caption = sanitizeChatCaption(video.caption, 120);
    const tagPreview = (video.hashtags ?? []).slice(0, 3).join(", ");
    const score = Number.isFinite(video.fused_score) ? video.fused_score.toFixed(2) : "n/a";
    const tagSuffix = tagPreview ? ` | tags: ${tagPreview}` : "";
    return `${index + 1}. "${caption}" (score ${score}${tagSuffix})`;
  });

  if (exampleLines.length === 0) {
    return "";
  }

  return `Relevant examples from similar videos:\n${exampleLines.join("\n")}`;
}

function chatWantsExamples(question: string): boolean {
  const normalized = question.toLowerCase();
  return (
    normalized.includes("example") ||
    normalized.includes("examples") ||
    normalized.includes("similar video") ||
    normalized.includes("reference")
  );
}

function chatNeedsKnowledgeBase(question: string): boolean {
  const normalized = question.toLowerCase();
  return (
    normalized.includes("algorithm") ||
    normalized.includes("what performs") ||
    normalized.includes("what works") ||
    normalized.includes("top creator") ||
    normalized.includes("top creators") ||
    normalized.includes("top hashtag") ||
    normalized.includes("top hashtags") ||
    normalized.includes("trend")
  );
}

function uniqueStrings(values: string[], maxItems: number): string[] {
  const output: string[] = [];
  const seen = new Set<string>();
  for (const value of values) {
    const normalized = value.trim();
    if (!normalized) {
      continue;
    }
    const dedupeKey = normalized.toLowerCase();
    if (seen.has(dedupeKey)) {
      continue;
    }
    seen.add(dedupeKey);
    output.push(normalized);
    if (output.length >= maxItems) {
      break;
    }
  }
  return output;
}

function chatNeedsHashtags(question: string): boolean {
  const q = question.toLowerCase();
  return q.includes("hashtag") || q.includes("tag") || q.includes("#");
}

function resolveChatObjective(params: {
  objectiveEffective?: string;
  report: ReportOutput | null;
}): string {
  const objectiveCandidate =
    params.objectiveEffective ||
    params.report?.meta.objective_effective ||
    params.report?.meta.objective;
  const mapped = mapObjectiveForRecommender(objectiveCandidate);
  return mapped === "community" ? "engagement" : mapped;
}

function inferMetricFocus(question: string, objective: string): string {
  const normalized = question.toLowerCase();
  if (normalized.includes("retention")) return "retention";
  if (normalized.includes("hook")) return "hook_strength";
  if (normalized.includes("share")) return "share_rate";
  if (normalized.includes("comment")) return "comment_rate";
  if (normalized.includes("save")) return "save_rate";
  if (normalized.includes("view")) return "views";
  if (normalized.includes("watch time")) return "watch_time";
  return objective === "conversion" ? "conversion_rate" : objective === "reach" ? "reach" : "engagement_rate";
}

function buildHistoryContext(history: ChatHistoryMessage[]): {
  summary: string;
  recentTurns: string[];
} {
  const filtered = history
    .map((item) => ({
      role: item.role,
      content: truncateForPrompt(item.content, 360)
    }))
    .filter((item) => item.content.length > 0);

  const withoutIntro = filtered.filter(
    (item) => !item.content.toLowerCase().startsWith("report loaded. ask me")
  );
  const usable = withoutIntro.length > 0 ? withoutIntro : filtered;
  const recent = usable.slice(-8);
  const older = usable.slice(0, Math.max(0, usable.length - recent.length));
  const summary = older.length
    ? older
        .slice(-6)
        .map((item) => `${item.role}: ${item.content}`)
        .join(" | ")
    : "";
  const recentTurns = recent.map((item) => `${item.role}: ${item.content}`);
  return { summary, recentTurns };
}

function buildReportRagSignals(report: ReportOutput | null): {
  hashtags: string[];
  keywords: string[];
  language?: string;
  locale?: string;
  contentType?: string;
  primaryCta?: string;
  topicKey?: string;
} {
  if (!report) {
    return { hashtags: [], keywords: [] };
  }

  const topComparables = report.comparables.slice(0, 8);
  const hashtags = uniqueStrings(
    topComparables.flatMap((item) => item.hashtags || []),
    24
  );
  const keywords = uniqueStrings(
    [
      ...report.executive_summary.extracted_keywords,
      ...topComparables.flatMap((item) => item.matched_keywords || [])
    ],
    30
  );
  const querySummary = report.reasoning?.evidence_pack?.query_summary;
  const contentType =
    typeof querySummary?.content_type === "string" ? querySummary.content_type : undefined;
  const primaryCta =
    typeof querySummary?.primary_cta === "string" ? querySummary.primary_cta : undefined;
  const language =
    typeof querySummary?.language === "string" ? querySummary.language : undefined;
  const locale =
    typeof querySummary?.locale === "string" ? querySummary.locale : undefined;

  return {
    hashtags,
    keywords,
    language,
    locale,
    contentType,
    primaryCta,
    topicKey: keywords[0] || hashtags[0]?.replace(/^#/, "")
  };
}

function buildEvidenceRefs(report: ReportOutput | null, tools: ToolResult[]): string[] {
  const refs: string[] = [];
  if (report) {
    refs.push(...report.recommendations.items.flatMap((item) => item.evidence_refs || []));
    refs.push(...report.comparables.slice(0, 5).map((item) => `candidate:${item.candidate_id}`));
  }
  const corpusTool = tools.find((tool) => tool.source === "corpus_search");
  if (corpusTool?.videos) {
    refs.push(...corpusTool.videos.slice(0, 8).map((item) => `retrieved:${item.video_id}`));
  }
  return uniqueStrings(refs, 24);
}

async function callRAGRetrieval(params: {
  question: string;
  objective: string;
  report: ReportOutput | null;
  videoAnalysis: Record<string, unknown> | null;
  history: ChatHistoryMessage[];
}): Promise<ToolResult> {
  const reportSignals = buildReportRagSignals(params.report);
  const recentUserQuestions = params.history
    .filter((item) => item.role === "user")
    .slice(-3)
    .map((item) => truncateForPrompt(item.content, 240));
  const transcriptHint =
    typeof params.videoAnalysis?.transcript === "string"
      ? truncateForPrompt(params.videoAnalysis.transcript, 800)
      : undefined;

  const payload = {
    question: params.question,
    top_k: 8,
    objective: params.objective,
    report_hashtags: reportSignals.hashtags,
    report_keywords: reportSignals.keywords,
    topic_key: reportSignals.topicKey,
    language: reportSignals.language,
    locale: reportSignals.locale,
    content_type: reportSignals.contentType,
    primary_cta: reportSignals.primaryCta,
    transcript_hint: transcriptHint,
    recent_user_questions: recentUserQuestions
  };

  try {
    const res = await fetch(`${RECOMMENDER_BASE_URL}/v1/chat/rag`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(10_000),
    });
    if (!res.ok) return { source: "corpus_search", error: `HTTP ${res.status}` };
    const data = await res.json() as {
      retrieved_videos?: RAGVideo[];
      retrieval_meta?: Record<string, unknown>;
    };
    return {
      source: "corpus_search",
      videos: data.retrieved_videos ?? [],
      retrievalMeta: data.retrieval_meta ?? {}
    };
  } catch (err) {
    return { source: "corpus_search", error: String(err) };
  }
}

async function callHashtagSuggest(question: string): Promise<ToolResult> {
  try {
    const res = await fetch(`${RECOMMENDER_BASE_URL}/v1/hashtags/suggest`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ caption: question, k: 10, top_n: 10 }),
      signal: AbortSignal.timeout(10_000),
    });
    if (!res.ok) return { source: "hashtags", error: `HTTP ${res.status}` };
    const data = await res.json() as { hashtags: string[] };
    return { source: "hashtags", hashtags: data.hashtags ?? [] };
  } catch (err) {
    return { source: "hashtags", error: String(err) };
  }
}

async function callKnowledgeBaseSearch(params: {
  question: string;
  objective: string;
  priority: "high" | "low";
}): Promise<ToolResult> {
  try {
    const result = searchKnowledgeBase(knowledgeBaseStore, {
      question: params.question,
      objective: params.objective,
      maxResults: 3,
      priority: params.priority
    });
    return {
      source: "knowledge_base",
      knowledgeBaseEntries: result.entries,
      knowledgeBaseMeta: {
        priority: result.priority,
        matched_categories: result.matched_categories
      }
    };
  } catch (error) {
    return {
      source: "knowledge_base",
      error: error instanceof Error ? error.message : String(error)
    };
  }
}

interface TimelineFrame {
  timestamp_sec: number;
  relevance_score: number;
  motion_score: number;
  face_count: number;
  is_scene_change: boolean;
  ocr_text: string;
}

function buildAgenticPrompt(params: {
  question: string;
  objectiveEffective: string;
  metricFocus: string;
  report: ReportOutput | null;
  videoAnalysis: Record<string, unknown> | null;
  history: ChatHistoryMessage[];
  toolResults: ToolResult[];
}): string {
  const sections: string[] = [];
  const historyContext = buildHistoryContext(params.history);

  sections.push(
    `## Coaching Goal\nObjective: ${params.objectiveEffective}\nMetric focus: ${params.metricFocus}`
  );

  if (historyContext.summary || historyContext.recentTurns.length > 0) {
    const memoryLines: string[] = [];
    if (historyContext.summary) {
      memoryLines.push(`Earlier summary: ${historyContext.summary}`);
    }
    if (historyContext.recentTurns.length > 0) {
      memoryLines.push(
        "Recent turns:\n" + historyContext.recentTurns.map((line) => `- ${line}`).join("\n")
      );
    }
    sections.push(`## Conversation Memory\n${memoryLines.join("\n")}`);
  }

  // Video analysis context (timeline-aware)
  if (params.videoAnalysis) {
    const va = params.videoAnalysis;
    const dur = va.duration_seconds ?? "unknown";
    const faces = va.face_count ?? 0;
    const scenes = va.scene_cuts ?? 0;
    const transcript =
      typeof va.transcript === "string" ? truncateForPrompt(va.transcript, 900) : "none";
    const caption =
      typeof va.video_caption === "string" ? truncateForPrompt(va.video_caption, 280) : "none";

    sections.push(
      `## Your Video Analysis\nDuration: ${dur}s | Faces detected: ${faces} | Scene cuts: ${scenes}\nTranscript: ${transcript}\nVLM Caption: ${caption}`
    );

    const timeline = va.timeline as TimelineFrame[] | undefined;
    if (timeline && timeline.length > 0) {
      const selectedTimeline = timeline.slice(0, 24);
      const lines = selectedTimeline.map((f) => {
        let line = `[${Number(f.timestamp_sec).toFixed(1)}s] relevance=${Math.round(f.relevance_score * 100)}% motion=${Math.round(f.motion_score * 100)}% faces=${f.face_count}`;
        if (f.is_scene_change) line += " SCENE_CHANGE";
        if (f.ocr_text) line += ` text="${truncateForPrompt(f.ocr_text, 90)}"`;
        return line;
      });
      sections.push(`## Frame-by-Frame Timeline\n${lines.join("\n")}`);
    }
  }

  // Report context
  if (params.report) {
    const r = params.report;
    const metricLines = r.executive_summary.metrics
      .slice(0, 8)
      .map((metric) => `- ${metric.label}: ${metric.value}`);
    const topComps = r.comparables
      .slice(0, 5)
      .map(
        (c, idx) =>
          `${idx + 1}. "${truncateForPrompt(c.caption, 160)}" [${c.hashtags
            .slice(0, 5)
            .join(", ")}] | support=${c.support_level} | why=${truncateForPrompt(c.why_this_was_chosen, 160)} | reasons=${c.ranking_reasons.slice(0, 3).join(", ")}`
      )
      .join("\n  ");
    const recs = r.recommendations.items
      .slice(0, 5)
      .map((item) => `- ${item.title}: ${truncateForPrompt(item.rationale, 190)} (evidence: ${item.evidence})`)
      .join("\n");
    const reasoningRecs = r.reasoning.recommendation_units
      .slice(0, 4)
      .map(
        (item) =>
          `- ${truncateForPrompt(item.action, 150)} | rationale=${truncateForPrompt(item.rationale, 150)} | area=${item.expected_effect_area} | confidence=${item.confidence.toFixed(2)}`
      )
      .join("\n");
    sections.push(
      `## Recommendation Report\nSummary: ${truncateForPrompt(r.executive_summary.summary_text, 320)}\n\nKey metrics:\n${metricLines.join("\n")}\n\nTop comparables:\n  ${topComps}\n\nRecommendations:\n${recs}\n\nReasoning recommendation units:\n${reasoningRecs}`
    );
  }

  // Tool results
  for (const tool of params.toolResults) {
    if (tool.source === "corpus_search" && tool.videos && tool.videos.length > 0) {
      const videoLines = tool.videos.map(
        (v) =>
          `- "${truncateForPrompt(v.caption, 140)}" [hashtags: ${v.hashtags.join(", ")}] score=${v.fused_score} type=${v.content_type} language=${v.language ?? "unknown"} branch_scores=${JSON.stringify(v.branch_scores ?? {})}`
      );
      sections.push(
        `## Similar Videos Retrieved from Corpus (${tool.videos.length} results)\n${videoLines.join("\n")}`
      );
      if (tool.retrievalMeta) {
        sections.push(`## Retrieval Metadata\n${JSON.stringify(tool.retrievalMeta, null, 2)}`);
      }
    }
    if (tool.source === "hashtags" && tool.hashtags && tool.hashtags.length > 0) {
      sections.push(`## AI-Suggested Hashtags\n${tool.hashtags.join(", ")}`);
    }
    if (
      tool.source === "knowledge_base" &&
      tool.knowledgeBaseEntries &&
      tool.knowledgeBaseEntries.length > 0
    ) {
      const lines = tool.knowledgeBaseEntries.map((entry, index) => {
        const facts = entry.content.slice(0, 2).map((line) => truncateForPrompt(line, 120)).join(" | ");
        return `${index + 1}. ${entry.title} [${entry.category}] -> action: ${truncateForPrompt(entry.action_hint, 120)} | facts: ${facts}`;
      });
      sections.push(`## TikTok Knowledge Base Insights\n${lines.join("\n")}`);
    }
  }

  sections.push(
    "## Response Format Requirements\n" +
      "Respond in this exact structure:\n" +
      "1) Quick diagnosis (one sentence)\n" +
      "2) Top 3 actions (numbered)\n" +
      "3) Expected impact (short lines tied to metrics)\n" +
      "4) One follow-up question\n" +
      "Do not include a weekly test plan.\n" +
      "Do not include extra sections unless the user explicitly requests them."
  );

  sections.push(`## User Question\n${params.question}`);

  return sections.join("\n\n");
}

// ---------------------------------------------------------------------------
// /chat — agentic chatbot with RAG + timeline awareness + tool use
// ---------------------------------------------------------------------------

app.post("/chat", async (request, response) => {
  try {
    const parsed = parseChatRequest(request.body);
    if (!parsed.ok) {
      response.status(400).json({ error: parsed.error });
      return;
    }
    const { question, report, videoAnalysis, history } = parsed.value;
    const objectiveEffective = resolveChatObjective({
      objectiveEffective: parsed.value.objectiveEffective,
      report
    });
    const metricFocus = parsed.value.metricFocus || inferMetricFocus(question, objectiveEffective);

    // --- Tool calls: corpus RAG + KB retrieval (blend mode) ---
    const toolPromises: Promise<ToolResult>[] = [];
    toolPromises.push(
      callRAGRetrieval({
        question,
        objective: objectiveEffective,
        report,
        videoAnalysis,
        history
      })
    );
    if (chatNeedsHashtags(question)) {
      toolPromises.push(callHashtagSuggest(question));
    }
    toolPromises.push(
      callKnowledgeBaseSearch({
        question,
        objective: objectiveEffective,
        priority: chatNeedsKnowledgeBase(question) ? "high" : "low"
      })
    );

    const toolResults = await Promise.allSettled(toolPromises);
    const resolvedTools: ToolResult[] = toolResults
      .filter((r): r is PromiseFulfilledResult<ToolResult> => r.status === "fulfilled")
      .map((r) => r.value);

    // --- Fallback if no LLM ---
    if (!DEEPSEEK_ENABLED) {
      // Enhanced local fallback that includes tool results
      const knowledgeTool = resolvedTools.find((tool) => tool.source === "knowledge_base");
      const knowledgeEntries = knowledgeTool?.knowledgeBaseEntries ?? [];
      let fallbackAnswer: string;
      fallbackAnswer = buildLocalChatAnswer({
        report,
        question,
        knowledgeBaseEntries: knowledgeEntries
      });

      const hashtagTool = resolvedTools.find((t) => t.source === "hashtags");
      if (hashtagTool?.hashtags && hashtagTool.hashtags.length > 0) {
        const topHashtags = uniqueStrings(hashtagTool.hashtags, 6);
        if (topHashtags.length > 0) {
          fallbackAnswer += `\n\nSuggested hashtags: ${topHashtags.join(", ")}`;
        }
      }

      const ragTool = resolvedTools.find((t) => t.source === "corpus_search");
      if (chatWantsExamples(question) && ragTool?.videos && ragTool.videos.length > 0) {
        const examples = formatCorpusExamples(ragTool.videos);
        if (examples) {
          fallbackAnswer += `\n\n${examples}`;
        }
      }

      response.setHeader("x-chat-source", "baseline-local-with-tools");
      response.json({
        answer: removeEmoji(fallbackAnswer)
      });
      return;
    }

    // --- Build agentic prompt ---
    const agenticPrompt = buildAgenticPrompt({
      question,
      objectiveEffective,
      metricFocus,
      report,
      videoAnalysis,
      history,
      toolResults: resolvedTools,
    });

    const client = ensureDeepSeekClient();

    try {
      const completion = await client.chat.completions.create({
        model: DEEPSEEK_MODEL,
        temperature: 0.4,
        messages: [
          {
            role: "system",
            content:
              "You are a TikTok video content strategist with access to frame-by-frame video analysis, " +
              "a recommendation report with comparable videos, and a searchable corpus of 13,000+ TikTok videos. " +
              "Reference specific timestamps, metrics, and video data when answering. " +
              "Be concrete and actionable. No emojis. No generic filler. " +
              "When discussing the user's video, cite timeline data (timestamps, relevance scores, scene changes). " +
              "When suggesting content strategy, reference comparable videos and their engagement patterns. " +
              "Always explain expected metric impact and keep recommendations prioritized. " +
              "Use this output structure: Quick diagnosis (one sentence), Top 3 actions (numbered), " +
              "Expected impact (short metric-linked lines), and one follow-up question. " +
              "Do not include a weekly test plan. " +
              "Do not add a 'relevant examples' section unless the user explicitly asks for examples."
          },
          {
            role: "user",
            content: agenticPrompt
          }
        ]
      });

      const rawContent = extractTextContent(completion.choices[0]?.message?.content ?? "");
      const answer = removeEmoji(rawContent || "I do not have an answer right now.");

      response.setHeader("x-chat-source", "deepseek-agentic");
      response.json({ answer });
    } catch (providerError) {
      console.error(providerError);
      const fallback = buildLocalChatAnswer({
        report,
        question,
        knowledgeBaseEntries:
          resolvedTools.find((tool) => tool.source === "knowledge_base")?.knowledgeBaseEntries ?? []
      });
      response.setHeader("x-chat-source", "baseline-local-provider-error");
      response.json({
        answer: removeEmoji(fallback)
      });
    }
  } catch (error) {
    console.error(error);
    response.status(500).json({
      error: "The chat request could not be completed."
    });
  }
});

// ---------------------------------------------------------------------------
// Video analysis proxy – forwards multipart upload to Python service
// ---------------------------------------------------------------------------
app.post("/video/analyze", express.raw({ type: "multipart/form-data", limit: "100mb" }), async (request, response) => {
  try {
    const pythonUrl = `${RECOMMENDER_BASE_URL}/v1/video/analyze`;
    const contentType = request.headers["content-type"] ?? "application/octet-stream";

    const upstream = await fetch(pythonUrl, {
      method: "POST",
      headers: { "content-type": contentType },
      body: request.body as Buffer
    });

    if (!upstream.ok) {
      const text = await upstream.text();
      response.status(upstream.status).json({
        error: "video_analysis_upstream_error",
        status: upstream.status,
        detail: text
      });
      return;
    }

    const payload = await upstream.json();
    response.json(payload);
  } catch (error) {
    console.error("Video analysis proxy error:", error);
    response.status(502).json({
      error: "video_analysis_proxy_failed",
      reason: error instanceof Error ? error.message : String(error)
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
void feedbackGateway.init();

app.listen(SERVER_PORT, () => {
  console.log(`Local API running on http://localhost:${SERVER_PORT}`);
  console.log(
    `DeepSeek enabled: ${DEEPSEEK_ENABLED ? "yes" : "no"} | model: ${DEEPSEEK_MODEL}`
  );
  console.log(`Recommender enabled: ${RECOMMENDER_ENABLED ? "yes" : "no"}`);
  const feedbackStatus = feedbackGateway.status();
  console.log(`Feedback store ready: ${feedbackStatus.ready ? "yes" : "no"}`);
});
