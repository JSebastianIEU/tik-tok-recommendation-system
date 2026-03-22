import cors from "cors";
import express from "express";
import OpenAI from "openai";
import {
  DEEPSEEK_API_KEY,
  DEEPSEEK_BASE_URL,
  DEEPSEEK_ENABLED,
  DEEPSEEK_MODEL,
  RECOMMENDER_ENABLED,
  SERVER_PORT
} from "./config";
import { getSeedVideo } from "../src/services/data/selectDemoSlices";
import type { DemoVideoRecord } from "../src/services/data/types";
import type { ReportOutput } from "../src/features/report/types";
import { loadDatasetFromFile } from "./dataset/loadDatasetFromFile";
import { buildLocalBaselineReport } from "./fallback/buildLocalBaselineReport";
import { buildLocalChatAnswer } from "./fallback/buildLocalChatAnswer";
import { enrichComparableMedia } from "./formatters/enrichComparableMedia";
import { normalizeReportOutput } from "./formatters/normalizeReportOutput";
import {
  buildCandidateProfileCore,
  buildComparableNeighborhood,
  buildNeighborhoodContrast,
  extractCandidateSignals
} from "./modeling";
import { buildChatPrompt } from "./prompts/buildChatPrompt";
import { buildReportPrompt } from "./prompts/buildReportPrompt";
import { HARD_CODED_EXTRACTED_KEYWORDS } from "./prompts/seedVideoContext";
import { parseGenerateReportRequest } from "./validation/parseGenerateReportRequest";
import { parseRecommendationsRequest } from "./validation/parseRecommendationsRequest";
import { validateReportOutput } from "./validation/validateReportOutput";
import {
  requestRecommendations,
  type RecommenderCandidatePayload,
  type RecommenderRequestPayload,
  type RecommenderResult
} from "./recommender/client";

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
          : undefined
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
  }
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
  return {
    ...enriched,
    header: {
      ...enriched.header,
      subtitle: `${enriched.header.subtitle} | ${suffix}`.slice(0, 220),
      disclaimer: `${enriched.header.disclaimer} ${suffix}`.trim()
    }
  };
}

function ensureDeepSeekClient(): OpenAI {
  if (!deepSeekClient) {
    throw new Error("DEEPSEEK_API_KEY is not configured in .env.local.");
  }
  return deepSeekClient;
}

async function fetchRecommenderResult(params: {
  objective: string | undefined;
  description: string;
  hashtags: string[];
  mentions: string[];
  candidates: DemoVideoRecord[];
  asOfTimeIso: string;
  topK?: number;
  retrieveK?: number;
  debug?: boolean;
}): Promise<RecommenderResult> {
  if (!RECOMMENDER_ENABLED) {
    return { ok: false, error: "Recommender disabled by config." };
  }
  const payload: RecommenderRequestPayload = {
    objective: mapObjectiveForRecommender(params.objective),
    as_of_time: params.asOfTimeIso,
    query: {
      description: params.description,
      hashtags: params.hashtags,
      mentions: params.mentions,
      text: [params.description, ...params.hashtags, ...params.mentions].join(" ").trim(),
      topic_key: params.hashtags[0]?.replace(/^#/, "") || "general",
      as_of_time: params.asOfTimeIso
    },
    candidates: buildRecommenderCandidates(params.candidates, params.asOfTimeIso),
    top_k: Math.max(1, params.topK ?? 20),
    retrieve_k: Math.max(1, params.retrieveK ?? 200),
    debug: Boolean(params.debug)
  };
  return requestRecommendations(payload);
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
    const parsedRequest = parseRecommendationsRequest(request.body);
    if (!parsedRequest.ok) {
      response.status(400).json({ error: parsedRequest.error });
      return;
    }
    const body = parsedRequest.value;
    const candidateProfile = buildCandidateProfileCore({
      description: body.description,
      mentions: body.mentions,
      hashtags: body.hashtags,
      objective: body.objective,
      audience: body.audience,
      content_type: body.content_type,
      primary_cta: body.primary_cta,
      locale: body.locale
    });
    const candidateSignals = extractCandidateSignals(candidateProfile, body.signal_hints);

    const dataset = await loadDatasetFromFile();
    if (dataset.length === 0) {
      response.status(400).json({ error: "Local dataset is empty or invalid." });
      return;
    }
    const seedVideoId = body.seed_video_id;
    const candidates = getCombinedCandidates(dataset, seedVideoId);
    if (candidates.length === 0) {
      response.status(400).json({ error: "No comparable candidates were found." });
      return;
    }

    const recommenderResult = await fetchRecommenderResult({
      objective: body.objective,
      description: body.description,
      hashtags: body.hashtags,
      mentions: body.mentions,
      candidates,
      asOfTimeIso: body.as_of_time,
      topK: body.top_k,
      retrieveK: body.retrieve_k,
      debug: body.debug
    });

    if (recommenderResult.ok) {
      response.setHeader("x-recommender-source", "python-service");
      response.json(recommenderResult.payload);
      return;
    }

    const fallbackNeighborhood = buildComparableNeighborhood({
      candidateProfile,
      candidateSignals,
      records: candidates
    });
    const fallbackItems = [
      ...fallbackNeighborhood.content_twins,
      ...fallbackNeighborhood.similar_overperformers,
      ...fallbackNeighborhood.similar_underperformers
    ]
      .slice(0, body.top_k)
      .map((candidate, index) => ({
        candidate_id: candidate.record.video_id,
        rank: index + 1,
        score: Number(candidate.composite_score.toFixed(6)),
        similarity: {
          sparse: Number(candidate.score_components.text_similarity.toFixed(6)),
          dense: Number(candidate.score_components.signal_match.toFixed(6)),
          fused: Number(candidate.similarity.toFixed(6))
        },
        trace: {
          objective_model:
            mapObjectiveForRecommender(body.objective) === "community"
              ? "engagement"
              : mapObjectiveForRecommender(body.objective),
          ranker_backend: "deterministic-fallback"
        }
      }));

    response.setHeader("x-recommender-source", "fallback-deterministic");
    response.status(200).json({
      objective: mapObjectiveForRecommender(body.objective),
      objective_effective:
        mapObjectiveForRecommender(body.objective) === "community"
          ? "engagement"
          : mapObjectiveForRecommender(body.objective),
      generated_at: new Date().toISOString(),
      fallback_mode: true,
      fallback_reason: recommenderResult.error,
      items: fallbackItems,
      debug: body.debug
        ? {
            candidate_pool_size: candidates.length,
            neighborhood_confidence: fallbackNeighborhood.confidence
          }
        : undefined
    });
  } catch (error) {
    console.error(error);
    response.status(500).json({
      error: "Recommendations are unavailable right now.",
      fallback_mode: true
    });
  }
});

app.post("/generate-report", async (request, response) => {
  try {
    const parsedRequest = parseGenerateReportRequest(request.body);
    if (!parsedRequest.ok) {
      response.status(400).json({ error: parsedRequest.error });
      return;
    }
    const body = parsedRequest.value;
    const candidateProfile = buildCandidateProfileCore({
      description: body.description,
      mentions: body.mentions,
      hashtags: body.hashtags,
      objective: body.objective,
      audience: body.audience,
      content_type: body.content_type,
      primary_cta: body.primary_cta,
      locale: body.locale
    });
    const candidateSignals = extractCandidateSignals(candidateProfile, body.signal_hints);
    const description = candidateProfile.raw.description;
    const mentions = candidateProfile.normalized.mentions.map((mention) => `@${mention}`);
    const hashtags = candidateProfile.normalized.hashtags.map((tag) => `#${tag}`);
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

    const uploadedSeed = buildUploadedSeedRecord(seed, description, hashtags);
    const candidates = getCombinedCandidates(dataset, seedVideoId);
    if (candidates.length === 0) {
      response.status(400).json({ error: "No comparable candidates were found." });
      return;
    }

    const recommenderResult = await fetchRecommenderResult({
      objective: body.objective,
      description,
      hashtags,
      mentions,
      candidates,
      asOfTimeIso: new Date().toISOString(),
      topK: 20,
      retrieveK: 200
    });
    const rankedCandidates = applyRecommenderOrder(candidates, recommenderResult);
    response.setHeader(
      "x-recommender-source",
      recommenderResult.ok ? "python-service" : "fallback-deterministic"
    );

    const comparableNeighborhood = buildComparableNeighborhood({
      candidateProfile,
      candidateSignals,
      records: rankedCandidates
    });
    const neighborhoodContrast = buildNeighborhoodContrast({
      candidateProfile,
      candidateSignals,
      neighborhood: comparableNeighborhood
    });

    const reportPrompt = buildReportPrompt({
      seed: uploadedSeed,
      candidates: rankedCandidates,
      mentions,
      hashtags,
      description,
      candidatesK: rankedCandidates.length,
      candidateProfile,
      candidateSignals,
      comparableNeighborhood,
      neighborhoodContrast
    });

    const localFallbackReport = buildLocalBaselineReport({
      seed: uploadedSeed,
      candidates: rankedCandidates,
      mentions,
      hashtags,
      description,
      candidatesK: rankedCandidates.length,
      candidateProfile,
      candidateSignals,
      comparableNeighborhood,
      neighborhoodContrast
    });

    if (!DEEPSEEK_ENABLED) {
      response.setHeader("x-report-source", "baseline-local-no-key");
      const report = await normalizeAndEnrichReport(
        sanitizeUnknownStrings(localFallbackReport) as ReportOutput,
        rankedCandidates,
        {
          source: recommenderResult.ok ? "recommender" : "fallback",
          note: recommenderResult.ok ? undefined : recommenderResult.error
        }
      );
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
          sanitizeUnknownStrings(localFallbackReport) as ReportOutput,
          rankedCandidates,
          {
            source: recommenderResult.ok ? "recommender" : "fallback",
            note: recommenderResult.ok ? undefined : recommenderResult.error
          }
        );
        response.json({
          report
        });
        return;
      }

      const parsed = extractFirstJsonObject(rawContent);
      const sanitizedReport = sanitizeUnknownStrings(parsed);

      if (!validateReportOutput(sanitizedReport)) {
        response.setHeader("x-report-source", "baseline-local-invalid-provider-schema");
        const report = await normalizeAndEnrichReport(
          sanitizeUnknownStrings(localFallbackReport) as ReportOutput,
          rankedCandidates,
          {
            source: recommenderResult.ok ? "recommender" : "fallback",
            note: recommenderResult.ok ? undefined : recommenderResult.error
          }
        );
        response.json({
          report
        });
        return;
      }

      response.setHeader("x-report-source", "deepseek");
      const report = await normalizeAndEnrichReport(
        sanitizedReport as ReportOutput,
        rankedCandidates,
        {
          source: recommenderResult.ok ? "recommender" : "fallback",
          note: recommenderResult.ok ? undefined : recommenderResult.error
        }
      );
      response.json({
        report
      });
    } catch (providerError) {
      console.error(providerError);
      response.setHeader("x-report-source", "baseline-local-provider-error");
      const report = await normalizeAndEnrichReport(
        sanitizeUnknownStrings(localFallbackReport) as ReportOutput,
        rankedCandidates,
        {
          source: recommenderResult.ok ? "recommender" : "fallback",
          note: recommenderResult.ok ? undefined : recommenderResult.error
        }
      );
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

app.listen(SERVER_PORT, () => {
  console.log(`Local API running on http://localhost:${SERVER_PORT}`);
  console.log(
    `DeepSeek enabled: ${DEEPSEEK_ENABLED ? "yes" : "no"} | model: ${DEEPSEEK_MODEL}`
  );
  console.log(`Recommender enabled: ${RECOMMENDER_ENABLED ? "yes" : "no"}`);
});
