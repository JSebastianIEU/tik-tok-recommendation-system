import cors from "cors";
import express from "express";
import OpenAI from "openai";
import {
  DEEPSEEK_API_KEY,
  DEEPSEEK_BASE_URL,
  DEEPSEEK_ENABLED,
  DEEPSEEK_MODEL,
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
import { buildChatPrompt } from "./prompts/buildChatPrompt";
import { buildReportPrompt } from "./prompts/buildReportPrompt";
import { HARD_CODED_EXTRACTED_KEYWORDS } from "./prompts/seedVideoContext";
import { validateReportOutput } from "./validation/validateReportOutput";

interface GenerateReportRequestBody {
  seed_video_id?: string;
  mentions?: unknown;
  hashtags?: unknown;
  description?: string;
}

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

function normalizeArray(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value
    .map((entry) => (typeof entry === "string" ? entry.trim() : ""))
    .filter(Boolean);
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
  candidates: DemoVideoRecord[]
): Promise<ReportOutput> {
  const normalized = normalizeReportOutput(report, {
    candidatesK: candidates.length,
    extractedKeywords: [...HARD_CODED_EXTRACTED_KEYWORDS],
    modelLabel: DEEPSEEK_MODEL
  });

  return enrichComparableMedia(normalized, candidates);
}

function ensureDeepSeekClient(): OpenAI {
  if (!deepSeekClient) {
    throw new Error("DEEPSEEK_API_KEY is not configured in .env.local.");
  }
  return deepSeekClient;
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

app.post("/generate-report", async (request, response) => {
  try {
    const body = request.body as GenerateReportRequestBody;
    const description = typeof body.description === "string" ? body.description : "";
    const mentions = normalizeArray(body.mentions);
    const hashtags = normalizeArray(body.hashtags);
    const seedVideoId = typeof body.seed_video_id === "string" ? body.seed_video_id : "s001";

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

    const reportPrompt = buildReportPrompt({
      seed: uploadedSeed,
      candidates,
      mentions,
      hashtags,
      description,
      candidatesK: candidates.length
    });

    const localFallbackReport = buildLocalBaselineReport({
      seed: uploadedSeed,
      candidates,
      mentions,
      hashtags,
      description,
      candidatesK: candidates.length
    });

    if (!DEEPSEEK_ENABLED) {
      response.setHeader("x-report-source", "baseline-local-no-key");
      const report = await normalizeAndEnrichReport(
        sanitizeUnknownStrings(localFallbackReport) as ReportOutput,
        candidates
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
          candidates
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
          candidates
        );
        response.json({
          report
        });
        return;
      }

      response.setHeader("x-report-source", "deepseek");
      const report = await normalizeAndEnrichReport(
        sanitizedReport as ReportOutput,
        candidates
      );
      response.json({
        report
      });
    } catch (providerError) {
      console.error(providerError);
      response.setHeader("x-report-source", "baseline-local-provider-error");
      const report = await normalizeAndEnrichReport(
        sanitizeUnknownStrings(localFallbackReport) as ReportOutput,
        candidates
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
});
