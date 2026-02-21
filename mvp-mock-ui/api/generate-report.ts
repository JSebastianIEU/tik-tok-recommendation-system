// @ts-nocheck
import fs from "node:fs/promises";
import path from "node:path";
import { getSeedVideo } from "../src/services/data/selectDemoSlices";
import { parseDemoDatasetJsonl } from "../src/services/data/parseDemoDatasetJsonl";
import type { DemoVideoRecord } from "../src/services/data/types";
import type { ReportOutput } from "../src/features/report/types";
import { buildLocalBaselineReport } from "../server/fallback/buildLocalBaselineReport";
import { enrichComparableMedia } from "../server/formatters/enrichComparableMedia";
import { normalizeReportOutput } from "../server/formatters/normalizeReportOutput";
import { buildReportPrompt } from "../server/prompts/buildReportPrompt";
import { HARD_CODED_EXTRACTED_KEYWORDS } from "../server/prompts/seedVideoContext";
import { validateReportOutput } from "../server/validation/validateReportOutput";

interface GenerateReportRequestBody {
  seed_video_id?: string;
  mentions?: unknown;
  hashtags?: unknown;
  description?: string;
}

interface ApiRequest {
  method?: string;
  body?: unknown;
}

interface ApiResponse {
  setHeader(name: string, value: string): void;
  status(code: number): ApiResponse;
  json(payload: unknown): void;
}

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
  candidates: DemoVideoRecord[],
  modelLabel: string
): Promise<ReportOutput> {
  const normalized = normalizeReportOutput(report, {
    candidatesK: candidates.length,
    extractedKeywords: [...HARD_CODED_EXTRACTED_KEYWORDS],
    modelLabel
  });

  return enrichComparableMedia(normalized, candidates);
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

    if (character === '"') {
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

async function loadDatasetFromFile(): Promise<DemoVideoRecord[]> {
  const candidatePaths = [
    path.resolve(process.cwd(), "src/data/demodata.jsonl"),
    path.resolve(process.cwd(), "mvp-mock-ui/src/data/demodata.jsonl")
  ];

  for (const datasetPath of candidatePaths) {
    try {
      const raw = await fs.readFile(datasetPath, "utf-8");
      return parseDemoDatasetJsonl(raw);
    } catch {
    }
  }

  return [];
}

export default async function handler(request: ApiRequest, response: ApiResponse) {
  if (request.method !== "POST") {
    response.setHeader("Allow", "POST");
    response.status(405).json({ error: "Method not allowed." });
    return;
  }

  try {
    const body = (request.body ?? {}) as GenerateReportRequestBody;
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

    const apiKey = (process.env.DEEPSEEK_API_KEY ?? "").trim();
    const model = (process.env.DEEPSEEK_MODEL ?? "deepseek-reasoner").trim() || "deepseek-reasoner";
    const baseUrl = (process.env.DEEPSEEK_BASE_URL ?? "https://api.deepseek.com").trim() || "https://api.deepseek.com";
    const deepSeekEnabled = Boolean(apiKey) && apiKey !== "your_key_here";

    if (!deepSeekEnabled) {
      response.setHeader("x-report-source", "baseline-local-no-key");
      const report = await normalizeAndEnrichReport(
        sanitizeUnknownStrings(localFallbackReport) as ReportOutput,
        candidates,
        model
      );
      response.json({ report });
      return;
    }

    try {
      const providerResponse = await fetch(`${baseUrl}/chat/completions`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${apiKey}`
        },
        body: JSON.stringify({
          model,
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
        })
      });

      if (!providerResponse.ok) {
        throw new Error(`DeepSeek error: ${providerResponse.status}`);
      }

      const completion = (await providerResponse.json()) as {
        choices?: Array<{ message?: { content?: unknown } }>;
      };

      const rawContent = extractTextContent(completion.choices?.[0]?.message?.content ?? "");
      if (!rawContent) {
        response.setHeader("x-report-source", "baseline-local-empty-provider-response");
        const report = await normalizeAndEnrichReport(
          sanitizeUnknownStrings(localFallbackReport) as ReportOutput,
          candidates,
          model
        );
        response.json({ report });
        return;
      }

      const parsed = extractFirstJsonObject(rawContent);
      const sanitizedReport = sanitizeUnknownStrings(parsed);

      if (!validateReportOutput(sanitizedReport)) {
        response.setHeader("x-report-source", "baseline-local-invalid-provider-schema");
        const report = await normalizeAndEnrichReport(
          sanitizeUnknownStrings(localFallbackReport) as ReportOutput,
          candidates,
          model
        );
        response.json({ report });
        return;
      }

      response.setHeader("x-report-source", "deepseek");
      const report = await normalizeAndEnrichReport(
        sanitizedReport as ReportOutput,
        candidates,
        model
      );
      response.json({ report });
    } catch (providerError) {
      console.error(providerError);
      response.setHeader("x-report-source", "baseline-local-provider-error");
      const report = await normalizeAndEnrichReport(
        sanitizeUnknownStrings(localFallbackReport) as ReportOutput,
        candidates,
        model
      );
      response.json({ report });
    }
  } catch (error) {
    console.error(error);
    response.status(500).json({ error: "The report could not be generated right now." });
  }
}
