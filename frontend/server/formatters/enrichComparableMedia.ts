import type { ComparableItem, ReportOutput } from "../../src/features/report/types";
import type { DemoVideoRecord } from "../../src/services/data/types";

const TIKTOK_OEMBED_URL = "https://www.tiktok.com/oembed?url=";
const thumbnailCache = new Map<string, string>();

interface TikTokOEmbedResponse {
  thumbnail_url?: unknown;
}

function normalizeVideoId(value: string): string {
  const cleaned = value.toLowerCase().replace(/[^a-z0-9]/g, "");
  const match = cleaned.match(/^([a-z]+)(\d+)$/);

  if (!match) {
    return cleaned;
  }

  const prefix = match[1][0] ?? "";
  const digits = match[2].padStart(3, "0");
  return `${prefix}${digits}`;
}

function extractRecordVideoUrl(record: DemoVideoRecord): string {
  const value = record.video_url;
  return typeof value === "string" ? value : "";
}

function buildCandidateLookup(candidates: DemoVideoRecord[]): Map<string, DemoVideoRecord> {
  const lookup = new Map<string, DemoVideoRecord>();

  for (const candidate of candidates) {
    const normalizedId = normalizeVideoId(candidate.video_id);
    if (normalizedId) {
      lookup.set(normalizedId, candidate);
    }
  }

  return lookup;
}

function resolveCandidateByComparableId(
  item: ComparableItem,
  lookup: Map<string, DemoVideoRecord>
): DemoVideoRecord | null {
  const normalizedComparableId = normalizeVideoId(item.id);
  if (!normalizedComparableId) {
    return null;
  }

  return lookup.get(normalizedComparableId) ?? null;
}

function extractAuthorLabel(record: DemoVideoRecord): string {
  if (typeof record.author === "string") {
    return record.author;
  }

  if (record.author && typeof record.author === "object") {
    const username = (record.author as Record<string, unknown>).username;
    if (typeof username === "string" && username.trim()) {
      return `@${username.replace(/^@/, "")}`;
    }
  }

  return "@creator";
}

async function fetchTikTokThumbnail(videoUrl: string): Promise<string> {
  const normalizedUrl = videoUrl.trim();
  if (!normalizedUrl) {
    return "";
  }

  const cached = thumbnailCache.get(normalizedUrl);
  if (cached !== undefined) {
    return cached;
  }

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 3000);

  try {
    const response = await fetch(
      `${TIKTOK_OEMBED_URL}${encodeURIComponent(normalizedUrl)}`,
      {
        signal: controller.signal,
        headers: {
          Accept: "application/json"
        }
      }
    );

    if (!response.ok) {
      thumbnailCache.set(normalizedUrl, "");
      return "";
    }

    const parsed = (await response.json()) as TikTokOEmbedResponse;
    const thumbnailUrl =
      typeof parsed.thumbnail_url === "string" ? parsed.thumbnail_url : "";
    thumbnailCache.set(normalizedUrl, thumbnailUrl);
    return thumbnailUrl;
  } catch {
    thumbnailCache.set(normalizedUrl, "");
    return "";
  } finally {
    clearTimeout(timeoutId);
  }
}

async function enrichComparable(
  item: ComparableItem,
  lookup: Map<string, DemoVideoRecord>
): Promise<ComparableItem> {
  const candidate = resolveCandidateByComparableId(item, lookup);
  const candidateVideoUrl = candidate ? extractRecordVideoUrl(candidate) : "";
  const videoUrl = (item.video_url ?? "").trim() || candidateVideoUrl;
  const thumbnailFromItem = (item.thumbnail_url ?? "").trim();
  const thumbnailUrl = thumbnailFromItem || (await fetchTikTokThumbnail(videoUrl));

  return {
    ...item,
    id: candidate ? candidate.video_id.toUpperCase() : item.id,
    author: item.author.trim() || (candidate ? extractAuthorLabel(candidate) : "@creator"),
    video_url: videoUrl,
    thumbnail_url: thumbnailUrl
  };
}

export async function enrichComparableMedia(
  report: ReportOutput,
  candidates: DemoVideoRecord[]
): Promise<ReportOutput> {
  const lookup = buildCandidateLookup(candidates);
  const comparables = await Promise.all(
    report.comparables.map((item) => enrichComparable(item, lookup))
  );

  return {
    ...report,
    comparables
  };
}
