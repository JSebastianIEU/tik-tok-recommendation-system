import type {
  DemoVideoAuthor,
  DemoVideoMetrics,
  DemoVideoRecord
} from "./types";

const SUSPICIOUS_ENCODING_PATTERN = /[\u00c2\u00c3\u00e2\u00f0]/;

function fixPotentialMojibake(value: string): string {
  if (!SUSPICIOUS_ENCODING_PATTERN.test(value)) {
    return value;
  }

  try {
    const bytes = Uint8Array.from(
      Array.from(value),
      (character) => character.charCodeAt(0) & 0xff
    );
    const decoded = new TextDecoder("utf-8").decode(bytes);

    if (!decoded || decoded.includes("\uFFFD")) {
      return value;
    }

    return decoded;
  } catch {
    return value;
  }
}

function toStringArray(value: unknown): string[] {
  if (Array.isArray(value)) {
    return value
      .map((item) =>
        typeof item === "string" ? fixPotentialMojibake(item).trim() : ""
      )
      .filter(Boolean);
  }

  if (typeof value === "string") {
    return value
      .split(",")
      .map((item) => fixPotentialMojibake(item).trim())
      .filter(Boolean);
  }

  return [];
}

function toNumber(value: unknown): number {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }

  if (typeof value === "string") {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) {
      return parsed;
    }
  }

  return 0;
}

function toMetrics(
  metricsValue: unknown,
  fallback: Record<string, unknown>
): DemoVideoMetrics {
  const metricsObject =
    metricsValue && typeof metricsValue === "object"
      ? (metricsValue as Record<string, unknown>)
      : ({} as Record<string, unknown>);

  return {
    views: toNumber(metricsObject.views ?? fallback.views),
    likes: toNumber(metricsObject.likes ?? fallback.likes),
    comments_count: toNumber(
      metricsObject.comments_count ?? fallback.comments_count
    ),
    shares: toNumber(metricsObject.shares ?? fallback.shares)
  };
}

function toAuthor(value: unknown): string | DemoVideoAuthor {
  if (typeof value === "string") {
    return fixPotentialMojibake(value);
  }

  if (value && typeof value === "object") {
    return value as DemoVideoAuthor;
  }

  return "autor_desconocido";
}

function toCommentArray(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return [];
  }

  const comments: string[] = [];

  for (const entry of value) {
    if (typeof entry === "string") {
      const normalized = fixPotentialMojibake(entry).trim();
      if (normalized) {
        comments.push(normalized);
      }
      continue;
    }

    if (entry && typeof entry === "object") {
      const text = (entry as Record<string, unknown>).text;
      if (typeof text === "string" && text.trim()) {
        comments.push(fixPotentialMojibake(text).trim());
      }
    }
  }

  return comments;
}

function parseLineToRecord(line: string): DemoVideoRecord | null {
  if (!line.trim()) {
    return null;
  }

  try {
    const rawValue = JSON.parse(line);
    if (!rawValue || typeof rawValue !== "object") {
      return null;
    }

    const record = rawValue as Record<string, unknown>;
    const videoId = typeof record.video_id === "string" ? record.video_id.trim() : "";

    if (!videoId) {
      return null;
    }

    const caption =
      typeof record.caption === "string"
        ? fixPotentialMojibake(record.caption)
        : "";
    const author = toAuthor(record.author);

    return {
      ...record,
      video_id: videoId,
      caption,
      hashtags: toStringArray(record.hashtags),
      keywords: toStringArray(record.keywords),
      metrics: toMetrics(record.metrics, record),
      author,
      comments: toCommentArray(record.comments)
    };
  } catch {
    return null;
  }
}

export function parseDemoDatasetJsonl(rawJsonl: string): DemoVideoRecord[] {
  const lines = rawJsonl.split(/\r?\n/);
  const records: DemoVideoRecord[] = [];

  for (const line of lines) {
    const parsed = parseLineToRecord(line);
    if (parsed) {
      records.push(parsed);
    }
  }

  return records;
}
