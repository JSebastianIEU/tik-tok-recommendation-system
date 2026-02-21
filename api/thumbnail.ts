const TIKTOK_OEMBED_URL = "https://www.tiktok.com/oembed?url=";
const THUMBNAIL_FETCH_TIMEOUT_MS = 7000;
const ALLOWED_TIKTOK_VIDEO_HOSTS = new Set(["tiktok.com", "www.tiktok.com"]);

interface ThumbnailRequest {
  method?: string;
  query: Record<string, unknown>;
}

interface ThumbnailResponse {
  setHeader(name: string, value: string): void;
  status(code: number): ThumbnailResponse;
  json(payload: unknown): void;
  send(payload: unknown): void;
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
    if (!parsedThumbnailUrl || !isAllowedThumbnailHost(parsedThumbnailUrl.hostname)) {
      return "";
    }

    return parsedThumbnailUrl.toString();
  } catch {
    return "";
  }
}

async function fetchThumbnailImage(
  thumbnailUrl: string
): Promise<{ body: Uint8Array; contentType: string } | null> {
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
      body: new Uint8Array(arrayBuffer),
      contentType
    };
  } catch {
    return null;
  }
}

export default async function handler(
  request: ThumbnailRequest,
  response: ThumbnailResponse
) {
  if (request.method !== "GET") {
    response.setHeader("Allow", "GET");
    response.status(405).json({ error: "Method not allowed." });
    return;
  }

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
      response.status(200).send(image.body);
      return;
    }

    response.status(404).json({ error: "Thumbnail is unavailable." });
  } catch (error) {
    console.error(error);
    response.status(500).json({ error: "Could not load thumbnail." });
  }
}
