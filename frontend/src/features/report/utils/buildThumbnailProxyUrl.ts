import { buildApiUrl } from "../../../services/api/runtimeConfig";

const THUMBNAIL_PROXY_URL = buildApiUrl("/thumbnail");

/**
 * Proxies TikTok CDN thumbnail URLs through our serverless function
 * to bypass CORS and expired-signature restrictions.
 */
export function buildThumbnailProxyUrl(
  thumbnailUrl: string,
  _videoUrl: string
): string {
  const url = thumbnailUrl.trim();
  if (!url) return "";
  return `${THUMBNAIL_PROXY_URL}?url=${encodeURIComponent(url)}`;
}
