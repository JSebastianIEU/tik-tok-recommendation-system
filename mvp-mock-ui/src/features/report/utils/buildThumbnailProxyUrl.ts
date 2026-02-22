import { buildApiUrl } from "../../../services/api/runtimeConfig";

const THUMBNAIL_PROXY_URL = buildApiUrl("/api/thumbnail");

export function buildThumbnailProxyUrl(
  thumbnailUrl: string,
  videoUrl: string
): string {
  if (thumbnailUrl.startsWith("/")) {
    return thumbnailUrl;
  }

  const params = new URLSearchParams();
  const normalizedThumbnailUrl = thumbnailUrl.trim();
  const normalizedVideoUrl = videoUrl.trim();

  if (normalizedThumbnailUrl) {
    params.set("url", normalizedThumbnailUrl);
  }

  if (normalizedVideoUrl) {
    params.set("video", normalizedVideoUrl);
  }

  const queryString = params.toString();
  return queryString ? `${THUMBNAIL_PROXY_URL}?${queryString}` : "";
}
