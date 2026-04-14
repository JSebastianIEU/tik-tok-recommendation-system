/**
 * Returns the best thumbnail URL to display.
 * Uses the raw TikTok CDN URL directly (with no-referrer policy on the img tag).
 */
export function buildThumbnailProxyUrl(
  thumbnailUrl: string,
  _videoUrl: string
): string {
  return thumbnailUrl.trim();
}
