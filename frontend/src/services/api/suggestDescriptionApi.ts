import type { VideoAnalysisResult } from "../contracts/models";
import { buildApiUrl } from "./runtimeConfig";

const SUGGEST_DESCRIPTION_URL = buildApiUrl("/suggest-description");

export interface SuggestDescriptionResult {
  description: string;
  hashtags: string[];
}

export async function suggestDescription(
  videoAnalysis: VideoAnalysisResult,
  objective?: string,
  content_type?: string,
  locale?: string
): Promise<SuggestDescriptionResult> {
  const resp = await fetch(SUGGEST_DESCRIPTION_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      videoAnalysis,
      objective: objective || "engagement",
      content_type: content_type || "showcase",
      locale: locale || "en",
    }),
    signal: AbortSignal.timeout(25_000),
  });

  if (!resp.ok) {
    return { description: "", hashtags: [] };
  }

  return (await resp.json()) as SuggestDescriptionResult;
}
