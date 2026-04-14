import { buildApiUrl, MOCK_ONLY_MODE } from "./runtimeConfig";

const HASHTAG_SUGGEST_URL = buildApiUrl("/hashtags/suggest");

export interface HashtagSuggestion {
  hashtag: string;
  score: number;
  frequency: number;
  avg_engagement: number;
}

export interface HashtagSuggestRequest {
  caption: string;
  top_n?: number;
  exclude_tags?: string[];
  include_neighbours?: boolean;
}

export interface HashtagSuggestResponse {
  suggestions: HashtagSuggestion[];
  neighbours?: Array<{
    caption: string;
    hashtags: string[];
    similarity: number;
  }>;
}

const MOCK_SUGGESTIONS: HashtagSuggestion[] = [
  { hashtag: "fyp", score: 0.95, frequency: 12000, avg_engagement: 0.08 },
  { hashtag: "motivation", score: 0.88, frequency: 8500, avg_engagement: 0.06 },
  { hashtag: "viral", score: 0.85, frequency: 11000, avg_engagement: 0.07 },
  { hashtag: "trending", score: 0.82, frequency: 9200, avg_engagement: 0.05 },
  { hashtag: "foryou", score: 0.80, frequency: 10500, avg_engagement: 0.06 }
];

export async function suggestHashtags(
  payload: HashtagSuggestRequest
): Promise<HashtagSuggestResponse> {
  if (MOCK_ONLY_MODE) {
    return { suggestions: MOCK_SUGGESTIONS };
  }

  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 15000);

    const response = await fetch(HASHTAG_SUGGEST_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: controller.signal
    });

    clearTimeout(timeout);

    if (!response.ok) {
      console.warn("[hashtagApi] server returned", response.status);
      return { suggestions: MOCK_SUGGESTIONS };
    }

    const data = (await response.json()) as HashtagSuggestResponse;
    if (!data.suggestions || data.suggestions.length === 0) {
      return { suggestions: MOCK_SUGGESTIONS };
    }
    return data;
  } catch (err) {
    console.warn("[hashtagApi] fetch failed, using mock:", err);
    return { suggestions: MOCK_SUGGESTIONS };
  }
}
