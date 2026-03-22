import {
  RECOMMENDER_BASE_URL,
  RECOMMENDER_ENABLED,
  RECOMMENDER_RETRY_COUNT,
  RECOMMENDER_TIMEOUT_MS
} from "../config";

export interface RecommenderQueryPayload {
  query_id?: string;
  description?: string;
  hashtags?: string[];
  mentions?: string[];
  text?: string;
  topic_key?: string;
  author_id?: string;
  as_of_time?: string;
}

export interface RecommenderCandidatePayload {
  candidate_id: string;
  caption?: string;
  text?: string;
  hashtags?: string[];
  keywords?: string[];
  topic_key?: string;
  author_id?: string;
  as_of_time?: string;
  posted_at?: string;
}

export interface RecommenderRequestPayload {
  objective: string;
  as_of_time: string;
  query: RecommenderQueryPayload;
  candidates: RecommenderCandidatePayload[];
  top_k: number;
  retrieve_k: number;
  debug?: boolean;
}

export interface RecommenderItem {
  candidate_id: string;
  rank: number;
  score: number;
  similarity: {
    sparse: number;
    dense: number;
    fused: number;
  };
  trace?: {
    objective_model?: string | null;
    ranker_backend?: string | null;
  };
}

export interface RecommenderResponsePayload {
  objective: string;
  objective_effective: string;
  generated_at: string;
  fallback_mode: boolean;
  items: RecommenderItem[];
  debug?: Record<string, unknown>;
}

interface RecommenderResultSuccess {
  ok: true;
  payload: RecommenderResponsePayload;
}

interface RecommenderResultFailure {
  ok: false;
  error: string;
  status?: number;
}

export type RecommenderResult = RecommenderResultSuccess | RecommenderResultFailure;

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function fetchWithTimeout(
  url: string,
  init: RequestInit,
  timeoutMs: number
): Promise<Response> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, {
      ...init,
      signal: controller.signal
    });
  } finally {
    clearTimeout(timeout);
  }
}

export async function requestRecommendations(
  payload: RecommenderRequestPayload
): Promise<RecommenderResult> {
  if (!RECOMMENDER_ENABLED) {
    return { ok: false, error: "Recommender is disabled by configuration." };
  }

  const attempts = Math.max(1, RECOMMENDER_RETRY_COUNT + 1);
  const url = `${RECOMMENDER_BASE_URL.replace(/\/+$/, "")}/v1/recommendations`;
  let lastError = "Unknown recommender error";
  let lastStatus: number | undefined;

  for (let attempt = 1; attempt <= attempts; attempt += 1) {
    try {
      const response = await fetchWithTimeout(
        url,
        {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify(payload)
        },
        Math.max(500, RECOMMENDER_TIMEOUT_MS)
      );
      if (!response.ok) {
        lastStatus = response.status;
        const raw = await response.text();
        lastError = raw || `Recommender request failed with status ${response.status}`;
      } else {
        const parsed = (await response.json()) as RecommenderResponsePayload;
        if (!parsed || !Array.isArray(parsed.items)) {
          lastError = "Recommender returned invalid payload.";
        } else {
          return { ok: true, payload: parsed };
        }
      }
    } catch (error) {
      lastError = error instanceof Error ? error.message : "Recommender request failed.";
    }

    if (attempt < attempts) {
      await sleep(160 * attempt);
    }
  }

  return { ok: false, error: lastError, status: lastStatus };
}

