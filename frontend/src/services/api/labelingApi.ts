import type {
  LabelingReviewLabel,
  LabelingSession,
  LabelingSessionListItem,
  LabelingSourceSummary
} from "../../features/labeling/types";
import { buildApiUrl, MOCK_ONLY_MODE } from "./runtimeConfig";

const LABELING_SOURCES_API_URL = buildApiUrl("/labeling/sources");
const LABELING_SESSIONS_API_URL = buildApiUrl("/labeling/sessions");

async function parseJson<T>(response: Response): Promise<T> {
  return (await response.json()) as T;
}

export async function listLabelingSources(): Promise<LabelingSourceSummary[]> {
  if (MOCK_ONLY_MODE) {
    return [];
  }
  const response = await fetch(LABELING_SOURCES_API_URL);
  if (!response.ok) {
    throw new Error("labeling_sources_request_failed");
  }
  const payload = await parseJson<{ sources: LabelingSourceSummary[] }>(response);
  return payload.sources;
}

export async function listLabelingSessions(): Promise<LabelingSessionListItem[]> {
  if (MOCK_ONLY_MODE) {
    return [];
  }
  const response = await fetch(LABELING_SESSIONS_API_URL);
  if (!response.ok) {
    throw new Error("labeling_sessions_request_failed");
  }
  const payload = await parseJson<{ sessions: LabelingSessionListItem[] }>(response);
  return payload.sessions;
}

export async function createLabelingSession(input?: {
  source_id?: string;
  session_name?: string;
}): Promise<LabelingSession> {
  if (MOCK_ONLY_MODE) {
    throw new Error("labeling_sessions_mock_only_unavailable");
  }
  const response = await fetch(LABELING_SESSIONS_API_URL, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(input ?? {})
  });
  if (!response.ok) {
    throw new Error("labeling_session_create_failed");
  }
  const payload = await parseJson<{ session: LabelingSession }>(response);
  return payload.session;
}

export async function loadLabelingSession(
  sessionId: string
): Promise<LabelingSession> {
  if (MOCK_ONLY_MODE) {
    throw new Error("labeling_sessions_mock_only_unavailable");
  }
  const response = await fetch(
    `${LABELING_SESSIONS_API_URL}/${encodeURIComponent(sessionId)}`
  );
  if (!response.ok) {
    throw new Error("labeling_session_load_failed");
  }
  const payload = await parseJson<{ session: LabelingSession }>(response);
  return payload.session;
}

export async function updateLabelingCandidateReview(input: {
  session_id: string;
  case_id: string;
  candidate_id: string;
  label: LabelingReviewLabel | null;
  note?: string;
}): Promise<LabelingSession> {
  if (MOCK_ONLY_MODE) {
    throw new Error("labeling_sessions_mock_only_unavailable");
  }
  const response = await fetch(
    `${LABELING_SESSIONS_API_URL}/${encodeURIComponent(input.session_id)}/cases/${encodeURIComponent(input.case_id)}/candidates/${encodeURIComponent(input.candidate_id)}`,
    {
      method: "PUT",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        label: input.label,
        note: input.note ?? ""
      })
    }
  );
  if (!response.ok) {
    throw new Error("labeling_session_update_failed");
  }
  const payload = await parseJson<{ session: LabelingSession }>(response);
  return payload.session;
}
