export type LabelingReviewLabel = "saved" | "relevant" | "not_relevant";

export interface LabelingSourceSummary {
  source_id: string;
  file_name: string;
  source_path: string;
  generated_at: string;
  case_count: number;
  objectives: string[];
}

export interface LabelingCandidateReview {
  label: LabelingReviewLabel | null;
  note: string;
  updated_at: string | null;
}

export interface LabelingSessionCandidate {
  candidate_id: string;
  display: Record<string, unknown>;
  candidate_payload: Record<string, unknown>;
  baseline_rank: number | null;
  baseline_score: number | null;
  support_level: string | null;
  ranking_reasons: string[];
  review: LabelingCandidateReview;
}

export interface LabelingSessionCase {
  case_id: string;
  objective: string;
  query: {
    query_id: string;
    display: Record<string, unknown>;
    query_payload: Record<string, unknown>;
  };
  retrieve_k: number;
  label_pool_size: number;
  source_candidate_pool_size: number;
  notes: string;
  candidates: LabelingSessionCandidate[];
}

export interface LabelingSessionSummary {
  case_count: number;
  candidate_count: number;
  reviewed_count: number;
  saved_count: number;
  relevant_count: number;
  not_relevant_count: number;
  completion_ratio: number;
}

export interface LabelingSession {
  version: string;
  session_id: string;
  session_name: string;
  created_at: string;
  updated_at: string;
  storage_path: string;
  source: LabelingSourceSummary;
  rubric: {
    version: string;
    labels: LabelingReviewLabel[];
    instructions: string[];
  };
  cases: LabelingSessionCase[];
  summary: LabelingSessionSummary;
}

export interface LabelingSessionListItem {
  session_id: string;
  session_name: string;
  created_at: string;
  updated_at: string;
  storage_path: string;
  source: LabelingSourceSummary;
  summary: LabelingSessionSummary;
}
