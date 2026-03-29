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
  language?: string;
  locale?: string;
  content_type?: string;
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
  duration_seconds?: number;
  language?: string;
  locale?: string;
  content_type?: string;
  signal_hints?: Record<string, unknown>;
}

export interface RoutingContractPayload {
  objective_requested: string;
  objective_effective: string;
  track?: "pre_publication" | "post_publication";
  allow_fallback?: boolean;
  required_compat?: Record<string, string>;
  model_family?: string;
  request_id?: string;
  experiment?: {
    id?: string;
    variant?: "control" | "treatment";
    unit_hash?: string;
  };
  stage_budgets_ms?: {
    retrieval?: number;
    ranking?: number;
    explainability?: number;
  };
}

export interface RecommenderRequestPayload {
  objective: string;
  as_of_time: string;
  query: RecommenderQueryPayload;
  candidates: RecommenderCandidatePayload[];
  language?: string;
  locale?: string;
  content_type?: string;
  candidate_ids?: string[];
  policy_overrides?: {
    strict_language?: boolean;
    strict_locale?: boolean;
    max_items_per_author?: number;
  };
  portfolio?: {
    enabled?: boolean;
    weights?: {
      reach?: number;
      conversion?: number;
      durability?: number;
    };
    risk_aversion?: number;
    candidate_pool_cap?: number;
  };
  graph_controls?: {
    enable_graph_branch?: boolean;
  };
  trajectory_controls?: {
    enabled?: boolean;
  };
  explainability?: {
    enabled?: boolean;
    top_features?: number;
    neighbor_k?: number;
    run_counterfactuals?: boolean;
  };
  routing?: RoutingContractPayload;
  top_k: number;
  retrieve_k: number;
  debug?: boolean;
}

export interface FabricExtractPayload {
  video_id: string;
  as_of_time: string;
  caption?: string;
  hashtags?: string[];
  keywords?: string[];
  transcript_text?: string;
  ocr_text?: string;
  duration_seconds?: number;
  content_type?: string;
  source_updated_at?: string;
  hints?: Record<string, unknown>;
}

export interface FabricExtractResponsePayload {
  fabric_version: string;
  generated_at: string;
  video_id: string;
  as_of_time: string;
  registry_signature: string;
  extractor_traces: Array<{
    extractor: string;
    extractor_version: string;
    model_version: string;
    output_schema_hash: string;
    input_digest: string;
    generated_at: string;
  }>;
  text: {
    token_count: number;
    unique_token_count: number;
    hashtag_count: number;
    keyphrase_count: number;
    cta_keyword_count: number;
    clarity_score: number;
    confidence: { raw: number; calibrated: number; calibration_version: string };
    missing: Record<string, { reason: string; source_age_hours?: number; detail?: string }>;
  };
  structure: {
    hook_window_start_sec?: number;
    hook_window_end_sec?: number;
    mid_window_start_sec?: number;
    mid_window_end_sec?: number;
    payoff_window_start_sec?: number;
    payoff_window_end_sec?: number;
    hook_timing_seconds: number;
    payoff_timing_seconds: number;
    step_density: number;
    pacing_score: number;
    confidence: { raw: number; calibrated: number; calibration_version: string };
    missing: Record<string, { reason: string; source_age_hours?: number; detail?: string }>;
  };
  audio: {
    speech_ratio?: number | null;
    tempo_bpm?: number | null;
    energy?: number | null;
    music_presence_score?: number | null;
    confidence: { raw: number; calibrated: number; calibration_version: string };
    missing: Record<string, { reason: string; source_age_hours?: number; detail?: string }>;
  };
  visual: {
    shot_change_rate?: number | null;
    visual_motion_score?: number | null;
    style_tags: string[];
    confidence: { raw: number; calibrated: number; calibration_version: string };
    missing: Record<string, { reason: string; source_age_hours?: number; detail?: string }>;
  };
  trace_ids: string[];
}

export interface RecommenderItem {
  candidate_id: string;
  rank: number;
  score: number;
  score_raw?: number;
  score_calibrated?: number;
  policy_penalty?: number;
  policy_bonus?: number;
  policy_adjusted_score?: number;
  calibration_trace?: Record<string, unknown>;
  policy_trace?: Record<string, unknown>;
  portfolio_trace?: Record<string, unknown>;
  similarity: {
    sparse: number;
    dense: number;
    fused: number;
  };
  retrieval_branch_scores?: {
    lexical?: number;
    dense_text?: number;
    multimodal?: number;
    graph_dense?: number;
    trajectory_dense?: number;
    fused?: number;
  };
  graph_trace?: {
    graph_bundle_id?: string;
    graph_version?: string;
    query_sources?: string[];
    candidate_overlap?: {
      hashtag_overlap?: number;
      style_overlap?: number;
      same_audio_motif?: boolean;
    };
  } | null;
  trajectory_trace?: {
    trajectory_manifest_id?: string;
    trajectory_version?: string;
    trajectory_mode?: string;
    similarity?: number;
    regime_pred?: string;
    regime_probabilities?: Record<string, number>;
    regime_confidence?: number;
    available?: boolean;
  } | null;
  trajectory_score?: number;
  trajectory_similarity?: number;
  trajectory_regime_pred?: string;
  trajectory_regime_probabilities?: Record<string, number>;
  trajectory_regime_confidence?: number;
  trace?: {
    objective_model?: string | null;
    ranker_backend?: string | null;
  };
  comment_trace?: {
    source?: string;
    taxonomy_version?: string;
    dominant_intents?: string[];
    confusion_index?: number;
    help_seeking_index?: number;
    sentiment_volatility?: number;
    sentiment_shift_early_late?: number;
    reply_depth_max?: number;
    reply_branch_factor?: number;
    reply_ratio?: number;
    root_thread_concentration?: number;
    alignment_score?: number;
    value_prop_coverage?: number;
    on_topic_ratio?: number;
    artifact_drift_ratio?: number;
    alignment_shift_early_late?: number;
    alignment_confidence?: number;
    alignment_method_version?: string;
    confidence?: number;
    missingness_flags?: string[];
  } | null;
  evidence_cards?: {
    feature_contributions?: Record<string, unknown>;
    neighbor_evidence?: Record<string, unknown>;
    temporal_confidence_band?: {
      p10?: number;
      p50?: number;
      p90?: number;
      source?: string;
    };
  };
  temporal_confidence_band?: {
    p10?: number;
    p50?: number;
    p90?: number;
    source?: string;
  };
  counterfactual_scenarios?: Array<{
    scenario_id: string;
    expected_rank_delta_band?: {
      p10?: number;
      p50?: number;
      p90?: number;
      source?: string;
    };
    feasibility?: string;
    reason?: string;
    applied_deltas?: Record<string, number>;
    trace?: Record<string, unknown>;
  }>;
}

export interface RecommenderResponsePayload {
  request_id?: string;
  experiment_id?: string;
  variant?: "control" | "treatment";
  objective: string;
  objective_effective: string;
  generated_at: string;
  fallback_mode: boolean;
  calibration_version?: string;
  policy_version?: string;
  feature_manifest_id?: string | null;
  comment_feature_manifest_id?: string | null;
  comment_intelligence_version?: string;
  fabric_version?: string;
  retrieval_mode?: "global" | "intersected";
  constraint_tier_used?: number;
  retriever_artifact_version?: string;
  graph_bundle_id?: string;
  graph_version?: string;
  graph_coverage?: number;
  graph_fallback_mode?: string;
  trajectory_manifest_id?: string;
  trajectory_version?: string;
  trajectory_mode?: string;
  trajectory_prediction?: {
    trajectory_score?: number;
    regime_pred?: string;
    regime_probabilities?: Record<string, number>;
    regime_confidence?: number;
    available?: boolean;
  };
  policy_metadata?: Record<string, unknown>;
  portfolio_mode?: boolean;
  portfolio_metadata?: Record<string, unknown>;
  calibration_metadata?: Record<string, unknown>;
  explainability_metadata?: Record<string, unknown>;
  routing_decision?: Record<string, unknown>;
  compatibility_status?: Record<string, unknown>;
  fallback_reason?: string | null;
  latency_breakdown_ms?: Record<string, number>;
  circuit_state?: Record<string, unknown>;
  served_by?: string;
  extraction_trace_ids?: string[];
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

export interface CompatibilityPayload {
  ok: boolean;
  bundle_id?: string;
  fingerprints?: Record<string, unknown>;
  mismatches?: Array<Record<string, unknown>>;
  required_compat?: Record<string, string>;
}

export type CompatibilityResult =
  | { ok: true; payload: CompatibilityPayload }
  | { ok: false; error: string; status?: number };

interface FabricResultSuccess {
  ok: true;
  payload: FabricExtractResponsePayload;
}

interface FabricResultFailure {
  ok: false;
  error: string;
  status?: number;
}

export type FabricResult = FabricResultSuccess | FabricResultFailure;

interface RequestOptions {
  timeoutMs?: number;
}

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
  payload: RecommenderRequestPayload,
  options?: RequestOptions
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
        Math.max(500, options?.timeoutMs ?? RECOMMENDER_TIMEOUT_MS)
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

export async function requestCompatibility(
  options?: RequestOptions
): Promise<CompatibilityResult> {
  if (!RECOMMENDER_ENABLED) {
    return { ok: false, error: "Recommender is disabled by configuration." };
  }
  const url = `${RECOMMENDER_BASE_URL.replace(/\/+$/, "")}/v1/compatibility`;
  try {
    const response = await fetchWithTimeout(
      url,
      {
        method: "GET",
        headers: { accept: "application/json" }
      },
      Math.max(400, options?.timeoutMs ?? RECOMMENDER_TIMEOUT_MS)
    );
    if (!response.ok) {
      const raw = await response.text();
      return {
        ok: false,
        error: raw || `Compatibility request failed with status ${response.status}`,
        status: response.status
      };
    }
    const payload = (await response.json()) as CompatibilityPayload;
    return { ok: true, payload };
  } catch (error) {
    return {
      ok: false,
      error: error instanceof Error ? error.message : "Compatibility request failed."
    };
  }
}

export async function requestFabricSignals(
  payload: FabricExtractPayload
): Promise<FabricResult> {
  if (!RECOMMENDER_ENABLED) {
    return { ok: false, error: "Recommender is disabled by configuration." };
  }
  const attempts = Math.max(1, RECOMMENDER_RETRY_COUNT + 1);
  const url = `${RECOMMENDER_BASE_URL.replace(/\/+$/, "")}/v1/fabric/extract`;
  let lastError = "Unknown fabric extraction error";
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
        lastError = raw || `Fabric extract request failed with status ${response.status}`;
      } else {
        const parsed = (await response.json()) as FabricExtractResponsePayload;
        if (!parsed || typeof parsed.video_id !== "string" || typeof parsed.fabric_version !== "string") {
          lastError = "Fabric service returned invalid payload.";
        } else {
          return { ok: true, payload: parsed };
        }
      }
    } catch (error) {
      lastError = error instanceof Error ? error.message : "Fabric extract request failed.";
    }

    if (attempt < attempts) {
      await sleep(160 * attempt);
    }
  }
  return { ok: false, error: lastError, status: lastStatus };
}
