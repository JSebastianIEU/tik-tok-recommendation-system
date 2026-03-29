import { createHash } from "node:crypto";

export interface FeedbackStoreConfig {
  enabled: boolean;
  dbUrl: string;
}

export interface ExperimentAssignmentRecord {
  experimentId: string;
  objectiveEffective: string;
  unitHash: string;
  variant: "control" | "treatment";
  requestId: string;
}

export interface RequestEventRecord {
  requestId: string;
  endpoint: string;
  receivedAt: string;
  servedAt: string;
  objectiveRequested: string;
  objectiveEffective: string;
  experimentId?: string;
  variant?: "control" | "treatment";
  assignmentUnitHash?: string;
  requestHash: string;
  routingDecision: Record<string, unknown>;
  compatibilityStatus: Record<string, unknown>;
  fallbackMode: boolean;
  fallbackReason?: string | null;
  circuitState: Record<string, unknown>;
  latencyBreakdownMs: Record<string, number>;
  policyVersion?: string;
  calibrationVersion?: string;
  bundleFingerprint: Record<string, unknown>;
}

export interface CandidateEventRecord {
  requestId: string;
  candidateId: string;
  stage: "retrieved_universe" | "ranked_universe" | "served_output";
  retrievedRank?: number | null;
  finalRank?: number | null;
  selected: boolean;
  retrievalBranchScores: Record<string, number>;
  similarity: Record<string, number>;
  scoreRaw?: number | null;
  scoreCalibrated?: number | null;
  policyAdjustedScore?: number | null;
  policyTrace?: Record<string, unknown> | null;
  calibrationTrace?: Record<string, unknown> | null;
  explainabilityAvailable: boolean;
}

export interface ServedOutputRecord {
  requestId: string;
  candidateId: string;
  rank: number;
  score: number;
  metadata: Record<string, unknown>;
}

interface PgPoolLike {
  query: (sql: string, params?: unknown[]) => Promise<{ rows: unknown[] }>;
}

const FORBIDDEN_PAYLOAD_KEYS = new Set([
  "text",
  "caption",
  "comment",
  "comments",
  "transcript",
  "ocr",
  "raw_text",
  "raw",
  "description"
]);

function sanitizeForDerivedOnly(value: unknown): unknown {
  if (value === null || value === undefined) {
    return value;
  }
  if (typeof value === "number" || typeof value === "boolean") {
    return value;
  }
  if (typeof value === "string") {
    if (value.length > 256) {
      return `${value.slice(0, 252)}...`;
    }
    return value;
  }
  if (Array.isArray(value)) {
    return value.slice(0, 64).map((inner) => sanitizeForDerivedOnly(inner));
  }
  if (typeof value === "object") {
    const source = value as Record<string, unknown>;
    const out: Record<string, unknown> = {};
    for (const [key, inner] of Object.entries(source)) {
      const normalizedKey = key.trim().toLowerCase();
      if (FORBIDDEN_PAYLOAD_KEYS.has(normalizedKey)) {
        continue;
      }
      out[key] = sanitizeForDerivedOnly(inner);
    }
    return out;
  }
  return null;
}

export function buildRequestHash(payload: {
  objectiveRequested: string;
  objectiveEffective: string;
  requestId: string;
  assignmentUnitHash?: string;
}): string {
  const seed = JSON.stringify(
    {
      objective_requested: payload.objectiveRequested,
      objective_effective: payload.objectiveEffective,
      request_id: payload.requestId,
      assignment_unit_hash: payload.assignmentUnitHash || ""
    },
    Object.keys(payload).sort()
  );
  return createHash("sha256").update(seed, "utf-8").digest("hex");
}

export class FeedbackEventStore {
  private readonly cfg: FeedbackStoreConfig;
  private pool: PgPoolLike | null = null;
  private initError: string | null = null;

  constructor(cfg: FeedbackStoreConfig) {
    this.cfg = cfg;
  }

  async init(): Promise<void> {
    if (!this.cfg.enabled || !this.cfg.dbUrl.trim()) {
      this.initError = "feedback_store_disabled";
      return;
    }
    try {
      const moduleName = "pg";
      const pgModule = (await import(moduleName)) as {
        Pool: new (args: { connectionString: string }) => PgPoolLike;
      };
      this.pool = new pgModule.Pool({
        connectionString: this.cfg.dbUrl
      });
      await this.ensureSchema();
      this.initError = null;
    } catch (error) {
      this.pool = null;
      this.initError = error instanceof Error ? error.message : "feedback_store_init_failed";
    }
  }

  isReady(): boolean {
    return this.pool !== null && !this.initError;
  }

  status(): { ready: boolean; error?: string } {
    return this.isReady()
      ? { ready: true }
      : { ready: false, ...(this.initError ? { error: this.initError } : {}) };
  }

  private async ensureSchema(): Promise<void> {
    if (!this.pool) {
      return;
    }
    await this.pool.query(`
CREATE TABLE IF NOT EXISTS rec_request_events (
  request_id UUID PRIMARY KEY,
  endpoint TEXT NOT NULL,
  received_at TIMESTAMPTZ NOT NULL,
  served_at TIMESTAMPTZ NOT NULL,
  objective_requested TEXT NOT NULL,
  objective_effective TEXT NOT NULL,
  experiment_id TEXT,
  variant TEXT,
  assignment_unit_hash TEXT,
  request_hash TEXT NOT NULL,
  routing_decision JSONB NOT NULL DEFAULT '{}'::jsonb,
  compatibility_status JSONB NOT NULL DEFAULT '{}'::jsonb,
  fallback_mode BOOLEAN NOT NULL DEFAULT FALSE,
  fallback_reason TEXT,
  circuit_state JSONB NOT NULL DEFAULT '{}'::jsonb,
  latency_breakdown_ms JSONB NOT NULL DEFAULT '{}'::jsonb,
  policy_version TEXT,
  calibration_version TEXT,
  bundle_fingerprint JSONB NOT NULL DEFAULT '{}'::jsonb
);
`);
    await this.pool.query(`
CREATE TABLE IF NOT EXISTS rec_candidate_events (
  request_id UUID NOT NULL,
  candidate_id TEXT NOT NULL,
  stage TEXT NOT NULL,
  retrieved_rank INTEGER,
  final_rank INTEGER,
  selected BOOLEAN NOT NULL DEFAULT FALSE,
  retrieval_branch_scores JSONB NOT NULL DEFAULT '{}'::jsonb,
  similarity JSONB NOT NULL DEFAULT '{}'::jsonb,
  score_raw DOUBLE PRECISION,
  score_calibrated DOUBLE PRECISION,
  policy_adjusted_score DOUBLE PRECISION,
  policy_trace JSONB NOT NULL DEFAULT '{}'::jsonb,
  calibration_trace JSONB NOT NULL DEFAULT '{}'::jsonb,
  explainability_available BOOLEAN NOT NULL DEFAULT FALSE,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  PRIMARY KEY (request_id, candidate_id, stage),
  FOREIGN KEY (request_id) REFERENCES rec_request_events(request_id) ON DELETE CASCADE
);
`);
    await this.pool.query(`
CREATE TABLE IF NOT EXISTS rec_served_outputs (
  request_id UUID NOT NULL,
  candidate_id TEXT NOT NULL,
  rank INTEGER NOT NULL,
  score DOUBLE PRECISION NOT NULL,
  metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  PRIMARY KEY (request_id, candidate_id),
  FOREIGN KEY (request_id) REFERENCES rec_request_events(request_id) ON DELETE CASCADE
);
`);
    await this.pool.query(`
CREATE TABLE IF NOT EXISTS rec_experiment_assignments (
  experiment_id TEXT NOT NULL,
  objective_effective TEXT NOT NULL,
  unit_hash TEXT NOT NULL,
  variant TEXT NOT NULL,
  request_id UUID NOT NULL,
  assigned_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  PRIMARY KEY (experiment_id, objective_effective, unit_hash),
  FOREIGN KEY (request_id) REFERENCES rec_request_events(request_id) ON DELETE CASCADE
);
`);
    await this.pool.query(`
CREATE TABLE IF NOT EXISTS rec_outcome_events (
  request_id UUID NOT NULL,
  objective_effective TEXT NOT NULL,
  window_hours INTEGER NOT NULL,
  matured BOOLEAN NOT NULL DEFAULT FALSE,
  censorship_reason TEXT,
  reach_value DOUBLE PRECISION,
  engagement_value DOUBLE PRECISION,
  conversion_value DOUBLE PRECISION,
  computed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  PRIMARY KEY (request_id, window_hours),
  FOREIGN KEY (request_id) REFERENCES rec_request_events(request_id) ON DELETE CASCADE
);
`);
    await this.pool.query(`
CREATE TABLE IF NOT EXISTS rec_drift_daily (
  drift_date DATE NOT NULL,
  objective_effective TEXT NOT NULL,
  segment_id TEXT NOT NULL,
  feature_drift JSONB NOT NULL DEFAULT '{}'::jsonb,
  label_drift JSONB NOT NULL DEFAULT '{}'::jsonb,
  policy_drift JSONB NOT NULL DEFAULT '{}'::jsonb,
  breach_severity TEXT NOT NULL DEFAULT 'ok',
  trigger_recommendation TEXT NOT NULL DEFAULT 'none',
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  PRIMARY KEY (drift_date, objective_effective, segment_id)
);
`);
    await this.pool.query(`
CREATE TABLE IF NOT EXISTS rec_retrain_runs (
  run_id TEXT PRIMARY KEY,
  trigger_source TEXT NOT NULL,
  status TEXT NOT NULL,
  selected_bundle_id TEXT,
  previous_bundle_id TEXT,
  drift_evidence JSONB NOT NULL DEFAULT '{}'::jsonb,
  decision_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
`);
  }

  async writeRecommendationTrace(params: {
    request: RequestEventRecord;
    candidates: CandidateEventRecord[];
    served: ServedOutputRecord[];
    assignment?: ExperimentAssignmentRecord;
  }): Promise<void> {
    if (!this.pool || this.initError) {
      return;
    }
    const request = params.request;
    const pool = this.pool;
    await pool.query(
      `
INSERT INTO rec_request_events (
  request_id, endpoint, received_at, served_at, objective_requested, objective_effective,
  experiment_id, variant, assignment_unit_hash, request_hash, routing_decision,
  compatibility_status, fallback_mode, fallback_reason, circuit_state, latency_breakdown_ms,
  policy_version, calibration_version, bundle_fingerprint
) VALUES (
  $1::uuid, $2, $3::timestamptz, $4::timestamptz, $5, $6,
  $7, $8, $9, $10, $11::jsonb,
  $12::jsonb, $13, $14, $15::jsonb, $16::jsonb,
  $17, $18, $19::jsonb
)
ON CONFLICT (request_id) DO UPDATE SET
  served_at = EXCLUDED.served_at,
  fallback_mode = EXCLUDED.fallback_mode,
  fallback_reason = EXCLUDED.fallback_reason,
  latency_breakdown_ms = EXCLUDED.latency_breakdown_ms,
  circuit_state = EXCLUDED.circuit_state,
  compatibility_status = EXCLUDED.compatibility_status;
`,
        [
          request.requestId,
          request.endpoint,
          request.receivedAt,
          request.servedAt,
          request.objectiveRequested,
          request.objectiveEffective,
          request.experimentId ?? null,
          request.variant ?? null,
          request.assignmentUnitHash ?? null,
          request.requestHash,
          JSON.stringify(sanitizeForDerivedOnly(request.routingDecision)),
          JSON.stringify(sanitizeForDerivedOnly(request.compatibilityStatus)),
          request.fallbackMode,
          request.fallbackReason ?? null,
          JSON.stringify(sanitizeForDerivedOnly(request.circuitState)),
          JSON.stringify(sanitizeForDerivedOnly(request.latencyBreakdownMs)),
          request.policyVersion ?? null,
            request.calibrationVersion ?? null,
            JSON.stringify(sanitizeForDerivedOnly(request.bundleFingerprint))
      ]
    );

    if (params.assignment) {
      await pool.query(
        `
INSERT INTO rec_experiment_assignments (
  experiment_id, objective_effective, unit_hash, variant, request_id
) VALUES ($1, $2, $3, $4, $5::uuid)
ON CONFLICT (experiment_id, objective_effective, unit_hash) DO UPDATE SET
  variant = EXCLUDED.variant,
  request_id = EXCLUDED.request_id,
  assigned_at = NOW();
`,
          [
            params.assignment.experimentId,
            params.assignment.objectiveEffective,
            params.assignment.unitHash,
            params.assignment.variant,
            params.assignment.requestId
        ]
      );
    }

    for (const candidate of params.candidates) {
      await pool.query(
        `
INSERT INTO rec_candidate_events (
  request_id, candidate_id, stage, retrieved_rank, final_rank, selected,
  retrieval_branch_scores, similarity, score_raw, score_calibrated,
  policy_adjusted_score, policy_trace, calibration_trace, explainability_available
) VALUES (
  $1::uuid, $2, $3, $4, $5, $6,
  $7::jsonb, $8::jsonb, $9, $10,
  $11, $12::jsonb, $13::jsonb, $14
)
ON CONFLICT (request_id, candidate_id, stage) DO UPDATE SET
  retrieved_rank = EXCLUDED.retrieved_rank,
  final_rank = EXCLUDED.final_rank,
  selected = EXCLUDED.selected,
  retrieval_branch_scores = EXCLUDED.retrieval_branch_scores,
  similarity = EXCLUDED.similarity,
  score_raw = EXCLUDED.score_raw,
  score_calibrated = EXCLUDED.score_calibrated,
  policy_adjusted_score = EXCLUDED.policy_adjusted_score,
  policy_trace = EXCLUDED.policy_trace,
  calibration_trace = EXCLUDED.calibration_trace,
  explainability_available = EXCLUDED.explainability_available;
`,
          [
            candidate.requestId,
            candidate.candidateId,
            candidate.stage,
            candidate.retrievedRank ?? null,
            candidate.finalRank ?? null,
            candidate.selected,
            JSON.stringify(sanitizeForDerivedOnly(candidate.retrievalBranchScores)),
            JSON.stringify(sanitizeForDerivedOnly(candidate.similarity)),
            candidate.scoreRaw ?? null,
            candidate.scoreCalibrated ?? null,
            candidate.policyAdjustedScore ?? null,
            JSON.stringify(sanitizeForDerivedOnly(candidate.policyTrace ?? {})),
            JSON.stringify(sanitizeForDerivedOnly(candidate.calibrationTrace ?? {})),
            candidate.explainabilityAvailable
        ]
      );
    }

    for (const item of params.served) {
      await pool.query(
        `
INSERT INTO rec_served_outputs (
  request_id, candidate_id, rank, score, metadata
) VALUES (
  $1::uuid, $2, $3, $4, $5::jsonb
)
ON CONFLICT (request_id, candidate_id) DO UPDATE SET
  rank = EXCLUDED.rank,
  score = EXCLUDED.score,
  metadata = EXCLUDED.metadata;
`,
          [
            item.requestId,
            item.candidateId,
            item.rank,
            item.score,
            JSON.stringify(sanitizeForDerivedOnly(item.metadata))
        ]
      );
    }
  }
}
