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
  policyMetadata?: Record<string, unknown>;
  requestContext?: Record<string, unknown>;
  bundleFingerprint: Record<string, unknown>;
  trafficClass?: "production" | "synthetic";
  isSynthetic?: boolean;
  injectedFailure?: boolean;
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

export interface UiFeedbackEventRecord {
  requestId: string;
  eventName: string;
  entityType: "report" | "comparable" | "recommendation" | "explainability" | "chat";
  entityId?: string | null;
  section: string;
  rank?: number | null;
  objectiveEffective: string;
  experimentId?: string | null;
  variant?: "control" | "treatment" | null;
  signalStrength: "strong" | "medium" | "weak" | "context";
  labelDirection: "positive" | "negative" | "neutral" | "context";
  metadata?: Record<string, unknown>;
  createdAt: string;
  userId?: string | null;
}

export interface CreatorPreferenceProfile {
  version: "creator_feedback_profile.v1";
  creator_id: string;
  objective_effective: string;
  built_at: string;
  last_feedback_at?: string;
  support: {
    explicit_positive_count: number;
    explicit_negative_count: number;
    explicit_request_count: number;
    objective_request_count: number;
  };
  global_preferences: Record<string, Record<string, number>>;
  objective_preferences: Record<string, Record<string, number>>;
  global_candidate_memory: {
    positive_candidate_ids: string[];
    negative_candidate_ids: string[];
  };
  objective_candidate_memory: {
    positive_candidate_ids: string[];
    negative_candidate_ids: string[];
  };
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

function normalizeTextToken(value: unknown): string | null {
  if (typeof value !== "string") {
    return null;
  }
  const normalized = value.trim().toLowerCase().replace(/^[@#]/, "");
  return normalized || null;
}

function pushWeighted(map: Map<string, number>, value: unknown, weight: number): void {
  const token = normalizeTextToken(value);
  if (!token || weight <= 0) {
    return;
  }
  map.set(token, (map.get(token) ?? 0) + weight);
}

function pushManyWeighted(map: Map<string, number>, values: unknown, weight: number): void {
  if (!Array.isArray(values)) {
    return;
  }
  for (const value of values) {
    pushWeighted(map, value, weight);
  }
}

function mapToSortedObject(map: Map<string, number>, limit = 24): Record<string, number> {
  return Object.fromEntries(
    [...map.entries()]
      .sort((left, right) => right[1] - left[1])
      .slice(0, Math.max(1, limit))
      .map(([key, value]) => [key, Number(value.toFixed(6))])
  );
}

function topKeysByWeight(map: Map<string, number>, limit = 24): string[] {
  return [...map.entries()]
    .sort((left, right) => right[1] - left[1])
    .slice(0, Math.max(1, limit))
    .map(([key]) => key);
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
  policy_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
  request_context JSONB NOT NULL DEFAULT '{}'::jsonb,
  bundle_fingerprint JSONB NOT NULL DEFAULT '{}'::jsonb,
  traffic_class TEXT NOT NULL DEFAULT 'production',
  is_synthetic BOOLEAN NOT NULL DEFAULT FALSE,
  injected_failure BOOLEAN NOT NULL DEFAULT FALSE
);
`);
    await this.pool.query(
      "ALTER TABLE rec_request_events ADD COLUMN IF NOT EXISTS traffic_class TEXT NOT NULL DEFAULT 'production';"
    );
    await this.pool.query(
      "ALTER TABLE rec_request_events ADD COLUMN IF NOT EXISTS is_synthetic BOOLEAN NOT NULL DEFAULT FALSE;"
    );
    await this.pool.query(
      "ALTER TABLE rec_request_events ADD COLUMN IF NOT EXISTS injected_failure BOOLEAN NOT NULL DEFAULT FALSE;"
    );
    await this.pool.query(
      "ALTER TABLE rec_request_events ADD COLUMN IF NOT EXISTS policy_metadata JSONB NOT NULL DEFAULT '{}'::jsonb;"
    );
    await this.pool.query(
      "ALTER TABLE rec_request_events ADD COLUMN IF NOT EXISTS request_context JSONB NOT NULL DEFAULT '{}'::jsonb;"
    );
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
    await this.pool.query(`
CREATE TABLE IF NOT EXISTS rec_ui_feedback_events (
  event_id BIGSERIAL PRIMARY KEY,
  request_id UUID NOT NULL,
  event_name TEXT NOT NULL,
  entity_type TEXT NOT NULL,
  entity_id TEXT,
  section TEXT NOT NULL,
  rank INTEGER,
  objective_effective TEXT NOT NULL,
  experiment_id TEXT,
  variant TEXT,
  signal_strength TEXT NOT NULL,
  label_direction TEXT NOT NULL,
  metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  FOREIGN KEY (request_id) REFERENCES rec_request_events(request_id) ON DELETE CASCADE
);
`);
    await this.pool.query(
      "ALTER TABLE rec_ui_feedback_events ADD COLUMN IF NOT EXISTS user_id TEXT;"
    );
    await this.pool.query(
      "CREATE INDEX IF NOT EXISTS idx_rec_ui_feedback_events_user_id ON rec_ui_feedback_events(user_id);"
    );
    await this.pool.query(
      "ALTER TABLE rec_request_events ADD COLUMN IF NOT EXISTS user_id TEXT;"
    );
    await this.pool.query(
      "CREATE INDEX IF NOT EXISTS idx_rec_request_events_user_id ON rec_request_events(user_id);"
    );
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
  policy_version, calibration_version, policy_metadata, request_context, bundle_fingerprint, traffic_class, is_synthetic, injected_failure
) VALUES (
  $1::uuid, $2, $3::timestamptz, $4::timestamptz, $5, $6,
  $7, $8, $9, $10, $11::jsonb,
  $12::jsonb, $13, $14, $15::jsonb, $16::jsonb,
  $17, $18, $19::jsonb, $20::jsonb, $21::jsonb, $22, $23, $24
)
ON CONFLICT (request_id) DO UPDATE SET
  served_at = EXCLUDED.served_at,
  fallback_mode = EXCLUDED.fallback_mode,
  fallback_reason = EXCLUDED.fallback_reason,
  latency_breakdown_ms = EXCLUDED.latency_breakdown_ms,
  circuit_state = EXCLUDED.circuit_state,
  compatibility_status = EXCLUDED.compatibility_status,
  policy_metadata = EXCLUDED.policy_metadata,
  request_context = EXCLUDED.request_context,
  traffic_class = EXCLUDED.traffic_class,
  is_synthetic = EXCLUDED.is_synthetic,
  injected_failure = EXCLUDED.injected_failure;
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
          JSON.stringify(sanitizeForDerivedOnly(request.policyMetadata ?? {})),
          JSON.stringify(sanitizeForDerivedOnly(request.requestContext ?? {})),
          JSON.stringify(sanitizeForDerivedOnly(request.bundleFingerprint)),
          request.trafficClass === "synthetic" ? "synthetic" : "production",
          Boolean(request.isSynthetic),
          Boolean(request.injectedFailure)
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

  async loadCreatorPreferenceProfile(params: {
    creatorId: string;
    objectiveEffective: string;
    historyDays?: number;
    maxFeedbackRows?: number;
    userId?: string;
  }): Promise<CreatorPreferenceProfile | null> {
    if (!this.pool || this.initError) {
      return null;
    }
    const creatorId = normalizeTextToken(params.creatorId);
    const userId = params.userId ? normalizeTextToken(params.userId) : null;
    const objectiveEffective = normalizeTextToken(params.objectiveEffective);
    if ((!creatorId && !userId) || !objectiveEffective) {
      return null;
    }
    const historyDays = Math.max(7, Math.min(365, Math.round(params.historyDays ?? 180)));
    const maxFeedbackRows = Math.max(50, Math.min(5000, Math.round(params.maxFeedbackRows ?? 750)));

    const userClause = userId
      ? `(uife.user_id = $1 OR COALESCE(re.request_context->>'creator_id', '') = $1)`
      : `COALESCE(re.request_context->>'creator_id', '') = $1`;
    const lookupId = userId ?? creatorId!;

    const rows = await this.pool.query(
      `
SELECT
  re.request_id::text,
  re.objective_effective,
  so.candidate_id,
  uife.event_name,
  uife.created_at,
  so.metadata
FROM rec_request_events re
JOIN rec_ui_feedback_events uife
  ON uife.request_id = re.request_id
JOIN rec_served_outputs so
  ON so.request_id = re.request_id
 AND so.candidate_id = uife.entity_id
WHERE ${userClause}
  AND uife.entity_type = 'comparable'
  AND uife.signal_strength = 'strong'
  AND uife.event_name IN (
    'comparable_saved',
    'comparable_marked_relevant',
    'comparable_marked_not_relevant'
  )
  AND uife.created_at >= NOW() - ($2::text || ' days')::interval
ORDER BY uife.created_at DESC
LIMIT $3
`,
      [lookupId, String(historyDays), maxFeedbackRows]
    );

    type AggregateEntry = {
      requestId: string;
      candidateId: string;
      objectiveEffective: string;
      events: string[];
      latestCreatedAt?: string;
      metadata: Record<string, unknown>;
    };

    const aggregates = new Map<string, AggregateEntry>();
    for (const row of rows.rows as Array<Record<string, unknown>>) {
      const requestId = typeof row.request_id === "string" ? row.request_id : "";
      const objective = typeof row.objective_effective === "string" ? row.objective_effective : "";
      const candidateId = typeof row.candidate_id === "string" ? row.candidate_id : "";
      const eventName = typeof row.event_name === "string" ? row.event_name : "";
      const createdAt =
        typeof row.created_at === "string"
          ? row.created_at
          : row.created_at instanceof Date
            ? row.created_at.toISOString()
            : undefined;
      const metadata =
        row.metadata && typeof row.metadata === "object" && !Array.isArray(row.metadata)
          ? (row.metadata as Record<string, unknown>)
          : {};
      const key = `${requestId}::${candidateId}`;
      const existing = aggregates.get(key);
      if (existing) {
        existing.events.push(eventName);
        if (createdAt && (!existing.latestCreatedAt || createdAt > existing.latestCreatedAt)) {
          existing.latestCreatedAt = createdAt;
        }
        continue;
      }
      aggregates.set(key, {
        requestId,
        candidateId,
        objectiveEffective: objective,
        events: [eventName],
        latestCreatedAt: createdAt,
        metadata
      });
    }

    const globalPositiveTopics = new Map<string, number>();
    const globalNegativeTopics = new Map<string, number>();
    const globalPositiveContentTypes = new Map<string, number>();
    const globalNegativeContentTypes = new Map<string, number>();
    const globalPositiveHashtags = new Map<string, number>();
    const globalNegativeHashtags = new Map<string, number>();
    const globalPositiveAuthors = new Map<string, number>();
    const globalNegativeAuthors = new Map<string, number>();

    const objectivePositiveTopics = new Map<string, number>();
    const objectiveNegativeTopics = new Map<string, number>();
    const objectivePositiveContentTypes = new Map<string, number>();
    const objectiveNegativeContentTypes = new Map<string, number>();
    const objectivePositiveHashtags = new Map<string, number>();
    const objectiveNegativeHashtags = new Map<string, number>();
    const objectivePositiveAuthors = new Map<string, number>();
    const objectiveNegativeAuthors = new Map<string, number>();
    const globalPositiveCandidates = new Map<string, number>();
    const globalNegativeCandidates = new Map<string, number>();
    const objectivePositiveCandidates = new Map<string, number>();
    const objectiveNegativeCandidates = new Map<string, number>();

    let explicitPositiveCount = 0;
    let explicitNegativeCount = 0;
    let explicitRequestCount = 0;
    let objectiveRequestCount = 0;
    let lastFeedbackAt = "";

    const requestsWithFeedback = new Set<string>();
    const objectiveRequestsWithFeedback = new Set<string>();

    for (const entry of aggregates.values()) {
      const hasSaved = entry.events.includes("comparable_saved");
      const hasRelevant = entry.events.includes("comparable_marked_relevant");
      const hasNegative = entry.events.includes("comparable_marked_not_relevant");
      if ((hasSaved || hasRelevant) && hasNegative) {
        continue;
      }
      const isPositive = hasSaved || hasRelevant;
      const isNegative = hasNegative;
      if (!isPositive && !isNegative) {
        continue;
      }
      const ageDays =
        entry.latestCreatedAt && !Number.isNaN(Date.parse(entry.latestCreatedAt))
          ? Math.max(
              0,
              (Date.now() - Date.parse(entry.latestCreatedAt)) / (1000 * 60 * 60 * 24)
            )
          : historyDays;
      const recencyWeight = Math.max(0.1, Math.exp((-Math.log(2) * ageDays) / 45));
      const labelWeight = hasSaved ? 1.25 : 1.0;
      const totalWeight = recencyWeight * labelWeight;
      const topicKey = entry.metadata.topic_key;
      const contentType = entry.metadata.content_type;
      const authorId = entry.metadata.author_id;
      const hashtags = entry.metadata.hashtags;

      if (isPositive) {
        explicitPositiveCount += 1;
        requestsWithFeedback.add(entry.requestId);
        pushWeighted(globalPositiveTopics, topicKey, totalWeight);
        pushWeighted(globalPositiveContentTypes, contentType, totalWeight);
        pushManyWeighted(globalPositiveHashtags, hashtags, totalWeight);
        pushWeighted(globalPositiveAuthors, authorId, totalWeight * 0.5);
        pushWeighted(globalPositiveCandidates, entry.candidateId, totalWeight);
        if (entry.objectiveEffective === objectiveEffective) {
          objectiveRequestsWithFeedback.add(entry.requestId);
          pushWeighted(objectivePositiveTopics, topicKey, totalWeight);
          pushWeighted(objectivePositiveContentTypes, contentType, totalWeight);
          pushManyWeighted(objectivePositiveHashtags, hashtags, totalWeight);
          pushWeighted(objectivePositiveAuthors, authorId, totalWeight * 0.5);
          pushWeighted(objectivePositiveCandidates, entry.candidateId, totalWeight);
        }
      } else if (isNegative) {
        explicitNegativeCount += 1;
        requestsWithFeedback.add(entry.requestId);
        pushWeighted(globalNegativeTopics, topicKey, totalWeight);
        pushWeighted(globalNegativeContentTypes, contentType, totalWeight);
        pushManyWeighted(globalNegativeHashtags, hashtags, totalWeight);
        pushWeighted(globalNegativeAuthors, authorId, totalWeight * 0.5);
        pushWeighted(globalNegativeCandidates, entry.candidateId, totalWeight);
        if (entry.objectiveEffective === objectiveEffective) {
          objectiveRequestsWithFeedback.add(entry.requestId);
          pushWeighted(objectiveNegativeTopics, topicKey, totalWeight);
          pushWeighted(objectiveNegativeContentTypes, contentType, totalWeight);
          pushManyWeighted(objectiveNegativeHashtags, hashtags, totalWeight);
          pushWeighted(objectiveNegativeAuthors, authorId, totalWeight * 0.5);
          pushWeighted(objectiveNegativeCandidates, entry.candidateId, totalWeight);
        }
      }

      if (entry.latestCreatedAt && entry.latestCreatedAt > lastFeedbackAt) {
        lastFeedbackAt = entry.latestCreatedAt;
      }
    }

    explicitRequestCount = requestsWithFeedback.size;
    objectiveRequestCount = objectiveRequestsWithFeedback.size;
    if (explicitPositiveCount + explicitNegativeCount <= 0) {
      return null;
    }

    return {
      version: "creator_feedback_profile.v1",
      creator_id: creatorId,
      objective_effective: objectiveEffective,
      built_at: new Date().toISOString(),
      ...(lastFeedbackAt ? { last_feedback_at: lastFeedbackAt } : {}),
      support: {
        explicit_positive_count: explicitPositiveCount,
        explicit_negative_count: explicitNegativeCount,
        explicit_request_count: explicitRequestCount,
        objective_request_count: objectiveRequestCount
      },
      global_preferences: {
        topics_positive: mapToSortedObject(globalPositiveTopics),
        topics_negative: mapToSortedObject(globalNegativeTopics),
        content_types_positive: mapToSortedObject(globalPositiveContentTypes),
        content_types_negative: mapToSortedObject(globalNegativeContentTypes),
        hashtags_positive: mapToSortedObject(globalPositiveHashtags),
        hashtags_negative: mapToSortedObject(globalNegativeHashtags),
        authors_positive: mapToSortedObject(globalPositiveAuthors),
        authors_negative: mapToSortedObject(globalNegativeAuthors)
      },
      objective_preferences: {
        topics_positive: mapToSortedObject(objectivePositiveTopics),
        topics_negative: mapToSortedObject(objectiveNegativeTopics),
        content_types_positive: mapToSortedObject(objectivePositiveContentTypes),
        content_types_negative: mapToSortedObject(objectiveNegativeContentTypes),
        hashtags_positive: mapToSortedObject(objectivePositiveHashtags),
        hashtags_negative: mapToSortedObject(objectiveNegativeHashtags),
        authors_positive: mapToSortedObject(objectivePositiveAuthors),
        authors_negative: mapToSortedObject(objectiveNegativeAuthors)
      },
      global_candidate_memory: {
        positive_candidate_ids: topKeysByWeight(globalPositiveCandidates),
        negative_candidate_ids: topKeysByWeight(globalNegativeCandidates)
      },
      objective_candidate_memory: {
        positive_candidate_ids: topKeysByWeight(objectivePositiveCandidates),
        negative_candidate_ids: topKeysByWeight(objectiveNegativeCandidates)
      }
    };
  }

  async writeUiFeedbackEvent(event: UiFeedbackEventRecord): Promise<void> {
    if (!this.pool || this.initError) {
      return;
    }
    await this.pool.query(
      `
INSERT INTO rec_ui_feedback_events (
  request_id, event_name, entity_type, entity_id, section, rank, objective_effective,
  experiment_id, variant, signal_strength, label_direction, metadata, created_at, user_id
) VALUES (
  $1::uuid, $2, $3, $4, $5, $6, $7,
  $8, $9, $10, $11, $12::jsonb, $13::timestamptz, $14
);
`,
      [
        event.requestId,
        event.eventName,
        event.entityType,
        event.entityId ?? null,
        event.section,
        event.rank ?? null,
        event.objectiveEffective,
        event.experimentId ?? null,
        event.variant ?? null,
        event.signalStrength,
        event.labelDirection,
        JSON.stringify(sanitizeForDerivedOnly(event.metadata ?? {})),
        event.createdAt,
        event.userId ?? null
      ]
    );
  }
}
