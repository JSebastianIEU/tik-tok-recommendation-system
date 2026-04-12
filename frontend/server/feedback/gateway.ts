import {
  FeedbackEventStore,
  buildRequestHash,
  type CandidateEventRecord,
  type ServedOutputRecord,
  type UiFeedbackEventRecord,
  type FeedbackStoreConfig
} from "./store";

export interface TrafficMeta {
  trafficClass: "production" | "synthetic";
  isSynthetic: boolean;
  injectedFailure: boolean;
}

export interface ExperimentAssignment {
  experiment_id: string;
  variant: "control" | "treatment";
  unit_hash: string;
}

export interface TraceRecommendationParams {
  requestId: string;
  endpoint: string;
  requestReceivedAt: string;
  objectiveRequested: string;
  objectiveEffective: string;
  experimentAssignment: ExperimentAssignment;
  creatorId?: string;
  seedVideoId?: string;
  trafficMeta: TrafficMeta;
  payload: Record<string, unknown>;
}

export interface TraceReportParams {
  requestId: string;
  requestReceivedAt: string;
  objectiveRequested: string;
  objectiveEffective: string;
  experimentAssignment: ExperimentAssignment;
  creatorId?: string;
  seedVideoId?: string;
  payload: Record<string, unknown>;
}

export interface RecordUiFeedbackResult {
  ok: boolean;
  ready: boolean;
  error?: string;
}

function toCandidateEventsFromResponse(params: {
  requestId: string;
  payload: Record<string, unknown>;
}): CandidateEventRecord[] {
  const out: CandidateEventRecord[] = [];
  const debug = params.payload.debug as Record<string, unknown> | undefined;
  const retrieved = Array.isArray(debug?.retrieved_universe)
    ? (debug?.retrieved_universe as Array<Record<string, unknown>>)
    : [];
  for (const item of retrieved) {
    const candidateId = String(item.candidate_id || "").trim();
    if (!candidateId) {
      continue;
    }
    out.push({
      requestId: params.requestId,
      candidateId,
      stage: "retrieved_universe",
      retrievedRank:
        typeof item.retrieved_rank === "number" ? Math.round(item.retrieved_rank) : null,
      finalRank: null,
      selected: false,
      retrievalBranchScores:
        (item.retrieval_branch_scores as Record<string, number> | undefined) || {},
      similarity: (item.similarity as Record<string, number> | undefined) || {},
      scoreRaw: null,
      scoreCalibrated: null,
      policyAdjustedScore: null,
      policyTrace: null,
      calibrationTrace: null,
      explainabilityAvailable: false
    });
  }

  const rankingUniverse = Array.isArray(debug?.ranking_universe)
    ? (debug?.ranking_universe as Array<Record<string, unknown>>)
    : [];
  for (const item of rankingUniverse) {
    const candidateId = String(item.candidate_id || "").trim();
    if (!candidateId) {
      continue;
    }
    const policyTrace =
      (item.policy_trace as Record<string, unknown> | undefined) || {};
    const portfolioTrace =
      (item.portfolio_trace as Record<string, unknown> | undefined) || null;
    out.push({
      requestId: params.requestId,
      candidateId,
      stage: "ranked_universe",
      retrievedRank:
        typeof item.retrieved_rank === "number" ? Math.round(item.retrieved_rank) : null,
      finalRank: typeof item.final_rank === "number" ? Math.round(item.final_rank) : null,
      selected: Boolean(item.selected),
      retrievalBranchScores:
        (item.retrieval_branch_scores as Record<string, number> | undefined) || {},
      similarity: (item.similarity as Record<string, number> | undefined) || {},
      scoreRaw: typeof item.score_raw === "number" ? item.score_raw : null,
      scoreCalibrated: typeof item.score_calibrated === "number" ? item.score_calibrated : null,
      policyAdjustedScore:
        typeof item.policy_adjusted_score === "number" ? item.policy_adjusted_score : null,
      policyTrace:
        portfolioTrace && Object.keys(portfolioTrace).length > 0
          ? { ...policyTrace, portfolio_trace: portfolioTrace }
          : policyTrace,
      calibrationTrace: (item.calibration_trace as Record<string, unknown> | undefined) || null,
      explainabilityAvailable: Boolean(item.explainability_available)
    });
  }
  return out;
}

function toServedOutputEvents(params: {
  requestId: string;
  payload: Record<string, unknown>;
}): ServedOutputRecord[] {
  const items = Array.isArray(params.payload.items)
    ? (params.payload.items as Array<Record<string, unknown>>)
    : [];
  const out: ServedOutputRecord[] = [];
  for (const item of items) {
    const candidateId = String(item.candidate_id || "").trim();
    const rank = Number(item.rank);
    const score = Number(item.score);
    if (!candidateId || !Number.isFinite(rank) || !Number.isFinite(score)) {
      continue;
    }
    out.push({
      requestId: params.requestId,
      candidateId,
      rank: Math.round(rank),
      score,
      metadata: {
        served_rank: Math.round(rank),
        visible_position: Math.round(rank),
        was_exposed: true,
        author_id:
          typeof item.author_id === "string" && item.author_id.trim()
            ? item.author_id.trim()
            : null,
        topic_key:
          typeof item.topic_key === "string" && item.topic_key.trim()
            ? item.topic_key.trim()
            : null,
        content_type:
          typeof item.content_type === "string" && item.content_type.trim()
            ? item.content_type.trim()
            : null,
        language:
          typeof item.language === "string" && item.language.trim()
            ? item.language.trim()
            : null,
        locale:
          typeof item.locale === "string" && item.locale.trim()
            ? item.locale.trim()
            : null,
        hashtags: Array.isArray(item.hashtags) ? item.hashtags : [],
        keywords: Array.isArray(item.keywords) ? item.keywords : [],
        retrieval_branch_scores:
          (item.retrieval_branch_scores as Record<string, unknown> | undefined) || {},
        similarity: (item.similarity as Record<string, unknown> | undefined) || {},
        selected_ranker_id: item.selected_ranker_id,
        confidence: item.confidence,
        user_affinity_score: item.user_affinity_score,
        creator_retrieval_score: item.creator_retrieval_score,
        user_affinity_trace:
          (item.user_affinity_trace as Record<string, unknown> | undefined) || {},
        creator_retrieval_trace:
          (item.creator_retrieval_trace as Record<string, unknown> | undefined) || {},
        portfolio_trace: (item.portfolio_trace as Record<string, unknown> | undefined) || {},
        portfolio_mode:
          typeof params.payload.portfolio_mode === "boolean"
            ? params.payload.portfolio_mode
            : false,
        portfolio_metadata:
          (params.payload.portfolio_metadata as Record<string, unknown> | undefined) || {},
        retrieval_personalization_metadata:
          (params.payload.retrieval_personalization_metadata as
            | Record<string, unknown>
            | undefined) || {},
        objective_model: (item.trace as Record<string, unknown> | undefined)?.objective_model ?? null,
        score_components:
          (item.score_components as Record<string, unknown> | undefined) || {},
        support_level: item.support_level ?? null,
        support_score: item.support_score ?? null,
        ranking_reasons: Array.isArray(item.ranking_reasons) ? item.ranking_reasons : []
      }
    });
  }
  return out;
}

function buildRequestRecord(
  p: TraceRecommendationParams | TraceReportParams,
  endpointMode: string
) {
  const objectiveRequested = p.objectiveRequested;
  const objectiveEffective = p.objectiveEffective;
  const requestId = p.requestId;
  const ea = p.experimentAssignment;

  const routingDecision =
    (p.payload.routing_decision as Record<string, unknown> | undefined) || {};
  const compatibilityStatus =
    (p.payload.compatibility_status as Record<string, unknown> | undefined) || {};
  const fallbackMode = Boolean(p.payload.fallback_mode);
  const fallbackReason =
    typeof p.payload.fallback_reason === "string" ? p.payload.fallback_reason : null;
  const circuitState =
    (p.payload.circuit_state as Record<string, unknown> | undefined) || {};
  const latencyBreakdownMs =
    (p.payload.latency_breakdown_ms as Record<string, number> | undefined) || {};
  const policyVersion =
    typeof p.payload.policy_version === "string" ? p.payload.policy_version : undefined;
  const calibrationVersion =
    typeof p.payload.calibration_version === "string"
      ? p.payload.calibration_version
      : undefined;
  const policyMetadata =
    (p.payload.policy_metadata as Record<string, unknown> | undefined) || {};

  const fingerprints = compatibilityStatus?.fingerprints;
  const bundleFingerprint =
    fingerprints && typeof fingerprints === "object"
      ? (fingerprints as Record<string, unknown>)
      : {};

  const trafficMeta: TrafficMeta = "trafficMeta" in p
    ? (p as TraceRecommendationParams).trafficMeta
    : { trafficClass: "production", isSynthetic: false, injectedFailure: false };

  return {
    request: {
      requestId,
      endpoint: "endpoint" in p ? (p as TraceRecommendationParams).endpoint : "/generate-report",
      receivedAt: p.requestReceivedAt,
      servedAt: new Date().toISOString(),
      objectiveRequested,
      objectiveEffective,
      experimentId: ea.experiment_id,
      variant: ea.variant,
      assignmentUnitHash: ea.unit_hash,
      requestHash: buildRequestHash({
        objectiveRequested,
        objectiveEffective,
        requestId,
        assignmentUnitHash: ea.unit_hash
      }),
      routingDecision,
      compatibilityStatus,
      fallbackMode,
      fallbackReason,
      circuitState,
      latencyBreakdownMs,
      policyVersion,
      calibrationVersion,
      policyMetadata,
      requestContext: {
        creator_id: p.creatorId ?? null,
        seed_video_id: p.seedVideoId ?? null,
        endpoint_mode: endpointMode
      },
      trafficClass: trafficMeta.trafficClass,
      isSynthetic: trafficMeta.isSynthetic,
      injectedFailure: trafficMeta.injectedFailure,
      bundleFingerprint
    },
    assignment: {
      experimentId: ea.experiment_id,
      objectiveEffective,
      unitHash: ea.unit_hash,
      variant: ea.variant,
      requestId
    }
  };
}

export class FeedbackGateway {
  private readonly store: FeedbackEventStore;

  constructor(cfg: FeedbackStoreConfig) {
    this.store = new FeedbackEventStore(cfg);
  }

  async init(): Promise<void> {
    await this.store.init();
  }

  isReady(): boolean {
    return this.store.isReady();
  }

  status(): { ready: boolean; error?: string } {
    return this.store.status();
  }

  traceRecommendation(params: TraceRecommendationParams): void {
    const { request, assignment } = buildRequestRecord(params, "recommendations");

    void this.store
      .writeRecommendationTrace({
        request,
        assignment,
        candidates: toCandidateEventsFromResponse({
          requestId: params.requestId,
          payload: params.payload
        }),
        served: toServedOutputEvents({
          requestId: params.requestId,
          payload: params.payload
        })
      })
      .catch((error) => {
        console.error("feedback_store_write_failed", error);
      });
  }

  traceReport(params: TraceReportParams): void {
    const effectiveRequestId =
      (typeof params.payload.request_id === "string" && params.payload.request_id) ||
      params.requestId;

    const { request, assignment } = buildRequestRecord(
      { ...params, requestId: effectiveRequestId },
      "generate_report"
    );

    void this.store
      .writeRecommendationTrace({
        request,
        assignment,
        candidates: toCandidateEventsFromResponse({
          requestId: effectiveRequestId,
          payload: params.payload
        }),
        served: toServedOutputEvents({
          requestId: effectiveRequestId,
          payload: params.payload
        })
      })
      .catch((error) => {
        console.error("feedback_store_report_trace_failed", error);
      });
  }

  async recordUiFeedback(
    event: UiFeedbackEventRecord
  ): Promise<RecordUiFeedbackResult> {
    if (!this.store.isReady()) {
      return { ok: true, ready: false };
    }
    try {
      await this.store.writeUiFeedbackEvent(event);
      return { ok: true, ready: true };
    } catch (error) {
      console.error("feedback_store_ui_write_failed", error);
      const message = error instanceof Error ? error.message : "feedback_write_failed";
      return { ok: false, ready: this.store.isReady(), error: message };
    }
  }

  async getCreatorContext(params: {
    creatorId?: string;
    objective?: string;
    mapObjective: (value: string | undefined) => string;
  }): Promise<Record<string, unknown> | undefined> {
    const creatorId =
      typeof params.creatorId === "string" && params.creatorId.trim()
        ? params.creatorId.trim().toLowerCase()
        : "";
    if (!creatorId || !this.store.isReady()) {
      return undefined;
    }
    try {
      return (
        (await this.store.loadCreatorPreferenceProfile({
          creatorId,
          objectiveEffective: params.mapObjective(params.objective),
          historyDays: 180,
          maxFeedbackRows: 750
        })) as Record<string, unknown> | null
      ) ?? undefined;
    } catch (error) {
      console.error("creator_preference_profile_failed", error);
      return undefined;
    }
  }
}
