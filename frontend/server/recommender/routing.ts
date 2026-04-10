export type RoutingTrack = "pre_publication" | "post_publication";

export interface RoutingRequestInput {
  track?: RoutingTrack;
  allow_fallback?: boolean;
  required_compat?: Record<string, string>;
  request_id?: string;
  experiment?: {
    id?: string;
    variant?: "control" | "treatment";
    unit_hash?: string;
  };
}

export interface RuntimeStageBudgets {
  retrieval: number;
  ranking: number;
  explainability: number;
}

export interface RoutingEnvelope {
  objective_requested: string;
  objective_effective: string;
  track: RoutingTrack;
  allow_fallback: boolean;
  required_compat: Record<string, string>;
  model_family: string;
  request_id?: string;
  experiment?: {
    id?: string;
    variant?: "control" | "treatment";
    unit_hash?: string;
  };
  stage_budgets_ms: RuntimeStageBudgets;
}

function normalizeObjective(value: string | undefined): string {
  const normalized = typeof value === "string" ? value.trim().toLowerCase() : "";
  if (!normalized) {
    return "engagement";
  }
  if (normalized === "community") {
    return "engagement";
  }
  if (normalized === "reach" || normalized === "engagement" || normalized === "conversion") {
    return normalized;
  }
  return "engagement";
}

function normalizeTrack(value: string | undefined): RoutingTrack {
  if (value === "pre_publication" || value === "post_publication") {
    return value;
  }
  return "post_publication";
}

export function buildRoutingEnvelope(params: {
  objectiveRequested: string;
  track?: string;
  allowFallback?: boolean;
  requiredCompat?: Record<string, string>;
  requestId?: string;
  modelFamilySuffix?: string;
  experiment?: {
    id?: string;
    variant?: "control" | "treatment";
    unit_hash?: string;
  };
  stageBudgetsMs: RuntimeStageBudgets;
}): RoutingEnvelope {
  const objectiveRequested = (params.objectiveRequested || "engagement").trim().toLowerCase();
  const objectiveEffective = normalizeObjective(objectiveRequested);
  return {
    objective_requested: objectiveRequested || "engagement",
    objective_effective: objectiveEffective,
    track: normalizeTrack(params.track),
    allow_fallback: params.allowFallback ?? true,
    required_compat: {
      component: "recommender-learning-v1",
      ...(params.requiredCompat ?? {})
    },
    model_family: `ranker_${objectiveEffective}${
      params.modelFamilySuffix ? `_${params.modelFamilySuffix}` : ""
    }`,
    request_id: params.requestId,
    experiment: params.experiment,
    stage_budgets_ms: {
      retrieval: Math.max(1, Number(params.stageBudgetsMs.retrieval) || 260),
      ranking: Math.max(1, Number(params.stageBudgetsMs.ranking) || 260),
      explainability: Math.max(1, Number(params.stageBudgetsMs.explainability) || 300)
    }
  };
}
