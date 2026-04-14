const ENTITY_TYPES = new Set(["report", "comparable", "recommendation", "explainability", "chat"]);
const SIGNAL_STRENGTHS = new Set(["strong", "medium", "weak", "context"]);
const LABEL_DIRECTIONS = new Set(["positive", "negative", "neutral", "context"]);

export interface ReportFeedbackRequest {
  request_id: string;
  event_name: string;
  entity_type: "report" | "comparable" | "recommendation" | "explainability" | "chat";
  entity_id?: string;
  section: string;
  rank?: number;
  objective_effective: string;
  experiment_id?: string;
  variant?: "control" | "treatment";
  signal_strength: "strong" | "medium" | "weak" | "context";
  label_direction: "positive" | "negative" | "neutral" | "context";
  metadata?: Record<string, unknown>;
  user_id?: string;
}

type ParseResult =
  | { ok: true; value: ReportFeedbackRequest }
  | { ok: false; error: string };

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function normalizeString(value: unknown): string | null {
  if (typeof value !== "string") {
    return null;
  }
  const normalized = value.trim();
  return normalized ? normalized : null;
}

export function parseReportFeedbackRequest(value: unknown): ParseResult {
  if (!isObject(value)) {
    return { ok: false, error: "Request body must be an object." };
  }
  const requestId = normalizeString(value.request_id);
  const eventName = normalizeString(value.event_name);
  const entityType = normalizeString(value.entity_type);
  const section = normalizeString(value.section);
  const objectiveEffective = normalizeString(value.objective_effective);
  const signalStrength = normalizeString(value.signal_strength);
  const labelDirection = normalizeString(value.label_direction);
  if (!requestId || !eventName || !entityType || !section || !objectiveEffective || !signalStrength || !labelDirection) {
    return {
      ok: false,
      error:
        "request_id, event_name, entity_type, section, objective_effective, signal_strength, and label_direction are required."
    };
  }
  if (!ENTITY_TYPES.has(entityType)) {
    return { ok: false, error: "entity_type is invalid." };
  }
  if (!SIGNAL_STRENGTHS.has(signalStrength)) {
    return { ok: false, error: "signal_strength is invalid." };
  }
  if (!LABEL_DIRECTIONS.has(labelDirection)) {
    return { ok: false, error: "label_direction is invalid." };
  }
  if (value.rank !== undefined && !isFiniteNumber(value.rank)) {
    return { ok: false, error: "rank must be a finite number when provided." };
  }
  if (value.variant !== undefined && value.variant !== "control" && value.variant !== "treatment") {
    return { ok: false, error: "variant must be control or treatment when provided." };
  }
  if (value.metadata !== undefined && !isObject(value.metadata)) {
    return { ok: false, error: "metadata must be an object when provided." };
  }
  const entityId = normalizeString(value.entity_id) ?? undefined;
  const experimentId = normalizeString(value.experiment_id) ?? undefined;
  const userId = normalizeString(value.user_id) ?? undefined;
  return {
    ok: true,
    value: {
      request_id: requestId,
      event_name: eventName,
      entity_type: entityType as ReportFeedbackRequest["entity_type"],
      entity_id: entityId,
      section,
      rank: value.rank === undefined ? undefined : Math.round(value.rank),
      objective_effective: objectiveEffective,
      experiment_id: experimentId,
      variant:
        value.variant === "control" || value.variant === "treatment"
          ? value.variant
          : undefined,
      signal_strength: signalStrength as ReportFeedbackRequest["signal_strength"],
      label_direction: labelDirection as ReportFeedbackRequest["label_direction"],
      metadata: (value.metadata as Record<string, unknown> | undefined) ?? undefined,
      user_id: userId
    }
  };
}
