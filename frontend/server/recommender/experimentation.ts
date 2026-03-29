import { createHash } from "node:crypto";

export type ExperimentVariant = "control" | "treatment";

export interface ExperimentRequest {
  id?: string;
  force_variant?: ExperimentVariant;
}

export interface ExperimentAssignment {
  experiment_id: string;
  objective_effective: string;
  unit_hash: string;
  variant: ExperimentVariant;
  assignment_key: string;
}

export interface ExperimentPolicyConfig {
  defaultExperimentId: string;
  treatmentRatio: number;
  salt: string;
  controlBundleId?: string;
  treatmentBundleId?: string;
}

export interface ExperimentRoutePolicy {
  requiredCompat: Record<string, string>;
  modelFamilySuffix: string;
}

function normalizeText(value: string | undefined): string {
  return (value || "").trim().toLowerCase();
}

function normalizeObjective(value: string | undefined): string {
  const normalized = normalizeText(value);
  if (normalized === "community") {
    return "engagement";
  }
  if (normalized === "reach" || normalized === "engagement" || normalized === "conversion") {
    return normalized;
  }
  return "engagement";
}

export function stableHashHex(value: string): string {
  return createHash("sha256").update(value, "utf-8").digest("hex");
}

function hashToUnitInterval(hex: string): number {
  const prefix = hex.slice(0, 8);
  const value = Number.parseInt(prefix, 16);
  if (!Number.isFinite(value) || value <= 0) {
    return 0;
  }
  return value / 0xffffffff;
}

function normalizeExperimentId(value: string | undefined, fallback: string): string {
  const normalized = normalizeText(value).replace(/[^a-z0-9._-]+/g, "_");
  return normalized || fallback;
}

export function buildExperimentAssignment(params: {
  objectiveRequested?: string;
  experiment?: ExperimentRequest;
  assignmentUnit: string;
  config: ExperimentPolicyConfig;
}): ExperimentAssignment {
  const objectiveEffective = normalizeObjective(params.objectiveRequested);
  const experimentId = normalizeExperimentId(params.experiment?.id, params.config.defaultExperimentId);
  const unitHash = stableHashHex(params.assignmentUnit);
  const ratio = Math.max(0, Math.min(1, Number(params.config.treatmentRatio) || 0.5));
  const forced = params.experiment?.force_variant;
  let variant: ExperimentVariant;
  if (forced === "control" || forced === "treatment") {
    variant = forced;
  } else {
    const bucketSeed = `${params.config.salt}::${experimentId}::${objectiveEffective}::${unitHash}`;
    const bucket = hashToUnitInterval(stableHashHex(bucketSeed));
    variant = bucket < ratio ? "treatment" : "control";
  }
  const assignmentKey = `${experimentId}::${objectiveEffective}::${unitHash}`;
  return {
    experiment_id: experimentId,
    objective_effective: objectiveEffective,
    unit_hash: unitHash,
    variant,
    assignment_key: assignmentKey
  };
}

export function buildExperimentRoutePolicy(params: {
  assignment: ExperimentAssignment;
  config: ExperimentPolicyConfig;
}): ExperimentRoutePolicy {
  const requiredCompat: Record<string, string> = {};
  if (params.assignment.variant === "control" && normalizeText(params.config.controlBundleId)) {
    requiredCompat.bundle_id = normalizeText(params.config.controlBundleId);
  }
  if (
    params.assignment.variant === "treatment" &&
    normalizeText(params.config.treatmentBundleId)
  ) {
    requiredCompat.bundle_id = normalizeText(params.config.treatmentBundleId);
  }
  return {
    requiredCompat,
    modelFamilySuffix:
      params.assignment.variant === "control" ? "baseline" : "treatment"
  };
}
