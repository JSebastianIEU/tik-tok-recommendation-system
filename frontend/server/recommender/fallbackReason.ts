export interface RecommenderFailureClassification {
  reason: "required_compat_mismatch" | "incompatible_artifact" | "core_stage_timeout" | "python_request_failed";
  compatibilityMismatch: boolean;
}

export function classifyRecommenderFailure(error: string): RecommenderFailureClassification {
  const normalized = String(error || "").trim().toLowerCase();
  if (
    normalized.includes("required_compat_mismatch") ||
    (normalized.includes("required") && normalized.includes("compat") && normalized.includes("mismatch"))
  ) {
    return {
      reason: "required_compat_mismatch",
      compatibilityMismatch: true
    };
  }
  if (normalized.includes("incompatible_artifact")) {
    return {
      reason: "incompatible_artifact",
      compatibilityMismatch: true
    };
  }
  if (normalized.includes("stage_timeout")) {
    return {
      reason: "core_stage_timeout",
      compatibilityMismatch: false
    };
  }
  return {
    reason: "python_request_failed",
    compatibilityMismatch: false
  };
}
