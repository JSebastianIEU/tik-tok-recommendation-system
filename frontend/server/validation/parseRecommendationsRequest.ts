import {
  parseGenerateReportRequest,
  type ParsedGenerateReportRequest
} from "./parseGenerateReportRequest";

export interface ParsedRecommendationsRequest extends ParsedGenerateReportRequest {
  as_of_time: string;
  top_k: number;
  retrieve_k: number;
  language?: string;
  candidate_ids: string[];
  policy_overrides: {
    strict_language?: boolean;
    strict_locale?: boolean;
    max_items_per_author?: number;
  };
  portfolio: {
    enabled?: boolean;
    weights?: {
      reach?: number;
      conversion?: number;
      durability?: number;
    };
    risk_aversion?: number;
    candidate_pool_cap?: number;
  };
  graph_controls: {
    enable_graph_branch?: boolean;
  };
  trajectory_controls: {
    enabled?: boolean;
  };
  explainability: {
    enabled?: boolean;
    top_features?: number;
    neighbor_k?: number;
    run_counterfactuals?: boolean;
  };
  routing: {
    track?: "pre_publication" | "post_publication";
    allow_fallback?: boolean;
    required_compat?: Record<string, string>;
  };
  experiment: {
    id?: string;
    force_variant?: "control" | "treatment";
  };
  debug: boolean;
}

export type ParseRecommendationsResult =
  | { ok: true; value: ParsedRecommendationsRequest }
  | { ok: false; error: string };

function parsePositiveInteger(value: unknown, fallback: number, min: number, max: number): number | null {
  if (value === undefined || value === null) {
    return fallback;
  }
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return null;
  }
  const rounded = Math.round(value);
  if (rounded < min || rounded > max) {
    return null;
  }
  return rounded;
}

function parseAsOfTime(value: unknown): string | null {
  if (value === undefined || value === null || value === "") {
    return new Date().toISOString();
  }
  if (typeof value !== "string") {
    return null;
  }
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return null;
  }
  return parsed.toISOString();
}

function parseLanguage(value: unknown): string | null | undefined {
  if (value === undefined || value === null || value === "") {
    return undefined;
  }
  if (typeof value !== "string") {
    return null;
  }
  const normalized = value.trim().toLowerCase();
  if (!normalized) {
    return undefined;
  }
  if (normalized.length > 8) {
    return null;
  }
  return normalized;
}

function parseCandidateIds(value: unknown): string[] | null {
  if (value === undefined || value === null) {
    return [];
  }
  if (!Array.isArray(value)) {
    return null;
  }
  const out: string[] = [];
  for (const item of value) {
    if (typeof item !== "string") {
      return null;
    }
    const normalized = item.trim();
    if (!normalized) {
      continue;
    }
    out.push(normalized);
    if (out.length > 5000) {
      return null;
    }
  }
  return out;
}

function parsePolicyOverrides(
  value: unknown
): { strict_language?: boolean; strict_locale?: boolean; max_items_per_author?: number } | null {
  if (value === undefined || value === null) {
    return {};
  }
  if (typeof value !== "object" || Array.isArray(value)) {
    return null;
  }
  const source = value as Record<string, unknown>;
  const out: { strict_language?: boolean; strict_locale?: boolean; max_items_per_author?: number } = {};
  if (source.strict_language !== undefined) {
    if (typeof source.strict_language !== "boolean") {
      return null;
    }
    out.strict_language = source.strict_language;
  }
  if (source.strict_locale !== undefined) {
    if (typeof source.strict_locale !== "boolean") {
      return null;
    }
    out.strict_locale = source.strict_locale;
  }
  if (source.max_items_per_author !== undefined) {
    if (
      typeof source.max_items_per_author !== "number" ||
      !Number.isFinite(source.max_items_per_author)
    ) {
      return null;
    }
    const rounded = Math.round(source.max_items_per_author);
    if (rounded < 1 || rounded > 20) {
      return null;
    }
    out.max_items_per_author = rounded;
  }
  return out;
}

function parsePortfolio(
  value: unknown
): {
  enabled?: boolean;
  weights?: {
    reach?: number;
    conversion?: number;
    durability?: number;
  };
  risk_aversion?: number;
  candidate_pool_cap?: number;
} | null {
  if (value === undefined || value === null) {
    return {};
  }
  if (typeof value !== "object" || Array.isArray(value)) {
    return null;
  }
  const source = value as Record<string, unknown>;
  const out: {
    enabled?: boolean;
    weights?: {
      reach?: number;
      conversion?: number;
      durability?: number;
    };
    risk_aversion?: number;
    candidate_pool_cap?: number;
  } = {};
  if (source.enabled !== undefined) {
    if (typeof source.enabled !== "boolean") {
      return null;
    }
    out.enabled = source.enabled;
  }
  if (source.risk_aversion !== undefined) {
    if (typeof source.risk_aversion !== "number" || !Number.isFinite(source.risk_aversion)) {
      return null;
    }
    if (source.risk_aversion < 0 || source.risk_aversion > 2) {
      return null;
    }
    out.risk_aversion = source.risk_aversion;
  }
  if (source.candidate_pool_cap !== undefined) {
    if (typeof source.candidate_pool_cap !== "number" || !Number.isFinite(source.candidate_pool_cap)) {
      return null;
    }
    const rounded = Math.round(source.candidate_pool_cap);
    if (rounded < 1 || rounded > 1000) {
      return null;
    }
    out.candidate_pool_cap = rounded;
  }
  if (source.weights !== undefined) {
    if (typeof source.weights !== "object" || source.weights === null || Array.isArray(source.weights)) {
      return null;
    }
    const rawWeights = source.weights as Record<string, unknown>;
    const parsedWeights: { reach?: number; conversion?: number; durability?: number } = {};
    for (const key of ["reach", "conversion", "durability"] as const) {
      const inner = rawWeights[key];
      if (inner === undefined) {
        continue;
      }
      if (typeof inner !== "number" || !Number.isFinite(inner)) {
        return null;
      }
      if (inner < 0 || inner > 10) {
        return null;
      }
      parsedWeights[key] = inner;
    }
    out.weights = parsedWeights;
  }
  return out;
}

function parseExplainability(
  value: unknown
): {
  enabled?: boolean;
  top_features?: number;
  neighbor_k?: number;
  run_counterfactuals?: boolean;
} | null {
  if (value === undefined || value === null) {
    return {};
  }
  if (typeof value !== "object" || Array.isArray(value)) {
    return null;
  }
  const source = value as Record<string, unknown>;
  const out: {
    enabled?: boolean;
    top_features?: number;
    neighbor_k?: number;
    run_counterfactuals?: boolean;
  } = {};
  if (source.enabled !== undefined) {
    if (typeof source.enabled !== "boolean") {
      return null;
    }
    out.enabled = source.enabled;
  }
  if (source.run_counterfactuals !== undefined) {
    if (typeof source.run_counterfactuals !== "boolean") {
      return null;
    }
    out.run_counterfactuals = source.run_counterfactuals;
  }
  if (source.top_features !== undefined) {
    if (typeof source.top_features !== "number" || !Number.isFinite(source.top_features)) {
      return null;
    }
    const rounded = Math.round(source.top_features);
    if (rounded < 1 || rounded > 12) {
      return null;
    }
    out.top_features = rounded;
  }
  if (source.neighbor_k !== undefined) {
    if (typeof source.neighbor_k !== "number" || !Number.isFinite(source.neighbor_k)) {
      return null;
    }
    const rounded = Math.round(source.neighbor_k);
    if (rounded < 1 || rounded > 6) {
      return null;
    }
    out.neighbor_k = rounded;
  }
  return out;
}

function parseGraphControls(
  value: unknown
): { enable_graph_branch?: boolean } | null {
  if (value === undefined || value === null) {
    return {};
  }
  if (typeof value !== "object" || Array.isArray(value)) {
    return null;
  }
  const source = value as Record<string, unknown>;
  const out: { enable_graph_branch?: boolean } = {};
  if (source.enable_graph_branch !== undefined) {
    if (typeof source.enable_graph_branch !== "boolean") {
      return null;
    }
    out.enable_graph_branch = source.enable_graph_branch;
  }
  return out;
}

function parseTrajectoryControls(value: unknown): { enabled?: boolean } | null {
  if (value === undefined || value === null) {
    return {};
  }
  if (typeof value !== "object" || Array.isArray(value)) {
    return null;
  }
  const source = value as Record<string, unknown>;
  const out: { enabled?: boolean } = {};
  if (source.enabled !== undefined) {
    if (typeof source.enabled !== "boolean") {
      return null;
    }
    out.enabled = source.enabled;
  }
  return out;
}

function parseRouting(
  value: unknown
): {
  track?: "pre_publication" | "post_publication";
  allow_fallback?: boolean;
  required_compat?: Record<string, string>;
} | null {
  if (value === undefined || value === null) {
    return {};
  }
  if (typeof value !== "object" || Array.isArray(value)) {
    return null;
  }
  const source = value as Record<string, unknown>;
  const out: {
    track?: "pre_publication" | "post_publication";
    allow_fallback?: boolean;
    required_compat?: Record<string, string>;
  } = {};
  if (source.track !== undefined) {
    if (source.track !== "pre_publication" && source.track !== "post_publication") {
      return null;
    }
    out.track = source.track;
  }
  if (source.allow_fallback !== undefined) {
    if (typeof source.allow_fallback !== "boolean") {
      return null;
    }
    out.allow_fallback = source.allow_fallback;
  }
  if (source.required_compat !== undefined) {
    if (
      typeof source.required_compat !== "object" ||
      source.required_compat === null ||
      Array.isArray(source.required_compat)
    ) {
      return null;
    }
    const compat = source.required_compat as Record<string, unknown>;
    const outCompat: Record<string, string> = {};
    for (const [key, inner] of Object.entries(compat)) {
      if (typeof inner !== "string") {
        return null;
      }
      const normalizedKey = key.trim();
      const normalizedValue = inner.trim();
      if (!normalizedKey || !normalizedValue) {
        continue;
      }
      outCompat[normalizedKey] = normalizedValue;
    }
    out.required_compat = outCompat;
  }
  return out;
}

function parseExperiment(
  value: unknown
): { id?: string; force_variant?: "control" | "treatment" } | null {
  if (value === undefined || value === null) {
    return {};
  }
  if (typeof value !== "object" || Array.isArray(value)) {
    return null;
  }
  const source = value as Record<string, unknown>;
  const out: { id?: string; force_variant?: "control" | "treatment" } = {};
  if (source.id !== undefined) {
    if (typeof source.id !== "string") {
      return null;
    }
    const normalized = source.id.trim();
    if (normalized.length > 80) {
      return null;
    }
    if (normalized) {
      out.id = normalized;
    }
  }
  if (source.force_variant !== undefined) {
    if (source.force_variant !== "control" && source.force_variant !== "treatment") {
      return null;
    }
    out.force_variant = source.force_variant;
  }
  return out;
}

export function parseRecommendationsRequest(body: unknown): ParseRecommendationsResult {
  const parsedCore = parseGenerateReportRequest(body);
  if (!parsedCore.ok) {
    return parsedCore;
  }

  const source = body as Record<string, unknown>;
  const asOfTime = parseAsOfTime(source.as_of_time);
  if (!asOfTime) {
    return { ok: false, error: "'as_of_time' must be a valid ISO-8601 datetime string." };
  }

  const topK = parsePositiveInteger(source.top_k, 20, 1, 200);
  if (topK === null) {
    return { ok: false, error: "'top_k' must be an integer between 1 and 200." };
  }
  const retrieveK = parsePositiveInteger(source.retrieve_k, 200, 1, 1000);
  if (retrieveK === null) {
    return { ok: false, error: "'retrieve_k' must be an integer between 1 and 1000." };
  }
  const language = parseLanguage(source.language);
  if (language === null) {
    return { ok: false, error: "'language' must be a short language code string." };
  }
  const candidateIds = parseCandidateIds(source.candidate_ids);
  if (candidateIds === null) {
    return { ok: false, error: "'candidate_ids' must be an array of up to 5000 strings." };
  }
  const policyOverrides = parsePolicyOverrides(source.policy_overrides);
  if (policyOverrides === null) {
    return {
      ok: false,
      error:
        "'policy_overrides' must be an object with optional strict_language/strict_locale booleans and max_items_per_author integer (1-20)."
    };
  }
  const portfolio = parsePortfolio(source.portfolio);
  if (portfolio === null) {
    return {
      ok: false,
      error:
        "'portfolio' must be an object with optional enabled(boolean), weights(reach|conversion|durability numeric >=0), risk_aversion(0-2), and candidate_pool_cap(1-1000)."
    };
  }
  const explainability = parseExplainability(source.explainability);
  if (explainability === null) {
    return {
      ok: false,
      error:
        "'explainability' must be an object with optional enabled/run_counterfactuals booleans and top_features(1-12)/neighbor_k(1-6) integers."
    };
  }
  const graphControls = parseGraphControls(source.graph_controls);
  if (graphControls === null) {
    return {
      ok: false,
      error: "'graph_controls' must be an object with optional enable_graph_branch boolean."
    };
  }
  const trajectoryControls = parseTrajectoryControls(source.trajectory_controls);
  if (trajectoryControls === null) {
    return {
      ok: false,
      error: "'trajectory_controls' must be an object with optional enabled boolean."
    };
  }
  const routing = parseRouting(source.routing);
  if (routing === null) {
    return {
      ok: false,
      error:
        "'routing' must be an object with optional track(pre_publication|post_publication), allow_fallback(boolean), and required_compat record."
    };
  }
  const experiment = parseExperiment(source.experiment);
  if (experiment === null) {
    return {
      ok: false,
      error:
        "'experiment' must be an object with optional id(string<=80) and force_variant(control|treatment)."
    };
  }
  const debug = source.debug === true;

  return {
    ok: true,
    value: {
      ...parsedCore.value,
      as_of_time: asOfTime,
      top_k: topK,
      retrieve_k: retrieveK,
      language,
      candidate_ids: candidateIds,
      policy_overrides: policyOverrides,
      portfolio,
      graph_controls: graphControls,
      trajectory_controls: trajectoryControls,
      explainability,
      routing,
      experiment,
      debug
    }
  };
}
