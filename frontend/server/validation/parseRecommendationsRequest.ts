import {
  parseGenerateReportRequest,
  type ParsedGenerateReportRequest
} from "./parseGenerateReportRequest";

export interface ParsedRecommendationsRequest extends ParsedGenerateReportRequest {
  as_of_time: string;
  top_k: number;
  retrieve_k: number;
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
  const debug = source.debug === true;

  return {
    ok: true,
    value: {
      ...parsedCore.value,
      as_of_time: asOfTime,
      top_k: topK,
      retrieve_k: retrieveK,
      debug
    }
  };
}

