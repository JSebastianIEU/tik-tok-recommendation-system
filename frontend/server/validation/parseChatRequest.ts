import type { ReportOutput } from "../../src/features/report/types";
import { validateReportOutput } from "./validateReportOutput";

export interface ChatHistoryMessage {
  role: "assistant" | "user";
  content: string;
  timestamp?: string;
}

export interface ParsedChatRequest {
  question: string;
  report: ReportOutput | null;
  videoAnalysis: Record<string, unknown> | null;
  history: ChatHistoryMessage[];
  objectiveEffective?: string;
  metricFocus?: string;
}

export type ParseChatRequestResult =
  | { ok: true; value: ParsedChatRequest }
  | { ok: false; error: string };

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function normalizeText(value: unknown, maxLength: number): string {
  if (typeof value !== "string") {
    return "";
  }
  return value.trim().slice(0, maxLength);
}

function parseHistory(value: unknown): ChatHistoryMessage[] | null {
  if (value === undefined || value === null) {
    return [];
  }
  if (!Array.isArray(value)) {
    return null;
  }

  const parsed: ChatHistoryMessage[] = [];
  for (const item of value) {
    if (!isObject(item)) {
      continue;
    }
    const role = item.role === "assistant" ? "assistant" : item.role === "user" ? "user" : null;
    if (!role) {
      continue;
    }

    const content = normalizeText(item.content, 2000);
    if (!content) {
      continue;
    }

    const timestamp = normalizeText(item.timestamp, 64);
    parsed.push({
      role,
      content,
      ...(timestamp ? { timestamp } : {}),
    });

    if (parsed.length >= 40) {
      break;
    }
  }
  return parsed;
}

export function parseChatRequest(body: unknown): ParseChatRequestResult {
  if (!isObject(body)) {
    return { ok: false, error: "Request body must be a JSON object." };
  }

  const question = normalizeText(body.question, 2000);
  if (!question) {
    return { ok: false, error: "A question is required." };
  }

  const history = parseHistory(body.history);
  if (history === null) {
    return { ok: false, error: "'history' must be an array of chat messages." };
  }

  const objectiveEffective = normalizeText(body.objective_effective, 64).toLowerCase() || undefined;
  const metricFocus = normalizeText(body.metric_focus, 64).toLowerCase() || undefined;

  const report = validateReportOutput(body.report)
    ? (body.report as ReportOutput)
    : null;

  const videoAnalysis = isObject(body.videoAnalysis)
    ? body.videoAnalysis
    : null;

  return {
    ok: true,
    value: {
      question,
      report,
      videoAnalysis,
      history,
      objectiveEffective,
      metricFocus,
    },
  };
}
