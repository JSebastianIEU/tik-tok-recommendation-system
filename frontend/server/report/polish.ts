import type { ReportOutput } from "../../src/features/report/types";

export interface NarrativePolishOutput {
  executive_summary?: {
    meaning_points?: string[];
    summary_text?: string;
  };
  relevant_comments?: Array<{
    id: string;
    relevance_note: string;
  }>;
  recommendations?: Array<{
    id: string;
    evidence: string;
  }>;
}

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isStringArray(value: unknown): value is string[] {
  return Array.isArray(value) && value.every((item) => typeof item === "string");
}

export function buildReportPrompt(params: {
  report: ReportOutput;
}): string {
  return JSON.stringify(
    {
      role: "Senior content strategist and growth analyst",
      objective:
        "Rewrite only the allowed narrative fields in a grounded, concise, evidence-led way.",
      constraints: [
        "Return valid JSON only.",
        "Do not add sections or fields that are not requested.",
        "Do not change metrics, priorities, effort, IDs, or evidence references.",
        "Do not invent claims, examples, or percentages.",
        "Do not mention internal IDs in user-facing text."
      ],
      report_context: params.report,
      required_schema: {
        executive_summary: {
          meaning_points: ["string"],
          summary_text: "string"
        },
        relevant_comments: [
          {
            id: "string",
            relevance_note: "string"
          }
        ],
        recommendations: [
          {
            id: "string",
            evidence: "string"
          }
        ]
      },
      output_rules: {
        strict_json_only: true,
        no_markdown: true,
        no_text_outside_json: true
      }
    },
    null,
    2
  );
}

export function validateNarrativePolish(value: unknown): value is NarrativePolishOutput {
  if (!isObject(value)) {
    return false;
  }
  if (value.executive_summary !== undefined) {
    if (!isObject(value.executive_summary)) {
      return false;
    }
    if (
      value.executive_summary.meaning_points !== undefined &&
      !isStringArray(value.executive_summary.meaning_points)
    ) {
      return false;
    }
    if (
      value.executive_summary.summary_text !== undefined &&
      typeof value.executive_summary.summary_text !== "string"
    ) {
      return false;
    }
  }
  if (value.relevant_comments !== undefined) {
    if (!Array.isArray(value.relevant_comments)) {
      return false;
    }
    if (
      !value.relevant_comments.every(
        (item) => isObject(item) && typeof item.id === "string" && typeof item.relevance_note === "string"
      )
    ) {
      return false;
    }
  }
  if (value.recommendations !== undefined) {
    if (!Array.isArray(value.recommendations)) {
      return false;
    }
    if (
      !value.recommendations.every(
        (item) => isObject(item) && typeof item.id === "string" && typeof item.evidence === "string"
      )
    ) {
      return false;
    }
  }
  return true;
}

export function applyNarrativePolish(
  report: ReportOutput,
  polish: NarrativePolishOutput
): ReportOutput {
  const commentNotes = new Map(
    (polish.relevant_comments ?? []).map((item) => [item.id, item.relevance_note])
  );
  const recommendationEvidence = new Map(
    (polish.recommendations ?? []).map((item) => [item.id, item.evidence])
  );
  return {
    ...report,
    executive_summary: {
      ...report.executive_summary,
      meaning_points:
        polish.executive_summary?.meaning_points?.slice(0, 6) ??
        report.executive_summary.meaning_points,
      summary_text:
        polish.executive_summary?.summary_text ?? report.executive_summary.summary_text
    },
    relevant_comments: {
      ...report.relevant_comments,
      items: report.relevant_comments.items.map((item) => ({
        ...item,
        relevance_note: commentNotes.get(item.id) ?? item.relevance_note
      }))
    },
    recommendations: {
      ...report.recommendations,
      items: report.recommendations.items.map((item) => ({
        ...item,
        evidence: recommendationEvidence.get(item.id) ?? item.evidence
      }))
    }
  };
}
