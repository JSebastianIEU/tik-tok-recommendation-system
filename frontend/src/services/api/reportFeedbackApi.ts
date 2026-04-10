import { buildApiUrl, MOCK_ONLY_MODE } from "./runtimeConfig";

const REPORT_FEEDBACK_API_URL = buildApiUrl("/report-feedback");

export interface ReportFeedbackPayload {
  request_id: string;
  event_name: string;
  entity_type: "report" | "comparable" | "recommendation" | "explainability" | "chat";
  entity_id?: string;
  section: string;
  rank?: number;
  objective_effective: string;
  experiment_id?: string | null;
  variant?: "control" | "treatment" | null;
  signal_strength: "strong" | "medium" | "weak" | "context";
  label_direction: "positive" | "negative" | "neutral" | "context";
  metadata?: Record<string, unknown>;
}

export async function sendReportFeedback(payload: ReportFeedbackPayload): Promise<void> {
  if (MOCK_ONLY_MODE || !payload.request_id.trim()) {
    return;
  }

  try {
    await fetch(REPORT_FEEDBACK_API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(payload)
    });
  } catch (error) {
    console.error("report_feedback_failed", error);
  }
}
