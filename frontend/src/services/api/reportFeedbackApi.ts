import { buildApiUrl, MOCK_ONLY_MODE } from "./runtimeConfig";
import { supabase } from "../supabase";

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
  user_id?: string;
}

export async function sendReportFeedback(payload: ReportFeedbackPayload): Promise<void> {
  if (MOCK_ONLY_MODE || !payload.request_id.trim()) {
    return;
  }

  try {
    const { data: { session } } = await supabase.auth.getSession();
    const body = { ...payload, user_id: session?.user?.id ?? payload.user_id };
    await fetch(REPORT_FEEDBACK_API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(body)
    });
  } catch (error) {
    console.error("report_feedback_failed", error);
  }
}
