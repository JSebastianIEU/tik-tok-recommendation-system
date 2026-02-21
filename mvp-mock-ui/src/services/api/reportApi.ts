import type { ReportOutput } from "../../features/report/types";
import { generateMockReport } from "../mock/mockReportApi";
import { buildApiUrl } from "./runtimeConfig";

const REPORT_API_URL = buildApiUrl("/api/generate-report");

export interface GenerateReportPayload {
  seed_video_id: string;
  mentions: string[];
  hashtags: string[];
  description: string;
}

interface GenerateReportResponse {
  report: ReportOutput;
}

function fallbackToMock(payload: GenerateReportPayload): ReportOutput {
  return generateMockReport({
    seedVideoId: payload.seed_video_id,
    mentions: payload.mentions,
    hashtags: payload.hashtags,
    description: payload.description
  });
}

export async function generateReport(
  payload: GenerateReportPayload
): Promise<ReportOutput> {
  let response: Response;

  try {
    response = await fetch(REPORT_API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(payload)
    });
  } catch {
    return fallbackToMock(payload);
  }

  if (!response.ok) {
    return fallbackToMock(payload);
  }

  const parsed = (await response.json()) as GenerateReportResponse;
  return parsed.report;
}
