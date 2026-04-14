import type { ReportOutput } from "../../features/report/types";
import type { SignalHintsPayload, UploadFormValues } from "../contracts/models";
import { generateMockReport } from "../mock/mockReportApi";
import { buildApiUrl, MOCK_ONLY_MODE } from "./runtimeConfig";

const REPORT_API_URL = buildApiUrl("/generate-report");

export interface GenerateReportPayload {
  asset_id?: string;
  seed_video_id?: string;
  mentions: string[];
  hashtags: string[];
  description: string;
  objective?: UploadFormValues["objective"];
  audience?: string;
  content_type?: UploadFormValues["content_type"];
  primary_cta?: UploadFormValues["primary_cta"];
  locale?: string;
  signal_hints?: SignalHintsPayload;
}

interface GenerateReportResponse {
  report: ReportOutput;
  suggested_hashtags?: Array<{
    hashtag: string;
    score: number;
    frequency: number;
    avg_engagement: number;
  }>;
}

export interface GenerateReportResult {
  report: ReportOutput;
  suggested_hashtags: GenerateReportResponse["suggested_hashtags"];
}

function fallbackToMock(payload: GenerateReportPayload): GenerateReportResult {
  return {
    report: generateMockReport({
      seedVideoId: payload.asset_id ?? payload.seed_video_id ?? "uploaded-asset",
      mentions: payload.mentions,
      hashtags: payload.hashtags,
      description: payload.description
    }),
    suggested_hashtags: []
  };
}

export async function generateReport(
  payload: GenerateReportPayload
): Promise<GenerateReportResult> {
  if (MOCK_ONLY_MODE) {
    return fallbackToMock(payload);
  }

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
  return {
    report: parsed.report,
    suggested_hashtags: parsed.suggested_hashtags ?? []
  };
}
