import type { IVideoAnalysisService } from "../contracts/IVideoAnalysisService";
import type {
  AnalyzeVideoRequest,
  UploadPhase,
  VideoAnalysisResult
} from "../contracts/models";
import { MockVideoAnalysisService } from "../mock/mockUploadApi";
import { buildApiUrl, MOCK_ONLY_MODE } from "./runtimeConfig";

const UPLOAD_ANALYSIS_API_URL = buildApiUrl("/upload-video");

interface UploadApiErrorPayload {
  error?: string;
}

async function parseErrorMessage(response: Response): Promise<string> {
  try {
    const payload = (await response.json()) as UploadApiErrorPayload;
    if (typeof payload.error === "string" && payload.error.trim()) {
      return payload.error.trim();
    }
  } catch {
    // Ignore response parsing failures and fall back to a generic message.
  }

  return "Video upload analysis failed.";
}

export class ApiVideoAnalysisService implements IVideoAnalysisService {
  private readonly mockService = new MockVideoAnalysisService();

  public async analyzeVideo(
    request: AnalyzeVideoRequest,
    onPhaseChange?: (phase: UploadPhase) => void
  ): Promise<VideoAnalysisResult> {
    if (MOCK_ONLY_MODE) {
      return this.mockService.analyzeVideo(request, onPhaseChange);
    }

    onPhaseChange?.("uploading");
    const fileBytes = await request.file.arrayBuffer();
    let response: Response;

    try {
      response = await fetch(UPLOAD_ANALYSIS_API_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/octet-stream",
          "x-file-name": request.file.name,
          "x-file-type": request.file.type || "application/octet-stream"
        },
        body: fileBytes
      });
    } catch {
      throw new Error("Could not reach the upload analysis service.");
    }

    onPhaseChange?.("analyzing");

    if (!response.ok) {
      throw new Error(await parseErrorMessage(response));
    }

    return (await response.json()) as VideoAnalysisResult;
  }
}
