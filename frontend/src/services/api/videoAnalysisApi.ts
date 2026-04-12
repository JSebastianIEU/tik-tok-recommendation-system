import type { IVideoAnalysisService } from "../contracts/IVideoAnalysisService";
import type {
  AnalyzeVideoRequest,
  UploadPhase,
  VideoAnalysisResult
} from "../contracts/models";
import { MockVideoAnalysisService } from "../mock/mockUploadApi";
import { buildApiUrl } from "./runtimeConfig";

const mockFallback = new MockVideoAnalysisService();

export class ApiVideoAnalysisService implements IVideoAnalysisService {
  public async analyzeVideo(
    request: AnalyzeVideoRequest,
    onPhaseChange?: (phase: UploadPhase) => void
  ): Promise<VideoAnalysisResult> {
    onPhaseChange?.("uploading");

    try {
      const formData = new FormData();
      formData.append("file", request.file);

      const response = await fetch(buildApiUrl("/video/analyze"), {
        method: "POST",
        body: formData
      });

      if (!response.ok) {
        throw new Error(`Video analysis failed: ${response.status}`);
      }

      onPhaseChange?.("analyzing");

      const payload = await response.json();

      return {
        summary: "",
        keyTopics: payload.keywords ?? [],
        suggestedEdits: [],
        metrics: {
          retention: 0,
          hookStrength: 0,
          clarity: 0
        },
        signal_hints: payload.signal_hints ?? undefined,
        transcript: payload.transcript ?? "",
        ocr_text: payload.ocr_text ?? "",
        video_caption: payload.video_caption ?? "",
        detected_language: payload.detected_language ?? "",
        visual_features: payload.visual_features ?? undefined,
        timeline: payload.timeline ?? [],
        duration_seconds: payload.duration_seconds ?? 0
      };
    } catch (error) {
      console.warn("Real video analysis failed, falling back to mock:", error);
      return mockFallback.analyzeVideo(request, onPhaseChange);
    }
  }
}
