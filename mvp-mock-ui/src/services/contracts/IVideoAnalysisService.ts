import type {
  AnalyzeVideoRequest,
  UploadPhase,
  VideoAnalysisResult
} from "./models";

export interface IVideoAnalysisService {
  analyzeVideo(
    request: AnalyzeVideoRequest,
    onPhaseChange?: (phase: UploadPhase) => void
  ): Promise<VideoAnalysisResult>;
}
