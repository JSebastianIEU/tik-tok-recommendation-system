import type { IVideoAnalysisService } from "../contracts/IVideoAnalysisService";
import type {
  AnalyzeVideoRequest,
  UploadPhase,
  VideoAnalysisResult
} from "../contracts/models";
import { MOCK_ANALYSIS_RESULT } from "./fixtures";

function randomDelay(minMs: number, maxMs: number): number {
  return Math.floor(Math.random() * (maxMs - minMs + 1)) + minMs;
}

function wait(ms: number): Promise<void> {
  return new Promise((resolve) => {
    window.setTimeout(resolve, ms);
  });
}

function cloneAnalysisResult(result: VideoAnalysisResult): VideoAnalysisResult {
  return {
    asset_id: result.asset_id,
    analysis_provider: result.analysis_provider,
    summary: result.summary,
    keyTopics: [...result.keyTopics],
    suggestedEdits: [...result.suggestedEdits],
    metrics: { ...result.metrics },
    signal_hints: result.signal_hints ? { ...result.signal_hints } : undefined,
    asset: result.asset ? { ...result.asset } : undefined
  };
}

export class MockVideoAnalysisService implements IVideoAnalysisService {
  public async analyzeVideo(
    _request: AnalyzeVideoRequest,
    onPhaseChange?: (phase: UploadPhase) => void
  ): Promise<VideoAnalysisResult> {
    onPhaseChange?.("uploading");
    await wait(randomDelay(800, 1200));

    onPhaseChange?.("analyzing");
    await wait(randomDelay(1200, 2000));

    return cloneAnalysisResult(MOCK_ANALYSIS_RESULT);
  }
}
