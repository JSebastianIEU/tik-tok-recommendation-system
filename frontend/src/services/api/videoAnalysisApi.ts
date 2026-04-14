import type { IVideoAnalysisService } from "../contracts/IVideoAnalysisService";
import type {
  AnalyzeVideoRequest,
  UploadPhase,
  VideoAnalysisResult
} from "../contracts/models";
import { MockVideoAnalysisService } from "../mock/mockUploadApi";
import { buildApiUrl, MOCK_ONLY_MODE } from "./runtimeConfig";

export interface FrameTimelineEntry {
  timestamp_sec: number;
  relevance_score: number;
  is_scene_change: boolean;
  caption?: string;
  thumbnail_b64?: string;
}

const UPLOAD_ANALYSIS_API_URL = buildApiUrl("/upload-video");

// Cloud Run URL for direct video upload (bypasses Vercel 4.5MB body limit)
const RECOMMENDER_URL = (
  import.meta.env.VITE_RECOMMENDER_URL ?? ""
).trim().replace(/\/+$/, "");

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

/**
 * Upload video directly to Cloud Run /v1/video/analyze for real analysis
 * (frame extraction, transcription, scene detection, etc.)
 */
async function analyzeViaCloudRun(
  file: File
): Promise<VideoAnalysisResult | null> {
  if (!RECOMMENDER_URL) return null;

  try {
    const formData = new FormData();
    formData.append("file", file, file.name);

    const resp = await fetch(`${RECOMMENDER_URL}/v1/video/analyze`, {
      method: "POST",
      body: formData,
      signal: AbortSignal.timeout(120_000),
    });

    if (!resp.ok) return null;

    const data = await resp.json() as Record<string, unknown>;

    // Map Python service response to VideoAnalysisResult
    const assetId = (data.asset_id as string) || crypto.randomUUID();
    const timeline = Array.isArray(data.timeline)
      ? (data.timeline as Array<Record<string, unknown>>).map((f) => ({
          timestamp_sec: Number(f.timestamp_sec) || 0,
          thumbnail_b64: (f.thumbnail_b64 as string) || "",
          ocr_text: (f.ocr_text as string) || "",
          face_count: Number(f.face_count) || 0,
          motion_score: Number(f.motion_score) || 0,
          is_scene_change: Boolean(f.is_scene_change),
          relevance_score: Number(f.relevance_score) || 0,
        }))
      : undefined;

    const vf = data.visual_features as Record<string, unknown> | undefined;

    return {
      asset_id: assetId,
      summary: (data.summary as string) || "",
      keyTopics: Array.isArray(data.key_topics) ? data.key_topics as string[] : [],
      suggestedEdits: [],
      metrics: {
        retention: Number((data as Record<string, unknown>).retention) || 0,
        hookStrength: Number((data as Record<string, unknown>).hook_strength) || 0,
        clarity: Number((data as Record<string, unknown>).clarity) || 0,
      },
      analysis_provider: "cloud-run",
      transcript: (data.transcript as string) || undefined,
      ocr_text: (data.ocr_text as string) || undefined,
      video_caption: (data.video_caption as string) || (data.caption as string) || undefined,
      detected_language: (data.detected_language as string) || undefined,
      visual_features: vf ? {
        avg_brightness: Number(vf.avg_brightness) || undefined,
        avg_saturation: Number(vf.avg_saturation) || undefined,
        avg_contrast: Number(vf.avg_contrast) || undefined,
        face_count: Number(vf.face_count) || undefined,
        aspect_ratio: (vf.aspect_ratio as string) || undefined,
        resolution: (vf.resolution as string) || undefined,
        blur_score: Number(vf.blur_score) || undefined,
      } : undefined,
      timeline,
      duration_seconds: Number(data.duration_seconds) || undefined,
      asset: {
        asset_id: assetId,
        original_filename: file.name,
        content_type: file.type || "video/mp4",
        size_bytes: file.size,
      },
    };
  } catch (err) {
    console.warn("[analyzeViaCloudRun] failed:", err);
    return null;
  }
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

    // Try Cloud Run first (real video analysis with frame extraction etc.)
    const cloudResult = await analyzeViaCloudRun(request.file);
    if (cloudResult) {
      onPhaseChange?.("analyzing");
      return cloudResult;
    }

    // Fallback: Vercel serverless (metadata-only, caption from DeepSeek)
    const fileName = request.file.name;
    const fileType = request.file.type || "application/octet-stream";
    let response: Response;

    try {
      response = await fetch(UPLOAD_ANALYSIS_API_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "x-file-name": fileName,
          "x-file-type": fileType
        },
        body: JSON.stringify({
          file_name: fileName,
          file_type: fileType,
          file_size: request.file.size
        })
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
