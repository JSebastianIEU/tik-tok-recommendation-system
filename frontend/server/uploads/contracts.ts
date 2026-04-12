import type { CandidateSignalHints } from "../contracts/query";

export interface UploadedVideoAsset {
  asset_id: string;
  checksum_sha256: string;
  file_name: string;
  mime_type: string;
  size_bytes: number;
  stored_at: string;
  duration_seconds?: number;
  width?: number;
  height?: number;
  fps?: number;
  has_audio: boolean;
  has_video: boolean;
  orientation: "portrait" | "landscape" | "square" | "unknown";
  storage_path: string;
}

export interface UploadedAssetAnalysis {
  summary: string;
  keyTopics: string[];
  suggestedEdits: string[];
  metrics: {
    retention: number;
    hookStrength: number;
    clarity: number;
  };
}

/** Optional VLM-style enrichment; aligns with frontend `VideoAnalysisResult`. */
export interface VisualFeaturesSummary {
  dominant_colors?: string[];
  avg_brightness?: number;
  avg_saturation?: number;
  avg_contrast?: number;
  face_count?: number;
  avg_face_area_ratio?: number;
  aspect_ratio?: string;
  resolution?: string;
  blur_score?: number;
  hook_motion_score?: number;
}

export interface FrameTimelineEntryPayload {
  timestamp_sec: number;
  thumbnail_b64: string;
  ocr_text: string;
  face_count: number;
  motion_score: number;
  is_scene_change: boolean;
  relevance_score: number;
}

export interface UploadedAssetRecord {
  asset_id: string;
  asset: UploadedVideoAsset;
  signal_hints: CandidateSignalHints;
  analysis: UploadedAssetAnalysis;
  analysis_provider?: string;
  transcript?: string;
  ocr_text?: string;
  video_caption?: string;
  detected_language?: string;
  visual_features?: VisualFeaturesSummary;
  timeline?: FrameTimelineEntryPayload[];
  /** Top-level duration when provided by analyzer; else use `asset.duration_seconds`. */
  duration_seconds?: number;
}

export interface UploadedVideoAnalysisPayload {
  asset_id: string;
  summary: string;
  keyTopics: string[];
  suggestedEdits: string[];
  metrics: {
    retention: number;
    hookStrength: number;
    clarity: number;
  };
  signal_hints: CandidateSignalHints;
  asset: Omit<UploadedVideoAsset, "storage_path">;
  analysis_provider?: string;
  transcript?: string;
  ocr_text?: string;
  video_caption?: string;
  detected_language?: string;
  visual_features?: VisualFeaturesSummary;
  timeline?: FrameTimelineEntryPayload[];
  duration_seconds?: number;
}

export interface AssetAnalysisResult {
  asset_updates: Pick<
    UploadedVideoAsset,
    | "duration_seconds"
    | "width"
    | "height"
    | "fps"
    | "has_audio"
    | "has_video"
    | "orientation"
  >;
  signal_hints: CandidateSignalHints;
  analysis: UploadedAssetAnalysis;
  transcript?: string;
  ocr_text?: string;
  video_caption?: string;
  detected_language?: string;
  visual_features?: VisualFeaturesSummary;
  timeline?: FrameTimelineEntryPayload[];
  duration_seconds?: number;
}

export interface AssetAnalysisProvider {
  readonly providerId: string;
  analyzeAsset(asset: UploadedVideoAsset): Promise<AssetAnalysisResult>;
}
