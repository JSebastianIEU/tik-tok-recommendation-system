import type { ReportOutput } from "../../features/report/types";

export type UploadPhase =
  | "idle"
  | "validating"
  | "processing"
  | "uploading"
  | "analyzing"
  | "done"
  | "error";

export interface UploadFormValues {
  mentions: string[];
  hashtags: string[];
  description: string;
  objective: "reach" | "engagement" | "conversion" | "community";
  audience: string;
  content_type: "tutorial" | "story" | "reaction" | "showcase" | "opinion";
  primary_cta: "follow" | "comment" | "save" | "share" | "link_click" | "none";
  locale: string;
}

export interface SignalHintsPayload {
  duration_seconds?: number;
  transcript_text?: string;
  ocr_text?: string;
  video_caption?: string;
  estimated_scene_cuts?: number;
  fps?: number;
  visual_motion_score?: number;
  speech_seconds?: number;
  music_seconds?: number;
  tempo_bpm?: number;
  audio_energy?: number;
  loudness_lufs?: number;
}

export interface AnalyzeVideoRequest {
  file: File;
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

export interface VideoMetrics {
  retention: number;
  hookStrength: number;
  clarity: number;
}

export interface VisualFeaturesResult {
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

export interface FrameTimelineEntry {
  timestamp_sec: number;
  thumbnail_b64: string;
  ocr_text: string;
  face_count: number;
  motion_score: number;
  is_scene_change: boolean;
  relevance_score: number;
}

export interface VideoAnalysisResult {
  asset_id: string;
  summary: string;
  keyTopics: string[];
  suggestedEdits: string[];
  metrics: VideoMetrics;
  signal_hints?: SignalHintsPayload;
  transcript?: string;
  ocr_text?: string;
  video_caption?: string;
  detected_language?: string;
  visual_features?: VisualFeaturesResult;
  timeline?: FrameTimelineEntry[];
  duration_seconds?: number;
}

export type ChatRole = "assistant" | "user";

export interface ChatMessage {
  id: string;
  role: ChatRole;
  content: string;
  timestamp: string;
}

export interface ChatRequest {
  question: string;
  report: ReportOutput;
  history: ChatMessage[];
  videoAnalysis?: VideoAnalysisResult | null;
}
