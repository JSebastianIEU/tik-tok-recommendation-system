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
}

export interface VideoAnalysisResult {
  asset_id: string;
  summary: string;
  keyTopics: string[];
  suggestedEdits: string[];
  metrics: VideoMetrics;
  signal_hints?: SignalHintsPayload;
  asset?: UploadedVideoAsset;
  analysis_provider?: string;
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
}
