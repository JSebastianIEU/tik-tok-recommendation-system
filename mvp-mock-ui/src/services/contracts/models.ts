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
}

export interface AnalyzeVideoRequest {
  file: File;
  mentions: string[];
  hashtags: string[];
  description: string;
}

export interface VideoMetrics {
  retention: number;
  hookStrength: number;
  clarity: number;
}

export interface VideoAnalysisResult {
  summary: string;
  keyTopics: string[];
  suggestedEdits: string[];
  metrics: VideoMetrics;
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
