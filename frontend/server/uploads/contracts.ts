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

export interface UploadedAssetRecord {
  asset_id: string;
  asset: UploadedVideoAsset;
  signal_hints: CandidateSignalHints;
  analysis: UploadedAssetAnalysis;
  analysis_provider?: string;
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
}

export interface AssetAnalysisProvider {
  readonly providerId: string;
  analyzeAsset(asset: UploadedVideoAsset): Promise<AssetAnalysisResult>;
}
