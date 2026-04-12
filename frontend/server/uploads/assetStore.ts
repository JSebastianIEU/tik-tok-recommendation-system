import { createHash } from "node:crypto";
import { promises as fs } from "node:fs";
import path from "node:path";

import type { CandidateSignalHints } from "../contracts/query";
import type {
  AssetAnalysisProvider,
  UploadedAssetRecord,
  UploadedVideoAnalysisPayload,
  UploadedVideoAsset
} from "./contracts";

const MIME_EXTENSION_MAP: Record<string, string> = {
  "video/mp4": ".mp4",
  "video/quicktime": ".mov",
  "video/x-m4v": ".m4v",
  "video/webm": ".webm",
  "video/x-matroska": ".mkv",
  "video/avi": ".avi",
  "video/x-msvideo": ".avi"
};

interface IngestUploadedVideoParams {
  fileBuffer: Buffer;
  fileName: string;
  mimeType: string;
  uploadsDir: string;
  analysisProvider: AssetAnalysisProvider;
}

function normalizeFileName(value: string): string {
  const trimmed = value.trim();
  const baseName = path.basename(trimmed || "upload");
  const sanitized = baseName.replace(/[^a-zA-Z0-9._-]+/g, "_").replace(/_+/g, "_");
  return sanitized.slice(0, 120) || "upload";
}

function inferExtension(fileName: string, mimeType: string): string {
  const fileNameExtension = path.extname(fileName).toLowerCase();
  if (fileNameExtension) {
    return fileNameExtension.slice(0, 12);
  }
  return MIME_EXTENSION_MAP[mimeType.toLowerCase()] ?? ".bin";
}

function buildAssetId(fileBuffer: Buffer): string {
  const checksum = createHash("sha256").update(fileBuffer).digest("hex");
  return `upl_${checksum.slice(0, 24)}`;
}

function buildChecksum(fileBuffer: Buffer): string {
  return createHash("sha256").update(fileBuffer).digest("hex");
}

function metadataPathFor(uploadsDir: string, assetId: string): string {
  return path.join(uploadsDir, `${assetId}.json`);
}

function toAnalysisPayload(record: UploadedAssetRecord): UploadedVideoAnalysisPayload {
  const { asset, analysis, signal_hints, analysis_provider } = record;
  const { storage_path: _storagePath, ...publicAsset } = asset;
  const durationSeconds =
    record.duration_seconds ?? publicAsset.duration_seconds;
  return {
    asset_id: record.asset_id,
    summary: analysis.summary,
    keyTopics: [...analysis.keyTopics],
    suggestedEdits: [...analysis.suggestedEdits],
    metrics: { ...analysis.metrics },
    signal_hints: { ...signal_hints },
    asset: publicAsset,
    analysis_provider,
    transcript: record.transcript,
    ocr_text: record.ocr_text,
    video_caption: record.video_caption,
    detected_language: record.detected_language,
    visual_features: record.visual_features,
    timeline: record.timeline,
    duration_seconds: durationSeconds
  };
}

export function mergeCandidateSignalHints(
  stored: CandidateSignalHints | undefined,
  provided: CandidateSignalHints | undefined
): CandidateSignalHints | undefined {
  const merged: CandidateSignalHints = {};
  const mutableMerged = merged as Record<
    keyof CandidateSignalHints,
    CandidateSignalHints[keyof CandidateSignalHints] | undefined
  >;

  for (const source of [stored, provided]) {
    if (!source) {
      continue;
    }
    for (const [key, value] of Object.entries(source)) {
      if (value === undefined || value === null) {
        continue;
      }
      if (typeof value === "string" && !value.trim()) {
        continue;
      }
      const typedKey = key as keyof CandidateSignalHints;
      mutableMerged[typedKey] = value as CandidateSignalHints[keyof CandidateSignalHints];
    }
  }

  return Object.keys(merged).length > 0 ? merged : undefined;
}

export async function readUploadedAssetRecord(
  uploadsDir: string,
  assetId: string
): Promise<UploadedAssetRecord | null> {
  try {
    const raw = await fs.readFile(metadataPathFor(uploadsDir, assetId), "utf8");
    return JSON.parse(raw) as UploadedAssetRecord;
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") {
      return null;
    }
    throw error;
  }
}

export async function ingestUploadedVideo(
  params: IngestUploadedVideoParams
): Promise<UploadedVideoAnalysisPayload> {
  const uploadsDir = path.resolve(params.uploadsDir);
  await fs.mkdir(uploadsDir, { recursive: true });

  const normalizedFileName = normalizeFileName(params.fileName);
  const checksum = buildChecksum(params.fileBuffer);
  const assetId = buildAssetId(params.fileBuffer);
  const existing = await readUploadedAssetRecord(uploadsDir, assetId);
  if (existing) {
    return toAnalysisPayload(existing);
  }

  const fileExtension = inferExtension(normalizedFileName, params.mimeType);
  const storedPath = path.join(uploadsDir, `${assetId}${fileExtension}`);
  await fs.writeFile(storedPath, params.fileBuffer);

  const asset: UploadedVideoAsset = {
    asset_id: assetId,
    checksum_sha256: checksum,
    file_name: normalizedFileName,
    mime_type: params.mimeType,
    size_bytes: params.fileBuffer.byteLength,
    stored_at: new Date().toISOString(),
    has_audio: false,
    has_video: true,
    orientation: "unknown",
    storage_path: storedPath
  };

  const analyzed = await params.analysisProvider.analyzeAsset(asset);
  const record: UploadedAssetRecord = {
    asset_id: assetId,
    asset: {
      ...asset,
      ...analyzed.asset_updates
    },
    signal_hints: analyzed.signal_hints,
    analysis: analyzed.analysis,
    analysis_provider: params.analysisProvider.providerId,
    transcript: analyzed.transcript,
    ocr_text: analyzed.ocr_text,
    video_caption: analyzed.video_caption,
    detected_language: analyzed.detected_language,
    visual_features: analyzed.visual_features,
    timeline: analyzed.timeline,
    duration_seconds: analyzed.duration_seconds
  };
  await fs.writeFile(metadataPathFor(uploadsDir, assetId), `${JSON.stringify(record, null, 2)}\n`, "utf8");

  return toAnalysisPayload(record);
}

export type {
  AssetAnalysisProvider,
  UploadedAssetRecord,
  UploadedVideoAnalysisPayload,
  UploadedVideoAsset
} from "./contracts";
