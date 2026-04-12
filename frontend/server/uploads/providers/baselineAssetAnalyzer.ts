import { execFile } from "node:child_process";
import { promisify } from "node:util";

import type { CandidateSignalHints } from "../../contracts/query";
import type {
  AssetAnalysisProvider,
  AssetAnalysisResult,
  UploadedAssetAnalysis,
  UploadedVideoAsset
} from "../contracts";

const execFileAsync = promisify(execFile);

interface BaselineAssetAnalysisProviderConfig {
  ffprobeBin: string;
}

interface FfprobeStream {
  codec_type?: string;
  width?: number;
  height?: number;
  avg_frame_rate?: string;
  r_frame_rate?: string;
  duration?: string;
  bit_rate?: string;
  sample_rate?: string;
}

interface FfprobeFormat {
  duration?: string;
  bit_rate?: string;
}

interface FfprobePayload {
  format?: FfprobeFormat;
  streams?: FfprobeStream[];
}

interface ProbeSummary {
  duration_seconds?: number;
  width?: number;
  height?: number;
  fps?: number;
  has_audio: boolean;
  has_video: boolean;
  container_bit_rate?: number;
  audio_bit_rate?: number;
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function round(value: number, decimals = 2): number {
  const factor = 10 ** decimals;
  return Math.round(value * factor) / factor;
}

function roundToInt(value: number): number {
  return Math.round(value);
}

function parseNumber(value: string | number | undefined): number | undefined {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value !== "string") {
    return undefined;
  }
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : undefined;
}

function parseFrameRate(value: string | undefined): number | undefined {
  if (!value) {
    return undefined;
  }
  const trimmed = value.trim();
  if (!trimmed || trimmed === "0/0") {
    return undefined;
  }
  if (!trimmed.includes("/")) {
    return parseNumber(trimmed);
  }
  const [numeratorRaw, denominatorRaw] = trimmed.split("/", 2);
  const numerator = Number(numeratorRaw);
  const denominator = Number(denominatorRaw);
  if (!Number.isFinite(numerator) || !Number.isFinite(denominator) || denominator === 0) {
    return undefined;
  }
  return numerator / denominator;
}

function deriveOrientation(
  width: number | undefined,
  height: number | undefined
): UploadedVideoAsset["orientation"] {
  if (!width || !height) {
    return "unknown";
  }
  if (Math.abs(width - height) <= 24) {
    return "square";
  }
  return height > width ? "portrait" : "landscape";
}

function normalizeTopic(value: string): string {
  return value.trim().replace(/\s+/g, " ");
}

function uniqueStrings(values: string[]): string[] {
  const seen = new Set<string>();
  const output: string[] = [];

  for (const value of values) {
    const normalized = normalizeTopic(value);
    if (!normalized) {
      continue;
    }
    const key = normalized.toLowerCase();
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    output.push(normalized);
  }

  return output;
}

function buildPacingLabel(
  sceneCuts: number | undefined,
  durationSeconds: number | undefined
): string {
  if (!sceneCuts || !durationSeconds || durationSeconds <= 0) {
    return "steady pacing";
  }
  const cutsPerTenSeconds = sceneCuts / Math.max(durationSeconds / 10, 1);
  if (cutsPerTenSeconds >= 7) {
    return "fast-cut pacing";
  }
  if (cutsPerTenSeconds >= 4) {
    return "balanced pacing";
  }
  return "slow pacing";
}

function buildResolutionLabel(width: number | undefined, height: number | undefined): string {
  if (!width || !height) {
    return "unknown resolution";
  }
  if (Math.max(width, height) >= 1920) {
    return "high-definition render";
  }
  if (Math.max(width, height) >= 1080) {
    return "full-frame export";
  }
  return "compressed export";
}

function deriveSignalHints(
  asset: UploadedVideoAsset,
  probe: ProbeSummary
): CandidateSignalHints {
  const durationSeconds = asset.duration_seconds;
  const fps = asset.fps;
  const width = asset.width;
  const height = asset.height;
  const pixelRate =
    width && height && fps ? Math.max(width * height * Math.max(fps, 1), 1) : undefined;
  const containerBitRate = probe.container_bit_rate;
  const bitsPerPixelFrame =
    pixelRate && containerBitRate ? containerBitRate / pixelRate : undefined;
  const visualMotionScore =
    bitsPerPixelFrame !== undefined
      ? round(clamp(bitsPerPixelFrame * 14, 0.08, 0.96), 4)
      : undefined;
  const sceneCutEstimate =
    durationSeconds && visualMotionScore !== undefined && fps
      ? roundToInt(
          clamp(
            durationSeconds * (0.18 + visualMotionScore * 0.42 + clamp(fps / 120, 0.08, 0.28)),
            2,
            320
          )
        )
      : undefined;
  const audioEnergy = probe.has_audio
    ? round(
        clamp(
          ((probe.audio_bit_rate ?? containerBitRate ?? 96000) / 192000) * 0.72,
          0.12,
          0.94
        ),
        4
      )
    : undefined;
  const speechRatio =
    durationSeconds && probe.has_audio
      ? clamp(
          0.34 +
            (visualMotionScore !== undefined ? (1 - visualMotionScore) * 0.18 : 0.08) +
            (asset.orientation === "portrait" ? 0.08 : 0),
          0.18,
          0.82
        )
      : undefined;
  const speechSeconds =
    durationSeconds && speechRatio !== undefined
      ? round(durationSeconds * speechRatio, 2)
      : undefined;
  const musicSeconds =
    durationSeconds && speechSeconds !== undefined
      ? round(Math.max(durationSeconds - speechSeconds, 0), 2)
      : undefined;
  const tempoBpm = probe.has_audio
    ? round(
        clamp(
          78 + (visualMotionScore ?? 0.35) * 74 + clamp((fps ?? 24) / 60, 0, 0.35) * 28,
          72,
          176
        ),
        1
      )
    : undefined;
  const loudnessLufs = audioEnergy !== undefined ? round(-28 + audioEnergy * 14, 1) : undefined;

  return {
    duration_seconds: durationSeconds !== undefined ? round(durationSeconds, 2) : undefined,
    fps: fps !== undefined ? round(fps, 3) : undefined,
    estimated_scene_cuts: sceneCutEstimate,
    visual_motion_score: visualMotionScore,
    speech_seconds: speechSeconds,
    music_seconds: musicSeconds,
    tempo_bpm: tempoBpm,
    audio_energy: audioEnergy,
    loudness_lufs: loudnessLufs
  };
}

function buildAnalysis(
  asset: UploadedVideoAsset,
  signalHints: CandidateSignalHints
): UploadedAssetAnalysis {
  const durationLabel =
    asset.duration_seconds !== undefined ? `${round(asset.duration_seconds, 1)}s` : "unknown length";
  const resolutionLabel =
    asset.width && asset.height ? `${asset.width}x${asset.height}` : "unknown resolution";
  const pacingLabel = buildPacingLabel(
    signalHints.estimated_scene_cuts,
    signalHints.duration_seconds
  );
  const resolutionTheme = buildResolutionLabel(asset.width, asset.height);
  const audioTheme = asset.has_audio ? "audio-backed delivery" : "silent-first playback";
  const motionTheme =
    signalHints.visual_motion_score !== undefined && signalHints.visual_motion_score >= 0.55
      ? "high visual motion"
      : signalHints.visual_motion_score !== undefined &&
          signalHints.visual_motion_score >= 0.3
        ? "moderate visual motion"
        : "low visual motion";
  const hookStrength = roundToInt(
    clamp(
      48 +
        (signalHints.visual_motion_score ?? 0.28) * 30 +
        (asset.orientation === "portrait" ? 8 : asset.orientation === "square" ? 5 : 2) +
        ((asset.duration_seconds ?? 45) <= 30 ? 7 : (asset.duration_seconds ?? 45) <= 60 ? 2 : -6),
      32,
      97
    )
  );
  const clarity = roundToInt(
    clamp(
      54 +
        (asset.has_audio ? 8 : -4) +
        ((asset.fps ?? 24) >= 24 ? 7 : 1) +
        (asset.width && asset.height ? clamp(Math.max(asset.width, asset.height) / 240, 0, 10) : 0),
      38,
      98
    )
  );
  const pacingBonus =
    signalHints.duration_seconds && signalHints.estimated_scene_cuts
      ? clamp(
          signalHints.estimated_scene_cuts / Math.max(signalHints.duration_seconds / 10, 1),
          1,
          8
        ) * 1.8
      : 4;
  const retention = roundToInt(
    clamp(
      42 +
        hookStrength * 0.34 +
        clarity * 0.22 +
        pacingBonus -
        Math.max((asset.duration_seconds ?? 0) - 75, 0) * 0.2,
      35,
      98
    )
  );
  const keyTopics = uniqueStrings([
    `${asset.orientation} framing`,
    pacingLabel,
    audioTheme,
    resolutionTheme,
    motionTheme
  ]).slice(0, 4);
  const suggestedEdits = uniqueStrings([
    hookStrength < 68
      ? "Tighten the first two seconds so the value proposition lands before the first pause."
      : "Keep the current opening momentum and reinforce it with a clearer first-frame title.",
    asset.has_audio
      ? "Balance captions with the existing audio track so silent viewers still catch the core message."
      : "Add voiceover or stronger captions so the message survives silent autoplay.",
    pacingLabel === "slow pacing"
      ? "Increase cut frequency or add pattern interrupts to keep scroll-stopping energy higher."
      : "Preserve the current pacing, but add one deliberate payoff beat near the end for a stronger finish.",
    asset.fps !== undefined && asset.fps < 24
      ? "Export at a steadier frame cadence to make motion read more cleanly on mobile."
      : "Use one or two cleaner hero shots so the strongest frames anchor the rest of the edit."
  ]).slice(0, 4);

  return {
    summary: `Uploaded asset ${asset.file_name} looks like a ${asset.orientation} ${durationLabel} clip at ${resolutionLabel}, ${asset.has_audio ? "with audio" : "without a detected audio track"}, and currently reads as ${pacingLabel} with ${motionTheme}.`,
    keyTopics,
    suggestedEdits,
    metrics: {
      retention,
      hookStrength,
      clarity
    }
  };
}

async function runFfprobe(filePath: string, ffprobeBin: string): Promise<ProbeSummary> {
  try {
    const { stdout } = await execFileAsync(ffprobeBin, [
      "-v",
      "error",
      "-print_format",
      "json",
      "-show_streams",
      "-show_format",
      filePath
    ]);
    const payload = JSON.parse(stdout) as FfprobePayload;
    const streams = payload.streams ?? [];
    const videoStream = streams.find((stream) => stream.codec_type === "video");
    const audioStream = streams.find((stream) => stream.codec_type === "audio");
    const durationSeconds =
      parseNumber(payload.format?.duration) ?? parseNumber(videoStream?.duration);
    const fps =
      parseFrameRate(videoStream?.avg_frame_rate) ?? parseFrameRate(videoStream?.r_frame_rate);

    return {
      duration_seconds: durationSeconds !== undefined ? round(durationSeconds, 4) : undefined,
      width: typeof videoStream?.width === "number" ? videoStream.width : undefined,
      height: typeof videoStream?.height === "number" ? videoStream.height : undefined,
      fps: fps !== undefined ? round(fps, 4) : undefined,
      has_audio: Boolean(audioStream),
      has_video: Boolean(videoStream),
      container_bit_rate:
        parseNumber(payload.format?.bit_rate) ?? parseNumber(videoStream?.bit_rate),
      audio_bit_rate: parseNumber(audioStream?.bit_rate)
    };
  } catch {
    return {
      has_audio: false,
      has_video: true
    };
  }
}

class BaselineAssetAnalysisProvider implements AssetAnalysisProvider {
  public readonly providerId = "baseline";

  public constructor(private readonly config: BaselineAssetAnalysisProviderConfig) {}

  public async analyzeAsset(asset: UploadedVideoAsset): Promise<AssetAnalysisResult> {
    const probe = await runFfprobe(asset.storage_path, this.config.ffprobeBin);
    const assetUpdates: AssetAnalysisResult["asset_updates"] = {
      duration_seconds: probe.duration_seconds,
      width: probe.width,
      height: probe.height,
      fps: probe.fps,
      has_audio: probe.has_audio,
      has_video: probe.has_video,
      orientation: deriveOrientation(probe.width, probe.height)
    };
    const enrichedAsset: UploadedVideoAsset = {
      ...asset,
      ...assetUpdates
    };
    const signal_hints = deriveSignalHints(enrichedAsset, probe);
    const analysis = buildAnalysis(enrichedAsset, signal_hints);

    return {
      asset_updates: assetUpdates,
      signal_hints,
      analysis
    };
  }
}

export function createBaselineAssetAnalysisProvider(
  config: BaselineAssetAnalysisProviderConfig
): AssetAnalysisProvider {
  return new BaselineAssetAnalysisProvider(config);
}
