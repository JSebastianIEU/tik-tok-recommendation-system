import type { CandidateProfileCore } from "../step1/types";

export interface CandidateSignalHints {
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

interface VisualSignalFeatures {
  confidence: number;
  shot_change_rate: number;
  visual_motion_score: number;
  visual_style_tags: string[];
  semantic_embedding_proxy: number[];
}

interface AudioSignalFeatures {
  confidence: number;
  speech_ratio: number;
  tempo_bpm_estimate: number;
  audio_energy: number;
  music_presence_score: number;
  audio_style_tags: string[];
}

interface TranscriptOcrSignalFeatures {
  confidence: number;
  transcript_text: string;
  ocr_text: string;
  combined_text: string;
  token_count: number;
  unique_token_count: number;
  clarity_score: number;
  cta_keywords_detected: string[];
}

interface StructureSignalFeatures {
  confidence: number;
  hook_timing_seconds: number;
  payoff_timing_seconds: number;
  step_density: number;
  pacing_score: number;
}

export interface CandidateSignalProfile {
  pipeline_version: "extractors.v1";
  generated_at: string;
  duration_seconds: number;
  visual: VisualSignalFeatures;
  audio: AudioSignalFeatures;
  transcript_ocr: TranscriptOcrSignalFeatures;
  structure: StructureSignalFeatures;
  overall_confidence: number;
  quality_flags: string[];
}

const CTA_TERMS = [
  "follow",
  "comment",
  "save",
  "share",
  "link",
  "bio",
  "join",
  "subscribe",
  "shop",
  "comenta",
  "guarda",
  "sigue"
];

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function round(value: number, decimals = 4): number {
  const factor = 10 ** decimals;
  return Math.round(value * factor) / factor;
}

function normalizeText(value: string): string {
  return value
    .toLowerCase()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[^\p{L}\p{N}\s]/gu, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function tokenize(value: string): string[] {
  return normalizeText(value)
    .split(" ")
    .map((token) => token.trim())
    .filter((token) => token.length >= 2);
}

function inferDurationSeconds(
  profile: CandidateProfileCore,
  hints?: CandidateSignalHints
): number {
  if (typeof hints?.duration_seconds === "number" && Number.isFinite(hints.duration_seconds)) {
    return clamp(Math.round(hints.duration_seconds), 3, 600);
  }
  const wordCount = profile.features.word_count;
  const contentType = profile.intent.content_type;
  const baseByType: Record<string, number> = {
    tutorial: 45,
    story: 50,
    reaction: 35,
    showcase: 30,
    opinion: 40
  };
  const baseline = baseByType[contentType] ?? 35;
  return clamp(Math.round(baseline + wordCount * 0.45), 12, 180);
}

function buildEmbeddingProxy(profile: CandidateProfileCore): number[] {
  const values = [
    profile.features.word_count / 120,
    profile.features.hashtag_density,
    profile.features.unique_word_count / 100,
    profile.tokens.keyphrases.length / 8,
    profile.tags.topic_tags.length / 5,
    profile.tags.format_tags.length / 5,
    profile.features.question_mark_present ? 1 : 0,
    profile.features.number_present ? 1 : 0
  ];
  return values.map((value) => round(clamp(value, 0, 1), 6));
}

function deriveVisualSignals(
  profile: CandidateProfileCore,
  durationSeconds: number,
  hints?: CandidateSignalHints
): VisualSignalFeatures {
  const baseShotChanges =
    profile.intent.content_type === "reaction"
      ? 24
      : profile.intent.content_type === "tutorial"
        ? 16
        : 12;
  const sceneCuts =
    typeof hints?.estimated_scene_cuts === "number" && Number.isFinite(hints.estimated_scene_cuts)
      ? Math.max(1, hints.estimated_scene_cuts)
      : clamp(baseShotChanges + profile.features.word_count * 0.15, 6, 80);
  const shotChangeRate = round(sceneCuts / Math.max(1, durationSeconds), 4);

  const motionHint =
    typeof hints?.visual_motion_score === "number" && Number.isFinite(hints.visual_motion_score)
      ? hints.visual_motion_score
      : shotChangeRate / 1.8;
  const visualMotionScore = round(clamp(motionHint, 0, 1), 4);

  const visualStyleTags: string[] = [];
  if (shotChangeRate >= 0.5) {
    visualStyleTags.push("rapid_cut");
  } else if (shotChangeRate <= 0.22) {
    visualStyleTags.push("slow_cut");
  } else {
    visualStyleTags.push("balanced_cut");
  }
  if (visualMotionScore >= 0.7) {
    visualStyleTags.push("high_motion");
  } else if (visualMotionScore <= 0.35) {
    visualStyleTags.push("static_framing");
  }
  visualStyleTags.push(`intent:${profile.intent.objective}`);

  let confidence = 0.66;
  if (typeof hints?.estimated_scene_cuts === "number") {
    confidence += 0.14;
  }
  if (typeof hints?.visual_motion_score === "number") {
    confidence += 0.12;
  }
  if (typeof hints?.fps === "number") {
    confidence += 0.06;
  }

  return {
    confidence: round(clamp(confidence, 0.25, 1), 2),
    shot_change_rate: shotChangeRate,
    visual_motion_score: visualMotionScore,
    visual_style_tags: visualStyleTags,
    semantic_embedding_proxy: buildEmbeddingProxy(profile)
  };
}

function deriveAudioSignals(
  profile: CandidateProfileCore,
  durationSeconds: number,
  hints?: CandidateSignalHints
): AudioSignalFeatures {
  const speechSeconds =
    typeof hints?.speech_seconds === "number" && Number.isFinite(hints.speech_seconds)
      ? clamp(hints.speech_seconds, 0, durationSeconds)
      : clamp(durationSeconds * (profile.intent.content_type === "tutorial" ? 0.65 : 0.5), 0, durationSeconds);
  const musicSeconds =
    typeof hints?.music_seconds === "number" && Number.isFinite(hints.music_seconds)
      ? clamp(hints.music_seconds, 0, durationSeconds)
      : clamp(durationSeconds - speechSeconds, 0, durationSeconds);

  const speechRatio = round(speechSeconds / Math.max(1, speechSeconds + musicSeconds), 4);
  const tempoBpmEstimate =
    typeof hints?.tempo_bpm === "number" && Number.isFinite(hints.tempo_bpm)
      ? clamp(hints.tempo_bpm, 60, 220)
      : clamp(
          95 +
            profile.features.hashtag_count * 3 +
            (profile.features.exclamation_present ? 8 : 0),
          75,
          180
        );

  const audioEnergyRaw =
    typeof hints?.audio_energy === "number" && Number.isFinite(hints.audio_energy)
      ? hints.audio_energy
      : (tempoBpmEstimate - 70) / 110 + (profile.features.exclamation_present ? 0.08 : 0);
  const audioEnergy = round(clamp(audioEnergyRaw, 0, 1), 4);
  const musicPresenceScore = round(clamp(1 - speechRatio + 0.2, 0, 1), 4);

  const audioStyleTags: string[] = [];
  if (speechRatio >= 0.62) {
    audioStyleTags.push("voice_forward");
  } else {
    audioStyleTags.push("music_forward");
  }
  if (tempoBpmEstimate >= 135) {
    audioStyleTags.push("high_tempo");
  } else if (tempoBpmEstimate <= 95) {
    audioStyleTags.push("low_tempo");
  }

  let confidence = 0.62;
  if (typeof hints?.tempo_bpm === "number") {
    confidence += 0.12;
  }
  if (typeof hints?.speech_seconds === "number" || typeof hints?.music_seconds === "number") {
    confidence += 0.16;
  }
  if (typeof hints?.audio_energy === "number" || typeof hints?.loudness_lufs === "number") {
    confidence += 0.08;
  }

  return {
    confidence: round(clamp(confidence, 0.25, 1), 2),
    speech_ratio: speechRatio,
    tempo_bpm_estimate: round(tempoBpmEstimate, 2),
    audio_energy: audioEnergy,
    music_presence_score: musicPresenceScore,
    audio_style_tags: audioStyleTags
  };
}

function deriveTranscriptOcrSignals(
  profile: CandidateProfileCore,
  hints?: CandidateSignalHints
): TranscriptOcrSignalFeatures {
  const transcriptText = (hints?.transcript_text ?? "").trim();
  const ocrText = (hints?.ocr_text ?? "").trim();
  const combinedText = [profile.raw.description, transcriptText, ocrText].filter(Boolean).join(" ");
  const tokens = tokenize(combinedText);
  const uniqueTokenCount = new Set(tokens).size;
  const ctaKeywords = CTA_TERMS.filter((term) => tokens.includes(term));

  const lexicalDiversity = uniqueTokenCount / Math.max(1, tokens.length);
  const clarityScore = round(clamp(0.45 + lexicalDiversity * 0.45 + (ctaKeywords.length > 0 ? 0.05 : 0), 0, 1), 4);

  let confidence = 0.55;
  if (transcriptText) {
    confidence += 0.18;
  }
  if (ocrText) {
    confidence += 0.12;
  }
  if (tokens.length >= 12) {
    confidence += 0.08;
  }

  return {
    confidence: round(clamp(confidence, 0.25, 1), 2),
    transcript_text: transcriptText,
    ocr_text: ocrText,
    combined_text: combinedText.trim(),
    token_count: tokens.length,
    unique_token_count: uniqueTokenCount,
    clarity_score: clarityScore,
    cta_keywords_detected: ctaKeywords
  };
}

function deriveStructureSignals(
  profile: CandidateProfileCore,
  durationSeconds: number,
  transcriptOcr: TranscriptOcrSignalFeatures,
  visual: VisualSignalFeatures,
  audio: AudioSignalFeatures
): StructureSignalFeatures {
  const transcriptTokens = tokenize(transcriptOcr.combined_text);
  const stepTerms = ["step", "first", "then", "next", "finally", "1", "2", "3"];
  const stepTermHits = transcriptTokens.filter((token) => stepTerms.includes(token)).length;
  const stepDensity = round(stepTermHits / Math.max(1, durationSeconds / 10), 4);

  const hookTimingSeconds =
    profile.features.question_mark_present || profile.features.number_present ? 1.2 : 2.4;
  const payoffTimingSeconds = clamp(
    durationSeconds * (profile.intent.content_type === "tutorial" ? 0.68 : 0.75),
    4,
    durationSeconds
  );

  const pacingRaw = visual.shot_change_rate * 0.55 + audio.audio_energy * 0.35 + stepDensity * 0.1;
  const pacingScore = round(clamp(pacingRaw, 0, 1), 4);

  const confidence = round(
    clamp((visual.confidence + audio.confidence + transcriptOcr.confidence) / 3, 0.25, 1),
    2
  );

  return {
    confidence,
    hook_timing_seconds: round(clamp(hookTimingSeconds, 0.8, 6), 2),
    payoff_timing_seconds: round(payoffTimingSeconds, 2),
    step_density: stepDensity,
    pacing_score: pacingScore
  };
}

export function extractCandidateSignals(
  profile: CandidateProfileCore,
  hints?: CandidateSignalHints
): CandidateSignalProfile {
  const durationSeconds = inferDurationSeconds(profile, hints);
  const visual = deriveVisualSignals(profile, durationSeconds, hints);
  const audio = deriveAudioSignals(profile, durationSeconds, hints);
  const transcriptOcr = deriveTranscriptOcrSignals(profile, hints);
  const structure = deriveStructureSignals(profile, durationSeconds, transcriptOcr, visual, audio);

  const qualityFlags: string[] = [];
  if (!hints?.transcript_text) {
    qualityFlags.push("missing_transcript_hint");
  }
  if (transcriptOcr.token_count < 10) {
    qualityFlags.push("low_text_signal_for_structure");
  }
  if (typeof hints?.duration_seconds !== "number") {
    qualityFlags.push("duration_estimated");
  }

  const overallConfidence = round(
    clamp(
      (visual.confidence + audio.confidence + transcriptOcr.confidence + structure.confidence) / 4,
      0.25,
      1
    ),
    2
  );

  return {
    pipeline_version: "extractors.v1",
    generated_at: new Date().toISOString(),
    duration_seconds: durationSeconds,
    visual,
    audio,
    transcript_ocr: transcriptOcr,
    structure,
    overall_confidence: overallConfidence,
    quality_flags: qualityFlags
  };
}
