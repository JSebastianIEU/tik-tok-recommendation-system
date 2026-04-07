export interface CandidateSignalProfile {
  pipeline_version: "extractors.v1";
  generated_at: string;
  duration_seconds: number;
  visual: {
    confidence: number;
    shot_change_rate: number;
    visual_motion_score: number;
    visual_style_tags: string[];
    semantic_embedding_proxy: number[];
  };
  audio: {
    confidence: number;
    speech_ratio: number;
    tempo_bpm_estimate: number;
    audio_energy: number;
    music_presence_score: number;
    audio_style_tags: string[];
  };
  transcript_ocr: {
    confidence: number;
    transcript_text: string;
    ocr_text: string;
    combined_text: string;
    token_count: number;
    unique_token_count: number;
    clarity_score: number;
    cta_keywords_detected: string[];
  };
  structure: {
    confidence: number;
    hook_timing_seconds: number;
    payoff_timing_seconds: number;
    step_density: number;
    pacing_score: number;
  };
  overall_confidence: number;
  quality_flags: string[];
}

interface SignalHints {
  duration_seconds?: number;
  transcript_text?: string;
  ocr_text?: string;
  estimated_scene_cuts?: number;
  visual_motion_score?: number;
  speech_seconds?: number;
  music_seconds?: number;
  tempo_bpm?: number;
  audio_energy?: number;
}

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

export function buildSignalProfileFallback(params: {
  description: string;
  contentType?: string;
  signalHints?: Record<string, unknown>;
}): CandidateSignalProfile {
  const hints = (params.signalHints ?? {}) as SignalHints;
  const description = params.description.trim();
  const transcriptText = typeof hints.transcript_text === "string" ? hints.transcript_text : "";
  const ocrText = typeof hints.ocr_text === "string" ? hints.ocr_text : "";
  const combinedText = [description, transcriptText, ocrText].filter(Boolean).join(" ").trim();
  const tokens = normalizeText(combinedText).split(" ").filter((token) => token.length >= 2);
  const uniqueTokenCount = new Set(tokens).size;
  const durationSeconds = clamp(Math.round(hints.duration_seconds ?? 35), 8, 600);
  const sceneCuts = clamp(
    Math.round(hints.estimated_scene_cuts ?? Math.max(8, tokens.length * 0.5)),
    4,
    80
  );
  const shotChangeRate = round(sceneCuts / Math.max(1, durationSeconds), 4);
  const visualMotionScore = round(
    clamp(hints.visual_motion_score ?? shotChangeRate / 1.8, 0, 1),
    4
  );
  const speechSeconds = clamp(
    hints.speech_seconds ?? durationSeconds * (params.contentType === "tutorial" ? 0.65 : 0.5),
    0,
    durationSeconds
  );
  const musicSeconds = clamp(
    hints.music_seconds ?? Math.max(0, durationSeconds - speechSeconds),
    0,
    durationSeconds
  );
  const speechRatio = round(speechSeconds / Math.max(1, speechSeconds + musicSeconds), 4);
  const clarityScore = round(clamp(uniqueTokenCount / Math.max(1, tokens.length || 1) + 0.35, 0, 1), 4);
  const pacingScore = round(
    clamp((shotChangeRate * 0.8) + (speechRatio * 0.2), 0, 1),
    4
  );
  const hookTimingSeconds = round(clamp(durationSeconds * 0.12, 0.5, 8), 2);
  const payoffTimingSeconds = round(clamp(durationSeconds * 0.72, 3, durationSeconds), 2);
  const ctaKeywordsDetected = ["follow", "comment", "save", "share", "link", "bio"].filter(
    (term) => normalizeText(combinedText).includes(term)
  );

  return {
    pipeline_version: "extractors.v1",
    generated_at: new Date().toISOString(),
    duration_seconds: durationSeconds,
    visual: {
      confidence: 0.42,
      shot_change_rate: shotChangeRate,
      visual_motion_score: visualMotionScore,
      visual_style_tags: [visualMotionScore > 0.55 ? "high_motion" : "balanced_motion"],
      semantic_embedding_proxy: [
        round(clamp(tokens.length / 80, 0, 1), 6),
        round(clamp(uniqueTokenCount / 60, 0, 1), 6),
        pacingScore,
        visualMotionScore
      ]
    },
    audio: {
      confidence: 0.4,
      speech_ratio: speechRatio,
      tempo_bpm_estimate: round(clamp(hints.tempo_bpm ?? 118, 60, 220), 2),
      audio_energy: round(clamp(hints.audio_energy ?? 0.55, 0, 1), 4),
      music_presence_score: round(clamp(1 - speechRatio, 0, 1), 4),
      audio_style_tags: [speechRatio >= 0.55 ? "voice_forward" : "music_forward"]
    },
    transcript_ocr: {
      confidence: 0.38,
      transcript_text: transcriptText,
      ocr_text: ocrText,
      combined_text: combinedText,
      token_count: tokens.length,
      unique_token_count: uniqueTokenCount,
      clarity_score: clarityScore,
      cta_keywords_detected: ctaKeywordsDetected
    },
    structure: {
      confidence: 0.4,
      hook_timing_seconds: hookTimingSeconds,
      payoff_timing_seconds: payoffTimingSeconds,
      step_density: round(clamp(tokens.length / Math.max(1, durationSeconds), 0, 1.25), 4),
      pacing_score: pacingScore
    },
    overall_confidence: 0.4,
    quality_flags: ["ts_signal_fallback"]
  };
}
