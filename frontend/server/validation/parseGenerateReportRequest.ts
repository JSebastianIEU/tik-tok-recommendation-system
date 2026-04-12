import {
  AUDIENCE_EXPERTISE_LEVELS,
  CONTENT_TYPES,
  OBJECTIVES,
  PRIMARY_CTAS,
  type CandidateSignalHints,
  type AudienceInput,
  type ContentType,
  type Objective,
  type PrimaryCta
} from "../contracts/query";

export interface ParsedGenerateReportRequest {
  asset_id?: string;
  seed_video_id: string;
  mentions: string[];
  hashtags: string[];
  description: string;
  objective?: Objective;
  audience: string | AudienceInput;
  content_type?: ContentType;
  primary_cta?: PrimaryCta;
  locale?: string;
  language?: string;
  signal_hints?: CandidateSignalHints;
}

export type ParseResult =
  | { ok: true; value: ParsedGenerateReportRequest }
  | { ok: false; error: string };

interface SignalHintFieldSpec {
  type: "number" | "string";
  min?: number;
  max?: number;
  maxLength?: number;
}

const SIGNAL_HINT_SPECS: Record<keyof CandidateSignalHints, SignalHintFieldSpec> = {
  duration_seconds: { type: "number", min: 0, max: 600 },
  transcript_text: { type: "string", maxLength: 15000 },
  ocr_text: { type: "string", maxLength: 15000 },
  estimated_scene_cuts: { type: "number", min: 0, max: 1000 },
  fps: { type: "number", min: 1, max: 240 },
  visual_motion_score: { type: "number", min: 0, max: 1 },
  speech_seconds: { type: "number", min: 0, max: 600 },
  music_seconds: { type: "number", min: 0, max: 600 },
  tempo_bpm: { type: "number", min: 30, max: 260 },
  audio_energy: { type: "number", min: 0, max: 1 },
  loudness_lufs: { type: "number", min: -80, max: 0 }
};

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function parseStringArray(value: unknown, maxItems: number): string[] | null {
  if (value === undefined) {
    return [];
  }
  if (!Array.isArray(value)) {
    return null;
  }
  const output: string[] = [];
  for (const item of value) {
    if (typeof item !== "string") {
      return null;
    }
    const normalized = item.trim();
    if (!normalized) {
      continue;
    }
    output.push(normalized);
    if (output.length > maxItems) {
      return null;
    }
  }
  return output;
}

function parseEnum<T extends readonly string[]>(
  value: unknown,
  allowed: T
): T[number] | undefined | null {
  if (value === undefined) {
    return undefined;
  }
  if (typeof value !== "string") {
    return null;
  }
  return (allowed as readonly string[]).includes(value) ? (value as T[number]) : null;
}

function parseSignalHints(value: unknown): CandidateSignalHints | null {
  if (value === undefined) {
    return undefined;
  }
  if (!isObject(value)) {
    return null;
  }
  const output: CandidateSignalHints = {};
  const mutableOutput = output as Record<
    keyof CandidateSignalHints,
    CandidateSignalHints[keyof CandidateSignalHints] | undefined
  >;
  for (const key of Object.keys(value)) {
    if (!(key in SIGNAL_HINT_SPECS)) {
      continue;
    }
    const typedKey = key as keyof CandidateSignalHints;
    const spec = SIGNAL_HINT_SPECS[typedKey];
    const raw = value[typedKey];

    if (raw === undefined || raw === null) {
      continue;
    }
    if (spec.type === "number") {
      if (typeof raw !== "number" || !Number.isFinite(raw)) {
        return null;
      }
      if (spec.min !== undefined && raw < spec.min) {
        return null;
      }
      if (spec.max !== undefined && raw > spec.max) {
        return null;
      }
      mutableOutput[typedKey] = raw;
      continue;
    }
    if (typeof raw !== "string") {
      return null;
    }
    const normalized = raw.trim();
    if (spec.maxLength !== undefined && normalized.length > spec.maxLength) {
      return null;
    }
    mutableOutput[typedKey] = normalized;
  }
  return output;
}

function parseAudience(value: unknown): string | AudienceInput | null {
  if (value === undefined) {
    return "";
  }
  if (typeof value === "string") {
    return value.trim().slice(0, 120);
  }
  if (!isObject(value)) {
    return null;
  }

  const label =
    typeof value.label === "string"
      ? value.label.trim().slice(0, 120)
      : undefined;
  const segments = parseStringArray(value.segments, 8);
  if (segments === null) {
    return null;
  }
  const expertiseLevel =
    typeof value.expertise_level === "string"
      ? value.expertise_level.trim()
      : undefined;
  if (expertiseLevel && !AUDIENCE_EXPERTISE_LEVELS.includes(expertiseLevel as (typeof AUDIENCE_EXPERTISE_LEVELS)[number])) {
    return null;
  }

  return {
    ...(label ? { label } : {}),
    ...(segments && segments.length > 0 ? { segments } : {}),
    ...(expertiseLevel ? { expertise_level: expertiseLevel as AudienceInput["expertise_level"] } : {})
  };
}

export function parseGenerateReportRequest(body: unknown): ParseResult {
  if (!isObject(body)) {
    return { ok: false, error: "Request body must be a JSON object." };
  }

  const mentions = parseStringArray(body.mentions, 30);
  if (mentions === null) {
    return { ok: false, error: "'mentions' must be an array of up to 30 strings." };
  }

  const hashtags = parseStringArray(body.hashtags, 30);
  if (hashtags === null) {
    return { ok: false, error: "'hashtags' must be an array of up to 30 strings." };
  }

  const rawDescription = typeof body.description === "string" ? body.description.trim() : "";
  const description = rawDescription.slice(0, 5000);

  const objective = parseEnum(body.objective, OBJECTIVES);
  if (objective === null) {
    return { ok: false, error: "'objective' is invalid." };
  }

  const contentType = parseEnum(body.content_type, CONTENT_TYPES);
  if (contentType === null) {
    return { ok: false, error: "'content_type' is invalid." };
  }

  const primaryCta = parseEnum(body.primary_cta, PRIMARY_CTAS);
  if (primaryCta === null) {
    return { ok: false, error: "'primary_cta' is invalid." };
  }

  const audience = parseAudience(body.audience);
  if (audience === null) {
    return { ok: false, error: "'audience' must be a string or object with valid fields." };
  }

  const locale =
    typeof body.locale === "string" && body.locale.trim()
      ? body.locale.trim().slice(0, 24)
      : undefined;

  const language =
    typeof body.language === "string" && body.language.trim()
      ? body.language.trim().toLowerCase().slice(0, 8)
      : undefined;

  const assetId =
    typeof body.asset_id === "string" && body.asset_id.trim()
      ? body.asset_id.trim().slice(0, 120)
      : undefined;

  const seedVideoId =
    typeof body.seed_video_id === "string" && body.seed_video_id.trim()
      ? body.seed_video_id.trim().slice(0, 120)
      : assetId ?? "s001";

  const signalHints = parseSignalHints(body.signal_hints);
  if (signalHints === null) {
    return { ok: false, error: "'signal_hints' has invalid field values." };
  }

  return {
    ok: true,
    value: {
      asset_id: assetId,
      seed_video_id: seedVideoId,
      mentions,
      hashtags,
      description,
      objective,
      audience,
      content_type: contentType,
      primary_cta: primaryCta,
      locale,
      language,
      signal_hints: signalHints
    }
  };
}
