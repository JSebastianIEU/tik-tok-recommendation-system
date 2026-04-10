export const OBJECTIVES = ["reach", "engagement", "conversion", "community"] as const;
export const CONTENT_TYPES = [
  "tutorial",
  "storytime",
  "reaction",
  "showcase",
  "commentary",
  "trend_participation",
  "opinion",
  "educational",
  "behind_the_scenes",
  "other"
] as const;
export const PRIMARY_CTAS = [
  "follow",
  "comment",
  "save",
  "share",
  "link_click",
  "profile_visit",
  "dm",
  "none"
] as const;
export const AUDIENCE_EXPERTISE_LEVELS = [
  "beginner",
  "intermediate",
  "advanced",
  "mixed"
] as const;

export type Objective = (typeof OBJECTIVES)[number];
export type ContentType = (typeof CONTENT_TYPES)[number];
export type PrimaryCta = (typeof PRIMARY_CTAS)[number];
export type AudienceExpertiseLevel = (typeof AUDIENCE_EXPERTISE_LEVELS)[number];

export interface AudienceInput {
  label?: string;
  segments?: string[];
  expertise_level?: AudienceExpertiseLevel;
}

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
