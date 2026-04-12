import type { VideoAnalysisResult } from "../contracts/models";

export const MOCK_ANALYSIS_RESULT: VideoAnalysisResult = {
  asset_id: "mock-upload-asset",
  analysis_provider: "mock",
  summary:
    "The concept has strong potential, but the opening hook should become more direct to improve first-second retention.",
  keyTopics: [
    "Opening hook",
    "Value narrative",
    "Cut rhythm",
    "Final CTA"
  ],
  suggestedEdits: [
    "Start with a one-line promise at second zero.",
    "Reduce dead space between cuts to keep momentum.",
    "Add dynamic subtitles for key claims.",
    "Close with a more specific call to action."
  ],
  metrics: {
    retention: 78,
    hookStrength: 71,
    clarity: 84
  },
  signal_hints: {
    duration_seconds: 32,
    transcript_text: "Mock transcript",
    estimated_scene_cuts: 14,
    visual_motion_score: 0.46,
    speech_seconds: 21,
    music_seconds: 11,
    fps: 30
  },
  asset: {
    asset_id: "mock-upload-asset",
    checksum_sha256: "mock",
    file_name: "mock-video.mp4",
    mime_type: "video/mp4",
    size_bytes: 1024,
    stored_at: "2026-01-01T00:00:00.000Z",
    duration_seconds: 32,
    width: 1080,
    height: 1920,
    fps: 30,
    has_audio: true,
    has_video: true,
    orientation: "portrait"
  }
};

export const CHAT_ASSISTANT_WELCOME =
  "Done. Your video is analyzed. Ask me about hashtags, retention, or summary.";

export const CHAT_KEYWORD_RESPONSES: Record<string, string> = {
  hashtags:
    "For hashtags, combine 2 niche + 2 medium + 1 broad. Example: #appdevelopment #buildinpublic #startuptips #productgrowth #foryou.",
  retention:
    "Retention improves when the intro is shorter and the result is shown before second two.",
  summary:
    "Quick summary: solid visual potential, high clarity, biggest opportunity is the opening hook.",
  fallback:
    "I can help with summary, hashtags, retention, hook, and clarity. Tell me what you want to optimize."
};
