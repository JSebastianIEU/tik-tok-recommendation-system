import type { VideoAnalysisResult } from "../contracts/models";

export const MOCK_ANALYSIS_RESULT: VideoAnalysisResult = {
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
