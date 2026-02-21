import type { ReportOutput } from "../../src/features/report/types";
import {
  HARD_CODED_EXTRACTED_KEYWORDS,
  SEED_VIDEO_ALGORITHM_DESCRIPTION
} from "./seedVideoContext";

export interface BuildChatPromptParams {
  report: ReportOutput;
  question: string;
}

export function buildChatPrompt(params: BuildChatPromptParams): string {
  const { report, question } = params;

  const payload = {
    role: "Strategic assistant for short-form video optimization",
    context: {
      language: "English",
      style: "actionable, specific, non-generic",
      constraints: [
        "No emojis.",
        "Do not claim official TikTok ranking access.",
        "Use report evidence directly when possible.",
        "Do not reference internal video IDs in user-facing explanation."
      ],
      extracted_keywords: HARD_CODED_EXTRACTED_KEYWORDS,
      uploaded_video_transcript_analysis: SEED_VIDEO_ALGORITHM_DESCRIPTION
    },
    report,
    user_question: question,
    answer_format:
      "plain text in English, concise steps, clear rationale, no markdown"
  };

  return JSON.stringify(payload, null, 2);
}