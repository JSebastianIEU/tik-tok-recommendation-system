import type { ReportOutput } from "../../src/features/report/types";

function normalize(value: string): string {
  return value
    .toLowerCase()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "");
}

export function buildLocalChatAnswer(
  report: ReportOutput,
  question: string
): string {
  const normalizedQuestion = normalize(question);

  if (normalizedQuestion.includes("hashtag")) {
    const top = report.comparables
      .flatMap((item) => item.hashtags)
      .slice(0, 6)
      .join(", ");

    return `From this local dataset, the most repeated hashtags are: ${top}. Use 3-5 tags, prioritize niche intent, and avoid broad generic tags.`;
  }

  if (
    normalizedQuestion.includes("retention") ||
    normalizedQuestion.includes("hook")
  ) {
    const metric = report.executive_summary.metrics.find(
      (item) => item.id === "retention-estimated"
    );

    return `Your current estimated retention is ${metric?.value ?? "N/A"}. The fastest win is to make the first 2 seconds outcome-driven with a concrete proof shot.`;
  }

  if (normalizedQuestion.includes("summary")) {
    return report.executive_summary.summary_text;
  }

  const firstRecommendation = report.recommendations.items[0]?.title;
  return `Next practical step: ${firstRecommendation ?? "strengthen hook clarity and CTA specificity"}.`;
}