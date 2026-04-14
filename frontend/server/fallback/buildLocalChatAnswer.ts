import type { ReportOutput } from "../../src/features/report/types";
import type { KnowledgeBaseEntry } from "../knowledgeBase/knowledgeBase";

function normalize(value: string): string {
  return value
    .toLowerCase()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "");
}

function cleanSentence(value: string): string {
  return value
    .replace(/\s+/g, " ")
    .trim()
    .replace(/[.!?]+$/, "");
}

function toAreaImpact(area: string): string {
  switch (area) {
    case "hook":
      return "Stronger first-second framing should raise early hold and watch-through.";
    case "cta":
      return "A clearer call to action should increase comments and interaction rate.";
    case "pacing":
      return "Cleaner pacing should reduce mid-video drop-off.";
    case "clarity":
      return "Higher message clarity should improve saves and completions.";
    case "format":
      return "Tighter format consistency should make the video easier to follow.";
    case "audience_alignment":
      return "Better audience alignment should improve meaningful engagement.";
    case "topic_alignment":
      return "Stronger topic alignment should improve relevance and discovery quality.";
    default:
      return "The change should improve engagement consistency across the video.";
  }
}

function buildGeneralDiagnosis(report: ReportOutput, normalizedQuestion: string): string {
  const rows = report.direct_comparison.rows ?? [];
  const weakestRow = [...rows]
    .map((row) => ({
      row,
      gap: row.your_value_pct - row.comparable_value_pct
    }))
    .sort((left, right) => left.gap - right.gap)[0];

  if (weakestRow && weakestRow.gap <= -5) {
    return `Quick diagnosis: your biggest gap is ${weakestRow.row.label.toLowerCase()} (${weakestRow.row.your_value_label} vs ${weakestRow.row.comparable_value_label} for top comparables).`;
  }

  const hookMetric = report.executive_summary.metrics.find((item) =>
    normalize(item.label).includes("hook")
  );
  if (hookMetric && normalizedQuestion.includes("engagement")) {
    return `Quick diagnosis: engagement upside is still available, and hook strength is the first lever to tighten (${hookMetric.value}).`;
  }

  return `Quick diagnosis: ${cleanSentence(report.executive_summary.summary_text || "you have a workable draft, but execution can be sharper in the opening and CTA")}.`;
}

function buildGeneralActions(
  report: ReportOutput,
  knowledgeBaseEntries: KnowledgeBaseEntry[]
): string[] {
  const reportActions = report.recommendations.items
    .slice(0, 2)
    .map((item) => cleanSentence(item.title))
    .filter(Boolean);

  const kbAction = knowledgeBaseEntries
    .map((entry) => cleanSentence(entry.action_hint))
    .find((value) => value.length > 0);

  if (kbAction && !reportActions.some((item) => item.toLowerCase() === kbAction.toLowerCase())) {
    reportActions.push(kbAction);
  }

  const fallbackActions = [
    "Make the first 1-2 seconds outcome-first with one clear promise",
    "Tighten mid-video pacing by removing one low-information beat",
    "End with a direct and specific comment CTA"
  ];

  const actions = reportActions.length > 0 ? reportActions : fallbackActions;
  while (actions.length < 3) {
    actions.push(fallbackActions[actions.length]);
  }
  return uniqueLines(actions, 3).slice(0, 3);
}

function buildGeneralImpact(
  report: ReportOutput,
  knowledgeBaseEntries: KnowledgeBaseEntry[]
): string[] {
  const impactLines = report.recommendations.items
    .slice(0, 2)
    .map((item) => toAreaImpact(item.effect_area))
    .filter(Boolean);

  const kbImpact = knowledgeBaseEntries
    .map((entry) => cleanSentence(entry.content[0] || ""))
    .find(Boolean);
  if (kbImpact) {
    impactLines.push(kbImpact);
  }

  if (impactLines.length === 0) {
    return [
      "Expected impact: stronger hook and pacing should improve hold and completion.",
      "Expected impact: a clearer CTA should lift comments and interaction quality."
    ];
  }

  return uniqueLines(impactLines, 3).map((line) => `Expected impact: ${line}`);
}

function uniqueLines(values: string[], maxItems: number): string[] {
  const output: string[] = [];
  const seen = new Set<string>();
  for (const value of values) {
    const normalizedValue = value.trim().toLowerCase();
    if (!normalizedValue || seen.has(normalizedValue)) {
      continue;
    }
    seen.add(normalizedValue);
    output.push(value.trim());
    if (output.length >= maxItems) {
      break;
    }
  }
  return output;
}

function buildHashtagAnswer(report: ReportOutput): string {
  const topTags = uniqueLines(
    report.comparables.flatMap((item) => item.hashtags || []),
    6
  );
  const topTagText = topTags.length > 0 ? topTags.join(", ") : "#nichetopic, #specificintent, #audiencefit";

  return [
    `Quick diagnosis: hashtag strategy can be tighter for discoverability and audience fit.`,
    "",
    "Top 3 actions:",
    `1. Start from repeated high-signal tags in your comparables: ${topTagText}.`,
    "2. Use 3-5 tags max: 1 broad context tag, 2-3 niche intent tags, and 1 format tag.",
    "3. Align the first hashtag with the opening promise so metadata and hook reinforce each other.",
    "",
    "Expected impact: cleaner tagging should improve relevance and early qualified impressions.",
    "Expected impact: tighter niche tagging should improve engagement quality over raw reach.",
    "",
    "Want me to suggest 8 hashtags ranked from safest to boldest?"
  ].join("\n");
}

function buildKnowledgeOnlyAnswer(
  question: string,
  knowledgeBaseEntries: KnowledgeBaseEntry[]
): string {
  const normalizedQuestion = normalize(question);
  const fallbackEntries = knowledgeBaseEntries.slice(0, 3);
  if (fallbackEntries.length === 0) {
    return "Upload a video and generate a report to start chatting.";
  }

  const diagnosisEntry = fallbackEntries[0];
  const diagnosisFact = cleanSentence(diagnosisEntry.content[0] || "TikTok performance usually improves with stronger hook clarity and a more explicit CTA.");
  const actions = fallbackEntries.map((entry) => cleanSentence(entry.action_hint)).filter(Boolean);
  const impact = fallbackEntries
    .map((entry) => cleanSentence(entry.content[0] || ""))
    .filter(Boolean)
    .map((line) => `Expected impact: ${line}`);

  const followUp = normalizedQuestion.includes("hashtag")
    ? "Want me to suggest hashtag sets for this topic?"
    : "Want me to turn this into a concrete script and shot list for your next post?";

  return [
    `Quick diagnosis: ${diagnosisFact}.`,
    "",
    "Top 3 actions:",
    `1. ${actions[0] ?? "Open with one explicit outcome in the first two seconds"}.`,
    `2. ${actions[1] ?? "Use one clear format shift to protect mid-video retention"}.`,
    `3. ${actions[2] ?? "End with a specific one-step CTA"}.`,
    "",
    ...(impact.length > 0 ? impact.slice(0, 3) : ["Expected impact: stronger structure should improve completion and engagement quality."]),
    "",
    followUp
  ].join("\n");
}

interface BuildLocalChatAnswerParams {
  report: ReportOutput | null;
  question: string;
  knowledgeBaseEntries?: KnowledgeBaseEntry[];
}

export function buildLocalChatAnswer(
  params: BuildLocalChatAnswerParams
): string {
  const { report, question } = params;
  const knowledgeBaseEntries = params.knowledgeBaseEntries ?? [];
  const normalizedQuestion = normalize(question);
  if (!report) {
    return buildKnowledgeOnlyAnswer(question, knowledgeBaseEntries);
  }

  if (normalizedQuestion.includes("hashtag")) {
    return buildHashtagAnswer(report);
  }

  const diagnosis = buildGeneralDiagnosis(report, normalizedQuestion);
  const actions = buildGeneralActions(report, knowledgeBaseEntries);
  const impact = buildGeneralImpact(report, knowledgeBaseEntries);
  const followUp = normalizedQuestion.includes("hook")
    ? "Want me to draft 3 hook options for your first 2 seconds?"
    : normalizedQuestion.includes("engagement")
      ? "Want me to rewrite your opening and CTA for engagement specifically?"
      : "Want me to turn these actions into exact script lines for your next cut?";

  return [
    diagnosis,
    "",
    "Top 3 actions:",
    `1. ${actions[0]}.`,
    `2. ${actions[1]}.`,
    `3. ${actions[2]}.`,
    "",
    ...impact,
    "",
    followUp
  ].join("\n");
}
