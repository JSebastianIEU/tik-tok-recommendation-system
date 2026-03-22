import type { ComparableItem, ReportOutput, ReportPolarity } from "../../src/features/report/types";

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

interface NormalizeReportOutputOptions {
  candidatesK: number;
  extractedKeywords: string[];
  modelLabel: string;
}

function stripComparableIdReferences(value: string): string {
  return value
    .replace(/\b[a-zA-Z]-?\d{2,4}\b/g, "top comparable")
    .replace(/\b[Cc]omparable\s*#?\d+\b/g, "top comparable")
    .replace(/\s{2,}/g, " ")
    .trim();
}

function normalizePolarity(value: string): ReportPolarity {
  const normalized = value.toLowerCase();

  if (normalized.includes("neg")) {
    return "Negative";
  }

  if (normalized.includes("quest") || normalized.includes("preg")) {
    return "Question";
  }

  return "Positive";
}

function normalizePriority(value: string): "High" | "Medium" | "Low" {
  const normalized = value.toLowerCase();
  if (normalized.includes("high") || normalized.includes("alta")) {
    return "High";
  }
  if (normalized.includes("low") || normalized.includes("baja")) {
    return "Low";
  }
  return "Medium";
}

function normalizeEffort(value: string): "Low" | "Medium" | "High" {
  const normalized = value.toLowerCase();
  if (normalized.includes("high") || normalized.includes("alto")) {
    return "High";
  }
  if (normalized.includes("low") || normalized.includes("bajo")) {
    return "Low";
  }
  return "Medium";
}

function normalizeComparable(item: ComparableItem): ComparableItem {
  return {
    ...item,
    similarity: Number(clamp(item.similarity, 0, 1).toFixed(2)),
    metrics: {
      views: Math.max(0, Math.round(item.metrics.views)),
      likes: Math.max(0, Math.round(item.metrics.likes)),
      comments_count: Math.max(0, Math.round(item.metrics.comments_count)),
      shares: Math.max(0, Math.round(item.metrics.shares)),
      engagement_rate: item.metrics.engagement_rate
    },
    caption: stripComparableIdReferences(item.caption),
    video_url: typeof item.video_url === "string" ? item.video_url : "",
    thumbnail_url: typeof item.thumbnail_url === "string" ? item.thumbnail_url : "",
    hashtags: item.hashtags.slice(0, 6),
    matched_keywords: item.matched_keywords.slice(0, 8),
    observations: item.observations.slice(0, 4).map(stripComparableIdReferences)
  };
}

export function normalizeReportOutput(
  report: ReportOutput,
  options: NormalizeReportOutputOptions
): ReportOutput {
  const { candidatesK, extractedKeywords, modelLabel } = options;
  const safeExtractedKeywords =
    report.executive_summary.extracted_keywords.length > 0
      ? report.executive_summary.extracted_keywords.slice(0, 10)
      : [...extractedKeywords];

  return {
    ...report,
    header: {
      ...report.header,
      title: "Report",
      subtitle: "Comparison based on local dataset candidates",
      badges: {
        ...report.header.badges,
        candidates_k: candidatesK,
        model: report.header.badges.model.trim() || modelLabel,
        mode: report.header.badges.mode.trim() || "Guided demo"
      },
      disclaimer:
        "Estimated insights built from local dataset evidence and uploaded-video context."
    },
    executive_summary: {
      ...report.executive_summary,
      extracted_keywords: safeExtractedKeywords,
      meaning_points: report.executive_summary.meaning_points
        .slice(0, 6)
        .map(stripComparableIdReferences),
      summary_text: stripComparableIdReferences(report.executive_summary.summary_text)
    },
    comparables: report.comparables.map(normalizeComparable),
    direct_comparison: {
      ...report.direct_comparison,
      rows: report.direct_comparison.rows.map((row) => ({
        ...row,
        your_value_pct: clamp(row.your_value_pct, 0, 100),
        comparable_value_pct: clamp(row.comparable_value_pct, 0, 100)
      })),
      note:
        "Estimated from local dataset candidates."
    },
    relevant_comments: {
      ...report.relevant_comments,
      items: report.relevant_comments.items.map((item, index) => ({
        ...item,
        id: item.id || `comment-${index + 1}`,
        text: stripComparableIdReferences(item.text),
        topic: item.topic.trim() || "content",
        polarity: normalizePolarity(item.polarity),
        relevance_note:
          item.relevance_note?.trim() ||
          "Use this signal to refine your hook clarity and call-to-action phrasing."
      })),
      disclaimer:
        "Comments are from the local test dataset."
    },
    recommendations: {
      ...report.recommendations,
      items: report.recommendations.items.map((item, index) => ({
        ...item,
        id: item.id || `rec-${index + 1}`,
        title: stripComparableIdReferences(item.title),
        priority: normalizePriority(item.priority),
        effort: normalizeEffort(item.effort),
        evidence: stripComparableIdReferences(item.evidence)
      }))
    }
  };
}
