// @ts-nocheck
import fs from "node:fs/promises";
import path from "node:path";

interface GenerateReportRequestBody {
  seed_video_id?: string;
  mentions?: unknown;
  hashtags?: unknown;
  description?: string;
}

interface ApiRequest {
  method?: string;
  body?: unknown;
}

interface ApiResponse {
  setHeader(name: string, value: string): void;
  status(code: number): ApiResponse;
  json(payload: unknown): void;
}

interface DatasetItem {
  video_id: string;
  video_url?: string;
  caption?: string;
  thumbnail_url?: string;
  hashtags?: string[];
  views?: number;
  likes?: number;
  comments_count?: number;
  shares?: number;
  metrics?: {
    views?: number;
    likes?: number;
    comments_count?: number;
    shares?: number;
  };
  author?: unknown;
}

interface MetricSnapshot {
  views: number;
  likes: number;
  comments_count: number;
  shares: number;
}

function normalizeArray(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value
    .map((entry) => (typeof entry === "string" ? entry.trim() : ""))
    .filter(Boolean);
}

function normalizeTagValues(values: string[]): string[] {
  return values
    .map((value) => value.trim().replace(/^#/, ""))
    .filter(Boolean)
    .map((value) => `#${value}`);
}

function removeEmoji(value: string): string {
  return value.replace(/\p{Extended_Pictographic}/gu, "").trim();
}

function extractTextContent(content: unknown): string {
  if (typeof content === "string") {
    return content;
  }

  if (Array.isArray(content)) {
    return content
      .map((part) => {
        if (part && typeof part === "object" && "text" in part) {
          const textValue = (part as { text?: unknown }).text;
          return typeof textValue === "string" ? textValue : "";
        }
        return "";
      })
      .join("")
      .trim();
  }

  return "";
}

function extractFirstJsonObject(rawContent: string): unknown {
  const trimmed = rawContent.trim();

  try {
    return JSON.parse(trimmed);
  } catch {
  }

  const fencedMatch = trimmed.match(/```(?:json)?\s*([\s\S]*?)\s*```/i);
  if (fencedMatch?.[1]) {
    return JSON.parse(fencedMatch[1]);
  }

  throw new Error("No valid JSON was found in the provider response.");
}

async function loadDatasetFromFile(): Promise<DatasetItem[]> {
  const candidatePaths = [
    path.resolve(process.cwd(), "mvp-mock-ui/src/data/demodata.jsonl"),
    path.resolve(process.cwd(), "src/data/demodata.jsonl")
  ];

  for (const datasetPath of candidatePaths) {
    try {
      const raw = await fs.readFile(datasetPath, "utf-8");
      return raw
        .split(/\r?\n/)
        .map((line) => line.trim())
        .filter(Boolean)
        .map((line) => JSON.parse(line) as DatasetItem);
    } catch {
    }
  }

  return [];
}

function toAuthorLabel(author: unknown): string {
  if (typeof author === "string") {
    return author.startsWith("@") ? author : `@${author}`;
  }

  if (author && typeof author === "object") {
    const username = (author as Record<string, unknown>).username;
    if (typeof username === "string" && username.trim()) {
      return `@${username.replace(/^@/, "")}`;
    }
  }

  return "@creator";
}

function engagementRate(item: DatasetItem): string {
  const views = Math.max(1, Number(item.metrics?.views ?? 0));
  const likes = Number(item.metrics?.likes ?? 0);
  const comments = Number(item.metrics?.comments_count ?? 0);
  const shares = Number(item.metrics?.shares ?? 0);
  return `${(((likes + comments + shares) / views) * 100).toFixed(2)}%`;
}

function toNumber(value: unknown): number {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  return 0;
}

function average(values: number[]): number {
  if (values.length === 0) {
    return 0;
  }
  return values.reduce((sum, current) => sum + current, 0) / values.length;
}

function toPercentScale(value: number, maxExpected: number): number {
  if (maxExpected <= 0) {
    return 0;
  }
  const scaled = (value / maxExpected) * 100;
  return Math.max(0, Math.min(100, Math.round(scaled)));
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function estimateInputQuality(
  description: string,
  hashtags: string[],
  mentions: string[]
): { qualityScore: number; hook: number; clarity: number; retention: number } {
  const descriptionWords = description.split(/\s+/).filter(Boolean).length;
  const hashtagCount = hashtags.length;
  const mentionCount = mentions.length;

  const hook = clamp(
    48 + Math.min(descriptionWords, 40) * 0.7 + hashtagCount * 4,
    35,
    92
  );
  const clarity = clamp(
    44 + Math.min(descriptionWords, 60) * 0.8 + mentionCount * 3,
    30,
    94
  );
  const qualityScore = clamp((hook * 0.45 + clarity * 0.55) / 100, 0.35, 0.95);
  const retention = clamp(40 + hook * 0.35 + clarity * 0.25, 35, 95);

  return {
    qualityScore,
    hook: Math.round(hook),
    clarity: Math.round(clarity),
    retention: Math.round(retention)
  };
}

function getMetrics(item: DatasetItem | undefined): MetricSnapshot {
  const nested = item?.metrics;
  const views = Number(nested?.views ?? item?.views ?? 0);
  const likes = Number(nested?.likes ?? item?.likes ?? 0);
  const comments = Number(nested?.comments_count ?? item?.comments_count ?? 0);
  const shares = Number(nested?.shares ?? item?.shares ?? 0);

  return {
    views: Number.isFinite(views) ? views : 0,
    likes: Number.isFinite(likes) ? likes : 0,
    comments_count: Number.isFinite(comments) ? comments : 0,
    shares: Number.isFinite(shares) ? shares : 0
  };
}

function buildFallbackReport(
  dataset: DatasetItem[],
  payload: GenerateReportRequestBody,
  model: string,
  deepseekSummary: string,
  deepseekRecommendations: string[]
) {
  const seedId = typeof payload.seed_video_id === "string" ? payload.seed_video_id : "s001";
  const description = typeof payload.description === "string" ? payload.description : "";
  const mentions = normalizeArray(payload.mentions);
  const hashtags = normalizeTagValues(normalizeArray(payload.hashtags));

  const seed = dataset.find((item) => item.video_id === seedId) ?? dataset[0];
  const totalCandidates = dataset.filter((item) => item.video_id !== seed?.video_id).length;
  const candidatePool = dataset.filter((item) => item.video_id !== seed?.video_id);
  const comparables = candidatePool
    .slice(0, 5)
    .map((item, index) => {
      const metrics = getMetrics(item);
      return {
      id: `${item.video_id || `cmp-${index + 1}`}-${index + 1}`,
      caption: removeEmoji(item.caption ?? "Comparable TikTok video"),
      author: toAuthorLabel(item.author),
      video_url: typeof item.video_url === "string" ? item.video_url : "",
      thumbnail_url: typeof item.thumbnail_url === "string" ? item.thumbnail_url : "",
      hashtags: Array.isArray(item.hashtags) ? item.hashtags.slice(0, 5) : [],
      similarity: Number((0.92 - index * 0.08).toFixed(2)),
      metrics: {
        views: metrics.views,
        likes: metrics.likes,
        comments_count: metrics.comments_count,
        shares: metrics.shares,
        engagement_rate: engagementRate(item)
      },
      matched_keywords: ["hook", "clarity", "CTA"],
      observations: [
        "Opening line communicates the payoff fast.",
        "Editing rhythm keeps attention high.",
        "CTA is explicit and easy to act on."
      ]
    };
    });

  const candidateViews = candidatePool.map((item) => toNumber(getMetrics(item).views));
  const candidateLikes = candidatePool.map((item) => toNumber(getMetrics(item).likes));
  const candidateComments = candidatePool.map((item) => toNumber(getMetrics(item).comments_count));
  const candidateShares = candidatePool.map((item) => toNumber(getMetrics(item).shares));
  const candidateEngagementRates = candidatePool.map((item) => {
    const metrics = getMetrics(item);
    const views = toNumber(metrics.views);
    const likes = toNumber(metrics.likes);
    const comments = toNumber(metrics.comments_count);
    const shares = toNumber(metrics.shares);
    return views > 0 ? ((likes + comments + shares) / views) * 100 : 0;
  });

  const avgViews = average(candidateViews);
  const avgLikes = average(candidateLikes);
  const avgComments = average(candidateComments);
  const avgShares = average(candidateShares);
  const avgEngagement = average(candidateEngagementRates);

  const totalAvgInteractions = Math.max(1, avgLikes + avgComments + avgShares);
  const avgLikeRatio = avgLikes / totalAvgInteractions;
  const avgCommentRatio = avgComments / totalAvgInteractions;
  const avgShareRatio = avgShares / totalAvgInteractions;

  const quality = estimateInputQuality(description, hashtags, mentions);
  const projectedViews = Math.round(avgViews * (0.55 + quality.qualityScore * 0.9));
  const projectedEngagement = clamp(avgEngagement * (0.78 + quality.qualityScore * 0.5), 2, 25);
  const projectedInteractions = Math.round(projectedViews * (projectedEngagement / 100));
  const projectedLikes = Math.round(projectedInteractions * avgLikeRatio);
  const projectedComments = Math.round(projectedInteractions * avgCommentRatio);
  const projectedShares = Math.round(projectedInteractions * avgShareRatio);

  const maxViews = Math.max(projectedViews, avgViews, 1);
  const maxLikes = Math.max(projectedLikes, avgLikes, 1);
  const maxComments = Math.max(projectedComments, avgComments, 1);
  const maxShares = Math.max(projectedShares, avgShares, 1);
  const maxEngagement = Math.max(projectedEngagement, avgEngagement, 0.1);

  const retention = quality.retention;
  const hook = quality.hook;
  const clarity = quality.clarity;

  const strengths: string[] = [];
  const improvements: string[] = [];

  if (hook >= 68) {
    strengths.push("Your hook already has strong potential for first-second attention.");
  } else {
    improvements.push("Make the first sentence more outcome-driven to increase scroll-stop power.");
  }

  if (clarity >= 70) {
    strengths.push("Your message is clear and should be easy to understand for cold viewers.");
  } else {
    improvements.push("Reduce ambiguity: one promise, one audience, one CTA in the first 6 seconds.");
  }

  if (hashtags.length >= 3 && hashtags.length <= 6) {
    strengths.push("Your hashtag strategy is within a healthy range for discoverability.");
  } else {
    improvements.push("Use 3-6 focused hashtags aligned to the exact value proposition.");
  }

  return {
    header: {
      title: "TikTok Performance Forecast",
      subtitle: "Comparative signal report",
      badges: {
        candidates_k: totalCandidates,
        model,
        mode: "serverless"
      },
      disclaimer: "Estimates are directional and based on historical comparables."
    },
    executive_summary: {
      metrics: [
        { id: "retention-estimated", label: "Estimated retention", value: `${retention}%` },
        { id: "hook-strength", label: "Hook strength", value: `${hook}%` },
        { id: "message-clarity", label: "Message clarity", value: `${clarity}%` }
      ],
      extracted_keywords: ["tiktok", "hook", "retention", "cta", "storytelling"],
      meaning_points: [
        ...strengths.slice(0, 2),
        ...(deepseekRecommendations.length > 0
          ? deepseekRecommendations.slice(0, 1)
          : improvements.slice(0, 1))
      ],
      summary_text: deepseekSummary || "Your concept is viable; the largest upside is a sharper first-second hook and a more explicit CTA."
    },
    comparables,
    direct_comparison: {
      rows: [
        {
          id: "engagement-rate",
          label: "Engagement rate",
          your_value_label: `${projectedEngagement.toFixed(2)}%`,
          comparable_value_label: `${avgEngagement.toFixed(2)}%`,
          your_value_pct: toPercentScale(projectedEngagement, maxEngagement),
          comparable_value_pct: toPercentScale(avgEngagement, maxEngagement)
        },
        {
          id: "likes",
          label: "Likes",
          your_value_label: `${projectedLikes}`,
          comparable_value_label: `${Math.round(avgLikes)}`,
          your_value_pct: toPercentScale(projectedLikes, maxLikes),
          comparable_value_pct: toPercentScale(avgLikes, maxLikes)
        },
        {
          id: "comments",
          label: "Comments",
          your_value_label: `${projectedComments}`,
          comparable_value_label: `${Math.round(avgComments)}`,
          your_value_pct: toPercentScale(projectedComments, maxComments),
          comparable_value_pct: toPercentScale(avgComments, maxComments)
        },
        {
          id: "shares",
          label: "Shares",
          your_value_label: `${projectedShares}`,
          comparable_value_label: `${Math.round(avgShares)}`,
          your_value_pct: toPercentScale(projectedShares, maxShares),
          comparable_value_pct: toPercentScale(avgShares, maxShares)
        },
        {
          id: "views",
          label: "Views",
          your_value_label: `${projectedViews}`,
          comparable_value_label: `${Math.round(avgViews)}`,
          your_value_pct: toPercentScale(projectedViews, maxViews),
          comparable_value_pct: toPercentScale(avgViews, maxViews)
        }
      ],
      note: "Target outperforming benchmark in hook clarity and CTA specificity."
    },
    relevant_comments: {
      items: [
        {
          id: "rc1",
          text: "Loved the idea, but get to the result faster.",
          topic: "hook",
          polarity: "Negative",
          relevance_note: "Indicates first seconds can be tighter."
        },
        {
          id: "rc2",
          text: "This is useful, where can I get the template?",
          topic: "cta",
          polarity: "Question",
          relevance_note: "High intent; add explicit CTA destination."
        }
      ],
      disclaimer: "Comment sample is indicative, not exhaustive."
    },
    recommendations: {
      items: (deepseekRecommendations.length ? deepseekRecommendations : [
        "Rewrite the first sentence to promise one measurable outcome.",
        "Show proof (before/after or result) within 2 seconds.",
        "Use one explicit CTA with a single next action."
      ]).slice(0, 3).map((text, index) => ({
        id: `rec-${index + 1}`,
        title: removeEmoji(text),
        priority: index === 0 ? "High" : "Medium",
        effort: index === 0 ? "Low" : "Medium",
        evidence: "Consistent with top comparable patterns in this dataset."
      }))
    }
  };
}

function parseDeepSeekEnhancement(raw: string): { summary: string; recommendations: string[] } {
  try {
    const parsed = extractFirstJsonObject(raw) as {
      summary?: unknown;
      recommendations?: unknown;
    };

    const summary = typeof parsed.summary === "string" ? parsed.summary.trim() : "";
    const recommendations = Array.isArray(parsed.recommendations)
      ? parsed.recommendations.filter((item) => typeof item === "string").map((item) => item.trim()).filter(Boolean)
      : [];

    return { summary, recommendations };
  } catch {
    return { summary: "", recommendations: [] };
  }
}

export default async function handler(request: ApiRequest, response: ApiResponse) {
  if (request.method !== "POST") {
    response.setHeader("Allow", "POST");
    response.status(405).json({ error: "Method not allowed." });
    return;
  }

  try {
    const body = (request.body ?? {}) as GenerateReportRequestBody;
    const dataset = await loadDatasetFromFile();
    const apiKey = (process.env.DEEPSEEK_API_KEY ?? "").trim();
    const model = (process.env.DEEPSEEK_MODEL ?? "deepseek-reasoner").trim() || "deepseek-reasoner";
    const baseUrl = (process.env.DEEPSEEK_BASE_URL ?? "https://api.deepseek.com").trim() || "https://api.deepseek.com";
    const deepSeekEnabled = Boolean(apiKey) && apiKey !== "your_key_here";

    let deepseekSummary = "";
    let deepseekRecommendations: string[] = [];

    if (deepSeekEnabled) {
      try {
        const prompt = {
          description: typeof body.description === "string" ? body.description : "",
          mentions: normalizeArray(body.mentions),
          hashtags: normalizeTagValues(normalizeArray(body.hashtags))
        };

        const providerResponse = await fetch(`${baseUrl}/chat/completions`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${apiKey}`
          },
          body: JSON.stringify({
            model,
            temperature: 0.3,
            messages: [
              {
                role: "system",
                content:
                  "Return only JSON with keys: summary (string), recommendations (array of 3 concise strings). No markdown."
              },
              {
                role: "user",
                content: JSON.stringify(prompt)
              }
            ]
          })
        });

        if (providerResponse.ok) {
          const completion = (await providerResponse.json()) as {
            choices?: Array<{ message?: { content?: unknown } }>;
          };

          const rawContent = extractTextContent(completion.choices?.[0]?.message?.content ?? "");
          const parsed = parseDeepSeekEnhancement(rawContent);
          deepseekSummary = parsed.summary;
          deepseekRecommendations = parsed.recommendations;
          response.setHeader("x-report-source", "deepseek");
        } else {
          response.setHeader("x-report-source", "baseline-local-provider-error");
        }
      } catch {
        response.setHeader("x-report-source", "baseline-local-provider-error");
      }
    } else {
      response.setHeader("x-report-source", "baseline-local-no-key");
    }

    const report = buildFallbackReport(dataset, body, model, deepseekSummary, deepseekRecommendations);
    response.json({ report });
  } catch (error) {
    console.error(error);
    response.status(500).json({ error: "The report could not be generated right now." });
  }
}
