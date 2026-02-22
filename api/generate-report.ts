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
  hashtags?: string[];
  metrics?: {
    views?: number;
    likes?: number;
    comments_count?: number;
    shares?: number;
  };
  author?: unknown;
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
  const comparables = dataset
    .filter((item) => item.video_id !== seed?.video_id)
    .slice(0, 5)
    .map((item, index) => ({
      id: `${item.video_id || `cmp-${index + 1}`}-${index + 1}`,
      caption: removeEmoji(item.caption ?? "Comparable TikTok video"),
      author: toAuthorLabel(item.author),
      video_url: typeof item.video_url === "string" ? item.video_url : "",
      thumbnail_url:
        typeof (item as Record<string, unknown>).thumbnail_url === "string"
          ? String((item as Record<string, unknown>).thumbnail_url)
          : "",
      hashtags: Array.isArray(item.hashtags) ? item.hashtags.slice(0, 5) : [],
      similarity: Number((0.92 - index * 0.08).toFixed(2)),
      metrics: {
        views: Number(item.metrics?.views ?? 0),
        likes: Number(item.metrics?.likes ?? 0),
        comments_count: Number(item.metrics?.comments_count ?? 0),
        shares: Number(item.metrics?.shares ?? 0),
        engagement_rate: engagementRate(item)
      },
      matched_keywords: ["hook", "clarity", "CTA"],
      observations: [
        "Opening line communicates the payoff fast.",
        "Editing rhythm keeps attention high.",
        "CTA is explicit and easy to act on."
      ]
    }));

  const retention = Math.min(96, 62 + Math.round(description.split(/\s+/).filter(Boolean).length * 0.6));
  const hook = Math.min(95, 58 + hashtags.length * 6 + mentions.length * 2);
  const clarity = Math.min(96, 60 + Math.round(description.length / 14));

  return {
    header: {
      title: "TikTok Performance Forecast",
      subtitle: "Comparative signal report",
      badges: {
        candidates_k: comparables.length,
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
        "Opening promise should be concrete in the first 2 seconds.",
        "Keep one message per clip to avoid cognitive load.",
        "Close with one clear CTA tied to the promised outcome."
      ],
      summary_text: deepseekSummary || "Your concept is viable; the largest upside is a sharper first-second hook and a more explicit CTA."
    },
    comparables,
    direct_comparison: {
      rows: [
        {
          id: "engagement-rate",
          label: "Engagement rate",
          your_value_label: "6.20%",
          comparable_value_label: "7.10%",
          your_value_pct: 62,
          comparable_value_pct: 71
        },
        {
          id: "likes-per-view",
          label: "Likes per view",
          your_value_label: "0.051",
          comparable_value_label: "0.062",
          your_value_pct: 51,
          comparable_value_pct: 62
        },
        {
          id: "hashtag-density",
          label: "Hashtag density",
          your_value_label: "0.13",
          comparable_value_label: "0.11",
          your_value_pct: 65,
          comparable_value_pct: 55
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
