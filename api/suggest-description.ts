import type { VercelRequest, VercelResponse } from "@vercel/node";

const DEEPSEEK_API_KEY = process.env.DEEPSEEK_API_KEY ?? "";
const DEEPSEEK_MODEL = process.env.DEEPSEEK_MODEL ?? "deepseek-chat";
const DEEPSEEK_BASE_URL =
  process.env.DEEPSEEK_BASE_URL ?? "https://api.deepseek.com";

export const config = { maxDuration: 30 };

interface SuggestBody {
  videoAnalysis?: Record<string, unknown> | null;
  objective?: string;
  content_type?: string;
  locale?: string;
}

function buildAnalysisContext(va: Record<string, unknown>): string {
  const parts: string[] = [];

  if (va.transcript && typeof va.transcript === "string") {
    parts.push(`Audio transcript: "${va.transcript.slice(0, 800)}"`);
  }

  if (va.ocr_text && typeof va.ocr_text === "string") {
    parts.push(`On-screen text (OCR): "${va.ocr_text.slice(0, 400)}"`);
  }

  if (va.video_caption && typeof va.video_caption === "string") {
    parts.push(`VLM scene description: "${va.video_caption.slice(0, 400)}"`);
  }

  if (va.keyTopics && Array.isArray(va.keyTopics) && va.keyTopics.length > 0) {
    parts.push(`Key topics detected: ${(va.keyTopics as string[]).join(", ")}`);
  }

  if (va.duration_seconds) {
    parts.push(`Video duration: ${va.duration_seconds}s`);
  }

  if (va.visual_features && typeof va.visual_features === "object") {
    const vf = va.visual_features as Record<string, unknown>;
    const details: string[] = [];
    if (vf.face_count) details.push(`${vf.face_count} face(s) detected`);
    if (vf.resolution) details.push(`resolution: ${vf.resolution}`);
    if (vf.aspect_ratio) details.push(`aspect ratio: ${vf.aspect_ratio}`);
    if (details.length) parts.push(`Visual: ${details.join(", ")}`);
  }

  if (va.timeline && Array.isArray(va.timeline)) {
    const frames = va.timeline as Array<Record<string, unknown>>;
    const sceneCaptions = frames
      .filter((f) => f.caption || f.is_scene_change)
      .slice(0, 5)
      .map(
        (f) =>
          `[${Number(f.timestamp_sec || 0).toFixed(1)}s] ${f.caption || "scene change"}`
      );
    if (sceneCaptions.length) {
      parts.push(`Key frames:\n${sceneCaptions.join("\n")}`);
    }
  }

  if (va.detected_language) {
    parts.push(`Detected language: ${va.detected_language}`);
  }

  return parts.join("\n\n");
}

export default async function handler(
  req: VercelRequest,
  res: VercelResponse
) {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");

  if (req.method === "OPTIONS") return res.status(204).end();
  if (req.method !== "POST")
    return res.status(405).json({ error: "Method not allowed" });

  const body = req.body as SuggestBody;

  if (!body.videoAnalysis) {
    return res
      .status(400)
      .json({ error: "videoAnalysis payload is required." });
  }

  if (!DEEPSEEK_API_KEY) {
    return res.json({ description: "", hashtags: [] });
  }

  const analysisContext = buildAnalysisContext(body.videoAnalysis);
  if (!analysisContext.trim()) {
    return res.json({ description: "", hashtags: [] });
  }

  const objective = body.objective || "engagement";
  const contentType = body.content_type || "video";
  const locale = body.locale || "en";

  const prompt = `You are a TikTok content strategist. Based on the video analysis data below, write:

1. A compelling TikTok video description (max 150 characters, optimized for ${objective}). The description should hook viewers, be natural, and match the content. Do NOT use hashtags in the description itself.

2. A list of 5-8 relevant hashtags (without the # symbol).

VIDEO ANALYSIS:
${analysisContext}

Content type: ${contentType}
Optimization goal: ${objective}
Locale: ${locale}

Respond in this exact JSON format only, no markdown:
{"description": "your suggested description here", "hashtags": ["tag1", "tag2", "tag3"]}`;

  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 20_000);

    const response = await fetch(`${DEEPSEEK_BASE_URL}/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${DEEPSEEK_API_KEY}`,
      },
      signal: controller.signal,
      body: JSON.stringify({
        model: DEEPSEEK_MODEL,
        ...(DEEPSEEK_MODEL.includes("reasoner") ? {} : { temperature: 0.7 }),
        max_tokens: 512,
        messages: [
          ...(DEEPSEEK_MODEL.includes("reasoner")
            ? []
            : [
                {
                  role: "system" as const,
                  content:
                    "You are a TikTok content expert. You write viral descriptions. Respond only with valid JSON.",
                },
              ]),
          {
            role: "user",
            content: DEEPSEEK_MODEL.includes("reasoner")
              ? "You are a TikTok content expert. Respond only with valid JSON.\n\n" +
                prompt
              : prompt,
          },
        ],
      }),
    });

    clearTimeout(timeout);

    if (!response.ok) {
      console.error(
        "DeepSeek suggest-description error:",
        response.status,
        await response.text().catch(() => "")
      );
      return res.json({ description: "", hashtags: [] });
    }

    const data = (await response.json()) as {
      choices: { message: { content: string } }[];
    };
    const raw = data.choices?.[0]?.message?.content?.trim() ?? "";

    // Parse JSON from response (handle markdown code blocks)
    let cleaned = raw;
    if (cleaned.startsWith("```")) {
      cleaned = cleaned.replace(/^```(?:json)?\s*/, "").replace(/\s*```$/, "");
    }

    try {
      const parsed = JSON.parse(cleaned) as {
        description?: string;
        hashtags?: string[];
      };
      return res.json({
        description: parsed.description || "",
        hashtags: Array.isArray(parsed.hashtags) ? parsed.hashtags : [],
      });
    } catch {
      // If JSON parse fails, try to extract description from raw text
      return res.json({ description: raw.slice(0, 200), hashtags: [] });
    }
  } catch (err) {
    console.error("suggest-description error:", err);
    return res.json({ description: "", hashtags: [] });
  }
}
