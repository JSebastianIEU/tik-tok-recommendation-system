import type { VercelRequest, VercelResponse } from "@vercel/node";

const DEEPSEEK_API_KEY = process.env.DEEPSEEK_API_KEY ?? "";
const DEEPSEEK_MODEL = process.env.DEEPSEEK_MODEL ?? "deepseek-chat";
const DEEPSEEK_BASE_URL = process.env.DEEPSEEK_BASE_URL ?? "https://api.deepseek.com";

interface ChatBody {
  question?: string;
  report?: Record<string, unknown> | null;
  videoAnalysis?: Record<string, unknown> | null;
}

function buildContext(body: ChatBody): string {
  const parts: string[] = [];

  if (body.report) {
    const r = body.report as Record<string, unknown>;
    if (r.comparables && Array.isArray(r.comparables)) {
      const top = (r.comparables as Record<string, unknown>[]).slice(0, 5);
      parts.push(
        "## Recommendation Report (top 5 comparables)\n" +
          top
            .map((c, i) => {
              const caption = c.caption ?? c.candidate_id ?? `#${i + 1}`;
              const score = typeof c.score === "number" ? c.score.toFixed(3) : "n/a";
              const hashtags = Array.isArray(c.hashtags) ? (c.hashtags as string[]).join(", ") : "";
              return `${i + 1}. [${score}] ${caption}${hashtags ? ` | tags: ${hashtags}` : ""}`;
            })
            .join("\n")
      );
    }
    if (typeof r.summary === "string") {
      parts.push(`## Report Summary\n${r.summary}`);
    }
  }

  if (body.videoAnalysis) {
    const va = body.videoAnalysis as Record<string, unknown>;
    const snippets: string[] = [];
    if (va.video_caption) snippets.push(`Caption: ${va.video_caption}`);
    if (va.transcript) snippets.push(`Transcript: ${va.transcript}`);
    if (va.duration_seconds) snippets.push(`Duration: ${va.duration_seconds}s`);
    if (va.hashtags && Array.isArray(va.hashtags))
      snippets.push(`Hashtags: ${(va.hashtags as string[]).join(", ")}`);
    if (snippets.length > 0) {
      parts.push("## Video Analysis\n" + snippets.join("\n"));
    }
  }

  return parts.join("\n\n");
}

export default async function handler(req: VercelRequest, res: VercelResponse) {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");

  if (req.method === "OPTIONS") {
    return res.status(204).end();
  }
  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method not allowed" });
  }

  const body = req.body as ChatBody;
  const question = typeof body.question === "string" ? body.question.trim() : "";

  if (!question) {
    return res.status(400).json({ error: "A question is required." });
  }

  if (!DEEPSEEK_API_KEY) {
    return res.json({
      answer: "Upload a video and generate a report to start chatting.",
      sources: [],
    });
  }

  const context = buildContext(body);
  const userContent = context
    ? `${context}\n\n---\n\nUser question: ${question}`
    : question;

  try {
    const response = await fetch(`${DEEPSEEK_BASE_URL}/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${DEEPSEEK_API_KEY}`,
      },
      body: JSON.stringify({
        model: DEEPSEEK_MODEL,
        temperature: 0.4,
        max_tokens: 2048,
        messages: [
          {
            role: "system",
            content:
              "You are a TikTok video content strategist with access to frame-by-frame video analysis, " +
              "a recommendation report with comparable videos, and a searchable corpus of 13,000+ TikTok videos. " +
              "Reference specific data when answering. " +
              "Be concrete and actionable. No emojis. No generic filler. " +
              "When discussing the user's video, cite analysis data (timestamps, relevance scores, scene changes). " +
              "When suggesting content strategy, reference comparable videos and their engagement patterns.",
          },
          { role: "user", content: userContent },
        ],
      }),
    });

    if (!response.ok) {
      const text = await response.text();
      console.error("DeepSeek error:", response.status, text);
      return res.json({
        answer: "The AI assistant is currently unavailable. Please try again.",
        sources: [],
      });
    }

    const data = (await response.json()) as {
      choices: { message: { content: string } }[];
    };
    const answer = data.choices?.[0]?.message?.content?.trim() ?? "No response.";

    return res.json({ answer, sources: ["deepseek"] });
  } catch (err) {
    console.error("Chat error:", err);
    return res.json({
      answer: "The AI assistant is currently unavailable. Please try again.",
      sources: [],
    });
  }
}
