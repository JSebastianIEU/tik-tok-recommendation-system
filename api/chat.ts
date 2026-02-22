// @ts-nocheck

interface ChatRequestBody {
  report?: unknown;
  question?: string;
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

function hasValidReportShape(value: unknown): boolean {
  if (!value || typeof value !== "object") {
    return false;
  }

  const report = value as Record<string, unknown>;
  return (
    typeof report.header === "object" &&
    typeof report.executive_summary === "object" &&
    Array.isArray((report as { comparables?: unknown }).comparables)
  );
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

function buildLocalFallbackAnswer(question: string): string {
  const q = question.toLowerCase();
  if (q.includes("hashtag")) {
    return "Use 3-5 hashtags: 2 niche, 2 medium, 1 broad. Keep them directly tied to the promised outcome in your hook.";
  }

  if (q.includes("retention") || q.includes("hook")) {
    return "Retention improves when the first 2 seconds show the outcome before explanation. Shorten setup, show proof first.";
  }

  return "Focus next on hook clarity, tighter edit pacing, and one explicit CTA with a single action.";
}

export default async function handler(request: ApiRequest, response: ApiResponse) {
  if (request.method !== "POST") {
    response.setHeader("Allow", "POST");
    response.status(405).json({ error: "Method not allowed." });
    return;
  }

  try {
    const body = (request.body ?? {}) as ChatRequestBody;
    const question = typeof body.question === "string" ? body.question.trim() : "";

    if (!question) {
      response.status(400).json({ error: "A question is required." });
      return;
    }

    if (!hasValidReportShape(body.report)) {
      response.status(400).json({ error: "The provided report payload is invalid." });
      return;
    }

    const apiKey = (process.env.DEEPSEEK_API_KEY ?? "").trim();
    const model = (process.env.DEEPSEEK_MODEL ?? "deepseek-reasoner").trim() || "deepseek-reasoner";
    const baseUrl = (process.env.DEEPSEEK_BASE_URL ?? "https://api.deepseek.com").trim() || "https://api.deepseek.com";
    const deepSeekEnabled = Boolean(apiKey) && apiKey !== "your_key_here";

    if (!deepSeekEnabled) {
      response.setHeader("x-chat-source", "baseline-local-no-key");
      response.json({
        answer: removeEmoji(buildLocalFallbackAnswer(question))
      });
      return;
    }

    try {
      const providerResponse = await fetch(`${baseUrl}/chat/completions`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${apiKey}`
        },
        body: JSON.stringify({
          model,
          temperature: 0.4,
          messages: [
            {
              role: "system",
              content:
                "You are a strategic content assistant. Reply in English plain text with concrete recommendations, no emojis, and no generic filler."
            },
            {
              role: "user",
              content: JSON.stringify({
                question,
                summary:
                  typeof (body.report as any)?.executive_summary?.summary_text === "string"
                    ? (body.report as any).executive_summary.summary_text
                    : "",
                metrics: Array.isArray((body.report as any)?.executive_summary?.metrics)
                  ? (body.report as any).executive_summary.metrics
                  : []
              })
            }
          ]
        })
      });

      if (!providerResponse.ok) {
        throw new Error(`DeepSeek error: ${providerResponse.status}`);
      }

      const completion = (await providerResponse.json()) as {
        choices?: Array<{ message?: { content?: unknown } }>;
      };

      const rawContent = extractTextContent(completion.choices?.[0]?.message?.content ?? "");
      const answer = removeEmoji(rawContent || "I do not have an answer right now.");

      response.setHeader("x-chat-source", "deepseek");
      response.json({ answer });
    } catch (providerError) {
      console.error(providerError);
      response.setHeader("x-chat-source", "baseline-local-provider-error");
      response.json({
        answer: removeEmoji(buildLocalFallbackAnswer(question))
      });
    }
  } catch (error) {
    console.error(error);
    response.status(500).json({ error: "The chat request could not be completed." });
  }
}
