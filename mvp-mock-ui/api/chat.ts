// @ts-nocheck
import { buildLocalChatAnswer } from "../server/fallback/buildLocalChatAnswer";
import { buildChatPrompt } from "../server/prompts/buildChatPrompt";
import { validateReportOutput } from "../server/validation/validateReportOutput";
import type { ReportOutput } from "../src/features/report/types";

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

    if (!validateReportOutput(body.report)) {
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
        answer: removeEmoji(buildLocalChatAnswer(body.report as ReportOutput, question))
      });
      return;
    }

    const chatPrompt = buildChatPrompt({
      report: body.report as ReportOutput,
      question
    });

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
              content: chatPrompt
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
        answer: removeEmoji(buildLocalChatAnswer(body.report as ReportOutput, question))
      });
    }
  } catch (error) {
    console.error(error);
    response.status(500).json({ error: "The chat request could not be completed." });
  }
}
