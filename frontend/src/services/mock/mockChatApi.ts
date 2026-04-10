import type { IChatService } from "../contracts/IChatService";
import type { ChatMessage, ChatRequest } from "../contracts/models";
import { CHAT_KEYWORD_RESPONSES } from "./fixtures";

function randomDelay(minMs: number, maxMs: number): number {
  return Math.floor(Math.random() * (maxMs - minMs + 1)) + minMs;
}

function wait(ms: number): Promise<void> {
  return new Promise((resolve) => {
    window.setTimeout(resolve, ms);
  });
}

function createMessageId(): string {
  return `assistant-${Date.now()}-${Math.floor(Math.random() * 1000)}`;
}

function normalizeText(value: string): string {
  return value
    .toLowerCase()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "");
}

function pickAnswer(message: string): string {
  const normalized = normalizeText(message);

  if (normalized.includes("hashtag")) {
    return CHAT_KEYWORD_RESPONSES.hashtags;
  }

  if (normalized.includes("retention")) {
    return CHAT_KEYWORD_RESPONSES.retention;
  }

  if (normalized.includes("summary")) {
    return CHAT_KEYWORD_RESPONSES.summary;
  }

  return CHAT_KEYWORD_RESPONSES.fallback;
}

export class MockChatService implements IChatService {
  public async sendMessage(request: ChatRequest): Promise<ChatMessage> {
    await wait(randomDelay(650, 1050));

    return {
      id: createMessageId(),
      role: "assistant",
      content: pickAnswer(request.question),
      timestamp: new Date().toISOString()
    };
  }
}
