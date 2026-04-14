import { MockChatService } from "../mock/mockChatApi";
import type { IChatService } from "../contracts/IChatService";
import type { ChatMessage, ChatRequest } from "../contracts/models";
import { buildApiUrl, MOCK_ONLY_MODE } from "./runtimeConfig";

const CHAT_API_URL = buildApiUrl("/chat");

interface ChatApiResponse {
  answer: string;
  sources?: string[];
  evidence_refs?: string[];
}

function createAssistantMessage(content: string): ChatMessage {
  return {
    id: `assistant-${Date.now()}-${Math.floor(Math.random() * 1000)}`,
    role: "assistant",
    content,
    timestamp: new Date().toISOString()
  };
}

export class ApiChatService implements IChatService {
  private readonly mockService = new MockChatService();

  public async sendMessage(request: ChatRequest): Promise<ChatMessage> {
    if (MOCK_ONLY_MODE) {
      return this.mockService.sendMessage(request);
    }

    let response: Response;

    try {
      response = await fetch(CHAT_API_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          report: request.report,
          question: request.question,
          videoAnalysis: request.videoAnalysis ?? null,
          history: request.history,
          objective_effective: request.report.meta.objective_effective
        })
      });
    } catch {
      return this.mockService.sendMessage(request);
    }

    if (!response.ok) {
      return this.mockService.sendMessage(request);
    }

    const parsed = (await response.json()) as ChatApiResponse;
    return createAssistantMessage(parsed.answer);
  }
}
