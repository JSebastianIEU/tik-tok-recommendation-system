import type { ChatMessage, ChatRequest } from "./models";

export interface IChatService {
  sendMessage(request: ChatRequest): Promise<ChatMessage>;
}
