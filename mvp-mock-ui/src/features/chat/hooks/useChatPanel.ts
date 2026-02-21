import { useEffect, useState } from "react";
import type { ReportOutput } from "../../report/types";
import type { IChatService } from "../../../services/contracts/IChatService";
import type { ChatMessage } from "../../../services/contracts/models";

interface UseChatPanelParams {
  chatService: IChatService;
  report: ReportOutput | null;
  resetKey: number;
}

interface UseChatPanelResult {
  messages: ChatMessage[];
  isThinking: boolean;
  error: string | null;
  sendMessage: (content: string) => Promise<boolean>;
}

const CHAT_ASSISTANT_WELCOME =
  "Report loaded. Ask me about comparables, recommendations, metrics, or editing strategy.";

function createClientMessageId(prefix: string): string {
  return `${prefix}-${Date.now()}-${Math.floor(Math.random() * 1000)}`;
}

export function useChatPanel(params: UseChatPanelParams): UseChatPanelResult {
  const { chatService, report, resetKey } = params;

  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isThinking, setIsThinking] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setMessages([]);
    setIsThinking(false);
    setError(null);
  }, [resetKey]);

  useEffect(() => {
    if (!report) {
      setMessages([]);
      return;
    }

    setMessages([
      {
        id: createClientMessageId("assistant-intro"),
        role: "assistant",
        content: CHAT_ASSISTANT_WELCOME,
        timestamp: new Date().toISOString()
      }
    ]);
  }, [report, resetKey]);

  const sendMessage = async (content: string): Promise<boolean> => {
    if (!report || isThinking) {
      return false;
    }

    const trimmed = content.trim();
    if (!trimmed) {
      return false;
    }

    const userMessage: ChatMessage = {
      id: createClientMessageId("user"),
      role: "user",
      content: trimmed,
      timestamp: new Date().toISOString()
    };

    const history = [...messages, userMessage];

    setMessages(history);
    setIsThinking(true);
    setError(null);

    try {
      const assistantReply = await chatService.sendMessage({
        question: trimmed,
        report,
        history
      });

      setMessages((previous) => [...previous, assistantReply]);
      return true;
    } catch (chatError) {
      console.error(chatError);
      setError("I could not answer right now. Please try again.");
      return false;
    } finally {
      setIsThinking(false);
    }
  };

  return {
    messages,
    isThinking,
    error,
    sendMessage
  };
}
