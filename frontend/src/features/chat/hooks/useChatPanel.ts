import { useEffect, useState } from "react";
import { sendReportFeedback } from "../../../services/api/reportFeedbackApi";
import type { IChatService } from "../../../services/contracts/IChatService";
import type { ChatMessage, VideoAnalysisResult } from "../../../services/contracts/models";
import type { ReportOutput } from "../../report/types";

interface UseChatPanelParams {
  chatService: IChatService;
  report: ReportOutput | null;
  videoAnalysis?: VideoAnalysisResult | null;
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
  const { chatService, report, videoAnalysis, resetKey } = params;

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
      await sendReportFeedback({
        request_id: report.meta.request_id,
        event_name: "report_followup_asked",
        entity_type: "chat",
        section: "chat",
        objective_effective: report.meta.objective_effective,
        experiment_id: report.meta.experiment_id ?? undefined,
        variant: report.meta.variant ?? undefined,
        signal_strength: "medium",
        label_direction: "positive",
        metadata: {
          history_size: history.length
        }
      });

      const assistantReply = await chatService.sendMessage({
        question: trimmed,
        report,
        history,
        videoAnalysis: videoAnalysis ?? null,
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
