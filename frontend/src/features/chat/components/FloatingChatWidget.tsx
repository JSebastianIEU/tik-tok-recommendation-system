import { useState } from "react";
import type { IChatService } from "../../../services/contracts/IChatService";
import type { VideoAnalysisResult } from "../../../services/contracts/models";
import type { ReportOutput } from "../../report/types";
import { useChatPanel } from "../hooks/useChatPanel";
import { ChatComposer } from "./ChatComposer";
import { ChatMessageList } from "./ChatMessageList";

interface FloatingChatWidgetProps {
  report: ReportOutput | null;
  videoAnalysis?: VideoAnalysisResult | null;
  chatService: IChatService;
  resetKey: number;
}

function ChatBubbleIcon(): JSX.Element {
  return (
    <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path d="M5 7H19V15H9L5 19V7Z" />
      <path d="M8 10H16" />
      <path d="M8 13H14" />
    </svg>
  );
}

function CloseIcon(): JSX.Element {
  return (
    <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path d="M6 6L18 18" />
      <path d="M18 6L6 18" />
    </svg>
  );
}

export function FloatingChatWidget(props: FloatingChatWidgetProps): JSX.Element | null {
  const { report, videoAnalysis, chatService, resetKey } = props;
  const [isOpen, setIsOpen] = useState<boolean>(false);

  const { messages, isThinking, error, sendMessage } = useChatPanel({
    chatService,
    report,
    videoAnalysis,
    resetKey
  });

  if (!report) {
    return null;
  }

  return (
    <div className={`floating-chat-widget ${isOpen ? "floating-chat-open" : ""}`}>
      {isOpen ? (
        <section className="glass-card floating-chat-panel">
          <header className="floating-chat-header">
            <h3>video chat</h3>
            <button
              type="button"
              className="floating-chat-close"
              onClick={() => setIsOpen(false)}
              aria-label="Close chat"
            >
              <CloseIcon />
            </button>
          </header>

          <ChatMessageList messages={messages} />

          {isThinking ? (
            <div className="chat-thinking" aria-live="polite">
              <div className="spinner spinner-small" aria-hidden="true" />
              <span>Thinking...</span>
            </div>
          ) : null}

          {error ? (
            <p className="inline-error" role="alert">
              {error}
            </p>
          ) : null}

          <ChatComposer disabled={isThinking} onSend={sendMessage} />
        </section>
      ) : (
        <button
          type="button"
          className="floating-chat-toggle"
          onClick={() => setIsOpen(true)}
          aria-label="Open video chat"
        >
          <ChatBubbleIcon />
        </button>
      )}
    </div>
  );
}
