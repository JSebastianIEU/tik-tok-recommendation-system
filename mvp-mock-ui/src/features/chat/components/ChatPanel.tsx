import { ChatComposer } from "./ChatComposer";
import { ChatMessageList } from "./ChatMessageList";
import { useChatPanel } from "../hooks/useChatPanel";
import type { IChatService } from "../../../services/contracts/IChatService";
import type { ReportOutput } from "../../report/types";

interface ChatPanelProps {
  report: ReportOutput | null;
  chatService: IChatService;
  resetKey: number;
}

export function ChatPanel(props: ChatPanelProps): JSX.Element {
  const { report, chatService, resetKey } = props;

  const { messages, isThinking, error, sendMessage } = useChatPanel({
    chatService,
    report,
    resetKey
  });

  if (!report) {
    return (
      <section className="glass-card chat-panel placeholder-panel">
        <h3 className="panel-title">video chat</h3>
        <p className="placeholder-text">
          Upload a video first to unlock chat.
        </p>
      </section>
    );
  }

  return (
    <section className="glass-card chat-panel">
      <h3 className="panel-title">video chat</h3>

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
  );
}
