import { useState, type FormEvent } from "react";

interface ChatComposerProps {
  disabled: boolean;
  onSend: (message: string) => Promise<boolean>;
}

export function ChatComposer(props: ChatComposerProps): JSX.Element {
  const { disabled, onSend } = props;
  const [draft, setDraft] = useState<string>("");

  const handleSubmit = async (event: FormEvent<HTMLFormElement>): Promise<void> => {
    event.preventDefault();
    if (disabled) {
      return;
    }

    const trimmed = draft.trim();
    if (!trimmed) {
      return;
    }

    const sent = await onSend(trimmed);
    if (sent) {
      setDraft("");
    }
  };

  return (
    <form className="chat-composer" onSubmit={handleSubmit}>
      <label htmlFor="chat-input" className="sr-only">
        Type your message
      </label>
      <input
        id="chat-input"
        className="chat-input"
        value={draft}
        disabled={disabled}
        placeholder="Ask me anything about this video..."
        onChange={(event) => setDraft(event.target.value)}
      />
      <button
        type="submit"
        className="chat-send-button"
        disabled={disabled || !draft.trim()}
      >
        Send
      </button>
    </form>
  );
}
