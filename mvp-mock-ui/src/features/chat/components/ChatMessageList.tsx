import type { ChatMessage } from "../../../services/contracts/models";

interface ChatMessageListProps {
  messages: ChatMessage[];
}

export function ChatMessageList(props: ChatMessageListProps): JSX.Element {
  const { messages } = props;

  return (
    <ul className="chat-message-list">
      {messages.map((message) => (
        <li
          key={message.id}
          className={`chat-message chat-message-${message.role}`}
          aria-label={message.role === "assistant" ? "Assistant" : "User"}
        >
          <p>{message.content}</p>
        </li>
      ))}
    </ul>
  );
}
