import {
  useEffect,
  useRef,
  useState,
  type FormEvent,
  type KeyboardEvent,
  type MouseEvent
} from "react";

export type TagEditorMode = "mentions" | "hashtags";

interface TagInputOverlayProps {
  mode: TagEditorMode | null;
  initialValue: string;
  onConfirm: (value: string, mode: TagEditorMode) => void;
  onClose: () => void;
}

function normalizeValue(rawValue: string): string {
  return rawValue.trim().replace(/\s+/g, "");
}

export function TagInputOverlay(props: TagInputOverlayProps): JSX.Element | null {
  const { mode, initialValue, onConfirm, onClose } = props;
  const [draft, setDraft] = useState<string>("");
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!mode) {
      return;
    }

    setDraft(initialValue);
    window.setTimeout(() => {
      inputRef.current?.focus();
    }, 10);
  }, [mode, initialValue]);

  if (!mode) {
    return null;
  }

  const title = mode === "mentions" ? "New mention" : "New hashtag";
  const symbol = mode === "mentions" ? "@" : "#";

  const handleBackdropClick = (): void => {
    onClose();
  };

  const handleDialogClick = (event: MouseEvent<HTMLDivElement>): void => {
    event.stopPropagation();
  };

  const handleSubmit = (event: FormEvent<HTMLFormElement>): void => {
    event.preventDefault();
    const normalized = normalizeValue(draft);

    if (normalized) {
      onConfirm(normalized, mode);
    }

    onClose();
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLInputElement>): void => {
    if (event.key === "Escape") {
      onClose();
    }
  };

  return (
    <div className="tag-overlay" onClick={handleBackdropClick}>
      <div
        className="tag-overlay-dialog glass-card"
        onClick={handleDialogClick}
        role="dialog"
        aria-modal="true"
        aria-label={title}
      >
        <h3 className="tag-overlay-title">{title}</h3>
        <form onSubmit={handleSubmit} className="tag-overlay-form">
          <span className="tag-overlay-prefix" aria-hidden="true">
            {symbol}
          </span>
          <input
            ref={inputRef}
            className="tag-overlay-input"
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={mode === "mentions" ? "username" : "topic"}
            aria-label={title}
          />
        </form>
        <p className="tag-overlay-hint">Press Enter to add. Click outside to cancel.</p>
      </div>
    </div>
  );
}
