import {
  useState,
  type ClipboardEvent,
  type FormEvent,
  type KeyboardEvent,
  type MouseEvent
} from "react";
import type { UploadFormValues } from "../../../services/contracts/models";
import { TagInputOverlay, type TagEditorMode } from "./TagInputOverlay";

interface UploadFormProps {
  values: UploadFormValues;
  disabled: boolean;
  error: string | null;
  onDescriptionChange: (value: string) => void;
  onMentionsChange: (values: string[]) => void;
  onHashtagsChange: (values: string[]) => void;
  onSubmit: () => Promise<void>;
}

interface TagFieldProps {
  label: string;
  symbol: "@" | "#";
  items: string[];
  disabled: boolean;
  onOpen: (initialValue?: string) => void;
  onRemove: (index: number) => void;
}

function normalizeForCompare(value: string): string {
  return value.trim().toLowerCase();
}

function TagField(props: TagFieldProps): JSX.Element {
  const { label, symbol, items, disabled, onOpen, onRemove } = props;
  const [inlineDraft, setInlineDraft] = useState<string>("");

  const openFromTypedText = (typedValue: string): void => {
    const normalized = typedValue.trim();
    if (!normalized) {
      onOpen();
      return;
    }

    onOpen(normalized);
    setInlineDraft("");
  };

  const handleInlineKeyDown = (event: KeyboardEvent<HTMLInputElement>): void => {
    if (disabled) {
      return;
    }

    if (event.key === "Enter") {
      event.preventDefault();
      openFromTypedText(inlineDraft);
      return;
    }

    if (event.key === "Escape") {
      setInlineDraft("");
      return;
    }

    if (
      event.key.length === 1 &&
      !event.ctrlKey &&
      !event.metaKey &&
      !event.altKey
    ) {
      event.preventDefault();
      openFromTypedText(`${inlineDraft}${event.key}`);
    }
  };

  const handlePlusClick = (event: MouseEvent<HTMLButtonElement>): void => {
    event.stopPropagation();
    if (!disabled) {
      openFromTypedText(inlineDraft);
    }
  };

  const handlePaste = (event: ClipboardEvent<HTMLInputElement>): void => {
    if (disabled) {
      return;
    }

    event.preventDefault();
    const pasted = event.clipboardData.getData("text");
    openFromTypedText(pasted);
  };

  return (
    <div className="form-field">
      <label className="field-label">{label}</label>
      <div className="tag-field-box" aria-label={label}>
        <div className="tag-item-list" role="list" aria-label={`${label} list`}>
          {items.map((item, index) => (
            <span className="tag-chip-item" key={`${label}-${item}-${index}`} role="listitem">
              <span className="tag-chip-text">
                <span className="tag-chip-label">
                  {symbol}
                  {item}
                </span>
                <button
                  type="button"
                  className="tag-chip-remove"
                  onClick={() => onRemove(index)}
                  disabled={disabled}
                  aria-label={`Remove ${label} item`}
                >
                  x
                </button>
              </span>
            </span>
          ))}

          <button
            type="button"
            className="tag-add-inside"
            onClick={handlePlusClick}
            disabled={disabled}
            aria-label={`Add ${label}`}
          >
            +
          </button>

          <input
            className="tag-inline-input"
            value={inlineDraft}
            disabled={disabled}
            onChange={(event) => setInlineDraft(event.target.value)}
            onKeyDown={handleInlineKeyDown}
            onPaste={handlePaste}
            placeholder={items.length === 0 ? "no items" : "add item"}
            aria-label={`Type ${label}`}
          />
        </div>
      </div>
    </div>
  );
}

export function UploadForm(props: UploadFormProps): JSX.Element {
  const {
    values,
    disabled,
    error,
    onDescriptionChange,
    onMentionsChange,
    onHashtagsChange,
    onSubmit
  } = props;

  const [editorMode, setEditorMode] = useState<TagEditorMode | null>(null);
  const [editorInitialValue, setEditorInitialValue] = useState<string>("");

  const openEditor = (mode: TagEditorMode, initialValue = ""): void => {
    setEditorMode(mode);
    setEditorInitialValue(initialValue);
  };

  const closeEditor = (): void => {
    setEditorMode(null);
    setEditorInitialValue("");
  };

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    await onSubmit();
  };

  const handleAddTag = (value: string, mode: TagEditorMode): void => {
    const nextValue = normalizeForCompare(value);

    if (!nextValue) {
      return;
    }

    if (mode === "mentions") {
      const exists = values.mentions.some(
        (item) => normalizeForCompare(item) === nextValue
      );

      if (!exists) {
        onMentionsChange([...values.mentions, nextValue]);
      }

      return;
    }

    const exists = values.hashtags.some(
      (item) => normalizeForCompare(item) === nextValue
    );

    if (!exists) {
      onHashtagsChange([...values.hashtags, nextValue]);
    }
  };

  return (
    <section className="glass-card form-panel">
      <form onSubmit={handleSubmit} className="upload-form">
        <TagField
          label="mentions"
          symbol="@"
          items={values.mentions}
          disabled={disabled}
          onOpen={(initialValue) => openEditor("mentions", initialValue)}
          onRemove={(index) =>
            onMentionsChange(values.mentions.filter((_, itemIndex) => itemIndex !== index))
          }
        />

        <TagField
          label="hashtags"
          symbol="#"
          items={values.hashtags}
          disabled={disabled}
          onOpen={(initialValue) => openEditor("hashtags", initialValue)}
          onRemove={(index) =>
            onHashtagsChange(values.hashtags.filter((_, itemIndex) => itemIndex !== index))
          }
        />

        <div className="form-field form-field-description">
          <label htmlFor="description" className="field-label">
            description
          </label>
          <textarea
            id="description"
            className="glass-textarea"
            value={values.description}
            disabled={disabled}
            onChange={(event) => onDescriptionChange(event.target.value)}
          />
        </div>

        {error ? (
          <p className="inline-error" role="alert">
            {error}
          </p>
        ) : null}

        <button type="submit" className="upload-button" disabled={disabled}>
          upload
        </button>
      </form>

      <TagInputOverlay
        mode={editorMode}
        initialValue={editorInitialValue}
        onConfirm={handleAddTag}
        onClose={closeEditor}
      />
    </section>
  );
}
