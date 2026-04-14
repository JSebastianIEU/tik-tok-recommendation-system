import {
  useEffect,
  useRef,
  useState,
  type ClipboardEvent,
  type FormEvent,
  type KeyboardEvent,
  type MouseEvent
} from "react";
import { createPortal } from "react-dom";
import type { UploadFormValues } from "../../../services/contracts/models";
import {
  suggestHashtags,
  type HashtagSuggestion
} from "../../../services/api/hashtagApi";
import { TagInputOverlay, type TagEditorMode } from "./TagInputOverlay";

interface UploadFormProps {
  values: UploadFormValues;
  disabled: boolean;
  isAnalyzing?: boolean;
  error: string | null;
  onDescriptionChange: (value: string) => void;
  onMentionsChange: (values: string[]) => void;
  onHashtagsChange: (values: string[]) => void;
  onObjectiveChange: (value: UploadFormValues["objective"]) => void;
  onAudienceChange: (value: string) => void;
  onContentTypeChange: (value: UploadFormValues["content_type"]) => void;
  onPrimaryCtaChange: (value: UploadFormValues["primary_cta"]) => void;
  onLocaleChange: (value: string) => void;
  onSubmit: () => Promise<void>;
}

interface TagFieldProps {
  label: string;
  symbol: "@" | "#";
  items: string[];
  disabled: boolean;
  placeholder?: string;
  suggestions?: HashtagSuggestion[];
  loadingSuggestions?: boolean;
  onPickSuggestion?: (hashtag: string) => void;
  onOpen: (initialValue?: string) => void;
  onRemove: (index: number) => void;
}

function normalizeForCompare(value: string): string {
  return value.trim().toLowerCase();
}

function TagField(props: TagFieldProps): JSX.Element {
  const { label, symbol, items, disabled, placeholder, suggestions, loadingSuggestions, onPickSuggestion, onOpen, onRemove } = props;
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
            placeholder={placeholder ?? (items.length === 0 ? "no items" : "add item")}
            aria-label={`Type ${label}`}
          />

          {loadingSuggestions && (
            <span className="suggestion-loading">suggesting...</span>
          )}

          {suggestions && suggestions.length > 0 && (
            <>
              <span className="suggestion-divider" />
              <span className="suggestion-label">tap to add:</span>
              {suggestions.map((s) => (
                <button
                  key={s.hashtag}
                  type="button"
                  className="suggestion-chip"
                  disabled={disabled}
                  onClick={() => onPickSuggestion?.(s.hashtag)}
                >
                  + {symbol}{s.hashtag.replace(/^#/, "")}
                </button>
              ))}
            </>
          )}
        </div>
      </div>
    </div>
  );
}

export function UploadForm(props: UploadFormProps): JSX.Element {
  const {
    values,
    disabled,
    isAnalyzing = false,
    error,
    onDescriptionChange,
    onMentionsChange,
    onHashtagsChange,
    onObjectiveChange,
    onAudienceChange,
    onContentTypeChange,
    onPrimaryCtaChange,
    onLocaleChange,
    onSubmit
  } = props;

  const [editorMode, setEditorMode] = useState<TagEditorMode | null>(null);
  const [editorInitialValue, setEditorInitialValue] = useState<string>("");
  const [hashtagSuggestions, setHashtagSuggestions] = useState<HashtagSuggestion[]>([]);
  const [suggestingHashtags, setSuggestingHashtags] = useState(false);

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

  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const lastFetchedDesc = useRef<string>("");
  const isFetchingRef = useRef(false);

  useEffect(() => {
    const desc = values.description.trim();

    // Clear pending timer on every description change
    if (debounceRef.current) {
      clearTimeout(debounceRef.current);
      debounceRef.current = null;
    }

    // Don't clear existing suggestions — only clear if description is emptied
    if (!desc || desc.length < 5) {
      if (desc.length === 0) {
        setHashtagSuggestions([]);
        lastFetchedDesc.current = "";
      }
      return;
    }

    // Already fetched for this exact description
    if (desc === lastFetchedDesc.current) {
      return;
    }

    debounceRef.current = setTimeout(() => {
      if (isFetchingRef.current) {
        return;
      }
      isFetchingRef.current = true;
      lastFetchedDesc.current = desc;
      setSuggestingHashtags(true);

      suggestHashtags({
        caption: desc,
        top_n: 10,
        exclude_tags: values.hashtags.map((h) =>
          h.startsWith("#") ? h : `#${h}`
        ),
        include_neighbours: false
      })
        .then((result) => {
          setHashtagSuggestions(result.suggestions ?? []);
        })
        .catch(() => {
          // keep existing suggestions on error
        })
        .finally(() => {
          setSuggestingHashtags(false);
          isFetchingRef.current = false;
        });
    }, 3000);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [values.description]);

  const handlePickSuggestion = (tag: string): void => {
    const clean = tag.replace(/^#/, "").trim().toLowerCase();
    if (!clean) {
      return;
    }

    const exists = values.hashtags.some(
      (item) => normalizeForCompare(item) === clean
    );

    if (!exists) {
      onHashtagsChange([...values.hashtags, clean]);
    }

    setHashtagSuggestions((prev) =>
      prev.filter((s) => s.hashtag.replace(/^#/, "").toLowerCase() !== clean)
    );
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
          placeholder={isAnalyzing && values.hashtags.length === 0 ? "Suggesting hashtags..." : undefined}
          suggestions={hashtagSuggestions}
          loadingSuggestions={suggestingHashtags}
          onPickSuggestion={handlePickSuggestion}
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
            placeholder={isAnalyzing ? "AI is analyzing your video and writing a description..." : "Describe your video content"}
            onChange={(event) => onDescriptionChange(event.target.value)}
          />
        </div>

        <div className="form-field">
          <label htmlFor="objective" className="field-label">
            objective
          </label>
          <select
            id="objective"
            className="glass-textarea"
            value={values.objective}
            disabled={disabled}
            onChange={(event) => onObjectiveChange(event.target.value as UploadFormValues["objective"])}
          >
            <option value="reach">reach</option>
            <option value="engagement">engagement</option>
            <option value="conversion">conversion</option>
            <option value="community">community</option>
          </select>
        </div>

        <div className="form-field">
          <label htmlFor="content-type" className="field-label">
            content type
          </label>
          <select
            id="content-type"
            className="glass-textarea"
            value={values.content_type}
            disabled={disabled}
            onChange={(event) =>
              onContentTypeChange(event.target.value as UploadFormValues["content_type"])
            }
          >
            <option value="tutorial">tutorial</option>
            <option value="story">story</option>
            <option value="reaction">reaction</option>
            <option value="showcase">showcase</option>
            <option value="opinion">opinion</option>
          </select>
        </div>

        <div className="form-field">
          <label htmlFor="primary-cta" className="field-label">
            primary cta
          </label>
          <select
            id="primary-cta"
            className="glass-textarea"
            value={values.primary_cta}
            disabled={disabled}
            onChange={(event) =>
              onPrimaryCtaChange(event.target.value as UploadFormValues["primary_cta"])
            }
          >
            <option value="none">none</option>
            <option value="follow">follow</option>
            <option value="comment">comment</option>
            <option value="save">save</option>
            <option value="share">share</option>
            <option value="link_click">link_click</option>
          </select>
        </div>

        <div className="form-field">
          <label htmlFor="audience" className="field-label">
            audience
          </label>
          <input
            id="audience"
            className="tag-inline-input"
            value={values.audience}
            disabled={disabled}
            onChange={(event) => onAudienceChange(event.target.value)}
          />
        </div>

        <div className="form-field">
          <label htmlFor="locale" className="field-label">
            locale
          </label>
          <input
            id="locale"
            className="tag-inline-input"
            value={values.locale}
            disabled={disabled}
            onChange={(event) => onLocaleChange(event.target.value)}
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

      {editorMode && createPortal(
        <TagInputOverlay
          mode={editorMode}
          initialValue={editorInitialValue}
          onConfirm={handleAddTag}
          onClose={closeEditor}
        />,
        document.body
      )}
    </section>
  );
}
