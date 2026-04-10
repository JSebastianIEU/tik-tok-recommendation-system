from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterable, List, Sequence


_URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
_HASHTAG_RE = re.compile(r"(?<!\w)#([\w_]+)", re.UNICODE)
_MENTION_RE = re.compile(r"(?<!\w)@([\w_.]+)", re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"\b\w+\b", re.UNICODE)
_COMBINING_MARK_RE = re.compile(r"[\u0300-\u036f]+")
_INLINE_ENTITY_RE = re.compile(r"(?<!\w)([#@])[\w_.]+", re.UNICODE)


@dataclass(frozen=True)
class ProcessedText:
    raw_text: str
    semantic_text: str
    lexical_text: str
    hashtags: List[str]
    mentions: List[str]
    emoji_tokens: List[str]
    semantic_tokens: List[str]
    lexical_tokens: List[str]


def _safe_text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _normalize_whitespace(value: str) -> str:
    return _WHITESPACE_RE.sub(" ", value).strip()


def _normalize_identifier(value: Any) -> str:
    text = unicodedata.normalize("NFKC", _safe_text(value)).lower().strip()
    if not text:
        return ""
    text = _COMBINING_MARK_RE.sub("", text)
    text = re.sub(r"[^\w.-]+", "", text, flags=re.UNICODE)
    return text


def _unique(values: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = str(value).strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        out.append(cleaned)
    return out


def extract_hashtags(text: Any) -> List[str]:
    raw = _safe_text(text)
    if not raw:
        return []
    return _unique(
        _normalize_identifier(match.group(1))
        for match in _HASHTAG_RE.finditer(raw)
        if _normalize_identifier(match.group(1))
    )


def extract_mentions(text: Any) -> List[str]:
    raw = _safe_text(text)
    if not raw:
        return []
    return _unique(
        _normalize_identifier(match.group(1))
        for match in _MENTION_RE.finditer(raw)
        if _normalize_identifier(match.group(1))
    )


def _emoji_token_for_char(char: str) -> str | None:
    if not char or char in {"\u200d", "\ufe0f"}:
        return None
    codepoint = ord(char)
    category = unicodedata.category(char)
    in_emoji_block = (
        0x1F300 <= codepoint <= 0x1FAFF
        or 0x2600 <= codepoint <= 0x27BF
        or 0x1F1E6 <= codepoint <= 0x1F1FF
    )
    if not in_emoji_block and category not in {"So", "Sk"}:
        return None
    name = unicodedata.name(char, "").lower().strip()
    if not name or name.startswith("variation selector"):
        return None
    return re.sub(r"[^a-z0-9_]+", "_", name.replace("-", "_").replace(" ", "_")).strip("_")


def extract_emoji_tokens(text: Any) -> List[str]:
    raw = _safe_text(text)
    if not raw:
        return []
    return _unique(
        token
        for char in raw
        for token in [_emoji_token_for_char(char)]
        if token
    )


def demojize_text(text: Any) -> str:
    raw = _safe_text(text)
    if not raw:
        return ""
    parts: List[str] = []
    for char in raw:
        token = _emoji_token_for_char(char)
        if token:
            parts.extend([" ", token, " "])
            continue
        if char in {"\u200d", "\ufe0f"}:
            continue
        parts.append(char)
    return _normalize_whitespace("".join(parts))


def _normalize_lexical_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).lower()
    normalized = re.sub(r"[^\w\s-]+", " ", normalized, flags=re.UNICODE)
    return _normalize_whitespace(normalized)


def tokenize_text(text: Any) -> List[str]:
    raw = _safe_text(text)
    if not raw:
        return []
    return _unique(
        token
        for token in _TOKEN_RE.findall(raw.lower())
        if len(token) >= 2
    )


def process_text(
    *,
    text: Any,
    explicit_hashtags: Sequence[Any] = (),
    explicit_mentions: Sequence[Any] = (),
) -> ProcessedText:
    raw_text = _safe_text(text)
    hashtags = _unique(
        [
            *[
                _normalize_identifier(item).lstrip("#")
                for item in explicit_hashtags
                if _normalize_identifier(item).lstrip("#")
            ],
            *extract_hashtags(raw_text),
        ]
    )
    mentions = _unique(
        [
            *[
                _normalize_identifier(item).lstrip("@")
                for item in explicit_mentions
                if _normalize_identifier(item).lstrip("@")
            ],
            *extract_mentions(raw_text),
        ]
    )
    no_urls = _URL_RE.sub(" ", raw_text)
    semantic_text = _INLINE_ENTITY_RE.sub(" ", no_urls)
    semantic_text = _normalize_whitespace(semantic_text)
    emoji_tokens = extract_emoji_tokens(semantic_text)
    lexical_text = _normalize_lexical_text(demojize_text(semantic_text))
    semantic_tokens = tokenize_text(_normalize_lexical_text(semantic_text))
    lexical_tokens = tokenize_text(lexical_text)
    return ProcessedText(
        raw_text=raw_text,
        semantic_text=semantic_text,
        lexical_text=lexical_text,
        hashtags=hashtags,
        mentions=mentions,
        emoji_tokens=emoji_tokens,
        semantic_tokens=semantic_tokens,
        lexical_tokens=lexical_tokens,
    )
