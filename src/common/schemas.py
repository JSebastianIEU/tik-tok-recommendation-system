"""Pydantic models for **mock / local JSONL** TikTok-shaped posts.

These types are used by ``validate_record``, ``mock_generator``, and baseline docs.
They are **not** the production recommender contract; see ``src/recommendation/contracts.py``.

Engagement helpers (:func:`compute_engagement_total`, :func:`compute_engagement_rate`)
are shared with ``src/baseline/baseline_stats.py`` so reports and schema stay aligned.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import List

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator, HttpUrl

# ISO-like language tags: en, ja, zh, pt-br (normalized to lowercase)
_LANGUAGE_RE = re.compile(r"^[a-z]{2,3}([_-][a-z]{2,4})?$")
# TikTok-style hashtag: leading #, no whitespace or second #
_HASHTAG_RE = re.compile(r"^#[^\s#]+$")


def compute_engagement_total(*, likes: int, comments_count: int, shares: int) -> int:
    """likes + comments_count + shares (same as :attr:`TikTokPost.engagement_total`)."""
    return int(likes) + int(comments_count) + int(shares)


def compute_engagement_rate(
    *, likes: int, comments_count: int, shares: int, views: int
) -> float:
    """(likes + comments_count + shares) / max(views, 1)."""
    return compute_engagement_total(
        likes=likes, comments_count=comments_count, shares=shares
    ) / max(int(views), 1)


class Author(BaseModel):
    model_config = ConfigDict(extra="forbid")

    author_id: str
    username: str
    followers: int = Field(..., ge=0)


class Audio(BaseModel):
    model_config = ConfigDict(extra="forbid")

    audio_id: str
    audio_title: str


class VideoMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    duration_seconds: int = Field(..., ge=1)
    language: str

    @field_validator("language")
    @classmethod
    def normalize_language(cls, value: str) -> str:
        text = value.strip().lower().replace("_", "-")
        if not _LANGUAGE_RE.match(text):
            raise ValueError(
                f"language must be an ISO-like code (e.g. en, ja, pt-br); got {value!r}"
            )
        return text


class Comment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    comment_id: str
    text: str
    likes: int = Field(..., ge=0)
    created_at: datetime


class TikTokPost(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    video_id: str
    video_url: HttpUrl
    caption: str
    hashtags: List[str]
    keywords: List[str]
    search_query: str
    posted_at: datetime
    likes: int = Field(..., ge=0)
    comments_count: int = Field(..., ge=0)
    shares: int = Field(..., ge=0)
    views: int = Field(..., ge=0)
    author: Author
    audio: Audio
    video_meta: VideoMeta
    comments: List[Comment]

    @field_validator("hashtags")
    @classmethod
    def normalize_hashtags(cls, values: List[str]) -> List[str]:
        out: List[str] = []
        for raw in values:
            tag = raw.strip()
            if not tag.startswith("#"):
                tag = "#" + tag
            if not _HASHTAG_RE.match(tag):
                raise ValueError(
                    f"hashtag must be non-empty, start with #, and contain no spaces: {raw!r}"
                )
            out.append(tag)
        return out

    @computed_field
    @property
    def engagement_total(self) -> int:
        return compute_engagement_total(
            likes=self.likes,
            comments_count=self.comments_count,
            shares=self.shares,
        )

    @computed_field
    @property
    def engagement_rate(self) -> float:
        return compute_engagement_rate(
            likes=self.likes,
            comments_count=self.comments_count,
            shares=self.shares,
            views=self.views,
        )


__all__ = [
    "Author",
    "Audio",
    "VideoMeta",
    "Comment",
    "TikTokPost",
    "compute_engagement_total",
    "compute_engagement_rate",
]
