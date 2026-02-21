from typing import List
from pydantic import BaseModel, HttpUrl, Field
from datetime import datetime


class Author(BaseModel):
    author_id: str
    username: str
    followers: int = Field(..., ge=0)


class Audio(BaseModel):
    audio_id: str
    audio_title: str


class VideoMeta(BaseModel):
    duration_seconds: int = Field(..., ge=1)
    language: str


class Comment(BaseModel):
    comment_id: str
    text: str
    likes: int = Field(..., ge=0)
    created_at: datetime


class TikTokPost(BaseModel):
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

    class Config:
        arbitrary_types_allowed = True


__all__ = ["Author", "Audio", "VideoMeta", "Comment", "TikTokPost"]

# TODO: extend with engagement-rate helpers and stricter validators
# (language code shape, hashtag format, minimal comment count checks).
