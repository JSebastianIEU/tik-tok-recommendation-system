"""Pydantic contracts for video analysis results."""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class SignalHints(BaseModel):
    """Audio/visual signal hints consumed by the FeatureFabric."""

    speech_seconds: float = Field(ge=0.0)
    music_seconds: float = Field(ge=0.0)
    tempo_bpm: float = Field(ge=0.0)
    audio_energy: float = Field(ge=0.0)
    estimated_scene_cuts: int = Field(ge=0)
    visual_motion_score: float = Field(ge=0.0)
    fps: float = Field(ge=0.0)


class VisualFeatures(BaseModel):
    """Visual features extracted from video frames."""

    dominant_colors: List[str] = Field(default_factory=list)
    avg_brightness: float = 0.0
    avg_saturation: float = 0.0
    avg_contrast: float = 0.0
    face_count: int = Field(default=0, ge=0)
    avg_face_area_ratio: float = 0.0
    aspect_ratio: str = ""
    resolution: str = ""
    blur_score: float = 0.0
    hook_motion_score: float = 0.0


class FrameTimelineEntry(BaseModel):
    """A single frame in the video timeline with analysis data."""

    timestamp_sec: float = Field(ge=0.0)
    thumbnail_b64: str = ""
    ocr_text: str = ""
    face_count: int = Field(default=0, ge=0)
    motion_score: float = 0.0
    is_scene_change: bool = False
    relevance_score: float = Field(default=0.5, ge=0.0, le=1.0)


class VideoAnalysisResponse(BaseModel):
    """Full analysis response returned by the /v1/video/analyze endpoint."""

    signal_hints: SignalHints
    transcript: str = ""
    ocr_text: str = ""
    video_caption: str = ""
    detected_language: str = ""
    keywords: List[str] = Field(default_factory=list)
    visual_features: Optional[VisualFeatures] = None
    timeline: List[FrameTimelineEntry] = Field(default_factory=list)
    duration_seconds: float = Field(ge=0.0)
    processing_time_seconds: float = Field(ge=0.0)
