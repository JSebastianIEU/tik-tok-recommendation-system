"""Video analysis (frames, audio, transcription, optional VLM) for recommender-side tooling.

The Node `/upload-video` ingest path uses `AssetAnalysisProvider` in `frontend/server/uploads/`;
this package powers Python `/v1/video/analyze` when the serving stack proxies multipart there.
"""

from .models import (
    FrameTimelineEntry,
    SignalHints,
    VideoAnalysisResponse,
    VisualFeatures,
)

__all__ = [
    "FrameTimelineEntry",
    "SignalHints",
    "VideoAnalysisResponse",
    "VisualFeatures",
]
