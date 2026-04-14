"""Speech-to-text transcription using faster-whisper.

Uses CTranslate2 backend with INT8 quantization for fast CPU inference.
Whisper tiny model: ~75MB, ~3-5s for 30-second audio on CPU.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..models import TranscriptionResult


def transcribe(audio_path: Path, whisper_model: Any) -> TranscriptionResult:
    """Transcribe audio file using a pre-loaded faster-whisper model."""
    segments_iter, info = whisper_model.transcribe(
        str(audio_path),
        beam_size=1,
        best_of=1,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )

    segments = []
    full_text_parts = []
    total_speech = 0.0

    for seg in segments_iter:
        segments.append({
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "text": seg.text.strip(),
        })
        full_text_parts.append(seg.text.strip())
        total_speech += seg.end - seg.start

    return TranscriptionResult(
        text=" ".join(full_text_parts),
        language=info.language or "en",
        speech_seconds=round(total_speech, 2),
        segments=segments,
    )
