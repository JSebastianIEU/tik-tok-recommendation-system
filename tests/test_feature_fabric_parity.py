from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.recommendation.fabric import FeatureFabric


def _dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def test_python_fabric_extracts_stable_signals_on_fixed_fixture():
    repo_root = Path(__file__).resolve().parent.parent
    fixture = json.loads(
        (repo_root / "tests" / "fixtures" / "fabric_parity_fixture.json").read_text(
            encoding="utf-8"
        )
    )
    py_out = FeatureFabric().extract(
        {
            "video_id": "fixture-video",
            "as_of_time": _dt("2026-03-24T12:00:00Z"),
            "caption": fixture["description"],
            "hashtags": [f"#{item}" for item in fixture["hashtags"]],
            "keywords": fixture["hashtags"],
            "transcript_text": fixture["signal_hints"]["transcript_text"],
            "ocr_text": fixture["signal_hints"]["ocr_text"],
            "duration_seconds": fixture["signal_hints"]["duration_seconds"],
            "content_type": fixture["content_type"],
            "hints": fixture["signal_hints"],
        }
    )
    assert py_out.text.token_count >= 8
    assert py_out.text.unique_token_count >= 6
    assert 0.0 <= py_out.text.clarity_score <= 1.0
    assert 0.0 <= py_out.structure.hook_timing_seconds <= py_out.structure.payoff_timing_seconds
    assert py_out.audio.speech_ratio is not None
    assert 0.0 <= py_out.audio.speech_ratio <= 1.0
    assert py_out.visual.shot_change_rate is not None
    assert py_out.visual.shot_change_rate >= 0.0
