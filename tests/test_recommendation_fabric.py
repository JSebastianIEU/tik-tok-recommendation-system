from __future__ import annotations

from datetime import datetime, timezone

from src.recommendation.fabric import FeatureFabric


def _dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def test_fabric_is_deterministic_for_same_input() -> None:
    fabric = FeatureFabric()
    payload = {
        "video_id": "v1",
        "as_of_time": _dt("2026-03-24T12:00:00Z"),
        "caption": "How to build a content hook that converts",
        "hashtags": ["#marketing", "#content"],
        "keywords": ["hook", "retention", "cta"],
        "duration_seconds": 32,
        "content_type": "tutorial",
    }
    first = fabric.extract(payload)
    second = fabric.extract(payload)
    assert first.registry_signature == second.registry_signature
    assert first.generated_at == second.generated_at
    assert first.text.token_count == second.text.token_count
    assert first.structure.hook_timing_seconds == second.structure.hook_timing_seconds
    assert first.trace_ids == second.trace_ids


def test_fabric_marks_optional_modalities_missing_when_hints_absent() -> None:
    fabric = FeatureFabric()
    output = fabric.extract(
        {
            "video_id": "v2",
            "as_of_time": _dt("2026-03-24T12:00:00Z"),
            "caption": "Quick growth tips",
            "hashtags": ["#growth"],
            "keywords": ["growth"],
            "duration_seconds": 24,
            "content_type": "opinion",
        }
    )
    assert output.audio.speech_ratio is None
    assert "speech_ratio" in output.audio.missing
    assert output.visual.shot_change_rate is None
    assert "shot_change_rate" in output.visual.missing


def test_fabric_segment_windows_respect_content_type_overrides() -> None:
    fabric = FeatureFabric()
    tutorial = fabric.extract(
        {
            "video_id": "t1",
            "as_of_time": _dt("2026-03-24T12:00:00Z"),
            "caption": "Tutorial caption",
            "hashtags": [],
            "keywords": [],
            "duration_seconds": 40,
            "content_type": "tutorial",
        }
    )
    reaction = fabric.extract(
        {
            "video_id": "r1",
            "as_of_time": _dt("2026-03-24T12:00:00Z"),
            "caption": "Reaction caption",
            "hashtags": [],
            "keywords": [],
            "duration_seconds": 40,
            "content_type": "reaction",
        }
    )
    assert tutorial.structure.hook_timing_seconds != reaction.structure.hook_timing_seconds
    assert tutorial.structure.payoff_timing_seconds != reaction.structure.payoff_timing_seconds


def test_fabric_calibration_fit_and_reload() -> None:
    fabric = FeatureFabric()
    artifacts = fabric.fit_calibrators(
        {
            "text": [(0.1, 0.05), (0.3, 0.25), (0.6, 0.55), (0.9, 0.88)],
            "audio": [(0.1, 0.2), (0.5, 0.55), (0.8, 0.85)],
        }
    )
    assert "text" in artifacts
    reloaded = FeatureFabric(calibration_artifacts=artifacts)
    output = reloaded.extract(
        {
            "video_id": "cal-v1",
            "as_of_time": _dt("2026-03-24T12:00:00Z"),
            "caption": "Calibration test",
            "hashtags": ["#x"],
            "keywords": ["x"],
            "duration_seconds": 20,
            "content_type": "tutorial",
        }
    )
    assert output.text.confidence.calibration_version.startswith("text-cal.v2")
