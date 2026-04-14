"""Dense frame sampling + scene detection using Decord.

Samples frames at 1fps, detects scene changes via histogram difference,
computes motion score from frame-to-frame pixel deltas.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

from ..models import FrameExtractionResult, KeyFrame


def _histogram_diff(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    """Compute normalized histogram difference between two RGB frames."""
    hist_a = np.histogram(frame_a, bins=64, range=(0, 256))[0].astype(np.float32)
    hist_b = np.histogram(frame_b, bins=64, range=(0, 256))[0].astype(np.float32)
    hist_a /= hist_a.sum() + 1e-8
    hist_b /= hist_b.sum() + 1e-8
    return float(np.sum(np.abs(hist_a - hist_b)))


def extract_frames(
    video_path: Path,
    target_fps: float = 1.0,
    scene_threshold: float = 0.35,
) -> FrameExtractionResult:
    """Extract frames at target_fps, detect scenes, compute motion."""
    import decord

    vr = decord.VideoReader(str(video_path))
    total_frames = len(vr)
    native_fps = float(vr.get_avg_fps())
    duration = total_frames / max(native_fps, 1.0)

    # Sample at target_fps
    interval = max(1, int(native_fps / target_fps))
    sample_indices = list(range(0, total_frames, interval))

    # Limit to max 60 frames (for very long videos)
    if len(sample_indices) > 60:
        step = len(sample_indices) // 60
        sample_indices = sample_indices[::step][:60]

    frames_np = vr.get_batch(sample_indices).asnumpy()  # (N, H, W, 3)

    # Scene detection + motion score
    scene_cuts = 0
    motion_diffs: List[float] = []
    scene_cut_indices: List[int] = []

    for i in range(1, len(frames_np)):
        diff = _histogram_diff(frames_np[i - 1], frames_np[i])
        motion_diffs.append(diff)
        if diff > scene_threshold:
            scene_cuts += 1
            scene_cut_indices.append(i)

    motion_score = float(np.mean(motion_diffs)) if motion_diffs else 0.0
    # Normalize to 0-1 (typical range 0.0-0.8)
    motion_score = min(motion_score / 0.6, 1.0)

    # Build keyframes
    keyframes: List[KeyFrame] = []
    for i, frame_idx in enumerate(sample_indices):
        timestamp = frame_idx / max(native_fps, 1.0)
        img = Image.fromarray(frames_np[i])
        keyframes.append(KeyFrame(timestamp_sec=timestamp, image=img, index=i))

    return FrameExtractionResult(
        keyframes=keyframes,
        duration_seconds=round(duration, 2),
        fps=round(native_fps, 2),
        total_frames=total_frames,
        estimated_scene_cuts=scene_cuts,
        visual_motion_score=round(motion_score, 4),
        scene_cut_indices=scene_cut_indices,
    )
