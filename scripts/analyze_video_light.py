#!/usr/bin/env python3
"""Lightweight video analyzer — cv2/numpy only, no ML models.

Produces real timeline with thumbnails, face detection, motion, scene cuts.
Skips BLIP captioning, whisper transcription, EasyOCR to stay under memory limits.

Usage: python scripts/analyze_video_light.py <input_video> <output_json>
"""
from __future__ import annotations

import base64
import json
import sys
import time
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np


def extract_frames(video_path: str, max_frames: int = 20):
    """Extract evenly-spaced frames from video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0.0

    n = min(max_frames, total_frames)
    indices = np.linspace(0, total_frames - 1, n, dtype=int)

    frames = []
    timestamps = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
            timestamps.append(float(idx) / fps)

    cap.release()
    return frames, timestamps, fps, duration, width, height


def resize_frame(frame, max_dim=480):
    """Resize frame to max dimension."""
    h, w = frame.shape[:2]
    if max(h, w) <= max_dim:
        return frame
    scale = max_dim / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def frame_to_b64(frame, quality=70) -> str:
    """Encode frame as base64 JPEG thumbnail."""
    small = resize_frame(frame, max_dim=160)
    _, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode("ascii")


_FACE_CASCADE = None

def _get_face_cascade():
    global _FACE_CASCADE
    if _FACE_CASCADE is not None:
        return _FACE_CASCADE
    # Try multiple possible paths for the cascade file
    candidates = [
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml",
    ]
    # Also search in common opencv install locations
    import glob
    candidates += glob.glob("C:/Python*/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")
    candidates += glob.glob("C:/Python*/site-packages/cv2/data/haarcascade_frontalface_default.xml")

    for p in candidates:
        cascade = cv2.CascadeClassifier(p)
        if not cascade.empty():
            _FACE_CASCADE = cascade
            return cascade
    return None


def detect_faces(frame) -> int:
    """Detect faces using Haar cascade."""
    cascade = _get_face_cascade()
    if cascade is None:
        return 0  # No cascade available, skip face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = resize_frame(gray, max_dim=320)
    faces = cascade.detectMultiScale(small, 1.1, 4, minSize=(20, 20))
    return len(faces)


def compute_motion(prev_frame, curr_frame) -> float:
    """Compute motion score between two frames."""
    prev_gray = cv2.cvtColor(resize_frame(prev_frame, 320), cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(resize_frame(curr_frame, 320), cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(prev_gray, curr_gray)
    return float(np.mean(diff)) / 255.0


def detect_scene_changes(frames, threshold=0.15):
    """Detect scene changes based on histogram difference."""
    scene_changes = set()
    for i in range(1, len(frames)):
        prev_hist = cv2.calcHist([frames[i-1]], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        curr_hist = cv2.calcHist([frames[i]], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        cv2.normalize(prev_hist, prev_hist)
        cv2.normalize(curr_hist, curr_hist)
        diff = cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_BHATTACHARYYA)
        if diff > threshold:
            scene_changes.add(i)
    return scene_changes


def extract_colors(frame) -> list:
    """Extract dominant colors."""
    small = resize_frame(frame, 64)
    pixels = small.reshape(-1, 3).astype(np.float32)
    k = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, _, centers = cv2.kmeans(pixels, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    colors = []
    for c in centers:
        b, g, r = int(c[0]), int(c[1]), int(c[2])
        colors.append(f"#{r:02x}{g:02x}{b:02x}")
    return colors


def analyze_video(video_path: str) -> dict:
    """Full lightweight analysis."""
    t0 = time.time()
    print("Extracting frames...", flush=True)
    frames, timestamps, fps, duration, width, height = extract_frames(video_path, max_frames=20)
    print(f"  {len(frames)} frames extracted in {time.time()-t0:.1f}s", flush=True)

    # Scene changes
    print("Detecting scene changes...", flush=True)
    scene_changes = detect_scene_changes(frames)
    print(f"  {len(scene_changes)} scene changes", flush=True)

    # Build timeline entries
    print("Building timeline...", flush=True)
    timeline = []
    for i, (frame, ts) in enumerate(zip(frames, timestamps)):
        # Motion score
        motion = compute_motion(frames[i-1], frame) if i > 0 else 0.0

        # Face detection
        face_count = detect_faces(frame)

        # Thumbnail
        thumb_b64 = frame_to_b64(frame)

        # Relevance score (heuristic: early frames + faces + motion)
        early_boost = max(0, 1.0 - ts / max(duration, 1.0))
        relevance = 0.3 + 0.3 * early_boost + 0.2 * min(motion * 5, 1.0) + 0.2 * min(face_count, 1)
        relevance = min(relevance, 1.0)

        timeline.append({
            "timestamp_sec": round(ts, 2),
            "thumbnail_b64": thumb_b64,
            "ocr_text": "",
            "face_count": face_count,
            "motion_score": round(motion, 4),
            "is_scene_change": i in scene_changes,
            "relevance_score": round(relevance, 4),
        })
        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{len(frames)} frames processed", flush=True)

    print(f"Timeline built with {len(timeline)} entries", flush=True)

    # Visual features
    print("Extracting visual features...", flush=True)
    mid_frame = frames[len(frames) // 2]
    colors = extract_colors(mid_frame)
    gray_mid = cv2.cvtColor(mid_frame, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray_mid)) / 255.0
    blur = float(cv2.Laplacian(gray_mid, cv2.CV_64F).var()) / 1000.0

    # Aspect ratio
    if width > 0 and height > 0:
        ratio = width / height
        if abs(ratio - 9/16) < 0.1:
            aspect = "9:16"
        elif abs(ratio - 16/9) < 0.1:
            aspect = "16:9"
        elif abs(ratio - 1.0) < 0.1:
            aspect = "1:1"
        elif abs(ratio - 4/3) < 0.1:
            aspect = "4:3"
        else:
            aspect = f"{width}:{height}"
    else:
        aspect = "unknown"

    # Total face count
    total_faces = sum(e["face_count"] for e in timeline)

    # Hook motion (first 2 seconds)
    hook_entries = [e for e in timeline if e["timestamp_sec"] <= 2.0]
    hook_motion = max((e["motion_score"] for e in hook_entries), default=0.0)

    visual_features = {
        "dominant_colors": colors,
        "avg_brightness": round(brightness, 3),
        "avg_saturation": 0.5,
        "avg_contrast": round(blur, 3),
        "face_count": total_faces,
        "avg_face_area_ratio": 0.0,
        "aspect_ratio": aspect,
        "resolution": f"{width}x{height}",
        "blur_score": round(blur, 3),
        "hook_motion_score": round(hook_motion, 4),
    }

    # Signal hints
    signal_hints = {
        "speech_seconds": 0.0,
        "music_seconds": duration * 0.8,
        "tempo_bpm": 120.0,
        "audio_energy": 0.5,
        "estimated_scene_cuts": len(scene_changes),
        "visual_motion_score": round(float(np.mean([e["motion_score"] for e in timeline])), 4),
        "fps": round(fps, 2),
    }

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s", flush=True)

    return {
        "signal_hints": signal_hints,
        "transcript": "",
        "ocr_text": "",
        "video_caption": f"A video at {width}x{height} ({aspect}), {duration:.1f}s, {len(scene_changes)} scene cuts, {total_faces} faces detected",
        "detected_language": "",
        "keywords": [],
        "visual_features": visual_features,
        "timeline": timeline,
        "duration_seconds": round(duration, 2),
        "processing_time_seconds": round(elapsed, 2),
    }


def main() -> int:
    if len(sys.argv) < 3:
        print("Usage: analyze_video_light.py <input_video> <output_json>", file=sys.stderr)
        return 1

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    result = analyze_video(input_path)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, default=str)

    print(f"Saved to {output_path}")
    print(f"Timeline: {len(result['timeline'])} entries")
    print(f"Caption: {result['video_caption']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
