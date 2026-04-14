"""CLIP frame embeddings + Florence-2 scene captioning.

1. CLIP ViT-B/32 encodes ALL sampled frames (batch, ~1-2s for 30 frames on CPU)
2. K-Means clusters frames into unique scenes
3. Florence-2 captions representative frames with <DETAILED_CAPTION> task
4. Florence-2 also runs <OCR> on representative frames for on-screen text
5. CLIP text-matching identifies visual topics ("gym", "cooking", "dance")
6. Per-frame relevance scores computed from CLIP similarity to description
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from ..models import KeyFrame


VISUAL_CONCEPTS = [
    "person dancing", "cooking food", "gym workout", "makeup tutorial",
    "comedy skit", "vlog daily life", "music performance", "product review",
    "pet animal", "nature scenery", "sports activity", "fashion outfit",
    "art drawing painting", "gaming", "educational lecture", "motivational speech",
    "travel exploration", "reaction video", "challenge trend", "cleaning organizing",
    "car automotive", "baby toddler", "relationship couple", "unboxing haul",
]


@dataclass
class FrameTimelineEntry:
    """Per-frame data for the frontend timeline visualization."""
    timestamp_sec: float
    thumbnail_b64: str  # base64-encoded JPEG thumbnail
    caption: str
    relevance_score: float  # 0-1
    is_scene_change: bool
    cluster_id: int


@dataclass
class CaptionerResult:
    scene_captions: List[str]
    visual_topics: List[str]
    frame_embeddings: np.ndarray  # (N, embed_dim)
    cluster_labels: np.ndarray
    frame_timeline: List[Dict[str, Any]] = field(default_factory=list)
    ocr_texts: List[str] = field(default_factory=list)


def _frame_to_thumbnail_b64(image: Image.Image, max_size: int = 160) -> str:
    """Resize frame to thumbnail and encode as base64 JPEG."""
    thumb = image.copy()
    thumb.thumbnail((max_size, max_size), Image.LANCZOS)
    buf = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=70)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _cluster_frames(embeddings: np.ndarray, max_clusters: int = 6) -> np.ndarray:
    """Cluster frame embeddings to find unique scenes."""
    from sklearn.cluster import KMeans

    n_samples = embeddings.shape[0]
    n_clusters = min(max_clusters, max(1, n_samples // 3), n_samples)

    if n_clusters <= 1:
        return np.zeros(n_samples, dtype=int)

    kmeans = KMeans(n_clusters=n_clusters, n_init=3, max_iter=50, random_state=42)
    return kmeans.fit_predict(embeddings)


def _get_representative_frames(
    keyframes: List[KeyFrame],
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> List[Tuple[int, KeyFrame]]:
    """Pick the frame closest to each cluster center."""
    representatives = []
    for cluster_id in sorted(set(labels)):
        mask = labels == cluster_id
        cluster_embeddings = embeddings[mask]
        center = cluster_embeddings.mean(axis=0)
        distances = np.linalg.norm(cluster_embeddings - center, axis=1)
        best_local_idx = int(np.argmin(distances))
        global_indices = np.where(mask)[0]
        global_idx = int(global_indices[best_local_idx])
        representatives.append((global_idx, keyframes[global_idx]))
    return representatives


def _caption_frame(
    image: Image.Image,
    processor: Any,
    model: Any,
    model_type: str = "blip",
    device: str = "cpu",
) -> str:
    """Caption a single frame using the loaded model (Florence-2 or BLIP)."""
    import torch

    if model_type == "florence2":
        task = "<DETAILED_CAPTION>"
        inputs = processor(text=task, images=image, return_tensors="pt")
        if device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=100, num_beams=3, early_stopping=True)
        result = processor.batch_decode(output, skip_special_tokens=False)[0]
        parsed = processor.post_process_generation(result, task=task, image_size=image.size)
        if isinstance(parsed, dict) and task in parsed:
            return str(parsed[task]).strip()
        return result.strip()

    # BLIP-base
    inputs = processor(images=image, return_tensors="pt")
    if device == "cuda":
        inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=60, num_beams=3, early_stopping=True)
    return processor.decode(output[0], skip_special_tokens=True).strip()


def _ocr_frame(
    image: Image.Image,
    processor: Any,
    model: Any,
    model_type: str = "blip",
    device: str = "cpu",
) -> str:
    """Extract on-screen text (OCR). Only Florence-2 supports native OCR."""
    if model_type != "florence2":
        return ""
    import torch
    task = "<OCR>"
    inputs = processor(text=task, images=image, return_tensors="pt")
    if device == "cuda":
        inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=80)
    result = processor.batch_decode(output, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(result, task=task, image_size=image.size)
    if isinstance(parsed, dict) and task in parsed:
        text = str(parsed[task]).strip()
        return text if text.lower() not in ("", "<ocr>", "none") else ""
    return ""


def _detect_visual_topics(
    frame_embeddings: np.ndarray,
    clip_model: Any,
    top_k: int = 5,
) -> List[str]:
    """Match average frame embedding against concept vocabulary."""
    avg_embedding = frame_embeddings.mean(axis=0, keepdims=True)

    concept_embeddings = clip_model.encode(
        VISUAL_CONCEPTS, convert_to_numpy=True, normalize_embeddings=True,
    )

    avg_norm = avg_embedding / (np.linalg.norm(avg_embedding, axis=1, keepdims=True) + 1e-8)
    similarities = (avg_norm @ concept_embeddings.T).flatten()

    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [
        VISUAL_CONCEPTS[i]
        for i in top_indices
        if similarities[i] > 0.15
    ]


def _compute_relevance_scores(
    frame_embeddings: np.ndarray,
    clip_model: Any,
    description: str,
) -> np.ndarray:
    """Compute per-frame relevance as CLIP similarity to description + visual diversity."""
    n = frame_embeddings.shape[0]
    if n == 0:
        return np.array([])

    # Similarity to description (if provided)
    if description.strip():
        desc_emb = clip_model.encode(
            [description], convert_to_numpy=True, normalize_embeddings=True,
        )
        desc_sims = (frame_embeddings @ desc_emb.T).flatten()
    else:
        desc_sims = np.zeros(n)

    # Visual uniqueness: distance from mean embedding (more unique = more interesting)
    mean_emb = frame_embeddings.mean(axis=0, keepdims=True)
    uniqueness = np.linalg.norm(frame_embeddings - mean_emb, axis=1)
    if uniqueness.max() > 0:
        uniqueness = uniqueness / uniqueness.max()

    # Combine: 60% description relevance + 40% visual uniqueness
    scores = 0.6 * np.clip(desc_sims, 0, 1) + 0.4 * uniqueness
    # Normalize to 0-1
    if scores.max() > scores.min():
        scores = (scores - scores.min()) / (scores.max() - scores.min())
    return scores


def caption_and_analyze(
    keyframes: List[KeyFrame],
    clip_model: Any,
    caption_pipeline: Tuple[Any, Any, str],  # (processor, model, model_type)
    device: str = "cpu",
    description: str = "",
    scene_cut_indices: Optional[List[int]] = None,
) -> CaptionerResult:
    """Full captioning pipeline: CLIP embed → cluster → caption → topics → timeline."""
    if not keyframes:
        return CaptionerResult(
            scene_captions=[],
            visual_topics=[],
            frame_embeddings=np.zeros((0, 512)),
            cluster_labels=np.array([]),
            frame_timeline=[],
            ocr_texts=[],
        )

    scene_cut_set = set(scene_cut_indices or [])

    # Step 1: CLIP encode all frames (batched)
    images = [kf.image for kf in keyframes]
    frame_embeddings = clip_model.encode(
        images, batch_size=8, convert_to_numpy=True, normalize_embeddings=True,
    )

    # Step 2: Cluster to find unique scenes
    labels = _cluster_frames(frame_embeddings)

    # Step 3: Caption + OCR on representative frames
    representatives = _get_representative_frames(keyframes, frame_embeddings, labels)
    processor, model, model_type = caption_pipeline

    scene_captions = []
    ocr_texts = []
    caption_by_cluster: Dict[int, str] = {}

    for global_idx, kf in representatives:
        caption = _caption_frame(kf.image, processor, model, model_type, device)
        scene_captions.append(caption)
        caption_by_cluster[int(labels[global_idx])] = caption

        ocr = _ocr_frame(kf.image, processor, model, model_type, device)
        if ocr:
            ocr_texts.append(ocr)

    # Step 4: Detect visual topics
    visual_topics = _detect_visual_topics(frame_embeddings, clip_model)

    # Step 5: Per-frame relevance scores
    relevance_scores = _compute_relevance_scores(frame_embeddings, clip_model, description)

    # Step 6: Build frame timeline
    frame_timeline = []
    for i, kf in enumerate(keyframes):
        cluster_id = int(labels[i])
        frame_caption = caption_by_cluster.get(cluster_id, "")

        entry = {
            "timestamp_sec": round(kf.timestamp_sec, 2),
            "thumbnail_b64": _frame_to_thumbnail_b64(kf.image),
            "caption": frame_caption,
            "relevance_score": round(float(relevance_scores[i]), 4) if i < len(relevance_scores) else 0.0,
            "is_scene_change": i in scene_cut_set,
            "cluster_id": cluster_id,
        }
        frame_timeline.append(entry)

    return CaptionerResult(
        scene_captions=scene_captions,
        visual_topics=visual_topics,
        frame_embeddings=frame_embeddings,
        cluster_labels=labels,
        frame_timeline=frame_timeline,
        ocr_texts=ocr_texts,
    )
