"""CPU/GPU model loading abstraction.

Switch between CPU and GPU by setting VIDEO_ANALYZER_DEVICE=cpu|cuda.
CPU default: CLIP ViT-B/32 + BLIP-base + whisper-tiny-int8
GPU (Azure): DAM-3B-Video + whisper-base-float16
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Any, Literal, Optional


@dataclass
class DeviceConfig:
    device: Literal["cpu", "cuda"]
    clip_model: str
    caption_model: str
    whisper_model: str
    whisper_compute_type: str

    @classmethod
    def from_env(cls) -> DeviceConfig:
        device = os.getenv("VIDEO_ANALYZER_DEVICE", "cpu").lower().strip()
        if device == "cuda":
            return cls(
                device="cuda",
                clip_model=os.getenv("VIDEO_CLIP_MODEL", "clip-ViT-B-32"),
                caption_model=os.getenv("VIDEO_CAPTION_MODEL", "microsoft/Florence-2-base"),
                whisper_model=os.getenv("VIDEO_WHISPER_MODEL", "base"),
                whisper_compute_type="float16",
            )
        return cls(
            device="cpu",
            clip_model=os.getenv("VIDEO_CLIP_MODEL", "clip-ViT-B-32"),
            caption_model=os.getenv("VIDEO_CAPTION_MODEL", "microsoft/Florence-2-base"),
            whisper_model=os.getenv("VIDEO_WHISPER_MODEL", "tiny"),
            whisper_compute_type="int8",
        )


class ModelRegistry:
    """Thread-safe, lazy-loading model registry."""

    _instance: Optional[ModelRegistry] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self.config = DeviceConfig.from_env()
        self._clip_model: Any = None
        self._caption_pipeline: Any = None
        self._whisper_model: Any = None
        self._model_lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> ModelRegistry:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def get_clip_model(self) -> Any:
        if self._clip_model is None:
            with self._model_lock:
                if self._clip_model is None:
                    from sentence_transformers import SentenceTransformer
                    self._clip_model = SentenceTransformer(self.config.clip_model)
        return self._clip_model

    def get_caption_pipeline(self) -> Any:
        if self._caption_pipeline is None:
            with self._model_lock:
                if self._caption_pipeline is None:
                    model_name = self.config.caption_model
                    try:
                        # Try Florence-2 first (better captions)
                        if "florence" in model_name.lower():
                            from transformers import AutoProcessor, AutoModelForCausalLM
                            import torch
                            processor = AutoProcessor.from_pretrained(
                                model_name, trust_remote_code=True,
                            )
                            model = AutoModelForCausalLM.from_pretrained(
                                model_name,
                                trust_remote_code=True,
                                torch_dtype=torch.float32 if self.config.device == "cpu" else torch.float16,
                            )
                            if self.config.device == "cuda":
                                model = model.cuda()
                            model.eval()
                            self._caption_pipeline = (processor, model, "florence2")
                            return self._caption_pipeline
                    except Exception as exc:
                        print(f"[ModelRegistry] Florence-2 failed ({exc}), falling back to BLIP-base")
                        model_name = "Salesforce/blip-image-captioning-base"

                    # Fallback: BLIP-base
                    from transformers import BlipProcessor, BlipForConditionalGeneration
                    processor = BlipProcessor.from_pretrained(model_name)
                    model = BlipForConditionalGeneration.from_pretrained(model_name)
                    if self.config.device == "cuda":
                        model = model.cuda()
                    model.eval()
                    self._caption_pipeline = (processor, model, "blip")
        return self._caption_pipeline

    def get_whisper_model(self) -> Any:
        if self._whisper_model is None:
            with self._model_lock:
                if self._whisper_model is None:
                    from faster_whisper import WhisperModel
                    self._whisper_model = WhisperModel(
                        self.config.whisper_model,
                        device=self.config.device,
                        compute_type=self.config.whisper_compute_type,
                    )
        return self._whisper_model

    def warmup(self) -> None:
        """Preload all models. Call at startup to avoid cold-start latency."""
        import numpy as np

        print(f"[ModelRegistry] Warming up models (device={self.config.device})...")

        clip = self.get_clip_model()
        clip.encode(["warmup"], convert_to_numpy=True)
        print("[ModelRegistry] CLIP ready")

        self.get_caption_pipeline()
        print(f"[ModelRegistry] Caption model ready ({self.config.caption_model})")

        self.get_whisper_model()
        print("[ModelRegistry] Whisper ready")

        print("[ModelRegistry] All models loaded")
