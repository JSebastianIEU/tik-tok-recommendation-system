"""Video analyzer: frames, audio, transcription, scene cuts, keywords, VLM captioning.

Parallelizes transcription, audio analysis (demucs source separation), visual
analysis, OCR, and VLM captioning via ThreadPoolExecutor.

Improvements over v1:
  - OCR on extracted frames (easyocr)
  - KeyBERT singleton (no re-load per call)
  - 1-fps frames for scene cuts, 5 for VLM/heavy analysis
  - Single audio extraction at 44.1kHz, in-memory resample for whisper
  - Color/aesthetic features (HSV histograms)
  - Face detection (OpenCV Haar cascade)
  - Aspect ratio + resolution metadata
  - Whisper "small" for better transcript accuracy
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np

from .models import FrameTimelineEntry, SignalHints, VideoAnalysisResponse, VisualFeatures

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singletons & config
# ---------------------------------------------------------------------------

_WHISPER_MODEL = None
_WHISPER_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")
_DEVICE = os.getenv("VIDEO_ANALYZER_DEVICE", "cpu")

_KEYBERT_MODEL = None

_DEMUCS_MODEL = None
_DEMUCS_AVAILABLE: Optional[bool] = None
_DEMUCS_ENABLED = os.getenv("DEMUCS_ENABLED", "true").lower() in ("1", "true", "yes")

_OCR_READER = None
_OCR_AVAILABLE: Optional[bool] = None

_FACE_DETECTOR = None
_FACE_DETECTOR_AVAILABLE: Optional[bool] = None

_VLM_MODEL = None
_VLM_PROCESSOR = None
_VLM_AVAILABLE: Optional[bool] = None
_VLM_MODEL_ID = os.getenv("VLM_MODEL_ID", "lmms-lab/LLaVA-NeXT-Video-7B-DPO")
_VLM_QUANTIZE = os.getenv("VLM_QUANTIZE", "4bit")
_VLM_MAX_TOKENS = int(os.getenv("VLM_MAX_TOKENS", "256"))

_VLM_PROMPT = (
    "USER: <video>\nDescribe this TikTok video in detail: visual style, scene types, "
    "objects, people, on-screen text, transitions, and aesthetic quality. Be concise.\nASSISTANT:"
)

# CPU captioner (BLIP) — fallback when no GPU
_BLIP_MODEL = None
_BLIP_PROCESSOR = None
_BLIP_AVAILABLE: Optional[bool] = None
_BLIP_MODEL_ID = os.getenv("BLIP_MODEL_ID", "Salesforce/blip-image-captioning-base")

_SCENE_CUT_THRESHOLD_FACTOR = 2.0

# ---------------------------------------------------------------------------
# Lazy loaders
# ---------------------------------------------------------------------------


def _load_whisper_model():
    global _WHISPER_MODEL
    if _WHISPER_MODEL is not None:
        return _WHISPER_MODEL
    try:
        from faster_whisper import WhisperModel
        logger.info("Loading faster-whisper %s model...", _WHISPER_SIZE)
        _WHISPER_MODEL = WhisperModel(_WHISPER_SIZE, device=_DEVICE, compute_type="int8")
        return _WHISPER_MODEL
    except ImportError:
        logger.warning("faster-whisper not installed; transcription disabled")
        return None


def _load_keybert():
    global _KEYBERT_MODEL
    if _KEYBERT_MODEL is not None:
        return _KEYBERT_MODEL
    try:
        from keybert import KeyBERT
        _KEYBERT_MODEL = KeyBERT("paraphrase-multilingual-MiniLM-L12-v2")
        return _KEYBERT_MODEL
    except ImportError:
        logger.debug("keybert not installed")
        return None


def _load_demucs():
    global _DEMUCS_MODEL, _DEMUCS_AVAILABLE
    if _DEMUCS_AVAILABLE is False:
        return None
    if _DEMUCS_MODEL is not None:
        return _DEMUCS_MODEL
    if not _DEMUCS_ENABLED:
        _DEMUCS_AVAILABLE = False
        return None
    try:
        import torch
        from demucs.pretrained import get_model
        logger.info("Loading htdemucs model...")
        _DEMUCS_MODEL = get_model("htdemucs")
        _DEMUCS_MODEL.eval()
        _DEMUCS_AVAILABLE = True
        return _DEMUCS_MODEL
    except ImportError:
        logger.info("demucs not installed; using basic audio analysis")
        _DEMUCS_AVAILABLE = False
        return None
    except Exception:
        logger.warning("demucs loading failed", exc_info=True)
        _DEMUCS_AVAILABLE = False
        return None


def _load_ocr_reader():
    global _OCR_READER, _OCR_AVAILABLE
    if _OCR_AVAILABLE is False:
        return None
    if _OCR_READER is not None:
        return _OCR_READER
    try:
        import easyocr
        _OCR_READER = easyocr.Reader(["en", "es"], gpu=False, verbose=False)
        _OCR_AVAILABLE = True
        return _OCR_READER
    except ImportError:
        logger.debug("easyocr not installed; OCR disabled")
        _OCR_AVAILABLE = False
        return None
    except Exception:
        logger.warning("easyocr loading failed", exc_info=True)
        _OCR_AVAILABLE = False
        return None


def _load_face_detector():
    """Load OpenCV DNN face detector (avoids Haar cascade unicode path issues on Windows)."""
    global _FACE_DETECTOR, _FACE_DETECTOR_AVAILABLE
    if _FACE_DETECTOR_AVAILABLE is False:
        return None
    if _FACE_DETECTOR is not None:
        return _FACE_DETECTOR
    try:
        import cv2

        # Strategy 1: Try Haar cascade by reading XML bytes to avoid unicode path issues
        cascade_dir = cv2.data.haarcascades
        cascade_file = os.path.join(cascade_dir, "haarcascade_frontalface_default.xml")

        # Copy cascade to a temp path without unicode characters
        tmp_cascade = os.path.join(tempfile.gettempdir(), "haarcascade_frontalface.xml")
        if not os.path.exists(tmp_cascade):
            import shutil
            shutil.copy2(cascade_file, tmp_cascade)

        cascade = cv2.CascadeClassifier(tmp_cascade)
        if not cascade.empty():
            _FACE_DETECTOR = ("haar", cascade)
            _FACE_DETECTOR_AVAILABLE = True
            logger.info("Face detection: Haar cascade loaded from temp path")
            return _FACE_DETECTOR

        # Strategy 2: Try OpenCV DNN face detector (Caffe model)
        proto_path = os.path.join(cv2.data.haarcascades, "..", "deploy.prototxt")
        model_path = os.path.join(cv2.data.haarcascades, "..", "res10_300x300_ssd_iter_140000.caffemodel")
        if os.path.exists(proto_path) and os.path.exists(model_path):
            net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
            _FACE_DETECTOR = ("dnn", net)
            _FACE_DETECTOR_AVAILABLE = True
            logger.info("Face detection: DNN SSD detector loaded")
            return _FACE_DETECTOR

        logger.warning("No face detection model available")
        _FACE_DETECTOR_AVAILABLE = False
        return None
    except Exception:
        logger.debug("Face detector loading failed", exc_info=True)
        _FACE_DETECTOR_AVAILABLE = False
        return None


def _load_vlm():
    """Load GPU VLM (LLaVA-NeXT-Video) if CUDA available."""
    global _VLM_MODEL, _VLM_PROCESSOR, _VLM_AVAILABLE
    if _VLM_AVAILABLE is False:
        return None, None
    if _VLM_MODEL is not None and _VLM_PROCESSOR is not None:
        return _VLM_MODEL, _VLM_PROCESSOR
    try:
        import torch
        if not torch.cuda.is_available():
            logger.info("No CUDA GPU detected; GPU VLM disabled (will try CPU BLIP)")
            _VLM_AVAILABLE = False
            return None, None
        from transformers import LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor
        logger.info("Loading VLM %s...", _VLM_MODEL_ID)
        load_kwargs = {"device_map": "auto", "torch_dtype": torch.float16}
        if _VLM_QUANTIZE == "4bit":
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
            )
        elif _VLM_QUANTIZE == "8bit":
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        _VLM_PROCESSOR = LlavaNextVideoProcessor.from_pretrained(_VLM_MODEL_ID)
        _VLM_MODEL = LlavaNextVideoForConditionalGeneration.from_pretrained(
            _VLM_MODEL_ID, **load_kwargs
        )
        _VLM_AVAILABLE = True
        return _VLM_MODEL, _VLM_PROCESSOR
    except ImportError as exc:
        logger.info("VLM deps missing (%s)", exc)
        _VLM_AVAILABLE = False
        return None, None
    except Exception:
        logger.warning("VLM loading failed", exc_info=True)
        _VLM_AVAILABLE = False
        return None, None


def _load_blip():
    """Load CPU-friendly BLIP captioner (~250MB, runs on CPU in ~2-3s/frame)."""
    global _BLIP_MODEL, _BLIP_PROCESSOR, _BLIP_AVAILABLE
    if _BLIP_AVAILABLE is False:
        return None, None
    if _BLIP_MODEL is not None and _BLIP_PROCESSOR is not None:
        return _BLIP_MODEL, _BLIP_PROCESSOR
    try:
        import torch
        from transformers import BlipProcessor, BlipForConditionalGeneration
        logger.info("Loading BLIP captioner %s (CPU-friendly)...", _BLIP_MODEL_ID)
        _BLIP_PROCESSOR = BlipProcessor.from_pretrained(_BLIP_MODEL_ID)
        _BLIP_MODEL = BlipForConditionalGeneration.from_pretrained(
            _BLIP_MODEL_ID, torch_dtype=torch.float32,
        )
        _BLIP_MODEL.eval()
        _BLIP_AVAILABLE = True
        logger.info("BLIP captioner loaded successfully")
        return _BLIP_MODEL, _BLIP_PROCESSOR
    except ImportError as exc:
        logger.info("BLIP deps missing (%s) — install transformers", exc)
        _BLIP_AVAILABLE = False
        return None, None
    except Exception:
        logger.warning("BLIP loading failed", exc_info=True)
        _BLIP_AVAILABLE = False
        return None, None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_ffmpeg() -> Optional[str]:
    import shutil
    path = shutil.which("ffmpeg")
    if path:
        return path
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    for candidate in [
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        os.path.expanduser(r"~\scoop\apps\ffmpeg\current\bin\ffmpeg.exe"),
    ]:
        if os.path.isfile(candidate):
            return candidate
    return None


# ---------------------------------------------------------------------------
# Frame extraction — two tiers
# ---------------------------------------------------------------------------


def _extract_frames_decord(video_path: str, n: int) -> Tuple[List[np.ndarray], float, float, int, int]:
    """Returns (frames, fps, duration, width, height) using decord."""
    import decord
    decord.bridge.set_bridge("native")
    vr = decord.VideoReader(video_path)
    fps = float(vr.get_avg_fps())
    total = len(vr)
    duration = total / fps if fps > 0 else 0.0
    if total == 0:
        return [], fps, duration, 0, 0
    first = vr[0].asnumpy()
    h, w = first.shape[:2]
    indices = np.linspace(0, total - 1, min(n, total), dtype=int)
    frames = [vr[int(i)].asnumpy() for i in indices]
    return frames, fps, duration, w, h


def _extract_frames_cv2(video_path: str, n: int) -> Tuple[List[np.ndarray], float, float, int, int]:
    """Returns (frames, fps, duration, width, height) using OpenCV."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total / fps if fps > 0 else 0.0
    if total == 0:
        cap.release()
        return [], fps, duration, w, h
    indices = np.linspace(0, total - 1, min(n, total), dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames, fps, duration, w, h


def _extract_frames(
    video_path: str, n_vlm: int = 5, n_visual: int = 0
) -> Tuple[List[np.ndarray], List[np.ndarray], float, float, int, int]:
    """Extract two tiers of frames: n_vlm for heavy analysis, n_visual for scene cuts.

    n_visual=0 means auto (1 fps, capped at 120).
    Returns (vlm_frames, visual_frames, fps, duration, width, height).
    """
    # Try decord first for metadata
    fps = 30.0
    duration = 0.0
    width = height = 0

    try:
        import decord
        decord.bridge.set_bridge("native")
        vr = decord.VideoReader(video_path)
        fps = float(vr.get_avg_fps())
        total = len(vr)
        duration = total / fps if fps > 0 else 0.0
        if total == 0:
            return [], [], fps, duration, 0, 0
        first = vr[0].asnumpy()
        height, width = first.shape[:2]

        # VLM frames: n_vlm evenly spaced
        vlm_indices = np.linspace(0, total - 1, min(n_vlm, total), dtype=int)
        vlm_frames = [vr[int(i)].asnumpy() for i in vlm_indices]

        # Visual frames: 1 fps for accurate scene cuts + motion
        if n_visual <= 0:
            n_visual = min(int(duration), 120)
        vis_indices = np.linspace(0, total - 1, min(n_visual, total), dtype=int)
        visual_frames = [vr[int(i)].asnumpy() for i in vis_indices]

        return vlm_frames, visual_frames, fps, duration, width, height
    except Exception:
        logger.debug("decord unavailable, falling back to OpenCV", exc_info=True)

    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total / fps if fps > 0 else 0.0
        if total == 0:
            cap.release()
            return [], [], fps, duration, width, height

        # Gather all unique indices
        vlm_indices = set(np.linspace(0, total - 1, min(n_vlm, total), dtype=int).tolist())
        if n_visual <= 0:
            n_visual = min(int(duration), 120)
        vis_indices = set(np.linspace(0, total - 1, min(n_visual, total), dtype=int).tolist())
        all_indices = sorted(vlm_indices | vis_indices)

        frame_map: Dict[int, np.ndarray] = {}
        for idx in all_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if ok:
                frame_map[idx] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cap.release()

        vlm_frames = [frame_map[i] for i in sorted(vlm_indices) if i in frame_map]
        visual_frames = [frame_map[i] for i in sorted(vis_indices) if i in frame_map]
        return vlm_frames, visual_frames, fps, duration, width, height
    except Exception:
        logger.warning("No frame extraction available", exc_info=True)
        return [], [], 30.0, 0.0, 0, 0


# ---------------------------------------------------------------------------
# Audio extraction — single pass at 44.1kHz stereo
# ---------------------------------------------------------------------------


def _extract_audio_track(video_path: str) -> Optional[str]:
    """Extract audio to temporary WAV at 44100Hz stereo (native for demucs).

    Whisper resamples in-memory from this same file.
    """
    ffmpeg_bin = _find_ffmpeg()
    if ffmpeg_bin:
        try:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.close()
            result = subprocess.run(
                [
                    ffmpeg_bin, "-y", "-i", video_path,
                    "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2",
                    tmp.name,
                ],
                capture_output=True,
                timeout=30,
            )
            if result.returncode == 0 and os.path.getsize(tmp.name) > 0:
                return tmp.name
            os.unlink(tmp.name)
        except Exception:
            logger.debug("ffmpeg audio extraction failed", exc_info=True)

    try:
        from moviepy import VideoFileClip
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        clip = VideoFileClip(video_path)
        if clip.audio is not None:
            clip.audio.write_audiofile(tmp.name, fps=44100, nbytes=2, codec="pcm_s16le", logger=None)
            clip.close()
            if os.path.getsize(tmp.name) > 0:
                return tmp.name
        else:
            clip.close()
        os.unlink(tmp.name)
    except Exception:
        logger.debug("moviepy audio extraction also failed", exc_info=True)

    logger.warning("No audio extraction method available")
    return None


# ---------------------------------------------------------------------------
# Transcription (whisper small, resamples from 44.1kHz in-memory)
# ---------------------------------------------------------------------------


def _transcribe(wav_path: str) -> Tuple[str, float, str]:
    """Transcribe audio with faster-whisper. Returns (transcript, speech_seconds, language)."""
    model = _load_whisper_model()
    if model is None or wav_path is None:
        return "", 0.0, ""
    try:
        segments, info = model.transcribe(wav_path, beam_size=1, language=None)
        detected_lang = getattr(info, "language", "") or ""
        texts = []
        speech_seconds = 0.0
        for seg in segments:
            texts.append(seg.text.strip())
            speech_seconds += seg.end - seg.start
        return " ".join(texts), speech_seconds, detected_lang
    except Exception:
        logger.warning("Transcription failed", exc_info=True)
        return "", 0.0, ""


# ---------------------------------------------------------------------------
# Audio analysis (demucs source separation + librosa fallback)
# ---------------------------------------------------------------------------


class AudioAnalysisResult:
    __slots__ = ("tempo_bpm", "audio_energy", "speech_seconds", "music_seconds",
                 "has_source_separation")

    def __init__(self, tempo_bpm=0.0, audio_energy=0.0, speech_seconds=0.0,
                 music_seconds=0.0, has_source_separation=False):
        self.tempo_bpm = tempo_bpm
        self.audio_energy = audio_energy
        self.speech_seconds = speech_seconds
        self.music_seconds = music_seconds
        self.has_source_separation = has_source_separation


def _analyze_audio_demucs(wav_path: str, duration: float) -> Optional[AudioAnalysisResult]:
    """Deep audio analysis via demucs source separation."""
    model = _load_demucs()
    if model is None or wav_path is None:
        return None
    try:
        import torch
        import librosa
        from demucs.apply import apply_model

        sr = model.samplerate
        y, _ = librosa.load(wav_path, sr=sr, mono=False)
        if y.ndim == 1:
            y = np.stack([y, y])
        wav = torch.from_numpy(y).float()

        ref = wav.mean(0)
        wav_norm = (wav - ref.mean()) / ref.std()

        with torch.no_grad():
            sources = apply_model(model, wav_norm[None], device="cpu")
        sources = sources[0]

        src_names = model.sources
        instrumental = (sources[src_names.index("drums")]
                        + sources[src_names.index("bass")]
                        + sources[src_names.index("other")])
        vocals = sources[src_names.index("vocals")]

        n_seconds = min(instrumental.shape[1] // sr, int(duration) + 1)
        music_secs = speech_secs = 0.0
        threshold = 0.005
        for s in range(n_seconds):
            sl = slice(s * sr, (s + 1) * sr)
            if float(np.sqrt(np.mean(instrumental[:, sl].numpy() ** 2))) > threshold:
                music_secs += 1.0
            if float(np.sqrt(np.mean(vocals[:, sl].numpy() ** 2))) > threshold:
                speech_secs += 1.0

        inst_mono = instrumental.mean(0).numpy()
        tempo, _ = librosa.beat.beat_track(y=inst_mono, sr=sr)
        tempo_val = float(np.atleast_1d(tempo)[0])

        rms = librosa.feature.rms(y=y[0])
        energy = float(np.mean(rms)) if rms is not None and rms.size > 0 else 0.0

        return AudioAnalysisResult(
            tempo_bpm=tempo_val, audio_energy=energy,
            speech_seconds=speech_secs, music_seconds=music_secs,
            has_source_separation=True,
        )
    except Exception:
        logger.warning("Demucs audio analysis failed", exc_info=True)
        return None


def _analyze_audio_basic(wav_path: str) -> AudioAnalysisResult:
    """Basic audio analysis using librosa only (fast fallback)."""
    if wav_path is None:
        return AudioAnalysisResult()
    try:
        import librosa
        y, sr = librosa.load(wav_path, sr=16000, mono=True)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo_val = float(np.atleast_1d(tempo)[0])
        rms = librosa.feature.rms(y=y)
        energy = float(np.mean(rms)) if rms is not None and rms.size > 0 else 0.0
        return AudioAnalysisResult(tempo_bpm=tempo_val, audio_energy=energy)
    except Exception:
        logger.warning("Basic audio analysis failed", exc_info=True)
        return AudioAnalysisResult()


def _analyze_audio(wav_path: str, duration: float = 0.0) -> AudioAnalysisResult:
    result = _analyze_audio_demucs(wav_path, duration)
    if result is not None:
        return result
    return _analyze_audio_basic(wav_path)


# ---------------------------------------------------------------------------
# Visual analysis — scene cuts, motion, color, faces, blur, hook
# ---------------------------------------------------------------------------


def _detect_scene_cuts(frames: List[np.ndarray]) -> int:
    if len(frames) < 2:
        return 0
    diffs = []
    for i in range(1, len(frames)):
        prev_gray = np.mean(frames[i - 1], axis=2).astype(np.float32)
        curr_gray = np.mean(frames[i], axis=2).astype(np.float32)
        diffs.append(np.mean(np.abs(curr_gray - prev_gray)))
    if not diffs:
        return 0
    median_diff = float(np.median(diffs))
    threshold = median_diff * _SCENE_CUT_THRESHOLD_FACTOR
    return sum(1 for d in diffs if d > threshold)


def _compute_motion_score(frames: List[np.ndarray]) -> float:
    if len(frames) < 2:
        return 0.0
    diffs = []
    for i in range(1, len(frames)):
        prev_gray = np.mean(frames[i - 1], axis=2).astype(np.float32)
        curr_gray = np.mean(frames[i], axis=2).astype(np.float32)
        diffs.append(np.mean(np.abs(curr_gray - prev_gray)) / 255.0)
    return float(np.mean(diffs)) if diffs else 0.0


def _compute_hook_motion(frames: List[np.ndarray], fps: float) -> float:
    """Motion score specifically in the first 3 seconds (hook window)."""
    if len(frames) < 2 or fps <= 0:
        return 0.0
    # frames are 1-fps, so first 3 frames = first 3 seconds
    hook_frames = frames[:min(4, len(frames))]
    return _compute_motion_score(hook_frames)


def _analyze_colors(frames: List[np.ndarray]) -> Tuple[List[str], float, float, float]:
    """Compute dominant colors, avg brightness, saturation, contrast.

    Returns (dominant_color_hexes, brightness, saturation, contrast).
    """
    if not frames:
        return [], 0.0, 0.0, 0.0
    try:
        import cv2
    except ImportError:
        return [], 0.0, 0.0, 0.0

    brightness_vals = []
    saturation_vals = []
    contrast_vals = []
    all_hue_bins: Counter = Counter()

    # Sample up to 10 evenly-spaced frames
    sample_indices = np.linspace(0, len(frames) - 1, min(10, len(frames)), dtype=int)

    for idx in sample_indices:
        frame = frames[int(idx)]
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        brightness_vals.append(float(np.mean(hsv[:, :, 2])) / 255.0)
        saturation_vals.append(float(np.mean(hsv[:, :, 1])) / 255.0)

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        contrast_vals.append(float(np.std(gray)) / 128.0)  # normalized 0-1

        # Quantize hues into 12 bins (30° each)
        hue_bins = (hsv[:, :, 0].flatten() // 15).astype(int)
        mask = hsv[:, :, 1].flatten() > 30  # ignore desaturated pixels
        for h in hue_bins[mask]:
            all_hue_bins[h] += 1

    # Map top 3 hue bins to color names
    hue_names = {
        0: "#FF0000", 1: "#FF8000", 2: "#FFFF00", 3: "#80FF00",
        4: "#00FF00", 5: "#00FF80", 6: "#00FFFF", 7: "#0080FF",
        8: "#0000FF", 9: "#8000FF", 10: "#FF00FF", 11: "#FF0080",
    }
    top_hues = [hue_names.get(h, "#808080") for h, _ in all_hue_bins.most_common(3)]

    return (
        top_hues,
        float(np.mean(brightness_vals)) if brightness_vals else 0.0,
        float(np.mean(saturation_vals)) if saturation_vals else 0.0,
        float(np.mean(contrast_vals)) if contrast_vals else 0.0,
    )


def _detect_faces_haar(cascade, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Detect faces using Haar cascade. Returns list of (x, y, w, h)."""
    import cv2
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # Equalize histogram for better detection in varying lighting
    gray = cv2.equalizeHist(gray)
    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    if isinstance(faces, np.ndarray) and faces.ndim == 2:
        return [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]
    return []


def _detect_faces_dnn(net, frame: np.ndarray, conf_threshold: float = 0.5) -> List[Tuple[int, int, int, int]]:
    """Detect faces using DNN SSD. Returns list of (x, y, w, h)."""
    import cv2
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
    )
    net.setInput(blob)
    detections = net.forward()
    faces = []
    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        if confidence > conf_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            faces.append((x1, y1, x2 - x1, y2 - y1))
    return faces


def _detect_faces(frames: List[np.ndarray]) -> Tuple[int, float]:
    """Detect faces across frames. Returns (max_face_count, avg_face_area_ratio)."""
    detector = _load_face_detector()
    if detector is None or not frames:
        return 0, 0.0
    try:
        import cv2
    except ImportError:
        return 0, 0.0

    detector_type, model = detector
    max_count = 0
    area_ratios = []

    # Sample up to 5 frames
    sample_indices = np.linspace(0, len(frames) - 1, min(5, len(frames)), dtype=int)

    for idx in sample_indices:
        frame = frames[int(idx)]
        if detector_type == "haar":
            faces = _detect_faces_haar(model, frame)
        else:
            faces = _detect_faces_dnn(model, frame)

        n_faces = len(faces)
        max_count = max(max_count, n_faces)

        if n_faces > 0:
            frame_area = frame.shape[0] * frame.shape[1]
            total_face_area = sum(w * h for (_, _, w, h) in faces)
            area_ratios.append(total_face_area / frame_area)

    avg_ratio = float(np.mean(area_ratios)) if area_ratios else 0.0
    return max_count, avg_ratio


def _compute_blur_score(frames: List[np.ndarray]) -> float:
    """Compute average blur score (higher = sharper). Uses Laplacian variance."""
    if not frames:
        return 0.0
    try:
        import cv2
    except ImportError:
        return 0.0

    scores = []
    sample_indices = np.linspace(0, len(frames) - 1, min(5, len(frames)), dtype=int)
    for idx in sample_indices:
        gray = cv2.cvtColor(frames[int(idx)], cv2.COLOR_RGB2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        scores.append(float(lap_var))
    return float(np.mean(scores)) if scores else 0.0


def _get_aspect_ratio_label(width: int, height: int) -> str:
    if width <= 0 or height <= 0:
        return ""
    ratio = width / height
    if abs(ratio - 9 / 16) < 0.05:
        return "9:16"
    if abs(ratio - 16 / 9) < 0.05:
        return "16:9"
    if abs(ratio - 1.0) < 0.05:
        return "1:1"
    if abs(ratio - 4 / 5) < 0.05:
        return "4:5"
    return f"{width}:{height}"


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------


def _preprocess_frame_for_ocr(frame: np.ndarray) -> np.ndarray:
    """Preprocess frame for better OCR: grayscale, resize, contrast enhancement."""
    try:
        import cv2
    except ImportError:
        return frame

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # Upscale small frames (OCR works better on larger text)
    h, w = gray.shape[:2]
    if max(h, w) < 1000:
        scale = 1000 / max(h, w)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # CLAHE for adaptive contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    return gray


_OCR_CONFIDENCE_THRESHOLD = 0.4
_OCR_MIN_TEXT_LENGTH = 2


def _extract_ocr_text(frames: List[np.ndarray]) -> str:
    """Run OCR on sampled frames with confidence filtering and preprocessing."""
    reader = _load_ocr_reader()
    if reader is None or not frames:
        return ""
    try:
        seen = set()
        texts = []
        # Sample up to 3 evenly-spaced frames
        sample_indices = np.linspace(0, len(frames) - 1, min(3, len(frames)), dtype=int)
        for idx in sample_indices:
            preprocessed = _preprocess_frame_for_ocr(frames[int(idx)])
            # detail=1 returns list of (bbox, text, confidence)
            results = reader.readtext(preprocessed, detail=1, paragraph=False)
            for bbox, text, confidence in results:
                if confidence < _OCR_CONFIDENCE_THRESHOLD:
                    continue
                t_clean = text.strip()
                if len(t_clean) < _OCR_MIN_TEXT_LENGTH:
                    continue
                # Skip strings that are mostly non-alphanumeric (noise)
                alnum_ratio = sum(c.isalnum() or c.isspace() for c in t_clean) / len(t_clean)
                if alnum_ratio < 0.5:
                    continue
                t_lower = t_clean.lower()
                if t_lower not in seen:
                    seen.add(t_lower)
                    texts.append(t_clean)
        return " | ".join(texts)
    except Exception:
        logger.debug("OCR failed", exc_info=True)
        return ""


# ---------------------------------------------------------------------------
# VLM captioning
# ---------------------------------------------------------------------------


def _generate_video_caption_gpu(frames: List[np.ndarray]) -> Optional[str]:
    """Generate caption using GPU LLaVA-NeXT-Video model."""
    model, processor = _load_vlm()
    if model is None or processor is None:
        return None
    try:
        import torch
        from PIL import Image
        pil_frames = [Image.fromarray(f) for f in frames]
        inputs = processor(text=_VLM_PROMPT, videos=[pil_frames], return_tensors="pt").to(model.device)
        with torch.inference_mode():
            output_ids = model.generate(**inputs, max_new_tokens=_VLM_MAX_TOKENS, do_sample=False)
        prompt_len = inputs["input_ids"].shape[1]
        return processor.decode(output_ids[0][prompt_len:], skip_special_tokens=True).strip()
    except Exception:
        logger.warning("GPU VLM caption failed", exc_info=True)
        return None


def _generate_video_caption_cpu(frames: List[np.ndarray]) -> Optional[str]:
    """Generate captions using CPU BLIP model. Captions 3 key frames and merges."""
    model, processor = _load_blip()
    if model is None or processor is None:
        return None
    try:
        import torch
        from PIL import Image

        # Caption up to 5 evenly-spaced frames
        n = min(5, len(frames))
        indices = np.linspace(0, len(frames) - 1, n, dtype=int)
        captions = []

        for idx in indices:
            pil_img = Image.fromarray(frames[int(idx)])
            inputs = processor(
                images=pil_img,
                text="a tiktok video showing",
                return_tensors="pt",
            )
            with torch.inference_mode():
                output_ids = model.generate(**inputs, max_new_tokens=64)
            caption = processor.decode(output_ids[0], skip_special_tokens=True).strip()
            if caption and caption not in captions:
                captions.append(caption)

        return " | ".join(captions) if captions else None
    except Exception:
        logger.warning("CPU BLIP caption failed", exc_info=True)
        return None


def _generate_video_caption(frames: List[np.ndarray]) -> str:
    """Generate video caption — tries GPU LLaVA first, falls back to CPU BLIP."""
    if not frames:
        return ""
    # Tier 1: GPU (LLaVA-NeXT-Video, rich multi-frame)
    result = _generate_video_caption_gpu(frames)
    if result:
        return result
    # Tier 2: CPU (BLIP, per-frame captions merged)
    result = _generate_video_caption_cpu(frames)
    if result:
        return result
    return ""


# ---------------------------------------------------------------------------
# Keywords (KeyBERT singleton + YAKE fallback)
# ---------------------------------------------------------------------------


def _extract_keywords(text: str, language: str = "", max_keywords: int = 10) -> List[str]:
    if not text or not text.strip():
        return []
    # KeyBERT (singleton)
    kw_model = _load_keybert()
    if kw_model is not None:
        try:
            keywords = kw_model.extract_keywords(
                text, keyphrase_ngram_range=(1, 2), stop_words=None,
                top_n=max_keywords, use_mmr=True, diversity=0.5,
            )
            result = [kw for kw, _ in keywords]
            if result:
                return result
        except Exception:
            logger.debug("KeyBERT extraction failed", exc_info=True)
    # YAKE fallback
    try:
        import yake
        yake_lang = language if language and len(language) == 2 else "en"
        extractor = yake.KeywordExtractor(lan=yake_lang, n=2, top=max_keywords)
        return [kw for kw, _ in extractor.extract_keywords(text)]
    except Exception:
        logger.debug("YAKE also failed", exc_info=True)
        return []


# ---------------------------------------------------------------------------
# Timeline generation (CPU-friendly)
# ---------------------------------------------------------------------------

_TIMELINE_MAX_FRAMES = 10
_TIMELINE_THUMB_WIDTH = 160


def _generate_timeline(
    frames: List[np.ndarray],
    fps: float,
    duration: float,
) -> List[FrameTimelineEntry]:
    """Generate a temporal timeline with per-frame analysis (thumbnails, OCR, faces, motion, scene changes).

    Uses the visual_frames (1-fps) to produce a timeline with evenly-spaced entries.
    All analysis is CPU-only — no GPU models required.
    """
    if not frames or fps <= 0 or duration <= 0:
        return []

    import base64
    import io

    try:
        import cv2
    except ImportError:
        return []

    # Sample up to _TIMELINE_MAX_FRAMES evenly spaced
    n = min(_TIMELINE_MAX_FRAMES, len(frames))
    indices = np.linspace(0, len(frames) - 1, n, dtype=int)

    # Pre-compute frame diffs for scene cut & motion detection
    diffs = []
    for i in range(1, len(frames)):
        prev_gray = np.mean(frames[i - 1], axis=2).astype(np.float32)
        curr_gray = np.mean(frames[i], axis=2).astype(np.float32)
        diffs.append(float(np.mean(np.abs(curr_gray - prev_gray))))
    median_diff = float(np.median(diffs)) if diffs else 0.0
    scene_threshold = median_diff * _SCENE_CUT_THRESHOLD_FACTOR

    entries = []
    for i_pos, frame_idx in enumerate(indices):
        frame_idx = int(frame_idx)
        frame = frames[frame_idx]
        timestamp = (frame_idx / max(len(frames) - 1, 1)) * duration if len(frames) > 1 else 0.0

        # Thumbnail: resize preserving aspect ratio
        h, w = frame.shape[:2]
        thumb_h = int(h * _TIMELINE_THUMB_WIDTH / w) if w > 0 else 90
        thumb = cv2.resize(frame, (_TIMELINE_THUMB_WIDTH, thumb_h), interpolation=cv2.INTER_AREA)
        # Encode as JPEG base64
        thumb_bgr = cv2.cvtColor(thumb, cv2.COLOR_RGB2BGR)
        _, buf = cv2.imencode(".jpg", thumb_bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])
        thumb_b64 = base64.b64encode(buf.tobytes()).decode("ascii")

        # Per-frame motion score
        if frame_idx > 0 and frame_idx - 1 < len(diffs):
            motion = diffs[frame_idx - 1] / 255.0
        else:
            motion = 0.0

        # Scene change detection
        is_scene_change = False
        if frame_idx > 0 and frame_idx - 1 < len(diffs):
            is_scene_change = diffs[frame_idx - 1] > scene_threshold

        # Relevance heuristic: motion + scene change (skip per-frame OCR/faces for speed)
        relevance = 0.3  # baseline
        if is_scene_change:
            relevance += 0.35
        relevance += min(motion * 2.0, 0.35)
        relevance = min(relevance, 1.0)

        entries.append(FrameTimelineEntry(
            timestamp_sec=round(timestamp, 2),
            thumbnail_b64=thumb_b64,
            ocr_text="",
            face_count=0,
            motion_score=round(motion, 4),
            is_scene_change=is_scene_change,
            relevance_score=round(relevance, 2),
        ))

    return entries


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


class VideoAnalyzer:
    """Orchestrates parallel video analysis with all feature extractors."""

    def __init__(self, max_workers: int = 6):
        self._max_workers = max_workers

    def analyze(self, video_path: str) -> VideoAnalysisResponse:
        started = time.perf_counter()

        # Step 1: Extract two tiers of frames + metadata
        vlm_frames, visual_frames, fps, duration, width, height = _extract_frames(video_path, n_vlm=6)

        # Step 2: Extract audio (single 44.1kHz stereo WAV for both demucs and whisper)
        wav_path = _extract_audio_track(video_path)

        # Step 3: Run all branches in parallel
        transcript = ""
        speech_seconds = 0.0
        detected_language = ""
        audio_result = AudioAnalysisResult()
        scene_cuts = 0
        motion_score = 0.0
        hook_motion = 0.0
        video_caption = ""
        ocr_text = ""
        face_count = 0
        face_area_ratio = 0.0
        dominant_colors: List[str] = []
        brightness = saturation = contrast = 0.0
        blur_score = 0.0
        timeline: List[FrameTimelineEntry] = []

        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            future_transcribe = pool.submit(_transcribe, wav_path)
            future_audio = pool.submit(_analyze_audio, wav_path, duration)
            future_visual = pool.submit(
                lambda: (
                    _detect_scene_cuts(visual_frames),
                    _compute_motion_score(visual_frames),
                    _compute_hook_motion(visual_frames, fps),
                )
            )
            future_vlm = pool.submit(_generate_video_caption, vlm_frames)
            future_ocr = pool.submit(_extract_ocr_text, vlm_frames)
            future_colors = pool.submit(_analyze_colors, visual_frames)
            future_blur = pool.submit(_compute_blur_score, vlm_frames)
            future_timeline = pool.submit(_generate_timeline, visual_frames, fps, duration)

            futures = {
                future_transcribe: "transcribe",
                future_audio: "audio",
                future_visual: "visual",
                future_vlm: "vlm",
                future_ocr: "ocr",
                future_colors: "colors",
                future_blur: "blur",
                future_timeline: "timeline",
            }

            for future in as_completed(futures):
                name = futures[future]
                branch_start = time.perf_counter()
                try:
                    if name == "transcribe":
                        transcript, speech_seconds, detected_language = future.result(timeout=120)
                    elif name == "audio":
                        audio_result = future.result(timeout=300)
                    elif name == "visual":
                        scene_cuts, motion_score, hook_motion = future.result(timeout=30)
                    elif name == "vlm":
                        video_caption = future.result(timeout=60)
                    elif name == "ocr":
                        ocr_text = future.result(timeout=60)
                    elif name == "colors":
                        dominant_colors, brightness, saturation, contrast = future.result(timeout=10)
                    elif name == "blur":
                        blur_score = future.result(timeout=10)
                    elif name == "timeline":
                        timeline = future.result(timeout=120)
                    logger.info("Branch [%s] completed in %.1fs", name, time.perf_counter() - branch_start)
                except Exception:
                    logger.warning("Branch %s failed after %.1fs", name, time.perf_counter() - branch_start, exc_info=True)

        # Clean up temp audio
        if wav_path is not None:
            try:
                os.unlink(wav_path)
            except OSError:
                pass

        # Resolve speech/music from demucs or whisper fallback
        if audio_result.has_source_separation:
            speech_seconds = audio_result.speech_seconds
            music_seconds = audio_result.music_seconds
        else:
            music_seconds = max(0.0, duration - speech_seconds)

        # Keywords from transcript + OCR + VLM caption
        keyword_source = " ".join(filter(None, [transcript, ocr_text, video_caption]))
        keywords = _extract_keywords(keyword_source, language=detected_language)

        elapsed = time.perf_counter() - started
        logger.info("Video analysis total: %.1fs (video duration: %.1fs)", elapsed, duration)

        aspect_ratio = _get_aspect_ratio_label(width, height)
        resolution = f"{width}x{height}" if width > 0 and height > 0 else ""

        return VideoAnalysisResponse(
            signal_hints=SignalHints(
                speech_seconds=round(speech_seconds, 2),
                music_seconds=round(music_seconds, 2),
                tempo_bpm=round(audio_result.tempo_bpm, 1),
                audio_energy=round(audio_result.audio_energy, 6),
                estimated_scene_cuts=scene_cuts,
                visual_motion_score=round(motion_score, 4),
                fps=round(fps, 2),
            ),
            transcript=transcript,
            ocr_text=ocr_text,
            video_caption=video_caption,
            detected_language=detected_language,
            keywords=keywords,
            timeline=timeline,
            visual_features=VisualFeatures(
                dominant_colors=dominant_colors,
                avg_brightness=round(brightness, 3),
                avg_saturation=round(saturation, 3),
                avg_contrast=round(contrast, 3),
                face_count=face_count,
                avg_face_area_ratio=round(face_area_ratio, 4),
                aspect_ratio=aspect_ratio,
                resolution=resolution,
                blur_score=round(blur_score, 1),
                hook_motion_score=round(hook_motion, 4),
            ),
            duration_seconds=round(duration, 2),
            processing_time_seconds=round(elapsed, 3),
        )
