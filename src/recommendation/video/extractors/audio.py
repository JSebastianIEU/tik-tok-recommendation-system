"""Audio extraction and analysis using PyAV + librosa.

Extracts audio from video (no ffmpeg CLI needed), then analyzes
tempo, energy, loudness, and speech/music duration.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from ..models import AudioAnalysisResult


def extract_audio(video_path: Path, output_dir: Path | None = None) -> Path:
    """Extract audio from video as 16kHz mono WAV using PyAV."""
    import av
    import soundfile as sf

    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="vidaudio_"))

    audio_path = output_dir / "audio.wav"

    container = av.open(str(video_path))
    audio_stream = next((s for s in container.streams if s.type == "audio"), None)

    if audio_stream is None:
        # No audio track — return silence
        sf.write(str(audio_path), np.zeros(16000, dtype=np.float32), 16000)
        return audio_path

    resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
    samples = []

    for frame in container.decode(audio_stream):
        resampled = resampler.resample(frame)
        for r in resampled:
            arr = r.to_ndarray().flatten()
            samples.append(arr)

    container.close()

    if not samples:
        sf.write(str(audio_path), np.zeros(16000, dtype=np.float32), 16000)
        return audio_path

    audio_data = np.concatenate(samples).astype(np.float32) / 32768.0
    sf.write(str(audio_path), audio_data, 16000)
    return audio_path


def analyze_audio(audio_path: Path) -> AudioAnalysisResult:
    """Analyze audio file for tempo, energy, loudness, speech/music split."""
    import librosa

    y, sr = librosa.load(str(audio_path), sr=16000, mono=True)
    duration = float(len(y) / sr)

    if duration < 0.5:
        return AudioAnalysisResult(
            audio_path=str(audio_path),
            duration_seconds=duration,
            tempo_bpm=0.0,
            audio_energy=0.0,
            loudness_lufs=-70.0,
            speech_seconds=0.0,
            music_seconds=0.0,
        )

    # Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo_val = float(tempo[0]) if hasattr(tempo, '__len__') else float(tempo)

    # RMS energy (normalized 0-1)
    rms = librosa.feature.rms(y=y)[0]
    energy = float(np.mean(rms))
    energy_norm = min(energy / 0.15, 1.0)  # normalize typical range

    # Approximate LUFS from RMS
    rms_db = 20.0 * np.log10(energy + 1e-10)
    loudness_lufs = float(rms_db - 0.691)  # rough approximation

    # Estimate speech vs music via zero-crossing rate
    zcr = librosa.feature.zero_crossing_rate(y=y)[0]
    speech_frames = np.sum(zcr > 0.05)
    total_frames = len(zcr)
    speech_ratio = speech_frames / max(total_frames, 1)
    speech_seconds = round(duration * speech_ratio, 2)
    music_seconds = round(duration - speech_seconds, 2)

    return AudioAnalysisResult(
        audio_path=str(audio_path),
        duration_seconds=round(duration, 2),
        tempo_bpm=round(tempo_val, 1),
        audio_energy=round(energy_norm, 4),
        loudness_lufs=round(loudness_lufs, 2),
        speech_seconds=speech_seconds,
        music_seconds=max(music_seconds, 0.0),
    )
