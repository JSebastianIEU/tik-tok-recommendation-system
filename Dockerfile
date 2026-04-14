FROM python:3.12-slim

WORKDIR /app

# System deps: compiler toolchain + ffmpeg + opencv runtime libs
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc g++ \
        ffmpeg \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps (service + training + video analysis)
COPY requirements-base.txt requirements-service.txt requirements-training.txt ./
RUN pip install --no-cache-dir \
    -r requirements-service.txt \
    -r requirements-training.txt \
    huggingface_hub

# Video analysis dependencies
RUN pip install --no-cache-dir \
    opencv-python-headless>=4.9 \
    faster-whisper>=1.0 \
    easyocr>=1.7 \
    keybert>=0.8 \
    librosa>=0.10 \
    decord>=0.6 \
    imageio-ffmpeg>=0.5 \
    Pillow>=10.0

# ---------------------------------------------------------------------------
# Pre-download ALL ML models at build time
# This avoids HuggingFace rate-limiting (429) at runtime on Cloud Run.
# Models are cached in /root/.cache and loaded offline at runtime.
# ---------------------------------------------------------------------------

# 1. faster-whisper "base" model (fast on CPU, good enough for keyword extraction)
RUN python -c "\
from faster_whisper import WhisperModel; \
WhisperModel('base', device='cpu', compute_type='int8')"

# 2. EasyOCR detection + recognition models for en & es (~200MB)
RUN mkdir -p /root/.EasyOCR/model && \
    python -c "\
import easyocr; \
easyocr.Reader(['en', 'es'], gpu=False, verbose=True)"

# 3. BLIP image captioning model (~1GB)
RUN python -c "\
from transformers import BlipProcessor, BlipForConditionalGeneration; \
p = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base'); \
m = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base'); \
print('BLIP loaded, params:', sum(x.numel() for x in m.parameters()) // 1_000_000, 'M')"

# 4. KeyBERT + sentence-transformers model (~500MB)
RUN python -c "\
from keybert import KeyBERT; \
kb = KeyBERT('paraphrase-multilingual-MiniLM-L12-v2'); \
print('KeyBERT loaded')"

# 5. SentenceTransformer for hashtag recommender
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
m = SentenceTransformer('all-MiniLM-L6-v2'); \
print('SentenceTransformer loaded, dim:', m.get_sentence_embedding_dimension())"

# Verify all models load in offline mode (catches cache misses at build time)
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
m = SentenceTransformer('all-MiniLM-L6-v2'); \
print('OFFLINE OK: SentenceTransformer'); \
from transformers import BlipProcessor, BlipForConditionalGeneration; \
BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base'); \
BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base'); \
print('OFFLINE OK: BLIP'); \
from keybert import KeyBERT; \
KeyBERT('paraphrase-multilingual-MiniLM-L12-v2'); \
print('OFFLINE OK: KeyBERT')"

# Copy source code and ensure package is importable
COPY src/ ./src/
RUN touch ./src/__init__.py

# Copy serve script
COPY scripts/serve_recommender.py ./scripts/serve_recommender.py

# Download recommender artifacts from HuggingFace at build time
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download( \
    repo_id='JSebastianIEU/tiktok-repo', \
    repo_type='model', \
    local_dir='.', \
    allow_patterns=[ \
        'artifacts/recommender/20260414T050542Z-phase2-bootstrap-feedback/**', \
        'artifacts/contracts/f0119270fe1433f1adea9f41fbfd6eae66124c85ec9618619d0646ae29858bce/bundle.json', \
        'artifacts/hashtag_recommender/**', \
    ] \
)"

# Environment
ENV RECOMMENDER_BUNDLE_DIR=artifacts/recommender/20260414T050542Z-phase2-bootstrap-feedback
ENV RECOMMENDER_CORPUS_BUNDLE_PATH=artifacts/contracts/f0119270fe1433f1adea9f41fbfd6eae66124c85ec9618619d0646ae29858bce/bundle.json
ENV HASHTAG_RECOMMENDER_DIR=artifacts/hashtag_recommender
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PORT=8081
ENV DEMUCS_ENABLED=false
ENV BLIP_MODEL_ID=Salesforce/blip-image-captioning-base
ENV WHISPER_MODEL_SIZE=base

EXPOSE 8081

CMD ["sh", "-c", "python scripts/serve_recommender.py --host 0.0.0.0 --port ${PORT:-8081} --bundle-dir artifacts/recommender/20260414T050542Z-phase2-bootstrap-feedback"]
