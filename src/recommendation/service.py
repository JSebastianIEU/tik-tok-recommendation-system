from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from src.common.config import settings

try:
    from fastapi import FastAPI, File, HTTPException, UploadFile
except Exception as exc:  # pragma: no cover - optional dependency
    raise RuntimeError(
        "FastAPI is required for recommender service. Install fastapi and uvicorn."
    ) from exc

from .corpus import CorpusResolver, CorpusScopeSpec
from .learning.inference import (
    ArtifactCompatibilityError,
    RecommenderRuntime,
    RecommenderStageTimeoutError,
    RoutingContractError,
)
from .fabric import FeatureFabric, FeatureFabricInput


class RecommendationCandidateInput(BaseModel):
    candidate_id: str
    caption: Optional[str] = None
    text: Optional[str] = None
    hashtags: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    topic_key: Optional[str] = None
    author_id: Optional[str] = None
    as_of_time: Optional[datetime] = None
    posted_at: Optional[datetime] = None
    duration_seconds: Optional[int] = Field(default=None, ge=0)
    language: Optional[str] = None
    locale: Optional[str] = None
    content_type: Optional[str] = None
    signal_hints: Dict[str, Any] = Field(default_factory=dict)


class RecommendationQueryInput(BaseModel):
    query_id: Optional[str] = None
    description: Optional[str] = None
    hashtags: List[str] = Field(default_factory=list)
    mentions: List[str] = Field(default_factory=list)
    text: Optional[str] = None
    topic_key: Optional[str] = None
    author_id: Optional[str] = None
    audience: Optional[Any] = None
    primary_cta: Optional[str] = None
    language: Optional[str] = None
    locale: Optional[str] = None
    content_type: Optional[str] = None
    as_of_time: Optional[datetime] = None
    signal_hints: Dict[str, Any] = Field(default_factory=dict)


class CorpusScope(BaseModel):
    topic_key: Optional[str] = None
    language: Optional[str] = None
    locale: Optional[str] = None
    content_type: Optional[str] = None
    max_candidates: int = Field(default=500, ge=1, le=2000)
    exclude_video_ids: List[str] = Field(default_factory=list)


class RecommendationRequest(BaseModel):
    objective: str
    as_of_time: Optional[datetime] = Field(default_factory=datetime.utcnow)
    query: RecommendationQueryInput
    candidates: List[RecommendationCandidateInput] = Field(default_factory=list)
    corpus_scope: Optional[CorpusScope] = None
    user_context: Dict[str, Any] = Field(default_factory=dict)
    language: Optional[str] = None
    locale: Optional[str] = None
    content_type: Optional[str] = None
    candidate_ids: List[str] = Field(default_factory=list)
    policy_overrides: Dict[str, Any] = Field(default_factory=dict)
    portfolio: Dict[str, Any] = Field(default_factory=dict)
    graph_controls: Dict[str, Any] = Field(default_factory=dict)
    trajectory_controls: Dict[str, Any] = Field(default_factory=dict)
    explainability: Dict[str, Any] = Field(default_factory=dict)
    routing: Dict[str, Any] = Field(default_factory=dict)
    top_k: int = Field(default=20, ge=1, le=200)
    retrieve_k: int = Field(default=200, ge=1, le=1000)
    debug: bool = False


class FabricExtractRequest(BaseModel):
    video_id: str
    as_of_time: Optional[datetime] = Field(default_factory=datetime.utcnow)
    caption: str = ""
    hashtags: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    transcript_text: Optional[str] = None
    ocr_text: Optional[str] = None
    duration_seconds: Optional[int] = Field(default=None, ge=0)
    content_type: Optional[str] = None
    source_updated_at: Optional[datetime] = None
    hints: Dict[str, Any] = Field(default_factory=dict)


def _bundle_dir() -> Path:
    return Path(settings.recommender_bundle_dir)


def _build_runtime() -> RecommenderRuntime:
    bundle_dir = _bundle_dir()
    if not bundle_dir.exists():
        raise RuntimeError(
            f"Recommender bundle directory not found: {bundle_dir}. "
            "Train recommender and point RECOMMENDER_BUNDLE_DIR to the bundle path."
        )
    return RecommenderRuntime(bundle_dir=bundle_dir)


def _load_fabric() -> FeatureFabric:
    calibration_path = settings.fabric_calibration_path
    if not calibration_path:
        return FeatureFabric()
    path = Path(calibration_path)
    if not path.exists():
        return FeatureFabric()
    try:
        payload = path.read_text(encoding="utf-8")
        return FeatureFabric(calibration_artifacts=json.loads(payload))
    except Exception:
        return FeatureFabric()


from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Recommendation Service", version="v1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import logging as _logging

_service_logger = _logging.getLogger("recommendation.service")

_runtime: Optional[RecommenderRuntime] = None
_runtime_marker: Optional[Tuple[str, int]] = None
_fabric = _load_fabric()
_corpus_resolver = CorpusResolver()
_metrics: Dict[str, Any] = {
    "recommendations_requests_total": 0,
    "recommendations_latency_ms": [],
    "recommendations_comment_trace_items_total": 0,
    "recommendations_alignment_trace_items_total": 0,
    "recommendations_low_alignment_items_total": 0,
    "fabric_extract_requests_total": 0,
    "fabric_extract_latency_ms": [],
    "fabric_missingness_reason_total": {},
}


def _record_latency(key: str, value_ms: float, max_keep: int = 512) -> None:
    bucket = _metrics.setdefault(key, [])
    if not isinstance(bucket, list):
        return
    bucket.append(float(value_ms))
    if len(bucket) > max_keep:
        del bucket[: len(bucket) - max_keep]


def _latency_summary(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"p50_ms": 0.0, "p95_ms": 0.0, "count": 0.0}
    ordered = sorted(values)
    idx50 = int(round((len(ordered) - 1) * 0.5))
    idx95 = int(round((len(ordered) - 1) * 0.95))
    return {
        "p50_ms": round(ordered[idx50], 4),
        "p95_ms": round(ordered[idx95], 4),
        "count": float(len(values)),
    }


def _bundle_marker() -> Tuple[str, int]:
    bundle_dir = _bundle_dir()
    try:
        resolved = str(bundle_dir.resolve())
    except Exception:
        resolved = str(bundle_dir)
    manifest_path = bundle_dir / "manifest.json"
    try:
        manifest_mtime_ns = int(manifest_path.stat().st_mtime_ns)
    except OSError:
        manifest_mtime_ns = -1
    return resolved, manifest_mtime_ns


def _ensure_runtime(force_reload: bool = False) -> RecommenderRuntime:
    global _runtime, _runtime_marker
    marker = _bundle_marker()
    if force_reload or _runtime is None or _runtime_marker != marker:
        try:
            runtime = _build_runtime()
        except Exception:
            _runtime = None
            _runtime_marker = None
            raise
        _runtime = runtime
        _runtime_marker = marker
    return _runtime


@app.on_event("startup")
def _startup_prewarm() -> None:
    """Eagerly load the recommender runtime so the first request is fast."""
    _service_logger.info("startup: pre-warming recommender runtime...")
    try:
        runtime = _ensure_runtime()
        _service_logger.info("startup: runtime loaded from %s", runtime.bundle_dir)
        if runtime.retriever is not None:
            warmup_query = {
                "row_id": "__warmup__",
                "query_id": "__warmup__",
                "text": "test",
                "caption": "test",
                "hashtags": [],
                "keywords": [],
                "search_query": "test",
                "topic_key": "",
                "content_type": "",
                "language": "",
                "locale": "",
                "author_id": None,
                "as_of_time": None,
            }
            try:
                runtime.retriever.retrieve(
                    query_row=warmup_query, top_k=1, objective="engagement",
                    return_metadata=False,
                )
                _service_logger.info("startup: warm-up retrieval complete")
            except Exception as warn:
                _service_logger.warning("startup: warm-up retrieval skipped: %s", warn)
    except Exception as error:
        _service_logger.error("startup: failed to pre-warm runtime: %s", error)

    # Pre-load video analysis ML models so first /v1/video/analyze is fast
    _service_logger.info("startup: pre-loading video analysis models...")
    try:
        from src.recommendation.video.analyzer import (
            _load_whisper_model,
            _load_ocr_reader,
            _load_blip,
            _load_keybert,
        )
        _load_whisper_model()
        _service_logger.info("startup: Whisper model loaded")
        _load_ocr_reader()
        _service_logger.info("startup: EasyOCR reader loaded")
        _load_blip()
        _service_logger.info("startup: BLIP captioner loaded")
        _load_keybert()
        _service_logger.info("startup: KeyBERT model loaded")
    except Exception as video_err:
        _service_logger.warning("startup: video model pre-load failed: %s", video_err)


@app.get("/v1/warmup")
def warmup_check() -> Dict[str, Any]:
    """Readiness probe — returns ready only when the runtime is loaded."""
    if _runtime is None:
        return {"ready": False, "reason": "runtime_not_loaded"}
    return {"ready": True, "bundle_dir": str(_runtime.bundle_dir)}


@app.get("/v1/health")
def health() -> Dict[str, Any]:
    try:
        runtime = _ensure_runtime()
    except Exception as error:
        return {
            "ok": False,
            "status": "degraded",
            "reason": str(error),
        }
    return {
        "ok": True,
        "status": "ready",
        "bundle_dir": str(runtime.bundle_dir),
        "retriever_loaded": runtime.retriever is not None,
        "retriever_load_warning": runtime.retriever_load_warning,
        "comment_index_loaded": runtime.comment_index is not None,
    }


@app.get("/v1/compatibility")
def compatibility() -> Dict[str, Any]:
    try:
        runtime = _ensure_runtime()
    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "recommender_unavailable",
                "fallback_mode": True,
                "reason": str(error),
            },
        ) from error
    return runtime.compatibility_payload()


@app.post("/v1/recommendations")
def recommendations(request: RecommendationRequest) -> Dict[str, Any]:
    try:
        runtime = _ensure_runtime()
    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "recommender_unavailable",
                "fallback_mode": True,
                "reason": str(error),
            },
        ) from error

    try:
        _metrics["recommendations_requests_total"] = int(
            _metrics.get("recommendations_requests_total", 0)
        ) + 1
        started = time.perf_counter()
        resolved_candidates = [item.model_dump(mode="python") for item in request.candidates]
        if not resolved_candidates and request.corpus_scope is not None:
            scope = CorpusScopeSpec(
                topic_key=request.corpus_scope.topic_key,
                language=request.corpus_scope.language,
                locale=request.corpus_scope.locale,
                content_type=request.corpus_scope.content_type,
                max_candidates=request.corpus_scope.max_candidates,
                exclude_video_ids=request.corpus_scope.exclude_video_ids,
            )
            resolved_candidates = _corpus_resolver.resolve_candidates(scope)
        payload = runtime.recommend(
            objective=request.objective,
            as_of_time=request.as_of_time,
            query=request.query.model_dump(mode="python"),
            candidates=resolved_candidates,
            user_context=request.user_context,
            top_k=request.top_k,
            retrieve_k=request.retrieve_k,
            language=request.language,
            locale=request.locale,
            content_type=request.content_type,
            candidate_ids=request.candidate_ids,
            policy_overrides=request.policy_overrides,
            portfolio=request.portfolio,
            graph_controls=request.graph_controls,
            trajectory_controls=request.trajectory_controls,
            explainability=request.explainability,
            routing=request.routing,
            debug=request.debug,
        )
        _record_latency(
            "recommendations_latency_ms",
            (time.perf_counter() - started) * 1000.0,
        )
        item_rows = payload.get("items") if isinstance(payload, dict) else []
        if isinstance(item_rows, list):
            trace_count = sum(
                1
                for item in item_rows
                if isinstance(item, dict) and isinstance(item.get("comment_trace"), dict)
            )
            alignment_trace_count = 0
            low_alignment_count = 0
            for item in item_rows:
                if not isinstance(item, dict):
                    continue
                trace = item.get("comment_trace")
                if not isinstance(trace, dict):
                    continue
                if trace.get("alignment_score") is not None:
                    alignment_trace_count += 1
                try:
                    alignment_score = float(trace.get("alignment_score") or 0.0)
                except (TypeError, ValueError):
                    alignment_score = 0.0
                if alignment_score < 0.35:
                    low_alignment_count += 1
            _metrics["recommendations_comment_trace_items_total"] = int(
                _metrics.get("recommendations_comment_trace_items_total", 0)
            ) + trace_count
            _metrics["recommendations_alignment_trace_items_total"] = int(
                _metrics.get("recommendations_alignment_trace_items_total", 0)
            ) + alignment_trace_count
            _metrics["recommendations_low_alignment_items_total"] = int(
                _metrics.get("recommendations_low_alignment_items_total", 0)
            ) + low_alignment_count
        return payload
    except RoutingContractError as error:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_routing_contract",
                "fallback_mode": True,
                "reason": str(error),
            },
        ) from error
    except ArtifactCompatibilityError as error:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "incompatible_artifact",
                "fallback_mode": True,
                "reason": str(error),
            },
        ) from error
    except RecommenderStageTimeoutError as error:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "recommender_stage_timeout",
                "fallback_mode": True,
                "reason": str(error),
                "stage": error.stage,
                "elapsed_ms": round(error.elapsed_ms, 6),
                "budget_ms": round(error.budget_ms, 6),
            },
        ) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "recommender_failed",
                "fallback_mode": True,
                "reason": str(error),
            },
        ) from error


@app.post("/v1/fabric/extract")
def fabric_extract(request: FabricExtractRequest) -> Dict[str, Any]:
    _metrics["fabric_extract_requests_total"] = int(
        _metrics.get("fabric_extract_requests_total", 0)
    ) + 1
    started = time.perf_counter()
    out = _fabric.extract(
        FeatureFabricInput.model_validate(request.model_dump(mode="python"))
    )
    _record_latency("fabric_extract_latency_ms", (time.perf_counter() - started) * 1000.0)
    reasons: Dict[str, int] = _metrics.setdefault("fabric_missingness_reason_total", {})
    for block in (out.text.missing, out.structure.missing, out.audio.missing, out.visual.missing):
        for _, detail in block.items():
            reason = detail.reason
            reasons[reason] = reasons.get(reason, 0) + 1
    return out.model_dump(mode="python")


@app.get("/v1/metrics")
def metrics() -> Dict[str, Any]:
    return {
        "recommendations_requests_total": int(
            _metrics.get("recommendations_requests_total", 0)
        ),
        "recommendations_comment_trace_items_total": int(
            _metrics.get("recommendations_comment_trace_items_total", 0)
        ),
        "recommendations_alignment_trace_items_total": int(
            _metrics.get("recommendations_alignment_trace_items_total", 0)
        ),
        "recommendations_low_alignment_items_total": int(
            _metrics.get("recommendations_low_alignment_items_total", 0)
        ),
        "fabric_extract_requests_total": int(
            _metrics.get("fabric_extract_requests_total", 0)
        ),
        "recommendations_latency": _latency_summary(
            list(_metrics.get("recommendations_latency_ms", []))
        ),
        "fabric_extract_latency": _latency_summary(
            list(_metrics.get("fabric_extract_latency_ms", []))
        ),
        "fabric_missingness_reason_total": dict(
            _metrics.get("fabric_missingness_reason_total", {})
        ),
    }


# ---------------------------------------------------------------------------
# Hashtag recommendation endpoint
# ---------------------------------------------------------------------------

class HashtagSuggestRequest(BaseModel):
    caption: str
    k: int = Field(default=10, ge=1, le=50)
    top_n: int = Field(default=10, ge=1, le=50)
    exclude_tags: List[str] = Field(default_factory=list)
    include_neighbours: bool = False


def _hashtag_recommender_dir() -> Path:
    return Path(settings.hashtag_recommender_dir)


_hashtag_recommender = None


@app.post("/v1/hashtags/suggest")
def suggest_hashtags(request: HashtagSuggestRequest) -> Dict[str, Any]:
    global _hashtag_recommender
    if _hashtag_recommender is None:
        try:
            from .hashtag_recommender import HashtagRecommender
            _hashtag_recommender = HashtagRecommender.load(_hashtag_recommender_dir())
        except Exception as error:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "hashtag_recommender_unavailable",
                    "reason": str(error),
                },
            ) from error

    started = time.perf_counter()
    if request.include_neighbours:
        result = _hashtag_recommender.recommend_with_neighbours(
            caption=request.caption,
            k=request.k,
            top_n=request.top_n,
        )
    else:
        hashtags = _hashtag_recommender.recommend(
            caption=request.caption,
            k=request.k,
            top_n=request.top_n,
            exclude_tags=request.exclude_tags,
        )
        result = {"hashtags": hashtags}

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    result["latency_ms"] = round(elapsed_ms, 2)
    result["corpus_size"] = len(_hashtag_recommender.corpus_captions)
    return result


# ---------------------------------------------------------------------------
# Video analysis endpoint
# ---------------------------------------------------------------------------

_video_analyzer = None


@app.post("/v1/video/analyze")
async def video_analyze(file: UploadFile = File(...)) -> Dict[str, Any]:
    global _video_analyzer
    if _video_analyzer is None:
        from .video.analyzer import VideoAnalyzer
        _video_analyzer = VideoAnalyzer()

    import tempfile
    suffix = Path(file.filename).suffix if file.filename else ".mp4"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        content = await file.read()
        tmp.write(content)
        tmp.close()
        result = _video_analyzer.analyze(tmp.name)
        return result.model_dump(mode="python")
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail={"error": "video_analysis_failed", "reason": str(error)},
        ) from error
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# RAG retrieval endpoint for agentic chatbot
# ---------------------------------------------------------------------------


class ChatRAGRequest(BaseModel):
    question: str
    top_k: int = Field(default=5, ge=1, le=50)
    objective: str = "engagement"
    report_hashtags: List[str] = Field(default_factory=list)
    report_keywords: List[str] = Field(default_factory=list)
    candidate_ids: List[str] = Field(default_factory=list)
    topic_key: Optional[str] = None
    language: Optional[str] = None
    locale: Optional[str] = None
    content_type: Optional[str] = None
    primary_cta: Optional[str] = None
    transcript_hint: Optional[str] = None
    recent_user_questions: List[str] = Field(default_factory=list)


@app.post("/v1/chat/rag")
def chat_rag(request: ChatRAGRequest) -> Dict[str, Any]:
    """Retrieve relevant videos from the corpus for RAG-grounded chat."""
    global _runtime
    if _runtime is None:
        try:
            _runtime = _build_runtime()
        except Exception as error:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "recommender_unavailable",
                    "reason": str(error),
                },
            ) from error

    started = time.perf_counter()
    deduped_hashtags: List[str] = []
    seen_hashtags: set[str] = set()
    for value in request.report_hashtags:
        cleaned = str(value or "").strip().lower()
        if not cleaned:
            continue
        if not cleaned.startswith("#"):
            cleaned = f"#{cleaned}"
        if cleaned in seen_hashtags:
            continue
        seen_hashtags.add(cleaned)
        deduped_hashtags.append(cleaned)
        if len(deduped_hashtags) >= 24:
            break

    deduped_keywords: List[str] = []
    seen_keywords: set[str] = set()
    for value in request.report_keywords:
        cleaned = str(value or "").strip().lower()
        if not cleaned or cleaned in seen_keywords:
            continue
        seen_keywords.add(cleaned)
        deduped_keywords.append(cleaned)
        if len(deduped_keywords) >= 30:
            break

    context_fragments: List[str] = [request.question]
    context_fragments.extend(
        str(item or "").strip() for item in request.recent_user_questions[:3]
    )
    context_fragments.extend(deduped_keywords[:10])
    if request.primary_cta:
        context_fragments.append(f"primary cta {request.primary_cta.strip()}")
    if request.transcript_hint:
        context_fragments.append(str(request.transcript_hint).strip()[:800])
    combined_text = " ".join(fragment for fragment in context_fragments if fragment).strip()

    query_row: Dict[str, Any] = {
        "text": combined_text or request.question,
        "caption": request.question,
        "hashtags": deduped_hashtags,
        "keywords": deduped_keywords,
        "topic_key": (
            request.topic_key.strip()
            if isinstance(request.topic_key, str) and request.topic_key.strip()
            else deduped_keywords[0]
            if deduped_keywords
            else ""
        ),
        "search_query": request.question,
        "language": request.language,
        "locale": request.locale,
        "content_type": request.content_type,
        "as_of_time": datetime.utcnow(),
    }
    retrieval_constraints = {
        "language": request.language,
        "locale": request.locale,
        "content_type": request.content_type,
    }
    candidate_ids = [str(value).strip() for value in request.candidate_ids if str(value).strip()]

    if _runtime.retriever is None:
        raise HTTPException(
            status_code=503,
            detail={"error": "retriever_unavailable", "reason": "retriever_artifact_not_loaded"},
        )

    try:
        results, meta = _runtime.retriever.retrieve(
            query_row=query_row,
            top_k=request.top_k,
            objective=request.objective,
            candidate_ids=candidate_ids or None,
            retrieval_constraints=retrieval_constraints,
            return_metadata=True,
        )
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail={"error": "retrieval_failed", "reason": str(error)},
        ) from error

    # Extract video metadata for grounding context
    retrieved_videos = []
    for item in results:
        candidate_id = item.get("candidate_id", "")
        row_id = item.get("candidate_row_id", "")
        row_meta = _runtime.retriever.row_metadata.get(row_id, {})
        retrieved_videos.append({
            "video_id": candidate_id,
            "caption": row_meta.get("caption", ""),
            "hashtags": list(row_meta.get("hashtags") or []),
            "keywords": list(row_meta.get("keywords") or []),
            "author_id": row_meta.get("author_id", ""),
            "content_type": row_meta.get("content_type", ""),
            "language": row_meta.get("language"),
            "fused_score": round(float(item.get("fused_score", 0)), 4),
            "branch_scores": {
                k: round(float(v), 4)
                for k, v in (item.get("retrieval_branch_scores") or {}).items()
                if isinstance(v, (int, float))
            },
        })

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    candidate_pool_total = (
        int(meta.get("candidate_pool_total", 0))
        if isinstance(meta.get("candidate_pool_total"), (int, float))
        else len(_runtime.retriever.row_ids)
    )
    return {
        "retrieved_videos": retrieved_videos,
        "retrieval_meta": {
            "objective": str(meta.get("objective_effective") or request.objective),
            "top_k": request.top_k,
            "query_topic_key": query_row.get("topic_key") or "",
            "candidate_ids_supplied": len(candidate_ids),
            "branches_used": {
                k: v for k, v in (meta.get("branch_coverage") or {}).items()
            },
            "weights": {
                k: round(float(v), 4)
                for k, v in (meta.get("weights") or {}).items()
                if isinstance(v, (int, float))
            },
            "constraint_tier_used": meta.get("constraint_tier_used"),
            "candidate_pool_total": candidate_pool_total,
            "candidate_pool_temporal": meta.get("candidate_pool_temporal"),
            "candidate_pool_constrained": meta.get("candidate_pool_constrained"),
            "candidate_pool_constrained_unique_candidates": meta.get(
                "candidate_pool_constrained_unique_candidates"
            ),
            "retriever_artifact_version": meta.get("retriever_artifact_version"),
        },
        "latency_ms": round(elapsed_ms, 2),
    }
