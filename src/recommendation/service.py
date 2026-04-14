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
    as_of_time: datetime
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
    as_of_time: datetime
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
    query_row: Dict[str, Any] = {
        "text": request.question,
        "caption": request.question,
        "hashtags": [],
        "keywords": [],
        "topic_key": "",
        "search_query": request.question,
        "as_of_time": datetime.utcnow(),
    }

    try:
        results, meta = _runtime.retriever.retrieve(
            query_row=query_row,
            top_k=request.top_k,
            objective=request.objective,
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
    return {
        "retrieved_videos": retrieved_videos,
        "retrieval_meta": {
            "objective": request.objective,
            "top_k": request.top_k,
            "branches_used": {
                k: v for k, v in (meta.get("branch_coverage") or {}).items()
            },
            "total_indexed": meta.get("total_indexed", 0),
        },
        "latency_ms": round(elapsed_ms, 2),
    }
