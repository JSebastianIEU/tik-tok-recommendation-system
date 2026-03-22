from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

try:
    from fastapi import FastAPI, HTTPException
except Exception as exc:  # pragma: no cover - optional dependency
    raise RuntimeError(
        "FastAPI is required for recommender service. Install fastapi and uvicorn."
    ) from exc

from .learning.inference import RecommenderRuntime


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


class RecommendationQueryInput(BaseModel):
    query_id: Optional[str] = None
    description: Optional[str] = None
    hashtags: List[str] = Field(default_factory=list)
    mentions: List[str] = Field(default_factory=list)
    text: Optional[str] = None
    topic_key: Optional[str] = None
    author_id: Optional[str] = None
    as_of_time: Optional[datetime] = None


class RecommendationRequest(BaseModel):
    objective: str
    as_of_time: datetime
    query: RecommendationQueryInput
    candidates: List[RecommendationCandidateInput]
    top_k: int = Field(default=20, ge=1, le=200)
    retrieve_k: int = Field(default=200, ge=1, le=1000)
    debug: bool = False


def _bundle_dir() -> Path:
    configured = os.getenv("RECOMMENDER_BUNDLE_DIR", "").strip()
    if configured:
        return Path(configured)
    return Path("artifacts/recommender/latest")


def _build_runtime() -> RecommenderRuntime:
    bundle_dir = _bundle_dir()
    if not bundle_dir.exists():
        raise RuntimeError(
            f"Recommender bundle directory not found: {bundle_dir}. "
            "Train recommender and point RECOMMENDER_BUNDLE_DIR to the bundle path."
        )
    return RecommenderRuntime(bundle_dir=bundle_dir)


app = FastAPI(title="Recommendation Service", version="v1")
_runtime: Optional[RecommenderRuntime] = None


@app.get("/v1/health")
def health() -> Dict[str, Any]:
    global _runtime
    if _runtime is None:
        try:
            _runtime = _build_runtime()
        except Exception as error:
            return {
                "ok": False,
                "status": "degraded",
                "reason": str(error),
            }
    return {
        "ok": True,
        "status": "ready",
        "bundle_dir": str(_runtime.bundle_dir),
    }


@app.post("/v1/recommendations")
def recommendations(request: RecommendationRequest) -> Dict[str, Any]:
    global _runtime
    if _runtime is None:
        try:
            _runtime = _build_runtime()
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
        return _runtime.recommend(
            objective=request.objective,
            as_of_time=request.as_of_time,
            query=request.query.model_dump(mode="python"),
            candidates=[item.model_dump(mode="python") for item in request.candidates],
            top_k=request.top_k,
            retrieve_k=request.retrieve_k,
            debug=request.debug,
        )
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

