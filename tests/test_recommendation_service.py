from __future__ import annotations

import pytest


fastapi = pytest.importorskip("fastapi")
testclient = pytest.importorskip("fastapi.testclient")

from src.recommendation import service  # noqa: E402
from src.recommendation.learning.inference import ArtifactCompatibilityError  # noqa: E402


class _FakeRuntime:
    def __init__(self) -> None:
        self.last_kwargs = None

    def recommend(self, **kwargs):
        self.last_kwargs = kwargs
        return {
            "objective": kwargs["objective"],
            "objective_effective": "engagement",
            "generated_at": "2026-03-22T00:00:00Z",
            "fallback_mode": False,
            "items": [
                {
                    "candidate_id": "c1",
                    "rank": 1,
                    "score": 0.9,
                    "similarity": {"sparse": 0.9, "dense": 0.8, "fused": 0.85},
                    "trace": {"objective_model": "engagement", "ranker_backend": "test"},
                }
            ],
        }


def test_health_degraded_when_bundle_missing(monkeypatch):
    client = testclient.TestClient(service.app)
    monkeypatch.setenv("RECOMMENDER_BUNDLE_DIR", "/tmp/non-existent-recommender-bundle")
    service._runtime = None
    response = client.get("/v1/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is False
    assert payload["status"] == "degraded"


def test_recommendations_endpoint_works_with_loaded_runtime(monkeypatch):
    client = testclient.TestClient(service.app)
    runtime = _FakeRuntime()
    service._runtime = runtime
    payload = {
        "objective": "community",
        "as_of_time": "2026-03-22T00:00:00Z",
        "query": {"text": "growth tutorial", "language": "en"},
        "candidates": [
            {"candidate_id": "c1", "text": "growth tips", "as_of_time": "2026-03-21T00:00:00Z", "language": "en"},
            {"candidate_id": "c2", "text": "cooking tips", "as_of_time": "2026-03-20T00:00:00Z", "language": "en"},
        ],
        "language": "en",
        "locale": "en-us",
        "content_type": "tutorial",
        "candidate_ids": ["c1", "c2"],
        "policy_overrides": {"strict_language": True, "max_items_per_author": 1},
        "portfolio": {
            "enabled": True,
            "weights": {"reach": 0.45, "conversion": 0.35, "durability": 0.20},
            "risk_aversion": 0.10,
            "candidate_pool_cap": 120,
        },
        "graph_controls": {"enable_graph_branch": True},
        "trajectory_controls": {"enabled": True},
        "explainability": {"enabled": True, "top_features": 4, "neighbor_k": 2},
        "routing": {
            "objective_requested": "community",
            "objective_effective": "engagement",
            "track": "post_publication",
            "allow_fallback": True,
            "required_compat": {"component": "recommender-learning-v1"},
        },
        "top_k": 2,
        "retrieve_k": 5,
    }
    response = client.post("/v1/recommendations", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["objective"] == "community"
    assert body["objective_effective"] == "engagement"
    assert len(body["items"]) == 1
    assert runtime.last_kwargs is not None
    assert runtime.last_kwargs["candidate_ids"] == ["c1", "c2"]
    assert runtime.last_kwargs["policy_overrides"] == {"strict_language": True, "max_items_per_author": 1}
    assert runtime.last_kwargs["portfolio"] == {
        "enabled": True,
        "weights": {"reach": 0.45, "conversion": 0.35, "durability": 0.2},
        "risk_aversion": 0.1,
        "candidate_pool_cap": 120,
    }
    assert runtime.last_kwargs["graph_controls"] == {"enable_graph_branch": True}
    assert runtime.last_kwargs["trajectory_controls"] == {"enabled": True}
    assert runtime.last_kwargs["explainability"] == {"enabled": True, "top_features": 4, "neighbor_k": 2}
    assert runtime.last_kwargs["routing"]["objective_effective"] == "engagement"


def test_fabric_extract_and_metrics_endpoints():
    client = testclient.TestClient(service.app)
    payload = {
        "video_id": "x1",
        "as_of_time": "2026-03-24T12:00:00Z",
        "caption": "Hook and payoff test",
        "hashtags": ["#growth"],
        "keywords": ["growth"],
        "duration_seconds": 30,
        "content_type": "tutorial",
    }
    response = client.post("/v1/fabric/extract", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["fabric_version"] == "fabric.v2"
    metrics = client.get("/v1/metrics")
    assert metrics.status_code == 200
    metric_body = metrics.json()
    assert metric_body["fabric_extract_requests_total"] >= 1


def test_compatibility_endpoint_works_with_loaded_runtime():
    client = testclient.TestClient(service.app)
    runtime = _FakeRuntime()
    runtime.bundle_dir = "artifacts/recommender/latest"
    runtime.compatibility_payload = lambda: {
        "ok": True,
        "bundle_id": "latest",
        "fingerprints": {"component": "recommender-learning-v1"},
        "mismatches": [],
    }
    service._runtime = runtime
    response = client.get("/v1/compatibility")
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["bundle_id"] == "latest"


def test_recommendations_endpoint_returns_409_for_incompatible_artifact():
    client = testclient.TestClient(service.app)

    class _CompatibilityErrorRuntime:
        def recommend(self, **kwargs):
            del kwargs
            raise ArtifactCompatibilityError("incompatible_artifact: feature_schema_hash mismatch")

    service._runtime = _CompatibilityErrorRuntime()
    payload = {
        "objective": "engagement",
        "as_of_time": "2026-03-22T00:00:00Z",
        "query": {"text": "growth tutorial"},
        "candidates": [{"candidate_id": "c1", "text": "growth tips"}],
        "routing": {
            "objective_requested": "engagement",
            "objective_effective": "engagement",
            "required_compat": {"feature_schema_hash": "x"},
        },
    }
    response = client.post("/v1/recommendations", json=payload)
    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["error"] == "incompatible_artifact"
