from __future__ import annotations

import pytest


fastapi = pytest.importorskip("fastapi")
testclient = pytest.importorskip("fastapi.testclient")

from src.recommendation import service  # noqa: E402


class _FakeRuntime:
    def recommend(self, **kwargs):
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
    service._runtime = _FakeRuntime()
    payload = {
        "objective": "community",
        "as_of_time": "2026-03-22T00:00:00Z",
        "query": {"text": "growth tutorial"},
        "candidates": [
            {"candidate_id": "c1", "text": "growth tips", "as_of_time": "2026-03-21T00:00:00Z"},
            {"candidate_id": "c2", "text": "cooking tips", "as_of_time": "2026-03-20T00:00:00Z"},
        ],
        "top_k": 2,
        "retrieve_k": 5,
    }
    response = client.post("/v1/recommendations", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["objective"] == "community"
    assert body["objective_effective"] == "engagement"
    assert len(body["items"]) == 1
