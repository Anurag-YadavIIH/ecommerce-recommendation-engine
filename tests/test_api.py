"""
FastAPI smoke tests via TestClient (no running server needed).

Run with:  pytest -q
"""
import pytest
from fastapi.testclient import TestClient

from app.api import app


@pytest.fixture(scope="module")
def client():
    # Context-manager form triggers the lifespan (model load) exactly once
    # for the whole module, then tears down after all tests finish.
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_recommend_returns_correct_shape(client):
    r = client.get("/recommend/1?top_n=5")
    assert r.status_code == 200
    body = r.json()
    assert body["user_id"] == 1
    assert body["top_n"] == 5
    assert len(body["recommendations"]) == 5


def test_recommend_response_fields(client):
    r = client.get("/recommend/1?top_n=1")
    item = r.json()["recommendations"][0]
    assert {"movie_id", "title", "collaborative_score", "content_score", "hybrid_score"} == set(item)
    assert 0.0 <= item["hybrid_score"] <= 1.0


def test_recommend_unknown_user_returns_404(client):
    r = client.get("/recommend/999999")
    assert r.status_code == 404
    assert "not found" in r.json()["detail"].lower()


def test_recommend_top_n_clamps_at_50(client):
    r = client.get("/recommend/1?top_n=999")
    assert r.status_code == 422   # FastAPI validation rejects > 50


def test_users_endpoint(client):
    r = client.get("/users")
    assert r.status_code == 200
    ids = r.json()["user_ids"]
    assert len(ids) > 100
    assert 1 in ids
