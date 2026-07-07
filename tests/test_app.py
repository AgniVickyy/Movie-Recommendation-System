"""API + model tests. Run: pytest tests/ -v (requires models/ from train.py)."""
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # triggers lifespan -> loads artifacts
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["movies_indexed"] == 250


def test_list_movies(client):
    r = client.get("/api/movies")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 250
    assert any(m["title"] == "The Shawshank Redemption" for m in body["movies"])


def test_recommend_exact_title(client):
    r = client.post("/api/recommend", json={"title": "The Godfather", "top_n": 5})
    assert r.status_code == 200
    body = r.json()
    assert len(body["recommendations"]) == 5
    titles = [m["title"] for m in body["recommendations"]]
    assert "The Godfather: Part II" in titles
    assert body["source_movie"]["title"] == "The Godfather"
    # scores sorted descending, within [0,1]
    scores = [m["similarity"] for m in body["recommendations"]]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_recommend_fuzzy_title(client):
    r = client.post("/api/recommend", json={"title": "godfathr", "top_n": 3})
    assert r.status_code == 200
    assert "Godfather" in r.json()["matched_title"]


def test_recommend_substring(client):
    r = client.post("/api/recommend", json={"title": "shawshank", "top_n": 3})
    assert r.status_code == 200
    assert r.json()["matched_title"] == "The Shawshank Redemption"


def test_recommend_unknown_title(client):
    r = client.post("/api/recommend", json={"title": "zzzzqqqq nonexistent 12345"})
    assert r.status_code == 404


def test_recommend_validation(client):
    assert client.post("/api/recommend", json={"title": ""}).status_code == 422
    assert client.post("/api/recommend", json={"title": "Heat", "top_n": 0}).status_code == 422
    assert client.post("/api/recommend", json={"title": "Heat", "top_n": 999}).status_code == 422


def test_no_self_recommendation(client):
    r = client.post("/api/recommend", json={"title": "Inception", "top_n": 10})
    titles = [m["title"] for m in r.json()["recommendations"]]
    assert "Inception" not in titles


def test_get_single_movie(client):
    r = client.get("/api/movie/Heat")
    assert r.status_code == 200
    assert r.json()["movie"]["director"] == "Michael Mann"


def test_index_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "CineMatch" in r.text
