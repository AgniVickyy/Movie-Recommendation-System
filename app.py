"""
Movie Recommendation System - Production API
=============================================
FastAPI service for content-based movie recommendations (IMDb Top 250).

Endpoints
---------
GET  /                      -> web UI (index.html)
GET  /health                -> liveness + model status
GET  /api/movies            -> all titles (for autocomplete)
GET  /api/movie/{title}     -> single movie details
POST /api/recommend         -> {"title": str, "top_n": int} -> ranked recommendations

Run: uvicorn app:app --host 0.0.0.0 --port 8000
"""

import difflib
import logging
import pickle
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("app")

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

# Populated at startup
STATE: dict = {"ready": False}


def load_artifacts() -> None:
    required = ["similarity_matrix.pkl", "movies_metadata.pkl", "title_index.pkl"]
    for name in required:
        path = MODELS_DIR / name
        if not path.exists():
            raise FileNotFoundError(f"Missing model artifact: {path}. Run train.py first.")
        with open(path, "rb") as f:
            STATE[name.replace(".pkl", "")] = pickle.load(f)
    STATE["ready"] = True
    log.info("Artifacts loaded: %d movies indexed.", len(STATE["movies_metadata"]))


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()
    yield
    STATE.clear()


app = FastAPI(
    title="Movie Recommendation API",
    description="Content-based recommender over the IMDb Top 250 (TF-IDF plot + weighted metadata similarity).",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class RecommendRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=300, description="Movie title (fuzzy-matched)")
    top_n: int = Field(10, ge=1, le=50, description="Number of recommendations")


def movie_payload(row: pd.Series, score: float | None = None) -> dict:
    payload = {
        "title": row["Title"],
        "year": str(row["Year"]),
        "rated": row["Rated"],
        "runtime": row["Runtime"],
        "genre": row["Genre"],
        "director": row["Director"],
        "actors": row["Actors"],
        "plot": row["Plot"],
        "imdb_rating": None if pd.isna(row["imdbRating"]) else float(row["imdbRating"]),
        "imdb_id": row["imdbID"],
        "poster": row["Poster"] if str(row["Poster"]).startswith("http") else None,
    }
    if score is not None:
        payload["similarity"] = round(float(score), 4)
    return payload


def resolve_title(query: str) -> tuple[int, str | None]:
    """Return (row_index, corrected_title_or_None). Exact -> substring -> fuzzy."""
    title_index: dict = STATE["title_index"]
    q = query.strip().lower()

    if q in title_index:
        return title_index[q], None

    substr = [t for t in title_index if q in t]
    if substr:
        best = min(substr, key=len)
        return title_index[best], STATE["movies_metadata"].loc[title_index[best], "Title"]

    fuzzy = difflib.get_close_matches(q, list(title_index), n=1, cutoff=0.6)
    if fuzzy:
        return title_index[fuzzy[0]], STATE["movies_metadata"].loc[title_index[fuzzy[0]], "Title"]

    raise HTTPException(
        status_code=404,
        detail=f"'{query}' not found in the catalog. Use GET /api/movies for available titles.",
    )


@app.get("/health")
def health():
    return {
        "status": "ok" if STATE.get("ready") else "loading",
        "movies_indexed": len(STATE.get("movies_metadata", [])),
        "version": app.version,
    }


@app.get("/api/movies")
def list_movies():
    meta: pd.DataFrame = STATE["movies_metadata"]
    return {
        "count": len(meta),
        "movies": [
            {"title": r["Title"], "year": str(r["Year"]), "imdb_rating": float(r["imdbRating"])}
            for _, r in meta.iterrows()
        ],
    }


@app.get("/api/movie/{title}")
def get_movie(title: str):
    idx, corrected = resolve_title(title)
    meta: pd.DataFrame = STATE["movies_metadata"]
    result = {"movie": movie_payload(meta.loc[idx])}
    if corrected:
        result["matched_title"] = corrected
    return result


@app.post("/api/recommend")
def recommend(req: RecommendRequest):
    idx, corrected = resolve_title(req.title)
    meta: pd.DataFrame = STATE["movies_metadata"]
    sim: np.ndarray = STATE["similarity_matrix"]

    scores = sim[idx]
    order = np.argsort(scores)[::-1]
    order = order[order != idx][: req.top_n]

    return {
        "query": req.title,
        "matched_title": corrected or meta.loc[idx, "Title"],
        "source_movie": movie_payload(meta.loc[idx]),
        "recommendations": [movie_payload(meta.loc[i], scores[i]) for i in order],
    }


@app.get("/")
def index():
    return FileResponse(BASE_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
