"""
FastAPI wrapper around the HybridRecommender.

Run with:
    uvicorn app.api:app --reload

Interactive docs:
    http://localhost:8000/docs          (Swagger UI)
    http://localhost:8000/redoc         (ReDoc)

Example requests:
    curl http://localhost:8000/health
    curl http://localhost:8000/recommend/42?top_n=5
    curl http://localhost:8000/users | python -m json.tool | head -20
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from src.data_loader import load_dataset, Dataset
from src.hybrid import HybridRecommender


# ── Startup: load data + train model once ─────────────────────────────────────

class _State:
    ds: Dataset | None = None
    model: HybridRecommender | None = None
    load_time: float = 0.0

state = _State()


@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.time()
    state.ds = load_dataset()
    state.model = HybridRecommender(state.ds).fit()
    state.load_time = round(time.time() - t0, 2)
    yield


app = FastAPI(
    title="Movie Recommendation API",
    description=(
        "Hybrid SVD + TF-IDF content recommender trained on MovieLens ml-latest-small. "
        "Each recommendation includes a per-component score breakdown."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ── Response schemas ──────────────────────────────────────────────────────────

class Recommendation(BaseModel):
    movie_id: int
    title: str
    collaborative_score: float
    content_score: float
    hybrid_score: float


class RecommendResponse(BaseModel):
    user_id: int
    top_n: int
    recommendations: list[Recommendation]


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["meta"])
def health():
    """Liveness check — also reports model load time."""
    return {
        "status": "ok",
        "model_loaded": state.model is not None,
        "load_time_s": state.load_time,
    }


@app.get("/users", tags=["meta"])
def list_users():
    """Return all valid user IDs (use one with /recommend/{user_id})."""
    if state.ds is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"user_ids": sorted(int(uid) for uid in state.ds.user_index.keys())}


@app.get(
    "/recommend/{user_id}",
    response_model=RecommendResponse,
    tags=["recommend"],
)
def recommend(
    user_id: int,
    top_n: Annotated[int, Query(ge=1, le=50, description="List length (1-50)")] = 10,
):
    """
    Return the top-N hybrid recommendations for a user, with the
    collaborative (SVD) and content (TF-IDF) score breakdown for each pick.
    """
    if state.model is None or state.ds is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if user_id not in state.ds.user_index:
        raise HTTPException(
            status_code=404,
            detail=f"user_id {user_id} not found. Call /users for valid IDs.",
        )

    recs = state.model.recommend(user_id, top_n=top_n)
    items = []
    for movie_id, _ in recs:
        ex = state.model.explain(user_id, movie_id)
        items.append(
            Recommendation(
                movie_id=movie_id,
                title=ex["movie"],
                collaborative_score=ex["collaborative"],
                content_score=ex["content"],
                hybrid_score=ex["final"],
            )
        )

    return RecommendResponse(user_id=user_id, top_n=top_n, recommendations=items)
