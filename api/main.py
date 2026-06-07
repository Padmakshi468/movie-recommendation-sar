"""
api/main.py
===========
FastAPI production backend for the SAR Movie Recommendation System.

Endpoints:
  GET  /health                          – system health + model stats
  GET  /recommend/{user_id}?k=          – personalised recommendations
  GET  /similar/{movie_id}?k=           – similar movies
  GET  /movies?q=&limit=                – movie search / autocomplete
  POST /recommend/items                 – cold-start from a list of movie IDs
  GET  /users?limit=                    – list known user IDs
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.persistence import load, exists, load_metrics

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ── app ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="SAR Movie Recommendation API",
    description="Microsoft SAR collaborative filtering on MovieLens 100K",
    version="2.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# ── global state ──────────────────────────────────────────────────────────────
_model      = None
_item_meta: Optional[pd.DataFrame] = None
_title_map:  dict[int, str]  = {}
_genres_map: dict[int, str]  = {}
_metrics:    dict             = {}


@app.on_event("startup")
def _startup():
    global _model, _item_meta, _title_map, _genres_map, _metrics
    if not exists():
        logger.warning("No trained model found. Run: python scripts/train.py")
        return
    _model, _item_meta = load()
    _title_map  = dict(zip(_item_meta["item_id"], _item_meta["title"]))
    _genres_map = dict(zip(_item_meta["item_id"], _item_meta.get("genres", pd.Series(dtype=str))))
    _metrics    = load_metrics()
    logger.info("Model loaded. %d users | %d items", len(_model.user2idx), len(_model.item2idx))


# ── schemas ───────────────────────────────────────────────────────────────────
class MovieOut(BaseModel):
    item_id: int
    title:   str
    genres:  str = ""
    score:   Optional[float] = None


class RecommendResponse(BaseModel):
    user_id:         int
    k:               int
    is_popular_fallback: bool = False
    recommendations: List[MovieOut]


class SimilarResponse(BaseModel):
    movie_id:     int
    movie_title:  str
    similar:      List[MovieOut]


class ColdStartRequest(BaseModel):
    item_ids: List[int] = Field(..., min_items=1, max_items=50)
    k:        int       = Field(10, ge=1, le=100)


class HealthResponse(BaseModel):
    status:       str
    model_loaded: bool
    n_users:      int
    n_items:      int
    similarity:   str = ""
    metrics:      dict = {}


# ── helpers ───────────────────────────────────────────────────────────────────
def _require_model():
    if _model is None:
        raise HTTPException(503, "Model not loaded. Run: python scripts/train.py")


def _enrich(item_id: int, score: Optional[float] = None) -> MovieOut:
    return MovieOut(
        item_id=item_id,
        title=_title_map.get(item_id, f"Movie #{item_id}"),
        genres=_genres_map.get(item_id, ""),
        score=round(score, 5) if score is not None else None,
    )


# ── endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health():
    loaded = _model is not None
    return HealthResponse(
        status="ok" if loaded else "model_not_loaded",
        model_loaded=loaded,
        n_users=len(_model.user2idx)  if loaded else 0,
        n_items=len(_model.item2idx)  if loaded else 0,
        similarity=getattr(_model, "similarity_type", "") if loaded else "",
        metrics=_metrics,
    )


@app.get("/recommend/{user_id}", response_model=RecommendResponse)
def recommend(
    user_id: int,
    k: int = Query(10, ge=1, le=100, description="Number of recommendations"),
):
    """Personalised top-K recommendations for a known user."""
    _require_model()
    is_fallback = False

    if user_id not in _model.user2idx:
        # popularity fallback for unknown users
        logger.info("User %d unknown — serving popularity fallback", user_id)
        item_ids = _model.popular_items(top_k=k)
        is_fallback = True
        recs = [_enrich(i) for i in item_ids]
    else:
        df = _model.recommend_k_items(user_id, top_k=k, remove_seen=True)
        if df.empty:
            item_ids = _model.popular_items(top_k=k,
                        exclude=_model.seen_items.get(user_id, set()))
            is_fallback = True
            recs = [_enrich(i) for i in item_ids]
        else:
            recs = [_enrich(int(r["item_id"]), r["score"]) for _, r in df.iterrows()]

    return RecommendResponse(user_id=user_id, k=k,
                             is_popular_fallback=is_fallback, recommendations=recs)


@app.get("/similar/{movie_id}", response_model=SimilarResponse)
def similar_movies(
    movie_id: int,
    k: int = Query(10, ge=1, le=100),
):
    """Top-K items most similar to a given movie."""
    _require_model()
    if movie_id not in _model.item2idx:
        raise HTTPException(404, f"Movie ID {movie_id} not in training catalogue.")
    df   = _model.similar_items(movie_id, top_k=k)
    sims = [_enrich(int(r["similar_item_id"]), r["score"]) for _, r in df.iterrows()]
    return SimilarResponse(
        movie_id=movie_id,
        movie_title=_title_map.get(movie_id, f"Movie #{movie_id}"),
        similar=sims,
    )


@app.get("/movies")
def search_movies(
    q:     str   = Query("", description="Title search query"),
    limit: int   = Query(20, ge=1, le=200),
):
    """Search movies by title substring (case-insensitive)."""
    _require_model()
    if _item_meta is None or _item_meta.empty:
        return {"movies": []}
    if q.strip():
        mask = _item_meta["title"].str.contains(q.strip(), case=False, na=False)
        results = _item_meta[mask].head(limit)
    else:
        results = _item_meta.head(limit)

    movies = [
        {"item_id": int(r["item_id"]),
         "title":   r["title"],
         "genres":  r.get("genres", "")}
        for _, r in results.iterrows()
    ]
    return {"query": q, "count": len(movies), "movies": movies}


@app.post("/recommend/items")
def recommend_from_items(payload: ColdStartRequest):
    """Cold-start: recommend based on a list of liked movie IDs."""
    _require_model()
    df = _model.recommend_from_items(payload.item_ids, top_k=payload.k)
    if df.empty:
        # fallback to popular
        item_ids = _model.popular_items(top_k=payload.k,
                                        exclude=set(payload.item_ids))
        recs = [_enrich(i) for i in item_ids]
    else:
        recs = [_enrich(int(r["item_id"]), r["score"]) for _, r in df.iterrows()]
    return {
        "seed_item_ids": payload.item_ids,
        "k": payload.k,
        "recommendations": [r.dict() for r in recs],
    }


@app.get("/users")
def list_users(limit: int = Query(50, ge=1, le=943)):
    """List a sample of known user IDs."""
    _require_model()
    users = sorted(_model.user2idx.keys())[:limit]
    return {"users": users, "total": len(_model.user2idx)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)
