"""FastAPI serving endpoint: POST /recommend {user_id, k}."""
import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Two-Stage RecSys")

# Global state loaded at startup
_state: dict = {}


class RecommendRequest(BaseModel):
    user_id: int
    k: int = 10


class RecommendResponse(BaseModel):
    user_id: int
    items: list
    scores: list


def _load_state():
    state_path = Path("reports/serve_state.pkl")
    if not state_path.exists():
        return False
    with open(state_path, "rb") as f:
        data = pickle.load(f)
    _state.update(data)
    return True


@app.on_event("startup")
def startup():
    loaded = _load_state()
    if not loaded:
        print("WARNING: serve_state.pkl not found — run pipeline first.")


@app.get("/health")
def health():
    return {"status": "ok", "loaded": bool(_state)}


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    if not _state:
        raise HTTPException(status_code=503, detail="Model not loaded")

    user_id = req.user_id
    k = req.k

    retriever = _state["retriever"]
    reranker = _state.get("reranker")
    train_df = _state["train_df"]
    item_popularity = _state["item_popularity"]
    genre_matrix = _state["genre_matrix"]

    if user_id not in set(train_df["user_id"].values):
        raise HTTPException(status_code=404, detail=f"User {user_id} not found in training data")

    # Stage 1: retrieve top-100
    candidates = retriever.retrieve([user_id], train_df, k=100, exclude_seen=True)

    # Stage 2: rerank with DCN-v2
    if reranker is not None and _state.get("feat_mean") is not None:
        items = reranker.rerank(
            user_id, candidates, train_df, item_popularity, genre_matrix, k=k
        )
        # scores from DCN
        from ts.models.dcn import _build_features, _build_item_first_seen
        item_first_seen = _state.get("item_first_seen") or _build_item_first_seen(train_df)
        cand = candidates[user_id]
        feats = _build_features(
            user_id, cand["items"], cand["scores"], train_df, item_popularity,
            genre_matrix, item_first_seen
        )
        dcn_scores = reranker.score(feats)
        top_k_idx = np.argsort(-dcn_scores)[:k]
        final_items = [cand["items"][i] for i in top_k_idx]
        final_scores = [float(dcn_scores[i]) for i in top_k_idx]
    else:
        cand = candidates[user_id]
        final_items = cand["items"][:k]
        final_scores = cand["scores"][:k]

    return RecommendResponse(
        user_id=user_id,
        items=final_items,
        scores=final_scores,
    )
