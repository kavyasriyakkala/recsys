"""Benchmark the serving endpoint: 100 sequential requests, report p50/p95 latency."""
import time
import pickle
import random
from pathlib import Path

import numpy as np
import httpx


def run_benchmark(
    base_url: str = "http://localhost:8000",
    n_requests: int = 100,
    k: int = 10,
    seed: int = 42,
) -> dict:
    random.seed(seed)

    state_path = Path("reports/serve_state.pkl")
    if state_path.exists():
        with open(state_path, "rb") as f:
            state = pickle.load(f)
        users = state["train_df"]["user_id"].unique().tolist()
    else:
        # Fallback: use hardcoded user range
        users = list(range(1, 200))

    sample_users = random.choices(users, k=n_requests)
    latencies = []

    with httpx.Client(base_url=base_url, timeout=30.0) as client:
        for uid in sample_users:
            t0 = time.perf_counter()
            resp = client.post("/recommend", json={"user_id": int(uid), "k": k})
            elapsed_ms = (time.perf_counter() - t0) * 1000
            if resp.status_code == 200:
                latencies.append(elapsed_ms)
            else:
                print(f"  Error for user {uid}: {resp.status_code}")

    if not latencies:
        return {"error": "No successful requests"}

    p50 = float(np.percentile(latencies, 50))
    p95 = float(np.percentile(latencies, 95))
    mean = float(np.mean(latencies))

    print(f"  n_requests={len(latencies)}  p50={p50:.1f}ms  p95={p95:.1f}ms  mean={mean:.1f}ms")
    return {"p50_ms": p50, "p95_ms": p95, "mean_ms": mean, "n_requests": len(latencies)}


def run_inline_benchmark(state: dict, k: int = 10, n_requests: int = 100, seed: int = 42) -> dict:
    """Benchmark without starting a server — calls pipeline directly."""
    import sys
    sys.path.insert(0, "src")

    from ts.models.dcn import _build_features

    random.seed(seed)
    np.random.seed(seed)

    retriever = state["retriever"]
    reranker = state.get("reranker")
    train_df = state["train_df"]
    item_popularity = state["item_popularity"]
    genre_matrix = state["genre_matrix"]

    users = train_df["user_id"].unique().tolist()
    sample_users = random.choices(users, k=n_requests)
    latencies = []

    for uid in sample_users:
        t0 = time.perf_counter()
        candidates = retriever.retrieve([uid], train_df, k=100, exclude_seen=True)
        if reranker is not None:
            cand = candidates[uid]
            feats = _build_features(
                uid, cand["items"], cand["scores"], train_df, item_popularity, genre_matrix
            )
            _ = reranker.score(feats)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed_ms)

    p50 = float(np.percentile(latencies, 50))
    p95 = float(np.percentile(latencies, 95))
    mean = float(np.mean(latencies))
    print(f"  Inline benchmark  p50={p50:.1f}ms  p95={p95:.1f}ms  mean={mean:.1f}ms")
    return {"p50_ms": p50, "p95_ms": p95, "mean_ms": mean, "n_requests": len(latencies)}
