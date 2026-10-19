# two-stage-recsys

Two-stage recommender system on MovieLens-1M: popularity baseline → two-tower retrieval → DCN-v2 reranking.

## Quick start

```bash
make setup    # install dependencies
make test     # run unit tests (fast, no GPU, no download needed)
make smoke    # full pipeline on 50 users (<60s)
make pipeline # full pipeline on all data (~90min CPU)
make serve    # start FastAPI server at :8000
```

## Architecture

| Stage | Model | Eval |
|-------|-------|------|
| Baseline | Popularity (top-K most rated) | Recall/NDCG @10,50,100 |
| Retrieval | Two-Tower (item emb + genre MLP, user mean-pool MLP) | Recall@100, Recall@500 |
| Reranking | DCN-v2 (3 cross + deep [128,64]) | AUC, Recall/NDCG @10 |
| Ablation | History strategies: last_10/50/all/attention | NDCG@10 ± std, latency ms/user |

## Results

All numbers are written to `reports/final_results.json` with git commit, seed, and timestamp.
Run `make pipeline` to generate them.

## Data splits

Strict time-based split — no leakage:
- **Train**: ratings before 90th percentile timestamp
- **Val**: ratings in [p90, p95)
- **Test**: ratings at or after p95

Positive label: rating ≥ 4.

## Serving

```bash
make serve
curl -X POST http://localhost:8000/recommend \
     -H 'Content-Type: application/json' \
     -d '{"user_id": 1, "k": 10}'
```

Pipeline: encode user → matmul retrieval (top-100) → DCN-v2 rerank (top-10).
