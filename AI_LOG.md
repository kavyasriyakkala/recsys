# AI_LOG.md — Two-Stage RecSys

## Design decisions

### Time-based splits
Used strict percentile splits (p90/p95) to prevent any future data leakage. All feature engineering (user history, popularity) is computed from train-only data.

### In-batch softmax with log-frequency correction
Subtracts `log(freq(item))` from logits before softmax to debias popular items that naturally appear more often in batches. This follows the correction used in the YouTube two-tower paper.

### DCN-v2 feature engineering
Six scalar features chosen to capture complementary signals:
- `retrieval_score`: raw two-tower dot product
- `log_item_popularity`: log-count of train ratings
- `user_prior_rating_count`: user engagement proxy
- `days_since_last_rating`: recency proxy
- `is_novel`: whether item was unseen in training
- `genre_match_score`: cosine between item and user's positive-genre profile

### History ablation strategies
- `last_10` / `last_50`: recency-biased pooling
- `all`: full history (capped at 200 for ML-1M)
- `attention`: target-aware (candidate queries over history keys)

## Run history

| Date | Command | Notes |
|------|---------|-------|
| — | `make test` | All unit tests pass |
| — | `make smoke` | Smoke run <60s on 50 users |
