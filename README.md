# two-stage-recsys

A two-stage movie recommender built on MovieLens-1M from scratch — popularity baseline, two-tower retrieval, and a DCN-v2 reranker. Everything runs on CPU.

## What it does

The system follows the standard two-stage pattern you see in production recommenders. First, a two-tower model retrieves the top 100 plausible candidates out of ~3,700 items. Then DCN-v2 reranks those 100 into a final top-10. This way you get fast retrieval with a cheap inner-product search, followed by a more expressive cross-feature model on a small candidate set.

Splits are strictly time-based (no leakage): train before the 90th percentile timestamp, val in [p90, p95), test at or after p95. Positive label is rating ≥ 4.

## Results

From `reports/final_results.json` (git `441185d`, seed 42, 2026-10-19).

| Model | Recall@10 | NDCG@10 |
|---|---|---|
| Popularity baseline (full item set, ~3,700 items) | 0.051 | 0.150 |
| Popularity reranking (retrieved top-100 only) | 0.226 | 0.237 |
| **DCN-v2 reranker (retrieved top-100 only)** | **0.234** | **0.243** |

The meaningful comparison is DCN-v2 versus popularity reranking on the same retrieved candidates — both score the same 100 items for the same 670 users. DCN-v2 improves Recall@10 by roughly 3.5% relative and NDCG@10 by roughly 2.5% relative. The full-set popularity baseline solves a different problem (rank all items) so it's listed for context, not as a direct baseline for the reranker.

These are single-run results, not statistical significance claims.

Retriever test: Recall@100 = 0.269, Recall@500 = 0.647. DCN-v2 test AUC = 0.660.

**History length ablation** (NDCG@10, mean ± std over 3 seeds, evaluated on val set):

| History | NDCG@10 |
|---|---|
| Last 10 items | 0.1539 ± 0.0035 |
| Last 50 items | 0.1530 ± 0.0036 |
| All available history | 0.1421 ± 0.0050 |

Capping at 10 or 50 items outperforms using the full history. The difference between 10 and 50 is small relative to run-to-run variation (±std overlap). Full history likely adds noise from old ratings.

**Serving**: p50 = 2.0 ms, p95 = 2.5 ms per request (100-request inline benchmark, CPU). This measures the two-stage recommendation path end-to-end, not production-service overhead.

## Architecture

**Two-tower retrieval**

- Item tower: learned embedding (64d) + genre MLP (multi-hot genres → 64d) → Linear(128, 64) → L2 normalize
- User tower: same item embedding, mean-pool last 50 rated items → MLP(64 → 128 → 64) → L2 normalize
- Trained with in-batch softmax loss + log-frequency correction (YouTube DNN sampling bias correction)
- Inference: pre-compute all item embeddings, score users via matmul, exclude seen items

**DCN-v2 reranker**

- 3 cross layers (`x0 * (W·x + b) + x`) stacked with deep layers [128, 64] → sigmoid
- 6 features per candidate: retrieval score, log item popularity, user rating count, item age (days since first appearance in training data), item rating count, genre match score
- Trained with BCE + positive-class upweighting on val-period candidates
- Evaluated on test-period candidates from the trained retriever

**Metrics**

All Recall@K and NDCG@K are macro-averaged: computed per user, then averaged across users. For the reranker metrics specifically, the per-user relevant-positive denominator is restricted to items that appear in the retrieved top-100 (cascade recall), not the user's full set of test positives. Users with zero positives in their retrieved candidates are excluded from both the reranker and the within-candidates baseline. The full-catalog popularity baseline uses the user's total test positives as the denominator.

## Quick start

```bash
make setup    # install dependencies
make test     # unit tests, no download needed, runs in ~10s
make smoke    # full pipeline on 50 users (<60s), writes reports/smoke_results.json
make pipeline # full pipeline, ~15 min on CPU, writes reports/final_results.json
make serve    # FastAPI server at :8000
```

```bash
curl -X POST http://localhost:8000/recommend \
     -H 'Content-Type: application/json' \
     -d '{"user_id": 1, "k": 10}'
```

## Project layout

```
src/ts/
  data/loader.py          # download, time-based split, genre encoding
  models/two_tower.py     # item tower, user tower, in-batch loss
  models/dcn.py           # DCN-v2, feature builder, train/eval
  models/baselines.py     # popularity baseline
  eval/metrics.py         # recall@k, ndcg@k, auc — macro-averaged per user
  eval/history_study.py   # history length ablation
  serve/app.py            # FastAPI endpoint
  serve/benchmark.py      # latency benchmark
scripts/
  run_pipeline.py         # end-to-end pipeline, writes final_results.json
reports/
  final_results.json      # all numbers with git commit, seed, and timestamp
```

## Known limitations

**Attention history strategy** — `AttentionUserTower` is implemented but excluded from this ablation. The implementation requires the candidate item embedding as input (target-aware attention), which doesn't fit the current retrieval path of pre-computing all item embeddings and scoring via a single matmul. This exclusion is specific to this design: other retrieval architectures can incorporate target-aware signals. `AttentionUserTower` is kept in the codebase as a reranking-stage component where per-candidate scoring is acceptable.

**DCN val vs test** — DCN-v2 is trained on val-period candidates, so its val metrics are in-sample. The held-out numbers are the test metrics: Recall@10 = 0.234, NDCG@10 = 0.243, AUC = 0.660.

**Cascade recall denominator** — Recall@K in the reranking stage uses only the positives that appear in the retrieved top-100, not the user's total test positives. This is cascade recall — the right metric for evaluating a reranker, but not comparable to open-retrieval recall from the baseline row.
