"""Tests for recall and NDCG metrics: vectorized vs brute-force, canary check."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ts.eval.metrics import (
    recall_at_k, ndcg_at_k,
    recall_at_k_brute, ndcg_at_k_brute,
)


def _make_scores_labels(n_users=20, n_items=100, n_pos=5, seed=7):
    rng = np.random.RandomState(seed)
    scores = rng.randn(n_users, n_items).astype(np.float32)
    labels = np.zeros((n_users, n_items), dtype=np.int32)
    for u in range(n_users):
        pos_idx = rng.choice(n_items, n_pos, replace=False)
        labels[u, pos_idx] = 1
    return scores, labels


@pytest.mark.parametrize("k", [5, 10, 50])
def test_recall_vs_brute_force(k):
    scores, labels = _make_scores_labels()
    vec = recall_at_k(scores, labels, k)
    brute = recall_at_k_brute(scores, labels, k)
    assert abs(vec - brute) < 1e-5, f"recall@{k}: vectorized={vec:.6f} brute={brute:.6f}"


@pytest.mark.parametrize("k", [5, 10, 50])
def test_ndcg_vs_brute_force(k):
    scores, labels = _make_scores_labels()
    vec = ndcg_at_k(scores, labels, k)
    brute = ndcg_at_k_brute(scores, labels, k)
    assert abs(vec - brute) < 1e-5, f"ndcg@{k}: vectorized={vec:.6f} brute={brute:.6f}"


def test_shuffled_canary():
    """Random scores on 1-positive-per-user should give recall@1 near 1/n_items."""
    rng = np.random.RandomState(99)
    n_users, n_items = 500, 200
    scores = rng.randn(n_users, n_items).astype(np.float32)
    labels = np.zeros((n_users, n_items), dtype=np.int32)
    for u in range(n_users):
        labels[u, rng.randint(n_items)] = 1

    r1 = recall_at_k(scores, labels, k=1)
    expected = 1.0 / n_items
    # Should be near chance; allow ±3 standard deviations
    std = np.sqrt(expected * (1 - expected) / n_users)
    assert abs(r1 - expected) < 4 * std, (
        f"Canary failed: recall@1={r1:.4f}, expected≈{expected:.4f} ± {4*std:.4f}"
    )


def test_no_positives_returns_zero():
    scores = np.ones((5, 10), dtype=np.float32)
    labels = np.zeros((5, 10), dtype=np.int32)
    assert recall_at_k(scores, labels, k=5) == 0.0
    assert ndcg_at_k(scores, labels, k=5) == 0.0


def test_perfect_recall():
    n_items = 20
    scores = np.arange(n_items, 0, -1, dtype=np.float32)[np.newaxis, :]
    labels = np.zeros((1, n_items), dtype=np.int32)
    labels[0, :5] = 1  # top-5 are positive
    assert recall_at_k(scores, labels, k=5) == 1.0
    assert ndcg_at_k(scores, labels, k=5) == 1.0


def test_ndcg_sensitive_to_rank():
    """Higher-ranked positive should give better NDCG."""
    n_items = 10
    labels = np.zeros((1, n_items), dtype=np.int32)
    labels[0, 0] = 1  # one positive

    scores_rank1 = np.zeros((1, n_items), dtype=np.float32)
    scores_rank1[0, 0] = 10.0  # positive ranked first

    scores_rank5 = np.zeros((1, n_items), dtype=np.float32)
    scores_rank5[0, 0] = -10.0  # positive ranked last
    scores_rank5[0, 4] = 10.0

    ndcg_high = ndcg_at_k(scores_rank1, labels, k=5)
    ndcg_low = ndcg_at_k(scores_rank5, labels, k=5)
    assert ndcg_high > ndcg_low, "NDCG not sensitive to rank"
