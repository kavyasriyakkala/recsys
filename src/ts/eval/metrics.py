"""Recall@K and NDCG@K — vectorized and brute-force implementations."""
import numpy as np


def _dcg(labels_ranked: np.ndarray) -> float:
    return float(sum(r / np.log2(i + 2) for i, r in enumerate(labels_ranked)))


def recall_at_k_single(scores: np.ndarray, labels: np.ndarray, k: int) -> float:
    n_pos = labels.sum()
    if n_pos == 0:
        return 0.0
    top_k = np.argsort(-scores)[:k]
    return float(labels[top_k].sum() / n_pos)


def ndcg_at_k_single(scores: np.ndarray, labels: np.ndarray, k: int) -> float:
    n_pos = int(labels.sum())
    if n_pos == 0:
        return 0.0
    top_k = np.argsort(-scores)[:k]
    dcg = _dcg(labels[top_k])
    idcg = _dcg(np.ones(min(n_pos, k)))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(
    scores_matrix: np.ndarray,  # [n_users, n_items]
    labels_matrix: np.ndarray,  # [n_users, n_items]
    k: int,
) -> float:
    n_users = scores_matrix.shape[0]
    top_k_idx = np.argpartition(-scores_matrix, min(k, scores_matrix.shape[1] - 1), axis=1)[:, :k]
    vals = []
    for u in range(n_users):
        n_pos = labels_matrix[u].sum()
        if n_pos == 0:
            continue
        vals.append(labels_matrix[u][top_k_idx[u]].sum() / n_pos)
    return float(np.mean(vals)) if vals else 0.0


def ndcg_at_k(
    scores_matrix: np.ndarray,
    labels_matrix: np.ndarray,
    k: int,
) -> float:
    n_users = scores_matrix.shape[0]
    top_k_idx = np.argsort(-scores_matrix, axis=1)[:, :k]
    vals = []
    for u in range(n_users):
        n_pos = int(labels_matrix[u].sum())
        if n_pos == 0:
            continue
        dcg = _dcg(labels_matrix[u][top_k_idx[u]])
        idcg = _dcg(np.ones(min(n_pos, k)))
        if idcg > 0:
            vals.append(dcg / idcg)
    return float(np.mean(vals)) if vals else 0.0


def recall_at_k_brute(
    scores_matrix: np.ndarray,
    labels_matrix: np.ndarray,
    k: int,
) -> float:
    """Reference implementation using single-user loops."""
    vals = []
    for u in range(scores_matrix.shape[0]):
        v = recall_at_k_single(scores_matrix[u], labels_matrix[u], k)
        n_pos = labels_matrix[u].sum()
        if n_pos > 0:
            vals.append(v)
    return float(np.mean(vals)) if vals else 0.0


def ndcg_at_k_brute(
    scores_matrix: np.ndarray,
    labels_matrix: np.ndarray,
    k: int,
) -> float:
    """Reference implementation using single-user loops."""
    vals = []
    for u in range(scores_matrix.shape[0]):
        v = ndcg_at_k_single(scores_matrix[u], labels_matrix[u], k)
        n_pos = labels_matrix[u].sum()
        if n_pos > 0:
            vals.append(v)
    return float(np.mean(vals)) if vals else 0.0
