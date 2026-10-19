"""History length ablation study for the UserTower."""
import math
import time
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from ts.models.two_tower import (
    ItemTower, UserTower, in_batch_softmax_loss, _build_history, _pad, TwoTowerRetriever
)
from ts.eval.metrics import ndcg_at_k


# "attention" excluded from retrieval ablation: target-aware
# attention requires candidate-specific scoring (one forward
# pass per user-item pair), incompatible with matmul retrieval.
# AttentionUserTower class is kept for potential reranking experiments.
HISTORY_STRATEGIES = ["last_10", "last_50", "all"]


def _strategy_max_len(strategy: str) -> int:
    if strategy == "last_10":
        return 10
    if strategy == "last_50":
        return 50
    if strategy == "all":
        return 200  # effectively all for ML-1M
    if strategy == "attention":
        return 50
    return 50


class AttentionUserTower(nn.Module):
    """Target-aware attention over history items."""

    def __init__(self, n_items: int, emb_dim: int = 64):
        super().__init__()
        self.item_emb = nn.Embedding(n_items + 1, emb_dim, padding_idx=0)
        self.attn_q = nn.Linear(emb_dim, emb_dim, bias=False)
        self.attn_k = nn.Linear(emb_dim, emb_dim, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 128),
            nn.ReLU(),
            nn.Linear(128, emb_dim),
        )

    def forward(
        self,
        history_ids: torch.Tensor,  # [B, L]
        candidate_emb: Optional[torch.Tensor] = None,  # [B, D]
    ) -> torch.Tensor:
        mask = (history_ids != 0).float()  # [B, L]
        embs = self.item_emb(history_ids)  # [B, L, D]

        if candidate_emb is not None:
            q = self.attn_q(candidate_emb)  # [B, D]
            k = self.attn_k(embs)  # [B, L, D]
            scores = torch.bmm(k, q.unsqueeze(2)).squeeze(2)  # [B, L]
        else:
            scores = torch.zeros(history_ids.shape, device=history_ids.device)

        scores = scores - (1 - mask) * 1e9
        weights = torch.softmax(scores, dim=-1) * mask
        weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-9)
        pooled = torch.bmm(weights.unsqueeze(1), embs).squeeze(1)  # [B, D]
        return F.normalize(self.mlp(pooled), dim=-1)


def _train_one(
    strategy: str,
    train_df: pd.DataFrame,
    n_items: int,
    genre_matrix: np.ndarray,
    epochs: int = 5,
    batch_size: int = 512,
    lr: float = 1e-3,
    seed: int = 42,
):
    """Train a single history strategy and return (item_tower, user_module)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_genres = genre_matrix.shape[1]
    genre_t = torch.tensor(genre_matrix, dtype=torch.float32)
    max_h = _strategy_max_len(strategy)

    item_tower = ItemTower(n_items, n_genres)
    use_attention = strategy == "attention"
    if use_attention:
        user_module = AttentionUserTower(n_items)
    else:
        user_module = UserTower(n_items, history_mode=strategy)

    params = list(item_tower.parameters()) + list(user_module.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-5)

    pos_df = train_df[train_df["label"] == 1]
    users_arr = pos_df["user_id"].values
    items_arr = pos_df["item_id"].values

    item_freq = train_df.groupby("item_id").size()
    max_id = int(train_df["item_id"].max()) + 1
    freq_vec = np.zeros(max_id, dtype=np.float32)
    for iid, cnt in item_freq.items():
        freq_vec[iid] = math.log(cnt + 1)

    histories = {}
    for uid in np.unique(users_arr):
        histories[uid] = _pad(_build_history(uid, train_df, max_h), max_h)

    for epoch in range(epochs):
        item_tower.train()
        user_module.train()
        idx = np.random.permutation(len(users_arr))
        for start in range(0, len(idx), batch_size):
            bi = idx[start : start + batch_size]
            b_users = users_arr[bi]
            b_items = items_arr[bi]

            hist_t = torch.tensor([histories[u] for u in b_users], dtype=torch.long)
            item_t = torch.tensor(b_items, dtype=torch.long)
            genre_bt = genre_t[b_items]

            item_embs = item_tower(item_t, genre_bt)
            if use_attention:
                user_embs = user_module(hist_t, item_embs.detach())
            else:
                user_embs = user_module(hist_t)

            log_freqs = torch.tensor(freq_vec[b_items], dtype=torch.float32)
            loss = in_batch_softmax_loss(user_embs, item_embs, log_freqs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return item_tower, user_module, genre_t, max_h


@torch.no_grad()
def _eval_strategy(
    item_tower: nn.Module,
    user_module,
    genre_t: torch.Tensor,
    max_h: int,
    strategy: str,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    k: int = 10,
):
    item_tower.eval()
    user_module.eval()
    use_attention = strategy == "attention"

    all_items = np.array(sorted(train_df["item_id"].unique()))
    item_t = torch.tensor(all_items, dtype=torch.long)
    genre_bt = genre_t[all_items]

    # Build item embeddings
    item_embs_list = []
    for s in range(0, len(all_items), 256):
        e = item_t[s : s + 256]
        g = genre_bt[s : s + 256]
        item_embs_list.append(item_tower(e, g))
    item_embs = torch.cat(item_embs_list, dim=0)  # [N, D]

    eval_users = [
        u for u in eval_df["user_id"].unique()
        if u in set(train_df["user_id"])
        and eval_df[(eval_df["user_id"] == u) & (eval_df["label"] == 1)].shape[0] > 0
    ][:200]  # cap for speed

    item_to_col = {item: i for i, item in enumerate(all_items)}
    scores_mat = np.zeros((len(eval_users), len(all_items)), dtype=np.float32)
    labels_mat = np.zeros((len(eval_users), len(all_items)), dtype=np.int32)

    latencies = []
    for i, uid in enumerate(eval_users):
        hist = _pad(_build_history(uid, train_df, max_h), max_h)
        hist_t = torch.tensor([hist], dtype=torch.long)

        t0 = time.perf_counter()
        if use_attention:
            # Two-pass: first rough pass without candidate, then we use mean pool as proxy
            user_e = user_module(hist_t)  # [1, D]
        else:
            user_e = user_module(hist_t)  # [1, D]
        latencies.append((time.perf_counter() - t0) * 1000)

        sc = torch.matmul(user_e, item_embs.T).squeeze(0).numpy()
        seen = set(train_df[train_df["user_id"] == uid]["item_id"])
        for item in seen:
            if item in item_to_col:
                sc[item_to_col[item]] = -1e9
        scores_mat[i] = sc

        pos_items = eval_df[(eval_df["user_id"] == uid) & (eval_df["label"] == 1)]["item_id"]
        for item in pos_items:
            if item in item_to_col:
                labels_mat[i, item_to_col[item]] = 1

    metric = ndcg_at_k(scores_mat, labels_mat, k)
    avg_lat = float(np.mean(latencies))
    return metric, avg_lat


def run_history_study(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    item_meta: pd.DataFrame,
    n_items: int,
    genre_matrix: np.ndarray,
    seeds: list = [42, 123, 456],
    epochs: int = 5,
    smoke: bool = False,
) -> dict:
    if smoke:
        seeds = [42]
        epochs = 2

    strategies = HISTORY_STRATEGIES
    results = {s: {"ndcg10": [], "latency_ms": []} for s in strategies}

    for strategy in strategies:
        print(f"\n  Strategy: {strategy}")
        for seed in seeds:
            print(f"    Seed {seed}...")
            item_t, user_m, genre_t, max_h = _train_one(
                strategy, train_df, n_items, genre_matrix,
                epochs=epochs, seed=seed,
            )
            ndcg, lat = _eval_strategy(
                item_t, user_m, genre_t, max_h, strategy,
                train_df, val_df, k=10,
            )
            results[strategy]["ndcg10"].append(ndcg)
            results[strategy]["latency_ms"].append(lat)

    summary = {}
    for s in strategies:
        vals = results[s]["ndcg10"]
        lats = results[s]["latency_ms"]
        summary[s] = {
            "ndcg10_mean": float(np.mean(vals)),
            "ndcg10_std": float(np.std(vals)),
            "latency_ms_mean": float(np.mean(lats)),
        }
        print(
            f"  {s:12s}  NDCG@10={np.mean(vals):.4f}±{np.std(vals):.4f}  "
            f"lat={np.mean(lats):.2f}ms"
        )

    return summary
