"""Two-tower retrieval model with in-batch softmax + log-frequency correction."""
import math
import time
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from ts.eval.metrics import recall_at_k


class ItemTower(nn.Module):
    def __init__(self, n_items: int, n_genres: int, emb_dim: int = 64):
        super().__init__()
        self.item_emb = nn.Embedding(n_items + 1, emb_dim, padding_idx=0)
        self.genre_mlp = nn.Sequential(
            nn.Linear(n_genres, 64),
            nn.ReLU(),
            nn.Linear(64, emb_dim),
        )
        self.proj = nn.Linear(emb_dim * 2, emb_dim)

    def forward(self, item_ids: torch.Tensor, genre_feats: torch.Tensor) -> torch.Tensor:
        ie = self.item_emb(item_ids)
        ge = self.genre_mlp(genre_feats)
        out = self.proj(torch.cat([ie, ge], dim=-1))
        return F.normalize(out, dim=-1)


class UserTower(nn.Module):
    def __init__(self, n_items: int, emb_dim: int = 64, history_mode: str = "last_50"):
        super().__init__()
        self.emb_dim = emb_dim
        self.history_mode = history_mode
        self.item_emb = nn.Embedding(n_items + 1, emb_dim, padding_idx=0)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 128),
            nn.ReLU(),
            nn.Linear(128, emb_dim),
        )

    def _pool(
        self,
        history_ids: torch.Tensor,  # [B, L]
        candidate_emb: Optional[torch.Tensor] = None,  # [B, D] for attention
    ) -> torch.Tensor:
        mask = (history_ids != 0).float()  # [B, L]
        embs = self.item_emb(history_ids)  # [B, L, D]
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1)

        if self.history_mode == "attention" and candidate_emb is not None:
            # Target-aware attention: candidate queries over history keys
            scores = torch.bmm(embs, candidate_emb.unsqueeze(2)).squeeze(2)  # [B, L]
            scores = scores - (1 - mask) * 1e9
            weights = torch.softmax(scores, dim=-1) * mask  # [B, L]
            weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-9)
            pooled = torch.bmm(weights.unsqueeze(1), embs).squeeze(1)  # [B, D]
        else:
            pooled = (embs * mask.unsqueeze(-1)).sum(dim=1) / denom  # [B, D]

        return pooled

    def forward(
        self,
        history_ids: torch.Tensor,
        candidate_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pooled = self._pool(history_ids, candidate_emb)
        out = self.mlp(pooled)
        return F.normalize(out, dim=-1)


def in_batch_softmax_loss(
    user_embs: torch.Tensor,  # [B, D]
    item_embs: torch.Tensor,  # [B, D]
    log_freqs: torch.Tensor,  # [B]
    temperature: float = 0.07,
) -> torch.Tensor:
    logits = torch.matmul(user_embs, item_embs.T) / temperature  # [B, B]
    logits = logits - log_freqs.unsqueeze(0)  # subtract log-freq correction
    labels = torch.arange(len(user_embs), device=user_embs.device)
    return F.cross_entropy(logits, labels)


def _build_history(
    user_id: int,
    train_df: pd.DataFrame,
    max_len: int,
) -> list:
    rows = train_df[train_df["user_id"] == user_id].sort_values("timestamp")
    return rows["item_id"].tolist()[-max_len:]


def _pad(seq: list, max_len: int) -> list:
    return [0] * (max_len - len(seq)) + seq


class TwoTowerRetriever:
    """Wraps ItemTower + UserTower with train/eval logic."""

    def __init__(
        self,
        n_items: int,
        genre_matrix: np.ndarray,
        emb_dim: int = 64,
        history_mode: str = "last_50",
    ):
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.history_mode = history_mode

        self.genre_matrix = torch.tensor(genre_matrix, dtype=torch.float32)
        n_genres = genre_matrix.shape[1]

        self.item_tower = ItemTower(n_items, n_genres, emb_dim)
        self.user_tower = UserTower(n_items, emb_dim, history_mode)

        self._history_len = int(history_mode.split("_")[1]) if "_" in history_mode else 50

        self.item_embeddings: Optional[torch.Tensor] = None
        self.item_index: Optional[np.ndarray] = None

    def _max_history(self) -> int:
        return self._history_len

    def train(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        epochs: int = 10,
        batch_size: int = 512,
        lr: float = 1e-3,
        seed: int = 42,
    ) -> list:
        torch.manual_seed(seed)
        np.random.seed(seed)

        params = list(self.item_tower.parameters()) + list(self.user_tower.parameters())
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-5)

        # Build training pairs: (user, positive_item)
        pos_df = train_df[train_df["label"] == 1].copy()
        users = pos_df["user_id"].values
        items = pos_df["item_id"].values

        # Item frequencies for log-freq correction
        item_freq = train_df.groupby("item_id").size()
        max_item = train_df["item_id"].max() + 1
        freq_vec = np.zeros(max_item, dtype=np.float32)
        for iid, cnt in item_freq.items():
            freq_vec[iid] = math.log(cnt + 1)

        max_h = self._max_history()
        user_histories = {}
        for uid in np.unique(users):
            user_histories[uid] = _pad(
                _build_history(uid, train_df, max_h), max_h
            )

        val_metrics = []
        for epoch in range(epochs):
            self.item_tower.train()
            self.user_tower.train()

            idx = np.random.permutation(len(users))
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, len(idx), batch_size):
                batch_idx = idx[start : start + batch_size]
                b_users = users[batch_idx]
                b_items = items[batch_idx]

                hist_tensor = torch.tensor(
                    [user_histories[u] for u in b_users], dtype=torch.long
                )
                item_tensor = torch.tensor(b_items, dtype=torch.long)
                genre_tensor = self.genre_matrix[b_items]

                item_embs = self.item_tower(item_tensor, genre_tensor)
                user_embs = self.user_tower(hist_tensor)

                log_freqs = torch.tensor(freq_vec[b_items], dtype=torch.float32)
                loss = in_batch_softmax_loss(user_embs, item_embs, log_freqs)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            print(f"  Epoch {epoch+1}/{epochs}  loss={avg_loss:.4f}")
            val_metrics.append(avg_loss)

        return val_metrics

    @torch.no_grad()
    def build_item_index(self, all_item_ids: np.ndarray):
        self.item_tower.eval()
        item_tensor = torch.tensor(all_item_ids, dtype=torch.long)
        genre_tensor = self.genre_matrix[all_item_ids]

        # Process in chunks to avoid OOM
        embs = []
        chunk = 256
        for s in range(0, len(all_item_ids), chunk):
            e = item_tensor[s : s + chunk]
            g = genre_tensor[s : s + chunk]
            embs.append(self.item_tower(e, g))
        self.item_embeddings = torch.cat(embs, dim=0)  # [N, D]
        self.item_index = all_item_ids

    @torch.no_grad()
    def encode_users(
        self,
        user_ids: list,
        train_df: pd.DataFrame,
    ) -> torch.Tensor:
        self.user_tower.eval()
        max_h = self._max_history()
        hists = [
            _pad(_build_history(u, train_df, max_h), max_h) for u in user_ids
        ]
        hist_tensor = torch.tensor(hists, dtype=torch.long)
        return self.user_tower(hist_tensor)

    @torch.no_grad()
    def retrieve(
        self,
        user_ids: list,
        train_df: pd.DataFrame,
        k: int = 100,
        exclude_seen: bool = True,
    ) -> dict:
        assert self.item_embeddings is not None, "Call build_item_index first"
        user_embs = self.encode_users(user_ids, train_df)  # [U, D]
        scores = torch.matmul(user_embs, self.item_embeddings.T).numpy()  # [U, N]

        results = {}
        for i, uid in enumerate(user_ids):
            sc = scores[i].copy()
            if exclude_seen:
                seen = set(train_df[train_df["user_id"] == uid]["item_id"])
                for item in seen:
                    j = np.where(self.item_index == item)[0]
                    if len(j):
                        sc[j[0]] = -1e9
            top_k = np.argsort(-sc)[:k]
            results[uid] = {
                "items": self.item_index[top_k].tolist(),
                "scores": sc[top_k].tolist(),
            }
        return results

    def evaluate(
        self,
        train_df: pd.DataFrame,
        eval_df: pd.DataFrame,
        ks: list = [100, 500],
    ) -> dict:
        all_items = np.array(sorted(train_df["item_id"].unique()))
        self.build_item_index(all_items)

        eval_users = [
            u for u in eval_df["user_id"].unique()
            if u in set(train_df["user_id"])
            and eval_df[(eval_df["user_id"] == u) & (eval_df["label"] == 1)].shape[0] > 0
        ]

        user_embs = self.encode_users(eval_users, train_df)  # [U, D]
        scores_mat = torch.matmul(user_embs, self.item_embeddings.T).numpy()

        item_to_col = {item: i for i, item in enumerate(all_items)}
        labels_mat = np.zeros((len(eval_users), len(all_items)), dtype=np.int32)
        for i, uid in enumerate(eval_users):
            pos = eval_df[(eval_df["user_id"] == uid) & (eval_df["label"] == 1)]["item_id"]
            for item in pos:
                if item in item_to_col:
                    labels_mat[i, item_to_col[item]] = 1
            # Mask seen items
            seen = set(train_df[train_df["user_id"] == uid]["item_id"])
            for item in seen:
                if item in item_to_col:
                    scores_mat[i, item_to_col[item]] = -1e9

        results = {}
        for k in ks:
            results[f"recall@{k}"] = recall_at_k(scores_mat, labels_mat, k)
        return results
