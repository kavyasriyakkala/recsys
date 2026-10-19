"""DCN-v2 reranker: 3 cross layers + deep [128, 64], trained on retriever candidates."""
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from ts.eval.metrics import recall_at_k, ndcg_at_k

FEATURE_NAMES = [
    "retrieval_score",
    "log_item_popularity",
    "user_prior_rating_count",
    "days_since_last_rating",
    "is_novel",
    "genre_match_score",
]
INPUT_DIM = len(FEATURE_NAMES)


class CrossLayer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.W = nn.Linear(dim, dim, bias=False)
        self.b = nn.Parameter(torch.zeros(dim))

    def forward(self, x0: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return x0 * (self.W(x) + self.b) + x


class DCNv2(nn.Module):
    def __init__(self, input_dim: int = INPUT_DIM, n_cross: int = 3, deep_dims=(128, 64)):
        super().__init__()
        self.cross_layers = nn.ModuleList([CrossLayer(input_dim) for _ in range(n_cross)])
        deep = []
        in_d = input_dim
        for d in deep_dims:
            deep += [nn.Linear(in_d, d), nn.ReLU()]
            in_d = d
        self.deep = nn.Sequential(*deep)
        self.output = nn.Linear(input_dim + deep_dims[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, input_dim] -> [B, 1]
        x0 = x
        xc = x
        for layer in self.cross_layers:
            xc = layer(x0, xc)
        xd = self.deep(x)
        return torch.sigmoid(self.output(torch.cat([xc, xd], dim=-1)))


def _build_features(
    user_id: int,
    candidate_items: list,
    candidate_scores: list,
    train_df: pd.DataFrame,
    item_popularity: dict,
    genre_matrix: np.ndarray,
) -> np.ndarray:
    """Build feature matrix [n_candidates, INPUT_DIM] for one user."""
    max_pop = max(item_popularity.values()) if item_popularity else 1

    user_train = train_df[train_df["user_id"] == user_id]
    user_items = set(user_train["item_id"])
    last_ts = user_train["timestamp"].max() if len(user_train) else 0
    n_prior = len(user_train)

    # Genre preference: mean genre vector over positive train items
    pos_train = user_train[user_train["label"] == 1]["item_id"].tolist()
    if pos_train and genre_matrix is not None:
        valid = [i for i in pos_train if i < len(genre_matrix)]
        user_genre = genre_matrix[valid].mean(axis=0) if valid else np.zeros(genre_matrix.shape[1])
    else:
        user_genre = np.zeros(genre_matrix.shape[1]) if genre_matrix is not None else np.array([])

    feats = []
    for item, score in zip(candidate_items, candidate_scores):
        pop = item_popularity.get(item, 0)
        log_pop = np.log1p(pop)
        is_novel = 0.0 if item in user_items else 1.0

        item_rows = user_train[user_train["item_id"] == item]
        if len(item_rows) > 0:
            item_last_ts = item_rows["timestamp"].max()
            days_since = (last_ts - item_last_ts) / 86400.0
        else:
            days_since = 999.0  # user never rated this item

        if genre_matrix is not None and item < len(genre_matrix) and len(user_genre):
            item_genre = genre_matrix[item]
            norm_u = np.linalg.norm(user_genre)
            norm_i = np.linalg.norm(item_genre)
            genre_match = float(
                np.dot(user_genre, item_genre) / (norm_u * norm_i + 1e-9)
            )
        else:
            genre_match = 0.0

        feats.append([
            float(score),
            float(log_pop),
            float(n_prior),
            float(days_since),
            float(is_novel),
            float(genre_match),
        ])
    return np.array(feats, dtype=np.float32)


class DCNReranker:
    def __init__(self):
        self.model = DCNv2(INPUT_DIM)
        self.feat_mean: Optional[np.ndarray] = None
        self.feat_std: Optional[np.ndarray] = None

    def _normalize(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        if fit:
            self.feat_mean = X.mean(axis=0)
            self.feat_std = X.std(axis=0) + 1e-8
        return (X - self.feat_mean) / self.feat_std

    def build_dataset(
        self,
        candidates: dict,  # {user_id: {"items": [...], "scores": [...]}}
        train_df: pd.DataFrame,
        eval_df: pd.DataFrame,
        item_popularity: dict,
        genre_matrix: np.ndarray,
        n_neg_per_pos: int = 4,
        rng: np.random.RandomState = None,
    ):
        if rng is None:
            rng = np.random.RandomState(42)

        all_X, all_y = [], []
        for uid, cand in candidates.items():
            user_eval = eval_df[eval_df["user_id"] == uid]
            if len(user_eval) == 0:
                continue
            pos_set = set(user_eval[user_eval["label"] == 1]["item_id"])
            items = cand["items"]
            scores = cand["scores"]

            item_to_score = dict(zip(items, scores))
            pos_items = [i for i in items if i in pos_set]
            neg_items = [i for i in items if i not in pos_set]

            if not pos_items:
                continue

            # Balance: pos + n_neg_per_pos negatives
            selected_neg = rng.choice(
                neg_items, size=min(len(pos_items) * n_neg_per_pos, len(neg_items)), replace=False
            ).tolist() if neg_items else []

            selected = pos_items + selected_neg
            sel_scores = [item_to_score[i] for i in selected]
            labels = [1] * len(pos_items) + [0] * len(selected_neg)

            feats = _build_features(uid, selected, sel_scores, train_df, item_popularity, genre_matrix)
            all_X.append(feats)
            all_y.extend(labels)

        if not all_X:
            return np.zeros((0, INPUT_DIM), dtype=np.float32), np.zeros(0, dtype=np.float32)
        return np.vstack(all_X), np.array(all_y, dtype=np.float32)

    def train(
        self,
        candidates: dict,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        item_popularity: dict,
        genre_matrix: np.ndarray,
        epochs: int = 10,
        batch_size: int = 512,
        lr: float = 1e-3,
        seed: int = 42,
    ) -> dict:
        torch.manual_seed(seed)
        np.random.seed(seed)
        rng = np.random.RandomState(seed)

        X, y = self.build_dataset(
            candidates, train_df, val_df, item_popularity, genre_matrix, rng=rng
        )
        if len(X) == 0:
            print("  No training data for DCN.")
            return {}

        X = self._normalize(X, fit=True)
        pos_weight = torch.tensor([(y == 0).sum() / max((y == 1).sum(), 1)], dtype=torch.float32)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)
        criterion = nn.BCELoss(reduction="none")

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)

        idx = np.arange(len(X_t))
        for epoch in range(epochs):
            self.model.train()
            rng.shuffle(idx)
            epoch_loss = 0.0
            n_b = 0
            for s in range(0, len(idx), batch_size):
                bi = torch.tensor(idx[s : s + batch_size])
                bx, by = X_t[bi], y_t[bi]
                pred = self.model(bx).squeeze(1)
                # Manual pos_weight scaling
                weights = torch.where(by == 1, pos_weight.squeeze(), torch.ones(1))
                loss = (criterion(pred, by) * weights).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_b += 1
            if (epoch + 1) % 5 == 0:
                print(f"  DCN epoch {epoch+1}/{epochs}  loss={epoch_loss/max(n_b,1):.4f}")

        return {}

    @torch.no_grad()
    def score(self, features: np.ndarray) -> np.ndarray:
        self.model.eval()
        X = self._normalize(features, fit=False)
        X_t = torch.tensor(X, dtype=torch.float32)
        return self.model(X_t).squeeze(1).numpy()

    def rerank(
        self,
        user_id: int,
        candidates: dict,
        train_df: pd.DataFrame,
        item_popularity: dict,
        genre_matrix: np.ndarray,
        k: int = 10,
    ) -> list:
        cand = candidates[user_id]
        feats = _build_features(
            user_id, cand["items"], cand["scores"], train_df, item_popularity, genre_matrix
        )
        scores = self.score(feats)
        top_k = np.argsort(-scores)[:k]
        return [cand["items"][i] for i in top_k]

    def evaluate(
        self,
        candidates: dict,
        train_df: pd.DataFrame,
        eval_df: pd.DataFrame,
        item_popularity: dict,
        genre_matrix: np.ndarray,
        ks: list = [10],
    ) -> dict:
        all_users = [u for u in candidates if u in eval_df["user_id"].values]
        all_items_sorted = sorted(
            {item for cand in candidates.values() for item in cand["items"]}
        )
        item_to_col = {item: i for i, item in enumerate(all_items_sorted)}
        n_items = len(all_items_sorted)

        scores_mat = np.full((len(all_users), n_items), -1e9, dtype=np.float32)
        labels_mat = np.zeros((len(all_users), n_items), dtype=np.int32)
        all_preds, all_labels = [], []

        for row, uid in enumerate(all_users):
            cand = candidates[uid]
            feats = _build_features(
                uid, cand["items"], cand["scores"], train_df, item_popularity, genre_matrix
            )
            dcn_scores = self.score(feats)
            for col, (item, sc) in enumerate(zip(cand["items"], dcn_scores)):
                if item in item_to_col:
                    scores_mat[row, item_to_col[item]] = sc

            user_pos = set(
                eval_df[(eval_df["user_id"] == uid) & (eval_df["label"] == 1)]["item_id"]
            )
            for item in cand["items"]:
                if item in item_to_col:
                    lbl = 1 if item in user_pos else 0
                    labels_mat[row, item_to_col[item]] = lbl
                    all_preds.append(scores_mat[row, item_to_col[item]])
                    all_labels.append(lbl)

        results = {}
        if len(set(all_labels)) > 1:
            results["auc"] = float(roc_auc_score(all_labels, all_preds))
        for k in ks:
            results[f"recall@{k}"] = recall_at_k(scores_mat, labels_mat, k)
            results[f"ndcg@{k}"] = ndcg_at_k(scores_mat, labels_mat, k)
        return results


