"""Popularity baseline: recommends top-K most-rated items in train."""
import numpy as np
import pandas as pd
from collections import Counter

from ts.eval.metrics import recall_at_k, ndcg_at_k


class PopularityBaseline:
    def __init__(self):
        self.item_counts: Counter = Counter()
        self.popular_items: list = []

    def fit(self, train_df: pd.DataFrame):
        self.item_counts = Counter(train_df["item_id"].tolist())
        self.popular_items = [item for item, _ in self.item_counts.most_common()]

    def recommend(self, user_id: int, train_df: pd.DataFrame, k: int) -> list:
        seen = set(train_df[train_df["user_id"] == user_id]["item_id"])
        recs = [item for item in self.popular_items if item not in seen]
        return recs[:k]

    def evaluate(
        self,
        train_df: pd.DataFrame,
        eval_df: pd.DataFrame,
        ks: list = [10, 50, 100],
    ) -> dict:
        all_items = np.array(self.popular_items)
        n_items = len(all_items)
        item_to_col = {item: i for i, item in enumerate(all_items)}

        # Build popularity scores vector (descending rank = decreasing score)
        pop_scores = np.array([self.item_counts[item] for item in all_items], dtype=np.float32)

        users = eval_df["user_id"].unique()
        # Only evaluate users who have train history and test positives
        eval_users = [
            u for u in users
            if u in train_df["user_id"].values
            and eval_df[eval_df["user_id"] == u]["label"].sum() > 0
        ]

        scores_mat = np.zeros((len(eval_users), n_items), dtype=np.float32)
        labels_mat = np.zeros((len(eval_users), n_items), dtype=np.int32)

        for i, u in enumerate(eval_users):
            seen = set(train_df[train_df["user_id"] == u]["item_id"])
            user_scores = pop_scores.copy()
            for item in seen:
                if item in item_to_col:
                    user_scores[item_to_col[item]] = -1e9  # exclude seen
            scores_mat[i] = user_scores

            user_pos = eval_df[(eval_df["user_id"] == u) & (eval_df["label"] == 1)]["item_id"]
            for item in user_pos:
                if item in item_to_col:
                    labels_mat[i, item_to_col[item]] = 1

        results = {}
        for k in ks:
            results[f"recall@{k}"] = recall_at_k(scores_mat, labels_mat, k)
            results[f"ndcg@{k}"] = ndcg_at_k(scores_mat, labels_mat, k)

        # Novel-only: items not seen by any user in train
        train_seen_global = set(train_df["item_id"])
        novel_mask = np.array([1 if item not in train_seen_global else 0 for item in all_items])

        novel_scores = scores_mat * novel_mask[np.newaxis, :]
        novel_scores[novel_scores == 0] = -1e9
        for k in ks:
            results[f"novel_recall@{k}"] = recall_at_k(novel_scores, labels_mat, k)
            results[f"novel_ndcg@{k}"] = ndcg_at_k(novel_scores, labels_mat, k)

        return results
