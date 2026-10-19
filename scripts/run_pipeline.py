"""End-to-end pipeline: download → split → baselines → retriever →
candidates → dcn → history_study → serving_benchmark → reports/final_results.json
"""
import argparse
import json
import os
import pickle
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ts.data.loader import load_and_split, build_genre_encoder, encode_genres
from ts.models.baselines import PopularityBaseline
from ts.models.two_tower import TwoTowerRetriever
from ts.models.dcn import DCNReranker
from ts.eval.history_study import run_history_study
from ts.serve.benchmark import run_inline_benchmark


def get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "no-git"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Fast run on 50 users")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs-retriever", type=int, default=10)
    parser.add_argument("--epochs-dcn", type=int, default=10)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    if args.smoke:
        args.epochs_retriever = 2
        args.epochs_dcn = 3
        out_file = args.out or "reports/smoke_results.json"
    else:
        out_file = args.out or "reports/final_results.json"

    Path("reports").mkdir(exist_ok=True)
    t_start = time.time()

    print("=" * 60)
    print("STEP 1: Load and split data")
    print("=" * 60)
    train_df, val_df, test_df, item_meta = load_and_split(smoke=args.smoke)

    genre_vocab, genre_to_idx = build_genre_encoder(item_meta)
    genre_matrix = encode_genres(item_meta, genre_to_idx)

    n_items = int(max(
        train_df["item_id"].max(),
        val_df["item_id"].max(),
        test_df["item_id"].max(),
    ))
    item_popularity = dict(train_df.groupby("item_id").size())

    print("\n" + "=" * 60)
    print("STEP 2: Popularity baseline")
    print("=" * 60)
    baseline = PopularityBaseline()
    baseline.fit(train_df)
    baseline_val = baseline.evaluate(train_df, val_df, ks=[10, 50, 100])
    baseline_test = baseline.evaluate(train_df, test_df, ks=[10, 50, 100])
    print("  Val:", {k: f"{v:.4f}" for k, v in baseline_val.items()})
    print("  Test:", {k: f"{v:.4f}" for k, v in baseline_test.items()})

    print("\n" + "=" * 60)
    print("STEP 3: Two-tower retriever")
    print("=" * 60)
    retriever = TwoTowerRetriever(n_items, genre_matrix, emb_dim=64, history_mode="last_50")
    retriever.train(
        train_df, val_df,
        epochs=args.epochs_retriever,
        seed=args.seed,
    )
    retriever_val = retriever.evaluate(train_df, val_df, ks=[100, 500])
    retriever_test = retriever.evaluate(train_df, test_df, ks=[100, 500])
    print("  Val:", {k: f"{v:.4f}" for k, v in retriever_val.items()})
    print("  Test:", {k: f"{v:.4f}" for k, v in retriever_test.items()})

    print("\n" + "=" * 60)
    print("STEP 4: Generate candidates for DCN")
    print("=" * 60)
    all_items = np.array(sorted(train_df["item_id"].unique()))
    retriever.build_item_index(all_items)

    val_users = [
        u for u in val_df["user_id"].unique()
        if u in set(train_df["user_id"].values)
    ]
    test_users = [
        u for u in test_df["user_id"].unique()
        if u in set(train_df["user_id"].values)
    ]

    print(f"  Generating candidates for {len(val_users)} val users...")
    val_candidates = retriever.retrieve(val_users, train_df, k=100)
    print(f"  Generating candidates for {len(test_users)} test users...")
    test_candidates = retriever.retrieve(test_users, train_df, k=100)

    print("\n" + "=" * 60)
    print("STEP 5: DCN-v2 reranker")
    print("=" * 60)
    reranker = DCNReranker()
    reranker.train(
        val_candidates, train_df, val_df,
        item_popularity, genre_matrix,
        epochs=args.epochs_dcn,
        seed=args.seed,
    )
    dcn_val = reranker.evaluate(
        val_candidates, train_df, val_df, item_popularity, genre_matrix, ks=[10]
    )
    dcn_test = reranker.evaluate(
        test_candidates, train_df, test_df, item_popularity, genre_matrix, ks=[10]
    )
    print("  Val:", {k: f"{v:.4f}" for k, v in dcn_val.items()})
    print("  Test:", {k: f"{v:.4f}" for k, v in dcn_test.items()})

    print("\n" + "=" * 60)
    print("STEP 6: History length ablation")
    print("=" * 60)
    history_results = run_history_study(
        train_df, val_df, item_meta, n_items, genre_matrix,
        seeds=[42, 123, 456],
        epochs=5,
        smoke=args.smoke,
    )

    print("\n" + "=" * 60)
    print("STEP 7: Serving benchmark (inline)")
    print("=" * 60)
    state = {
        "retriever": retriever,
        "reranker": reranker,
        "train_df": train_df,
        "item_popularity": item_popularity,
        "genre_matrix": genre_matrix,
        "feat_mean": reranker.feat_mean,
        "feat_std": reranker.feat_std,
    }
    # Save state for FastAPI app
    with open("reports/serve_state.pkl", "wb") as f:
        pickle.dump(state, f)

    bench = run_inline_benchmark(state, k=10, n_requests=100, seed=args.seed)

    elapsed = time.time() - t_start
    print(f"\nTotal pipeline time: {elapsed/60:.1f} min")

    results = {
        "meta": {
            "git_commit": get_git_commit(),
            "seed": args.seed,
            "smoke": args.smoke,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(elapsed, 1),
        },
        "data": {
            "n_train": len(train_df),
            "n_val": len(val_df),
            "n_test": len(test_df),
            "n_users": int(train_df["user_id"].nunique()),
            "n_items": n_items,
        },
        "baseline_val": baseline_val,
        "baseline_test": baseline_test,
        "retriever_val": retriever_val,
        "retriever_test": retriever_test,
        "dcn_val": dcn_val,
        "dcn_test": dcn_test,
        "history_study": history_results,
        "serving": bench,
    }

    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_file}")

    return results


if __name__ == "__main__":
    main()
