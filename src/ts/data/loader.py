"""MovieLens-1M loader with time-based splits and no data leakage."""
import os
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

DATA_DIR = Path("data")
ML1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
ML1M_ZIP = DATA_DIR / "ml-1m.zip"
ML1M_DIR = DATA_DIR / "ml-1m"
MIN_RATINGS = 20
POSITIVE_THRESHOLD = 4


def download_data():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not ML1M_ZIP.exists():
        print("Downloading MovieLens-1M...")
        urlretrieve(ML1M_URL, ML1M_ZIP)
        print("Download complete.")
    if not ML1M_DIR.exists():
        print("Extracting...")
        with zipfile.ZipFile(ML1M_ZIP) as zf:
            zf.extractall(DATA_DIR)
        print("Extraction complete.")


def load_ratings() -> pd.DataFrame:
    return pd.read_csv(
        ML1M_DIR / "ratings.dat",
        sep="::",
        names=["user_id", "item_id", "rating", "timestamp"],
        engine="python",
    )


def load_movies() -> pd.DataFrame:
    return pd.read_csv(
        ML1M_DIR / "movies.dat",
        sep="::",
        names=["item_id", "title", "genres"],
        engine="python",
        encoding="latin-1",
    )


def build_genre_encoder(item_meta: pd.DataFrame):
    all_genres = set()
    for g in item_meta["genres"]:
        all_genres.update(g.split("|"))
    genre_vocab = sorted(all_genres)
    genre_to_idx = {g: i for i, g in enumerate(genre_vocab)}
    return genre_vocab, genre_to_idx


def encode_genres(item_meta: pd.DataFrame, genre_to_idx: dict) -> np.ndarray:
    n_items = item_meta["item_id"].max() + 1
    n_genres = len(genre_to_idx)
    genre_matrix = np.zeros((n_items, n_genres), dtype=np.float32)
    for _, row in item_meta.iterrows():
        for g in row["genres"].split("|"):
            if g in genre_to_idx:
                genre_matrix[row["item_id"], genre_to_idx[g]] = 1.0
    return genre_matrix


def load_and_split(smoke: bool = False):
    """Load and split MovieLens-1M with strict time-based splits."""
    download_data()

    ratings = load_ratings()
    item_meta = load_movies()

    # Filter users with too few ratings
    counts = ratings.groupby("user_id").size()
    valid_users = counts[counts >= MIN_RATINGS].index
    ratings = ratings[ratings["user_id"].isin(valid_users)].copy()

    if smoke:
        rng = np.random.RandomState(42)
        smoke_users = rng.choice(
            ratings["user_id"].unique(),
            size=min(50, ratings["user_id"].nunique()),
            replace=False,
        )
        ratings = ratings[ratings["user_id"].isin(smoke_users)].copy()

    # Time-based split — no leakage by construction
    p90 = ratings["timestamp"].quantile(0.90)
    p95 = ratings["timestamp"].quantile(0.95)

    train_df = ratings[ratings["timestamp"] < p90].copy()
    val_df = ratings[(ratings["timestamp"] >= p90) & (ratings["timestamp"] < p95)].copy()
    test_df = ratings[ratings["timestamp"] >= p95].copy()

    for df in [train_df, val_df, test_df]:
        df["label"] = (df["rating"] >= POSITIVE_THRESHOLD).astype(int)

    def _stats(name, df):
        print(
            f"{name:6s}: {df['user_id'].nunique():5d} users, "
            f"{df['item_id'].nunique():5d} items, "
            f"{len(df):7d} interactions"
        )

    _stats("Train", train_df)
    _stats("Val", val_df)
    _stats("Test", test_df)

    return train_df, val_df, test_df, item_meta
