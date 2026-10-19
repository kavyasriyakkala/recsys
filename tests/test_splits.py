"""Tests for data splits: no leakage, time ordering, user coverage."""
import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ts.data.loader import load_and_split


@pytest.fixture(scope="module")
def splits():
    return load_and_split(smoke=True)


def test_no_leakage(splits):
    train_df, val_df, test_df, _ = splits
    assert train_df["timestamp"].max() < val_df["timestamp"].min(), (
        f"Leakage: train max ts={train_df['timestamp'].max()} >= "
        f"val min ts={val_df['timestamp'].min()}"
    )


def test_time_ordering(splits):
    train_df, val_df, test_df, _ = splits
    assert val_df["timestamp"].max() < test_df["timestamp"].min(), (
        f"Val/test ordering violated: val max={val_df['timestamp'].max()} "
        f">= test min={test_df['timestamp'].min()}"
    )


def test_user_coverage(splits):
    train_df, val_df, test_df, _ = splits
    train_users = set(train_df["user_id"])
    test_users = set(test_df["user_id"])
    missing = test_users - train_users
    assert len(missing) == 0 or len(missing) / len(test_users) < 0.05, (
        f"{len(missing)} test users have no training history "
        f"({100*len(missing)/len(test_users):.1f}%)"
    )


def test_positive_labels(splits):
    train_df, val_df, test_df, _ = splits
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        assert "label" in df.columns, f"No 'label' column in {name}"
        assert df["label"].isin([0, 1]).all(), f"Labels not binary in {name}"
        assert df["label"].sum() > 0, f"No positive labels in {name}"


def test_split_sizes(splits):
    train_df, val_df, test_df, _ = splits
    total = len(train_df) + len(val_df) + len(test_df)
    assert len(train_df) / total > 0.7, "Train split seems too small"
    assert len(val_df) > 0
    assert len(test_df) > 0
