"""Tests for model forward passes and loss behavior."""
import sys
import math
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ts.models.two_tower import (
    ItemTower, UserTower, in_batch_softmax_loss, TwoTowerRetriever
)
from ts.models.dcn import DCNv2, CrossLayer
from ts.eval.history_study import AttentionUserTower

N_ITEMS = 200
N_GENRES = 18
EMB_DIM = 64
B = 16  # batch size


@pytest.fixture
def item_tower():
    return ItemTower(N_ITEMS, N_GENRES, EMB_DIM)


@pytest.fixture
def user_tower():
    return UserTower(N_ITEMS, EMB_DIM, history_mode="last_50")


def test_two_tower_item_forward(item_tower):
    item_ids = torch.randint(1, N_ITEMS, (B,))
    genre_feats = torch.rand(B, N_GENRES)
    out = item_tower(item_ids, genre_feats)
    assert out.shape == (B, EMB_DIM), f"Expected [{B}, {EMB_DIM}], got {out.shape}"
    norms = torch.norm(out, dim=-1)
    assert torch.allclose(norms, torch.ones(B), atol=1e-5), "Output not L2-normalized"


def test_two_tower_user_forward(user_tower):
    history = torch.randint(0, N_ITEMS, (B, 50))
    out = user_tower(history)
    assert out.shape == (B, EMB_DIM), f"Expected [{B}, {EMB_DIM}], got {out.shape}"
    norms = torch.norm(out, dim=-1)
    assert torch.allclose(norms, torch.ones(B), atol=1e-5), "Output not L2-normalized"


def test_two_tower_user_padding():
    tower = UserTower(N_ITEMS, EMB_DIM)
    # All-padding history should not crash
    history = torch.zeros(B, 50, dtype=torch.long)
    out = tower(history)
    assert out.shape == (B, EMB_DIM)


def test_dcn_forward():
    model = DCNv2(input_dim=6, n_cross=3, deep_dims=(128, 64))
    x = torch.randn(B, 6)
    out = model(x)
    assert out.shape == (B, 1), f"Expected [{B}, 1], got {out.shape}"
    assert (out >= 0).all() and (out <= 1).all(), "DCN output not in [0, 1]"


def test_cross_layer_shape():
    layer = CrossLayer(dim=6)
    x0 = torch.randn(B, 6)
    x = torch.randn(B, 6)
    out = layer(x0, x)
    assert out.shape == (B, 6)


def test_in_batch_loss():
    """Loss should decrease during training and converge below log(batch_size)."""
    torch.manual_seed(0)
    item_t = ItemTower(N_ITEMS, N_GENRES, EMB_DIM)
    user_t = UserTower(N_ITEMS, EMB_DIM)
    optimizer = torch.optim.Adam(
        list(item_t.parameters()) + list(user_t.parameters()), lr=5e-2
    )

    item_ids = torch.randint(1, N_ITEMS, (B,))
    genres = torch.rand(B, N_GENRES)
    history = torch.randint(1, N_ITEMS, (B, 50))
    log_freqs = torch.zeros(B)

    def compute_loss():
        ie = item_t(item_ids, genres)
        ue = user_t(history)
        return in_batch_softmax_loss(ue, ie, log_freqs)

    loss_before = compute_loss().item()

    # Run 20 steps on the same batch
    for _ in range(20):
        loss = compute_loss()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss_after = compute_loss().item()

    assert loss_after < loss_before, (
        f"Loss did not decrease: before={loss_before:.4f} after={loss_after:.4f}"
    )
    upper_bound = math.log(B)
    assert loss_after < upper_bound, (
        f"Loss {loss_after:.4f} should be < log(B)={upper_bound:.4f} after 20 steps"
    )


def test_attention_output():
    """AttentionUserTower over 50-item history: shape correct and normalized."""
    tower = AttentionUserTower(N_ITEMS, EMB_DIM)
    history = torch.randint(0, N_ITEMS, (B, 50))
    candidate_emb = F.normalize(torch.randn(B, EMB_DIM), dim=-1)
    out = tower(history, candidate_emb)
    assert out.shape == (B, EMB_DIM), f"Expected [{B}, {EMB_DIM}], got {out.shape}"
    norms = torch.norm(out, dim=-1)
    assert torch.allclose(norms, torch.ones(B), atol=1e-5), "Attention output not L2-normalized"


def test_attention_without_candidate():
    """AttentionUserTower works without a candidate embedding."""
    tower = AttentionUserTower(N_ITEMS, EMB_DIM)
    history = torch.randint(1, N_ITEMS, (B, 50))
    out = tower(history, candidate_emb=None)
    assert out.shape == (B, EMB_DIM)
