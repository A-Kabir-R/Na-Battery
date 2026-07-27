"""Finite forward pass on both DNN-Q and NaPINN-Q."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.pinn.models import DNNQ, NaPINNQ


def test_pinn_forward_is_finite() -> None:
    model = NaPINNQ(feature_dim=4, solution_hidden_dims=(16, 8), rate_hidden_dims=(8,))
    stress = torch.randn(20, requires_grad=True)
    features = torch.randn(20, 4)
    u, r = model(stress, features)
    assert torch.isfinite(u).all()
    assert torch.isfinite(r).all()


def test_dnn_forward_is_finite() -> None:
    model = DNNQ(feature_dim=4, solution_hidden_dims=(16, 8))
    stress = torch.randn(20, requires_grad=True)
    features = torch.randn(20, 4)
    u = model(stress, features)
    assert torch.isfinite(u).all()
