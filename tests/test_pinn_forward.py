"""Finite forward pass on NaPINN-Q, including hybrid-rate variant."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.pinn.models import NaPINNQ
from src.pinn.degradation_laws import HybridRateModel


def test_pinn_forward_is_finite() -> None:
    model = NaPINNQ(feature_dim=4, solution_hidden_dims=(16, 8), rate_hidden_dims=(8,))
    stress = torch.randn(20, requires_grad=True)
    features = torch.randn(20, 4)
    u, r = model(stress, features)
    assert torch.isfinite(u).all()
    assert torch.isfinite(r).all()


def _hybrid_model(feature_dim: int = 5,
                  temp_idx: int = 2,
                  dod_idx: int = 3,
                  crate_idx: int = 4) -> NaPINNQ:
    return NaPINNQ(
        feature_dim=feature_dim,
        solution_hidden_dims=(8,),
        use_hybrid_rate=True,
        hybrid_temperature_index=temp_idx,
        hybrid_dod_index=dod_idx,
        hybrid_c_rate_index=crate_idx,
        hybrid_enable_cold_regime=True,
    )


def test_hybrid_rate_model_is_used() -> None:
    model = _hybrid_model()
    assert isinstance(model.rate, HybridRateModel)
    assert model.use_hybrid_rate


def test_hybrid_forward_is_finite() -> None:
    model = _hybrid_model()
    n = 10
    features = torch.zeros(n, 5)
    features[:, 2] = 25.0    # temperature_C
    features[:, 3] = 100.0   # DOD_pct
    features[:, 4] = 1.0     # c_rate
    stress = torch.linspace(0.0, 1.0, n).requires_grad_(True)
    u, r = model(stress, features)
    assert torch.isfinite(u).all()
    assert torch.isfinite(r).all()
    assert (r >= 0).all(), "hybrid rate must be nonnegative"


def test_hybrid_rate_components_are_accessible() -> None:
    model = _hybrid_model()
    n = 6
    features = torch.zeros(n, 5)
    features[:, 2] = 25.0
    features[:, 3] = 100.0
    features[:, 4] = 1.0
    stress = torch.linspace(0.0, 1.0, n)
    components = model.rate_components(stress, features)
    for name in ("cycling", "cold", "calendar", "diagnostic", "residual"):
        val = getattr(components, name)
        assert torch.isfinite(val).all(), f"{name} component has non-finite values"
        assert (val >= 0).all(), f"{name} component must be nonnegative"
    assert torch.isfinite(components.total).all()


def test_rate_components_raises_for_generic_model() -> None:
    model = NaPINNQ(feature_dim=4)
    with pytest.raises(RuntimeError, match="use_hybrid_rate"):
        model.rate_components(torch.zeros(2), torch.zeros(2, 4))


def test_hybrid_requires_temperature_and_dod_indices() -> None:
    with pytest.raises(ValueError, match="hybrid_temperature_index"):
        NaPINNQ(feature_dim=5, use_hybrid_rate=True)  # missing indices
