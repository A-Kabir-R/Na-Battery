"""Autograd derivative and physics-backpropagation tests for NaPINN-Q."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.pinn.models import NaPINNQ, count_parameters
from src.pinn.physics import autograd_du_ds, pde_residual


def test_stress_requires_grad_and_du_ds_finite() -> None:
    model = NaPINNQ(feature_dim=4, solution_hidden_dims=(16, 8), rate_hidden_dims=(8,))
    stress = torch.randn(10, requires_grad=True)
    features = torch.randn(10, 4)
    u_hat = model.solution(stress, features)
    du_ds = autograd_du_ds(u_hat, stress)
    assert stress.requires_grad
    assert du_ds is not None
    assert torch.isfinite(du_ds).all()


def test_pde_loss_reaches_solution_parameters() -> None:
    torch.manual_seed(0)
    model = NaPINNQ(feature_dim=3, solution_hidden_dims=(16, 8), rate_hidden_dims=(8,))
    stress = torch.randn(8, requires_grad=True)
    features = torch.randn(8, 3)
    u_hat = model.solution(stress, features)
    r_hat = model.rate(stress, features, u_hat)
    du_ds = autograd_du_ds(u_hat, stress)
    pde_loss = pde_residual(du_ds, r_hat).pow(2).mean()
    assert pde_loss.requires_grad
    pde_loss.backward()

    solution_grad_norm = sum(
        float(p.grad.detach().pow(2).sum().item())
        for p in model.solution.parameters() if p.grad is not None
    ) ** 0.5
    rate_grad_norm = sum(
        float(p.grad.detach().pow(2).sum().item())
        for p in model.rate.parameters() if p.grad is not None
    ) ** 0.5
    assert solution_grad_norm > 0
    assert rate_grad_norm > 0


def test_softplus_rate_is_nonnegative() -> None:
    model = NaPINNQ(feature_dim=2, solution_hidden_dims=(4,), rate_hidden_dims=(4,))
    stress = torch.randn(50, requires_grad=True)
    features = torch.randn(50, 2)
    u_hat = model.solution(stress, features)
    r_hat = model.rate(stress, features, u_hat)
    assert (r_hat >= -1e-7).all()


def test_parameter_count_within_budget() -> None:
    model = NaPINNQ(feature_dim=10, solution_hidden_dims=(64, 64, 32),
                    rate_hidden_dims=(32, 16))
    n = count_parameters(model)
    assert 500 < n < 50_000
