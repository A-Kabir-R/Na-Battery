"""Governing-equation residual and integral consistency tests."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.pinn.models import NaPINNQ
from src.pinn.physics import (
    autograd_du_ds, discrete_state_transition_residual, integral_transition,
    pde_residual,
)


def test_discrete_state_transition_residual() -> None:
    u_next = torch.tensor([0.9, 0.85])
    u_current = torch.tensor([0.95, 0.92])
    r_current = torch.tensor([0.05, 0.07])
    delta = torch.tensor([1.0, 1.0])
    residual = discrete_state_transition_residual(u_next, u_current, r_current, delta)
    expected = u_next - u_current + r_current * delta
    assert torch.allclose(residual, expected)


def test_integral_transition_recovers_constant_rate() -> None:
    """When r is constant k, integral is u_current - k * Delta_s."""
    torch.manual_seed(0)

    def solution_forward(stress, features):
        return torch.ones(stress.size(0), device=stress.device) * 0.5

    class ConstantRate:
        def __call__(self, stress, features, u):
            return torch.ones_like(u) * 0.1

    stress_current = torch.zeros(4)
    delta = torch.ones(4) * 2.0
    features = torch.zeros(4, 3)
    u_current = torch.ones(4)
    integrated = integral_transition(
        solution_forward, ConstantRate(), stress_current, delta, features, u_current,
        method="trapezoidal", n_nodes=32,
    )
    expected = u_current - 0.1 * delta
    assert torch.allclose(integrated, expected, atol=1e-4)


def test_residual_is_finite_end_to_end() -> None:
    torch.manual_seed(1)
    model = NaPINNQ(feature_dim=3, solution_hidden_dims=(8, 8), rate_hidden_dims=(8,))
    stress = torch.randn(5, requires_grad=True)
    features = torch.randn(5, 3)
    u = model.solution(stress, features)
    r = model.rate(stress, features, u)
    du = autograd_du_ds(u, stress)
    residual = pde_residual(du, r)
    assert torch.isfinite(residual).all()
