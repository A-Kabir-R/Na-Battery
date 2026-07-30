"""The initial condition must be exact by construction, not penalised."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.pinn.models import NaPINNQ, SolutionNet, count_parameters
from src.pinn.physics import autograd_du_ds, integral_transition, pde_residual


FEATURES = 6
N = 12


def _inputs(seed: int = 0):
    torch.manual_seed(seed)
    features = torch.randn(N, FEATURES)
    u_current = torch.rand(N) * 0.2 + 0.8
    stress_current = torch.rand(N)
    stress_delta = torch.rand(N) * 0.5 + 0.1
    return features, u_current, stress_current, stress_current + stress_delta


def _model(seed: int = 0, **kwargs) -> NaPINNQ:
    torch.manual_seed(seed)
    defaults = dict(
        feature_dim=FEATURES, solution_hidden_dims=(16, 16),
        rate_hidden_dims=(8, 8), predict_delta_u=True, rate_uses_u_hat=False,
    )
    defaults.update(kwargs)
    return NaPINNQ(**defaults)


def test_u_hat_at_anchor_equals_u_current_exactly():
    features, u_current, stress_current, _ = _inputs()
    model = _model()
    # Perturb away from the zero init so this is not trivially satisfied.
    with torch.no_grad():
        for parameter in model.solution.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.5)
    u_anchor = model.solution(stress_current, features, u_current, stress_current)
    assert torch.allclose(u_anchor, u_current, atol=1e-6)


def test_initial_condition_holds_for_a_trained_model():
    """After gradient steps the IC must still be exact, not merely small."""
    features, u_current, stress_current, stress_next = _inputs()
    model = _model()
    target = u_current - 0.05
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(25):
        optimizer.zero_grad()
        prediction = model.solution(
            stress_next, features, u_current, stress_current,
        )
        loss = torch.nn.functional.mse_loss(prediction, target)
        loss.backward()
        optimizer.step()
    u_anchor = model.solution(stress_current, features, u_current, stress_current)
    assert torch.allclose(u_anchor, u_current, atol=1e-6)


def test_zero_initialized_model_predicts_persistence():
    features, u_current, stress_current, stress_next = _inputs()
    model = _model()
    u_next = model.solution(stress_next, features, u_current, stress_current)
    assert torch.allclose(u_next, u_current, atol=1e-7)


def test_residual_mode_requires_stress_current():
    features, u_current, stress_current, stress_next = _inputs()
    model = _model()
    with pytest.raises(ValueError):
        model.solution(stress_next, features, u_current)
    with pytest.raises(ValueError):
        model.solution(stress_next, features, None, stress_current)


def test_prediction_scales_with_the_transition_horizon():
    """u_hat - u_current must be proportional to (s - s_current)."""
    features, u_current, stress_current, _ = _inputs()
    model = _model()
    with torch.no_grad():
        model.solution.body[-1].weight.fill_(0.0)
        model.solution.body[-1].bias.fill_(-0.1)   # constant f_theta = -0.1
    for horizon in (0.0, 0.25, 0.5):
        stress = stress_current + horizon
        u_hat = model.solution(stress, features, u_current, stress_current)
        expected = u_current + horizon * (-0.1)
        assert torch.allclose(u_hat, expected, atol=1e-6)


def test_autograd_derivative_is_valid_in_residual_mode():
    features, u_current, stress_current, stress_next = _inputs()
    model = _model()
    with torch.no_grad():
        for parameter in model.solution.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.2)
    stress = stress_next.clone().detach().requires_grad_(True)
    u_hat = model.solution(stress, features, u_current, stress_current)
    du_ds = autograd_du_ds(u_hat, stress)
    assert du_ds.shape == (N,)
    assert torch.isfinite(du_ds).all()
    assert du_ds.requires_grad          # graph stays live for the PDE loss


def test_derivative_matches_the_analytic_residual_form():
    """du/ds = f_theta + (s - s_current) * df/ds; with constant f it is f."""
    features, u_current, stress_current, stress_next = _inputs()
    model = _model()
    with torch.no_grad():
        model.solution.body[-1].weight.fill_(0.0)
        model.solution.body[-1].bias.fill_(-0.07)
    stress = stress_next.clone().detach().requires_grad_(True)
    u_hat = model.solution(stress, features, u_current, stress_current)
    du_ds = autograd_du_ds(u_hat, stress)
    assert torch.allclose(du_ds, torch.full((N,), -0.07), atol=1e-6)


def test_pde_residual_vanishes_for_a_consistent_rate():
    """H = du/ds + r is zero when r == -du/ds, with the correct sign."""
    features, u_current, stress_current, stress_next = _inputs()
    model = _model()
    with torch.no_grad():
        model.solution.body[-1].weight.fill_(0.0)
        model.solution.body[-1].bias.fill_(-0.07)
    stress = stress_next.clone().detach().requires_grad_(True)
    u_hat = model.solution(stress, features, u_current, stress_current)
    du_ds = autograd_du_ds(u_hat, stress)
    residual = pde_residual(du_ds, torch.full((N,), 0.07))
    assert torch.allclose(residual, torch.zeros(N), atol=1e-6)


def test_rate_is_nonnegative_via_softplus():
    features, _, stress_current, _ = _inputs()
    model = _model()
    with torch.no_grad():
        for parameter in model.rate.parameters():
            parameter.add_(torch.randn_like(parameter) * 3.0)
    r_hat = model.rate(stress_current, features, torch.zeros(N))
    assert (r_hat >= 0).all()


def test_rate_network_does_not_receive_u_hat():
    model = _model(rate_uses_u_hat=False)
    assert model.rate.uses_u_hat is False
    features, _, stress, _ = _inputs()
    a = model.rate(stress, features, torch.zeros(N))
    b = model.rate(stress, features, torch.ones(N) * 99.0)
    # Changing u_hat must not change the rate at all.
    assert torch.allclose(a, b)


def test_stays_within_the_parameter_budget():
    model = _model()
    assert count_parameters(model) < 2000


def test_integral_transition_broadcasts_anchor_stress_in_residual_mode():
    """Every quadrature node must satisfy the hard IC, so s_current is required."""
    features, u_current, stress_current, _ = _inputs()
    stress_delta = torch.full((N,), 2.0)
    model = _model()
    with torch.no_grad():
        for parameter in model.solution.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.3)

    class ConstantRate:
        def __call__(self, stress, features_, u):
            return torch.full_like(u, 0.1)

    integrated = integral_transition(
        model.solution, ConstantRate(), stress_current, stress_delta,
        features, u_current, method="trapezoidal", n_nodes=16,
    )
    # u(s_next) = u_current - integral(r) ds = u_current - 0.1 * delta.
    assert torch.allclose(integrated, u_current - 0.1 * stress_delta, atol=1e-4)


def test_integral_sign_convention_reduces_capacity():
    """A positive degradation rate must lower capacity, never raise it."""
    features, u_current, stress_current, _ = _inputs()
    stress_delta = torch.full((N,), 1.0)
    model = _model()

    class PositiveRate:
        def __call__(self, stress, features_, u):
            return torch.full_like(u, 0.05)

    integrated = integral_transition(
        model.solution, PositiveRate(), stress_current, stress_delta,
        features, u_current, method="trapezoidal", n_nodes=8,
    )
    assert (integrated < u_current).all()


def test_known_linear_degradation_law_is_pde_consistent():
    """For u(s) = u0 - k*(s - s0) the residual vanishes when r == k."""
    features, u_current, stress_current, _ = _inputs()
    k = 0.08
    model = _model()
    with torch.no_grad():
        model.solution.body[-1].weight.fill_(0.0)
        model.solution.body[-1].bias.fill_(-k)

    stress = (stress_current + 0.7).clone().detach().requires_grad_(True)
    u_hat = model.solution(stress, features, u_current, stress_current)
    # Exact linear solution.
    assert torch.allclose(u_hat, u_current - k * 0.7, atol=1e-6)
    du_ds = autograd_du_ds(u_hat, stress)
    assert torch.allclose(pde_residual(du_ds, torch.full((N,), k)),
                          torch.zeros(N), atol=1e-6)
