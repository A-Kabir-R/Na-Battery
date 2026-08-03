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


def test_anchor_du_ds_requires_detached_anchor_coordinate() -> None:
    """The anchor residual must be a real derivative, not identically zero.

    With the hard IC ``u = u_c + (s - s_c) f(s)``, passing the live autograd leaf
    as *both* the evaluation coordinate and the IC anchor makes the two
    occurrences cancel: ``du/ds = f*(1 - 1) + 0*df/ds == 0``. The anchor residual
    then collapses from ``du/ds + r`` to just ``r``, turning the physics term
    into a rate-shrinkage penalty that fights the observed-rate supervision.
    Detaching the anchor leaves the IC value untouched and restores
    ``du/ds == f_theta(s_c)``.
    """
    torch.manual_seed(0)
    model = NaPINNQ(
        feature_dim=3, solution_hidden_dims=(16, 8), rate_hidden_dims=(8,),
        predict_delta_u=True,  # the hard-IC head, as the paper configuration uses
    )
    # The hard-IC head zero-initialises its final layer, so f_theta == 0 and the
    # derivative is trivially zero at init. Perturb to a trained-like state.
    with torch.no_grad():
        for parameter in model.solution.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.1)

    features = torch.randn(6, 3)
    u_current = torch.full((6,), 0.98)

    shared = torch.randn(6, requires_grad=True)
    u_shared = model.solution(shared, features, u_current, shared)
    du_shared = autograd_du_ds(u_shared, shared)

    detached = shared.detach().clone().requires_grad_(True)
    u_detached = model.solution(detached, features, u_current, detached.detach())
    du_detached = autograd_du_ds(u_detached, detached)

    # The buggy form is degenerate...
    assert torch.allclose(du_shared, torch.zeros_like(du_shared))
    # ...while the corrected form recovers the intended derivative.
    assert not torch.allclose(du_detached, torch.zeros_like(du_detached))
    assert torch.isfinite(du_detached).all()

    # Detaching must not disturb the hard initial condition.
    assert torch.allclose(u_detached, u_current, atol=1e-6)
    assert torch.allclose(u_shared, u_current, atol=1e-6)


def test_trainer_anchor_solution_call_detaches_anchor() -> None:
    """Guard the two call sites in trainer.py against silent regression."""
    import inspect
    import re

    from src.pinn import trainer

    source = inspect.getsource(trainer)
    calls = re.findall(
        r"\.solution\(\s*stress_current_tensor,\s*batch_features,\s*"
        r"batch_u_current,\s*(stress_current_tensor(?:\.detach\(\))?)\s*,?\s*\)",
        source,
    )
    assert calls, "anchor solution call sites not found; update this guard"
    assert all(c.endswith(".detach()") for c in calls), (
        "anchor coordinate must be detached at every trainer call site, "
        f"found: {calls}"
    )
