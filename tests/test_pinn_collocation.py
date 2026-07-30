"""Collocation-point purity and backpropagation."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.pinn.collocation import (
    enforce_collocation_purity, sample_collocation_points,
)
from src.pinn.models import NaPINNQ
from src.pinn.physics import autograd_du_ds, pde_residual


def test_collocation_points_inside_interval() -> None:
    stress_current = torch.tensor([0.0, 1.0, 2.0])
    delta = torch.tensor([0.5, 1.0, 2.0])
    features = torch.randn(3, 2)
    cell_index = torch.tensor([0, 1, 2])
    batch = sample_collocation_points(
        stress_current=stress_current, stress_delta=delta,
        features=features, cell_index=cell_index, points_per_transition=4,
    )
    stress = batch.stress
    lower = stress_current.repeat_interleave(4)
    upper = (stress_current + delta).repeat_interleave(4)
    assert (stress > lower - 1e-6).all()
    assert (stress <= upper + 1e-6).all()


def test_collocation_purity_rejects_targets() -> None:
    with pytest.raises(ValueError):
        enforce_collocation_purity(["EFC_cum", "next_rpt_Q_Ah"])


def test_collocation_residual_backprops() -> None:
    torch.manual_seed(0)
    model = NaPINNQ(feature_dim=2, solution_hidden_dims=(8,), rate_hidden_dims=(4,))
    stress_current = torch.zeros(3)
    delta = torch.ones(3)
    features = torch.randn(3, 2)
    cell_index = torch.tensor([0, 1, 2])
    batch = sample_collocation_points(
        stress_current=stress_current, stress_delta=delta,
        features=features, cell_index=cell_index, points_per_transition=4,
    )
    stress = batch.stress.clone().detach().requires_grad_(True)
    u = model.solution(stress, batch.features)
    r = model.rate(stress, batch.features, u)
    du = autograd_du_ds(u, stress)
    loss = pde_residual(du, r).pow(2).mean()
    loss.backward()
    solution_grad = sum(float(p.grad.detach().abs().sum().item())
                         for p in model.solution.parameters() if p.grad is not None)
    rate_grad = sum(float(p.grad.detach().abs().sum().item())
                     for p in model.rate.parameters() if p.grad is not None)
    assert solution_grad > 0
    assert rate_grad > 0


def test_collocation_residual_mode_backprops() -> None:
    """Hard-IC residual mode propagates gradients through collocation points."""
    torch.manual_seed(1)
    model = NaPINNQ(feature_dim=2, solution_hidden_dims=(8,), rate_hidden_dims=(4,),
                    predict_delta_u=True, rate_uses_u_hat=False)
    # Perturb away from zero-init so f_theta is non-trivial.
    with torch.no_grad():
        for p in model.solution.parameters():
            p.add_(torch.randn_like(p) * 0.2)

    stress_current = torch.tensor([0.0, 1.0, 2.0])
    delta = torch.ones(3)
    features = torch.randn(3, 2)
    u_current = torch.rand(3) * 0.2 + 0.8
    cell_index = torch.tensor([0, 1, 2])

    # Hard IC: u_hat at the anchor must equal u_current exactly.
    u_anchor = model.solution(stress_current, features, u_current, stress_current)
    assert torch.allclose(u_anchor, u_current, atol=1e-6)

    batch = sample_collocation_points(
        stress_current=stress_current, stress_delta=delta,
        features=features, cell_index=cell_index, points_per_transition=4,
    )
    collo_u_current = u_current[batch.row_index]
    collo_stress_current = stress_current[batch.row_index]
    collo_stress = batch.stress.clone().detach().requires_grad_(True)

    u_collo = model.solution(collo_stress, batch.features, collo_u_current,
                              collo_stress_current)
    r_collo = model.rate(collo_stress, batch.features, u_collo)
    du = autograd_du_ds(u_collo, collo_stress)
    loss = pde_residual(du, r_collo).pow(2).mean()
    loss.backward()

    solution_grad = sum(float(p.grad.detach().abs().sum().item())
                        for p in model.solution.parameters() if p.grad is not None)
    rate_grad = sum(float(p.grad.detach().abs().sum().item())
                    for p in model.rate.parameters() if p.grad is not None)
    assert solution_grad > 0
    assert rate_grad > 0
