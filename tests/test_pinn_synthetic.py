"""Synthetic linear/nonlinear degradation-law recovery."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.pinn.models import NaPINNQ
from src.pinn.physics import autograd_du_ds, pde_residual


def _train_synthetic(stress_np, u_true_np, features_np, *, epochs: int = 800,
                     lr: float = 5e-3):
    torch.manual_seed(0)
    model = NaPINNQ(feature_dim=features_np.shape[1],
                    solution_hidden_dims=(32, 32),
                    rate_hidden_dims=(16, 8))
    features = torch.tensor(features_np, dtype=torch.float32)
    u_true = torch.tensor(u_true_np, dtype=torch.float32)
    stress = torch.tensor(stress_np, dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        optimizer.zero_grad()
        s = stress.clone().detach().requires_grad_(True)
        u_hat = model.solution(s, features)
        r_hat = model.rate(s, features, u_hat)
        du = autograd_du_ds(u_hat, s)
        pde = pde_residual(du, r_hat).pow(2).mean()
        data = (u_hat - u_true).pow(2).mean()
        loss = data + 0.5 * pde
        loss.backward()
        optimizer.step()
    return model


def test_synthetic_linear_recovery() -> None:
    """u(s) = 1 - k*s => du/ds = -k, rate_hat should recover k pointwise.

    Tolerances verify physics recovery, not just that training executed:
    - mean rate within 20 % of k_true (rel) and within 0.005 (abs)
    - trajectory RMSE < 0.01 (model must fit the data well)
    - PDE RMSE < 0.01 (du/ds + r must be near zero everywhere)
    """
    import numpy as np
    k_true = 0.02
    stress = np.linspace(0.0, 5.0, 60, dtype="float32")
    u_true = 1.0 - k_true * stress
    features = np.zeros((stress.size, 2), dtype="float32")
    model = _train_synthetic(stress, u_true, features, epochs=1500, lr=5e-3)

    with torch.enable_grad():
        s = torch.tensor(stress, dtype=torch.float32, requires_grad=True)
        u_hat = model.solution(s, torch.tensor(features))
        r_hat = model.rate(s, torch.tensor(features), u_hat)
        du = autograd_du_ds(u_hat, s)

    residual = du + r_hat  # PDE residual: should be ~0
    mean_rate = torch.mean(r_hat).item()
    trajectory_rmse = float((u_hat.detach() - torch.tensor(u_true)).pow(2).mean().sqrt().item())
    pde_rmse = float(residual.pow(2).mean().sqrt().item())

    assert mean_rate == pytest.approx(k_true, rel=0.20, abs=0.005)
    assert trajectory_rmse < 0.01
    assert pde_rmse < 0.01


def test_synthetic_nonlinear_recovery() -> None:
    """du/ds = -k*u => u(s) = u0 * exp(-k*s), r_hat should recover k*u.

    Tolerances verify the model recovers the known physical rate law:
    - rate RMSE vs k*u_true < 0.02 (rate field should match the true pointwise rate)
    - trajectory RMSE < 0.015 (model must fit the exponential decay)
    - PDE RMSE < 0.02 (du/ds + r must be near zero everywhere)
    """
    import numpy as np
    k_true = 0.05
    stress = np.linspace(0.0, 4.0, 60, dtype="float32")
    u_true = np.exp(-k_true * stress).astype("float32")
    features = np.zeros((stress.size, 2), dtype="float32")
    model = _train_synthetic(stress, u_true, features, epochs=2000, lr=5e-3)

    with torch.enable_grad():
        s = torch.tensor(stress, dtype=torch.float32, requires_grad=True)
        u_hat = model.solution(s, torch.tensor(features))
        r_hat = model.rate(s, torch.tensor(features), u_hat)
        du = autograd_du_ds(u_hat, s)

    expected_rate = torch.tensor(k_true * u_true, dtype=torch.float32)
    residual = du + r_hat  # PDE residual
    rate_rmse = float((r_hat.detach() - expected_rate).pow(2).mean().sqrt().item())
    trajectory_rmse = float((u_hat.detach() - torch.tensor(u_true)).pow(2).mean().sqrt().item())
    pde_rmse = float(residual.pow(2).mean().sqrt().item())

    assert rate_rmse < 0.02
    assert trajectory_rmse < 0.015
    assert pde_rmse < 0.02
