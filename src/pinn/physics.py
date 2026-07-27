"""Autograd-based governing-equation utilities for NaPINN-Q.

The plan requires:

    H = du_hat/ds + r_hat

to be computed with ``torch.autograd.grad`` while the graph is still live, so
that ``H`` participates in ``loss.backward()``. Finite-difference substitutes
are disallowed in the primary PINN.
"""
from __future__ import annotations

from typing import Callable

import torch


def autograd_du_ds(u_hat: torch.Tensor, stress: torch.Tensor) -> torch.Tensor:
    """Return ``du_hat/ds`` while keeping the computational graph alive.

    Parameters
    ----------
    u_hat : (N,) or (N, 1) tensor
        Solution-network output. Must be a differentiable function of ``stress``.
    stress : (N,) or (N, 1) tensor
        Stress coordinate; must have ``requires_grad=True``.
    """
    if not stress.requires_grad:
        raise ValueError("stress tensor must have requires_grad=True")
    grad, = torch.autograd.grad(
        outputs=u_hat,
        inputs=stress,
        grad_outputs=torch.ones_like(u_hat),
        create_graph=True,
        retain_graph=True,
    )
    if grad.dim() == 2 and grad.size(-1) == 1:
        grad = grad.squeeze(-1)
    return grad


def pde_residual(du_ds: torch.Tensor, r_hat: torch.Tensor) -> torch.Tensor:
    """Return the governing-equation residual ``H = du/ds + r``."""
    return du_ds + r_hat


def discrete_state_transition_residual(u_hat_next: torch.Tensor,
                                       u_current: torch.Tensor,
                                       r_hat_current: torch.Tensor,
                                       delta_s: torch.Tensor) -> torch.Tensor:
    """Ablation A5: R_k = u_hat_{k+1} - u_k + r_hat_k * Delta_s."""
    return u_hat_next - u_current + r_hat_current * delta_s


def _quadrature_nodes(method: str, n_nodes: int, device: torch.device,
                       dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(xi, weights)`` on [0, 1] summing to 1."""
    if n_nodes < 2:
        n_nodes = 2
    if method == "midpoint":
        xi = (torch.arange(n_nodes, device=device, dtype=dtype) + 0.5) / n_nodes
        w = torch.full((n_nodes,), 1.0 / n_nodes, device=device, dtype=dtype)
        return xi, w
    if method == "trapezoidal":
        xi = torch.linspace(0.0, 1.0, n_nodes, device=device, dtype=dtype)
        w = torch.full((n_nodes,), 1.0 / (n_nodes - 1), device=device, dtype=dtype)
        w[0] *= 0.5
        w[-1] *= 0.5
        return xi, w
    raise ValueError(f"unknown quadrature method: {method}")


def integral_transition(solution_forward: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                        rate_forward: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
                        stress_current: torch.Tensor,
                        stress_delta: torch.Tensor,
                        features: torch.Tensor,
                        u_current: torch.Tensor, *,
                        method: str = "trapezoidal",
                        n_nodes: int = 8) -> torch.Tensor:
    """Approximate ``u(s_{k+1}) = u_k - int r(s, x, u) ds`` by quadrature."""
    device = stress_current.device
    dtype = stress_current.dtype
    xi, weights = _quadrature_nodes(method, n_nodes, device, dtype)
    # (N, J) evaluation points s_k + xi_j * Delta s_k
    stress_current_bc = stress_current.unsqueeze(-1)
    delta_bc = stress_delta.unsqueeze(-1)
    s_grid = stress_current_bc + xi.unsqueeze(0) * delta_bc  # (N, J)
    features_bc = features.unsqueeze(1).expand(-1, xi.size(0), -1)  # (N, J, F)
    n, j, f = features_bc.shape
    s_flat = s_grid.reshape(-1, 1)
    x_flat = features_bc.reshape(-1, f)
    u_flat = solution_forward(s_flat, x_flat)
    r_flat = rate_forward(s_flat, x_flat, u_flat)
    r_grid = r_flat.view(n, j)
    integrand = (weights.unsqueeze(0) * r_grid).sum(dim=-1)  # (N,)
    return u_current - stress_delta * integrand
