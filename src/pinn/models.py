"""NaPINN-Q solution and degradation-dynamics networks.

The Stage-2 diagnosis fixes:

* :class:`RateNet` accepts a ``uses_u_hat`` flag. When ``False`` the rate
  network does NOT receive ``u_hat`` as an input; the PDE residual
  ``H = du/ds + r`` can then no longer be trivially satisfied by learning
  ``r ≈ -∂u/∂s`` (Stage 2 diagnosis fix #14).

* :class:`DNNQ` exposes an independent ``hidden_dims`` so that the
  data-only baseline can be widened to approximately match NaPINN-Q's
  trainable parameter count (Stage 2 diagnosis fix #11).
"""
from __future__ import annotations

from typing import Iterable

import torch
from torch import nn


def _activation(name: str) -> nn.Module:
    lookup = {
        "tanh": nn.Tanh(),
        "silu": nn.SiLU(),
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
    }
    if name not in lookup:
        raise ValueError(f"unsupported activation: {name}")
    return lookup[name]


class SolutionNet(nn.Module):
    """Continuous capacity solution u_hat = F_Phi(s, x).

    Inputs
    ------
    stress : (N, 1) tensor
        Scalar (normalized) stress coordinate. Must carry ``requires_grad=True``
        for the PINN forward path so ``du/ds`` participates in autograd.
    features : (N, F) tensor
        Concatenation of horizon, u_current, physical-state block and the
        preprocessing-specific auxiliary features.
    """

    def __init__(self, feature_dim: int, hidden_dims: Iterable[int] = (64, 64, 32),
                 activation: str = "tanh") -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = feature_dim + 1
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(_activation(activation))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.body = nn.Sequential(*layers)

    def forward(self, stress: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        if stress.dim() == 1:
            stress = stress.unsqueeze(-1)
        return self.body(torch.cat([stress, features], dim=-1)).squeeze(-1)


class RateNet(nn.Module):
    """Nonnegative degradation rate r_hat = Softplus(G_Theta(s, x[, u_hat])).

    When ``uses_u_hat`` is ``True`` (the default), the network receives
    ``u_hat`` as an input alongside stress and features. When ``False``, the
    rate is a pure function of stress and features, which prevents the PINN
    from satisfying ``H = du/ds + r ≈ 0`` tautologically by matching the
    negative derivative of its own solution head.
    """

    def __init__(self, feature_dim: int, hidden_dims: Iterable[int] = (32, 16),
                 activation: str = "tanh", uses_u_hat: bool = True) -> None:
        super().__init__()
        self.uses_u_hat = bool(uses_u_hat)
        in_dim = feature_dim + (2 if self.uses_u_hat else 1)
        layers: list[nn.Module] = []
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(_activation(activation))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.body = nn.Sequential(*layers)
        self.softplus = nn.Softplus()

    def forward(self, stress: torch.Tensor, features: torch.Tensor,
                u_hat: torch.Tensor) -> torch.Tensor:
        if stress.dim() == 1:
            stress = stress.unsqueeze(-1)
        if self.uses_u_hat:
            if u_hat.dim() == 1:
                u_hat = u_hat.unsqueeze(-1)
            inputs = torch.cat([stress, u_hat, features], dim=-1)
        else:
            inputs = torch.cat([stress, features], dim=-1)
        return self.softplus(self.body(inputs)).squeeze(-1)


class NaPINNQ(nn.Module):
    """Composite module bundling the solution + rate networks.

    ``forward`` returns ``(u_hat, r_hat)`` for a matched batch of stress /
    feature tensors.
    """

    def __init__(self, feature_dim: int,
                 solution_hidden_dims: Iterable[int] = (64, 64, 32),
                 rate_hidden_dims: Iterable[int] = (32, 16),
                 solution_activation: str = "tanh",
                 rate_activation: str = "tanh",
                 rate_uses_u_hat: bool = True) -> None:
        super().__init__()
        self.solution = SolutionNet(feature_dim, solution_hidden_dims, solution_activation)
        self.rate = RateNet(feature_dim, rate_hidden_dims, rate_activation,
                            uses_u_hat=rate_uses_u_hat)
        self.feature_dim = int(feature_dim)

    def forward(self, stress: torch.Tensor,
                features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        u_hat = self.solution(stress, features)
        r_hat = self.rate(stress, features, u_hat)
        return u_hat, r_hat


class DNNQ(nn.Module):
    """Parameter-matched data-only baseline.

    The rate network of :class:`NaPINNQ` adds a few thousand trainable
    parameters that :class:`DNNQ` does not carry. To keep the family
    comparison honest, ``hidden_dims`` should be tuned so the baseline's
    parameter count is comparable to the full PINN's (usually a widened
    solution stack — see ``pinn.model.dnn_solution_hidden_dims`` in the
    config).
    """

    def __init__(self, feature_dim: int,
                 hidden_dims: Iterable[int] | None = None,
                 solution_activation: str = "tanh",
                 solution_hidden_dims: Iterable[int] | None = None) -> None:
        super().__init__()
        # ``solution_hidden_dims`` is the pre-rewrite alias kept for
        # backward-compatible tests. Prefer ``hidden_dims``.
        if hidden_dims is None:
            hidden_dims = solution_hidden_dims if solution_hidden_dims is not None else (80, 80, 40)
        self.solution = SolutionNet(feature_dim, hidden_dims, solution_activation)
        self.feature_dim = int(feature_dim)

    def forward(self, stress: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        return self.solution(stress, features)


def count_parameters(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad))
