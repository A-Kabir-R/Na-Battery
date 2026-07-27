"""NaPINN-Q solution and degradation-dynamics networks."""
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
    """Nonnegative degradation rate r_hat = Softplus(G_Theta(s, x, u_hat))."""

    def __init__(self, feature_dim: int, hidden_dims: Iterable[int] = (32, 16),
                 activation: str = "tanh") -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = feature_dim + 2  # stress + u_hat + features
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
        if u_hat.dim() == 1:
            u_hat = u_hat.unsqueeze(-1)
        return self.softplus(
            self.body(torch.cat([stress, u_hat, features], dim=-1))
        ).squeeze(-1)


class NaPINNQ(nn.Module):
    """Composite module bundling the solution + rate networks.

    ``forward`` returns ``(u_hat, r_hat)`` for a matched batch of stress /
    feature tensors.
    """

    def __init__(self, feature_dim: int,
                 solution_hidden_dims: Iterable[int] = (64, 64, 32),
                 rate_hidden_dims: Iterable[int] = (32, 16),
                 solution_activation: str = "tanh",
                 rate_activation: str = "tanh") -> None:
        super().__init__()
        self.solution = SolutionNet(feature_dim, solution_hidden_dims, solution_activation)
        self.rate = RateNet(feature_dim, rate_hidden_dims, rate_activation)
        self.feature_dim = int(feature_dim)

    def forward(self, stress: torch.Tensor,
                features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        u_hat = self.solution(stress, features)
        r_hat = self.rate(stress, features, u_hat)
        return u_hat, r_hat


class DNNQ(nn.Module):
    """Parameter-matched data-only baseline: shares SolutionNet architecture."""

    def __init__(self, feature_dim: int,
                 solution_hidden_dims: Iterable[int] = (64, 64, 32),
                 solution_activation: str = "tanh") -> None:
        super().__init__()
        self.solution = SolutionNet(feature_dim, solution_hidden_dims, solution_activation)
        self.feature_dim = int(feature_dim)

    def forward(self, stress: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        return self.solution(stress, features)


def count_parameters(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad))
