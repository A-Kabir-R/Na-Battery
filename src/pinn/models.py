"""NaPINN-Q solution and degradation-dynamics networks.

Stage-2 diagnosis fixes:

* :class:`RateNet` accepts a ``uses_u_hat`` flag. When ``False`` the rate
  network does NOT receive ``u_hat`` as an input; the PDE residual
  ``H = du/ds + r`` can then no longer be trivially satisfied by learning
  ``r ≈ -∂u/∂s`` (Stage 2 diagnosis fix #14).
* :class:`DNNQ` exposes an independent ``hidden_dims`` so that the
  data-only baseline can be widened to approximately match NaPINN-Q's
  trainable parameter count (Stage 2 diagnosis fix #11).

Stage-3 diagnosis fixes:

* Networks now support ``dropout`` between hidden layers so a ~500-parameter
  model can still be regularized when trained on ~120 anchor samples.
* Both :class:`DNNQ` and :class:`NaPINNQ` support ``predict_delta_u``.
  When set, the SolutionNet output is interpreted as ``Δu`` and the caller
  reconstructs ``u_next = u_current + Δu``. This enforces the initial
  condition by construction and keeps typical network outputs near 0
  (residual predictor).
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


def _stack_mlp(in_dim: int, hidden_dims: Iterable[int], activation: str,
               dropout: float, output_dim: int = 1) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(d, h))
        layers.append(_activation(activation))
        if dropout and dropout > 0.0:
            layers.append(nn.Dropout(float(dropout)))
        d = h
    layers.append(nn.Linear(d, output_dim))
    return nn.Sequential(*layers)


class SolutionNet(nn.Module):
    """Continuous capacity solution head.

    When ``predict_delta_u`` is ``False`` (legacy) the output is ``u_hat``
    directly. When ``True``, the output is ``Δu`` (residual) and the caller
    is responsible for adding ``u_current`` to obtain ``u_next``.

    Inputs
    ------
    stress : (N, 1) tensor
        Scalar (normalized) stress coordinate. Must carry ``requires_grad=True``
        for the PINN forward path so ``du/ds`` participates in autograd.
    features : (N, F) tensor
        Concatenation of horizon, u_current, physical-state block and the
        preprocessing-specific auxiliary features.
    """

    def __init__(self, feature_dim: int, hidden_dims: Iterable[int] = (16, 16),
                 activation: str = "tanh", dropout: float = 0.0,
                 predict_delta_u: bool = False) -> None:
        super().__init__()
        self.predict_delta_u = bool(predict_delta_u)
        self.body = _stack_mlp(feature_dim + 1, hidden_dims, activation,
                                dropout=dropout, output_dim=1)

    def raw(self, stress: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """Return the raw MLP output (Δu when predict_delta_u, else u_hat)."""
        if stress.dim() == 1:
            stress = stress.unsqueeze(-1)
        return self.body(torch.cat([stress, features], dim=-1)).squeeze(-1)

    def forward(self, stress: torch.Tensor, features: torch.Tensor,
                 u_current: torch.Tensor | None = None) -> torch.Tensor:
        """Return u_hat. If predicting Δu, u_current must be provided.

        When predict_delta_u=False the u_current argument is ignored so the
        signature stays backward compatible with older call sites that only
        pass (stress, features).
        """
        raw = self.raw(stress, features)
        if not self.predict_delta_u:
            return raw
        if u_current is None:
            raise ValueError(
                "predict_delta_u=True requires u_current in SolutionNet.forward"
            )
        return u_current + raw


class RateNet(nn.Module):
    """Nonnegative degradation rate r_hat = Softplus(G_Theta(s, x[, u_hat]))."""

    def __init__(self, feature_dim: int, hidden_dims: Iterable[int] = (8, 8),
                 activation: str = "tanh", dropout: float = 0.0,
                 uses_u_hat: bool = True) -> None:
        super().__init__()
        self.uses_u_hat = bool(uses_u_hat)
        in_dim = feature_dim + (2 if self.uses_u_hat else 1)
        self.body = _stack_mlp(in_dim, hidden_dims, activation,
                                dropout=dropout, output_dim=1)
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
    """Composite module bundling the solution + rate networks."""

    def __init__(self, feature_dim: int,
                 solution_hidden_dims: Iterable[int] = (16, 16),
                 rate_hidden_dims: Iterable[int] = (8, 8),
                 solution_activation: str = "tanh",
                 rate_activation: str = "tanh",
                 solution_dropout: float = 0.0,
                 rate_dropout: float = 0.0,
                 rate_uses_u_hat: bool = True,
                 predict_delta_u: bool = False) -> None:
        super().__init__()
        self.solution = SolutionNet(feature_dim, solution_hidden_dims,
                                     solution_activation,
                                     dropout=solution_dropout,
                                     predict_delta_u=predict_delta_u)
        self.rate = RateNet(feature_dim, rate_hidden_dims, rate_activation,
                             dropout=rate_dropout, uses_u_hat=rate_uses_u_hat)
        self.feature_dim = int(feature_dim)
        self.predict_delta_u = bool(predict_delta_u)

    def forward(self, stress: torch.Tensor, features: torch.Tensor,
                 u_current: torch.Tensor | None = None
                 ) -> tuple[torch.Tensor, torch.Tensor]:
        u_hat = self.solution(stress, features, u_current)
        r_hat = self.rate(stress, features, u_hat)
        return u_hat, r_hat


class DNNQ(nn.Module):
    """Parameter-matched data-only baseline."""

    def __init__(self, feature_dim: int,
                 hidden_dims: Iterable[int] | None = None,
                 solution_activation: str = "tanh",
                 solution_hidden_dims: Iterable[int] | None = None,
                 solution_dropout: float = 0.0,
                 predict_delta_u: bool = False) -> None:
        super().__init__()
        # ``solution_hidden_dims`` is the pre-rewrite alias kept for
        # backward-compatible tests. Prefer ``hidden_dims``.
        if hidden_dims is None:
            hidden_dims = solution_hidden_dims if solution_hidden_dims is not None else (16, 16)
        self.solution = SolutionNet(feature_dim, hidden_dims,
                                     solution_activation,
                                     dropout=solution_dropout,
                                     predict_delta_u=predict_delta_u)
        self.feature_dim = int(feature_dim)
        self.predict_delta_u = bool(predict_delta_u)

    def forward(self, stress: torch.Tensor, features: torch.Tensor,
                 u_current: torch.Tensor | None = None) -> torch.Tensor:
        return self.solution(stress, features, u_current)


def count_parameters(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad))
