"""NaPINN-Q solution and degradation-dynamics networks.

* :class:`RateNet` accepts a ``uses_u_hat`` flag. When ``False`` the rate
  network does NOT receive ``u_hat`` as an input; the PDE residual
  ``H = du/ds + r`` can then no longer be trivially satisfied by learning
  ``r ≈ -∂u/∂s``.
* Networks support ``dropout`` between hidden layers.
* :class:`SolutionNet` supports ``predict_delta_u``, which selects the **hard
  initial-condition residual parameterisation**::

      local_stress = s - s_current
      u_hat(s)     = u_current + local_stress * f_theta(s, x)

  so ``u_hat(s_current) == u_current`` *exactly*, for any parameters, and

      u_hat(s_next) = u_current + stress_delta * f_theta(s_next, x).

  The earlier implementation used ``u_hat(s) = u_current + f_theta(s, x)``, which
  reproduces the initial condition only if ``f_theta(s_current, x)`` happens to
  vanish — it does not, so the IC was penalised rather than enforced. Because the
  ``local_stress`` factor vanishes at the anchor, the initial-condition loss is
  now redundant by construction and its residual is identically zero.

  The derivative the PDE needs stays well defined::

      d u_hat / ds = f_theta(s, x) + local_stress * d f_theta / ds

  Zero-initialising the final layer of ``f_theta`` makes an untrained model
  predict persistence (``u_hat == u_current``) everywhere.
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

    When ``predict_delta_u`` is ``False`` (legacy) the network output is ``u_hat``
    directly. When ``True`` the network output is the *rate-like* field
    ``f_theta(s, x)`` and ``u_hat`` is reconstructed as

        u_hat(s) = u_current + (s - s_current) * f_theta(s, x)

    which enforces ``u_hat(s_current) = u_current`` identically.

    Inputs
    ------
    stress : (N,) or (N, 1) tensor
        Scalar (normalized) stress coordinate. Must carry ``requires_grad=True``
        on the PINN forward path so ``du/ds`` participates in autograd.
    features : (N, F) tensor
        Auxiliary feature block (horizon, physical state, registry features).
    u_current : (N,) tensor
        Capacity at the anchor. Required in residual mode.
    stress_current : (N,) tensor
        Anchor stress coordinate. Required in residual mode — this is what makes
        the initial condition hard rather than penalised.
    """

    def __init__(self, feature_dim: int, hidden_dims: Iterable[int] = (16, 16),
                 activation: str = "tanh", dropout: float = 0.0,
                 predict_delta_u: bool = False) -> None:
        super().__init__()
        self.predict_delta_u = bool(predict_delta_u)
        self.body = _stack_mlp(feature_dim + 1, hidden_dims, activation,
                                dropout=dropout, output_dim=1)
        # Zero-init the final layer so an untrained residual model predicts
        # persistence (f_theta == 0 => u_hat == u_current).
        if self.predict_delta_u:
            final_linear = self.body[-1]
            nn.init.zeros_(final_linear.weight)
            nn.init.zeros_(final_linear.bias)

    def raw(self, stress: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """Raw MLP output: ``f_theta(s, x)`` in residual mode, else ``u_hat``."""
        if stress.dim() == 1:
            stress = stress.unsqueeze(-1)
        return self.body(torch.cat([stress, features], dim=-1)).squeeze(-1)

    def forward(self, stress: torch.Tensor, features: torch.Tensor,
                 u_current: torch.Tensor | None = None,
                 stress_current: torch.Tensor | None = None) -> torch.Tensor:
        """Return ``u_hat``.

        In residual mode both ``u_current`` and ``stress_current`` are mandatory;
        omitting either raises rather than silently degrading to the soft-IC
        behaviour that this revision removed.
        """
        raw = self.raw(stress, features)
        if not self.predict_delta_u:
            return raw
        if u_current is None or stress_current is None:
            raise ValueError(
                "predict_delta_u=True requires both u_current and stress_current "
                "in SolutionNet.forward; the hard initial condition cannot be "
                "formed without the anchor stress coordinate"
            )
        if stress.dim() == 2 and stress.shape[-1] == 1:
            stress = stress.squeeze(-1)
        local_stress = stress - stress_current
        return u_current + local_stress * raw


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
                 u_current: torch.Tensor | None = None,
                 stress_current: torch.Tensor | None = None
                 ) -> tuple[torch.Tensor, torch.Tensor]:
        u_hat = self.solution(stress, features, u_current, stress_current)
        r_hat = self.rate(stress, features, u_hat)
        return u_hat, r_hat


def count_parameters(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad))
