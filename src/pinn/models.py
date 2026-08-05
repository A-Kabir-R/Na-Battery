"""NaPINN-Q solution and degradation-dynamics networks.

* :class:`RateNet` accepts a ``uses_u_hat`` flag. When ``False`` the rate
  network does NOT receive ``u_hat`` as an input; the ODE residual
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

  The derivative the ODE needs stays well defined::

      d u_hat / ds = f_theta(s, x) + local_stress * d f_theta / ds

  Zero-initialising the final layer of ``f_theta`` makes an untrained model
  predict persistence (``u_hat == u_current``) everywhere.
* When ``use_hybrid_rate=True``, :class:`NaPINNQ` substitutes a
  :class:`~src.pinn.degradation_laws.HybridRateModel` for the generic
  :class:`RateNet`.  The hybrid model decomposes the rate into named physical
  components (cycling, cold, calendar, diagnostic, residual), each nonnegative,
  and exposes them via :meth:`NaPINNQ.rate_components`.
"""
from __future__ import annotations

from typing import Iterable

import torch
from torch import nn

from .degradation_laws import HybridRateModel, RateComponents, StressLawConfig


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
    """Degradation rate r_hat = Softplus(G_Theta(s, x[, u_hat])) - rate_floor.

    With the default ``rate_floor=0.0`` this is the original hard nonnegative
    rate (``r_hat >= 0`` everywhere, so the ODE coupling ``du/ds = -r_hat``
    forces the solution to be monotonically non-increasing along stress with
    no exception). ``rate_floor > 0`` shifts the floor down so ``r_hat`` can
    take small negative values, i.e. the solution can express small predicted
    *increases* in capacity between anchors -- needed because real RPT-to-RPT
    transitions include apparent regeneration (relaxation/measurement noise)
    a hard-nonnegative rate head can never fit, which biases every delta
    prediction negative regardless of the true sign.
    """

    def __init__(self, feature_dim: int, hidden_dims: Iterable[int] = (8, 8),
                 activation: str = "tanh", dropout: float = 0.0,
                 uses_u_hat: bool = True, rate_floor: float = 0.0) -> None:
        super().__init__()
        self.uses_u_hat = bool(uses_u_hat)
        self.rate_floor = float(rate_floor)
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
        return self.softplus(self.body(inputs)).squeeze(-1) - self.rate_floor


class NaPINNQ(nn.Module):
    """Composite module bundling the solution + rate networks.

    When ``use_hybrid_rate=True`` the generic :class:`RateNet` is replaced by
    a :class:`~src.pinn.degradation_laws.HybridRateModel` that decomposes the
    degradation rate into named physical components.  The caller must supply
    valid *indices* into the full feature matrix (``[horizon, u_current,
    *scaled_features]``) for temperature, DOD, and C-rate channels.
    ``rate_components()`` is then available to read the decomposition.
    """

    def __init__(self, feature_dim: int,
                 solution_hidden_dims: Iterable[int] = (16, 16),
                 rate_hidden_dims: Iterable[int] = (8, 8),
                 solution_activation: str = "tanh",
                 rate_activation: str = "tanh",
                 solution_dropout: float = 0.0,
                 rate_dropout: float = 0.0,
                 rate_uses_u_hat: bool = True,
                 predict_delta_u: bool = False,
                 rate_floor: float = 0.0,
                 # Hybrid rate model -------------------------------------------
                 use_hybrid_rate: bool = False,
                 hybrid_temperature_index: int | None = None,
                 hybrid_dod_index: int | None = None,
                 hybrid_c_rate_index: int | None = None,
                 hybrid_diagnostic_indices: Iterable[int] = (),
                 hybrid_enable_cold_regime: bool = True,
                 hybrid_fit_c_rate_exponent: bool = False,
                 hybrid_residual_hidden_dims: Iterable[int] = (8, 8),
                 hybrid_temperature_is_celsius: bool = True,
                 hybrid_dod_is_percent: bool = True) -> None:
        super().__init__()
        self.solution = SolutionNet(feature_dim, solution_hidden_dims,
                                     solution_activation,
                                     dropout=solution_dropout,
                                     predict_delta_u=predict_delta_u)
        self.use_hybrid_rate = bool(use_hybrid_rate)
        if use_hybrid_rate and rate_floor:
            raise ValueError(
                "rate_floor is not implemented for use_hybrid_rate=True; "
                "HybridRateModel sums named nonnegative components and has no "
                "single Softplus head to shift"
            )
        if use_hybrid_rate:
            if hybrid_temperature_index is None or hybrid_dod_index is None:
                raise ValueError(
                    "use_hybrid_rate=True requires hybrid_temperature_index "
                    "and hybrid_dod_index"
                )
            c_rate_idx = (
                hybrid_c_rate_index
                if hybrid_c_rate_index is not None
                else hybrid_dod_index
            )
            law_config = StressLawConfig(
                enable_cold_regime=hybrid_enable_cold_regime,
                fit_c_rate_exponent=hybrid_fit_c_rate_exponent,
            )
            self.rate: nn.Module = HybridRateModel(
                feature_dim=feature_dim,
                temperature_index=hybrid_temperature_index,
                dod_index=hybrid_dod_index,
                c_rate_index=c_rate_idx,
                diagnostic_indices=hybrid_diagnostic_indices,
                config=law_config,
                residual_hidden_dims=hybrid_residual_hidden_dims,
                temperature_is_celsius=hybrid_temperature_is_celsius,
                dod_is_percent=hybrid_dod_is_percent,
            )
        else:
            self.rate = RateNet(feature_dim, rate_hidden_dims, rate_activation,
                                dropout=rate_dropout, uses_u_hat=rate_uses_u_hat,
                                rate_floor=rate_floor)
        self.feature_dim = int(feature_dim)
        self.predict_delta_u = bool(predict_delta_u)
        self.rate_floor = float(rate_floor)

    def rate_components(self, stress: torch.Tensor, features: torch.Tensor,
                        u_hat: torch.Tensor | None = None) -> RateComponents:
        """Return named rate components; only available in hybrid-rate mode."""
        if not self.use_hybrid_rate or not isinstance(self.rate, HybridRateModel):
            raise RuntimeError(
                "rate_components() requires use_hybrid_rate=True"
            )
        return self.rate.components(stress, features, u_hat)

    def forward(self, stress: torch.Tensor, features: torch.Tensor,
                 u_current: torch.Tensor | None = None,
                 stress_current: torch.Tensor | None = None
                 ) -> tuple[torch.Tensor, torch.Tensor]:
        u_hat = self.solution(stress, features, u_current, stress_current)
        r_hat = self.rate(stress, features, u_hat)
        return u_hat, r_hat


def count_parameters(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad))
