"""Interpretable degradation-rate laws with a constrained neural residual.

The current rate head is a free MLP behind a Softplus: nonnegative, but with no
identifiable physics inside it. Nothing distinguishes a sodium-ion degradation
law from any other monotone fit, which is the substance of the "generic PINN"
objection this module answers.

The rate is decomposed into named, separately-reportable components::

    r_total = r_cycling + r_cold + r_calendar + r_diagnostic + r_residual

``r_cycling``
    Arrhenius temperature dependence x DOD power law x C-rate power law.
``r_cold``
    A smoothly-gated low-temperature term. Deliberately named for the *regime*,
    not for a mechanism: this dataset carries no plating or gassing label, and
    on the standard-cycling subset only two development anchors sit at -10 C, so
    its parameters are exploratory and must be reported with intervals.
``r_calendar``
    Time-driven fade, active only when an elapsed-time channel is supplied.
``r_diagnostic``
    A small head over measured diagnostics (resistance, dQ/dV, thermal).
``r_residual``
    Whatever the physics cannot express, penalised toward zero so that a
    hybrid model cannot quietly become a black box while keeping the label.

Every component is nonnegative by construction, so the total is too and the
hard-IC solution stays monotone without an extra penalty.

Identifiability warning, specific to this dataset: DOD20 and DOD80 are tested
only at 2C, so the DOD and C-rate exponents are not separately identifiable
(design condition number ~1.3e4). Klick et al. (Batteries & Supercaps 2025)
report that C-rate does not significantly affect the ageing trajectory, so the
default here **freezes the C-rate exponent at zero**. Setting
``fit_c_rate_exponent=True`` turns it into a test of that published result --
which is a legitimate experiment, provided the confounding is reported.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import torch
from torch import nn

#: Universal gas constant, J/(mol*K).
R_GAS = 8.314462618


@dataclass(frozen=True)
class StressLawConfig:
    """Reference points and admissible ranges for the parametric rate."""

    reference_temperature_K: float = 298.15
    reference_dod_fraction: float = 1.0
    reference_c_rate: float = 1.0
    #: Physically plausible window for the Arrhenius activation energy. Sodium-ion
    #: cycling-ageing values reported in the literature sit well inside this.
    activation_energy_bounds_J_mol: tuple[float, float] = (-6.0e4, 1.2e5)
    dod_exponent_bounds: tuple[float, float] = (0.0, 3.0)
    c_rate_exponent_bounds: tuple[float, float] = (0.0, 2.0)
    cold_transition_temperature_bounds_K: tuple[float, float] = (243.15, 293.15)
    #: Upper bound on the residual's share of the total rate, enforced by a
    #: penalty rather than a hard clamp so gradients stay informative.
    residual_fraction_limit: float = 0.5
    fit_c_rate_exponent: bool = False
    enable_cold_regime: bool = True
    enable_calendar: bool = False


@dataclass
class RateComponents:
    """Named rate contributions. All nonnegative, all per unit stress."""

    cycling: torch.Tensor
    cold: torch.Tensor
    calendar: torch.Tensor
    diagnostic: torch.Tensor
    residual: torch.Tensor

    @property
    def total(self) -> torch.Tensor:
        return self.cycling + self.cold + self.calendar + self.diagnostic + self.residual

    def residual_fraction(self, epsilon: float = 1e-12) -> torch.Tensor:
        """Share of the rate the neural term is carrying.

        The headline interpretability number: a hybrid whose residual explains
        most of the rate is a black box wearing a physics label.
        """
        return self.residual / (self.total + epsilon)

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {
            "cycling": self.cycling, "cold": self.cold,
            "calendar": self.calendar, "diagnostic": self.diagnostic,
            "residual": self.residual, "total": self.total,
        }


def _bounded(raw: torch.Tensor, low: float, high: float) -> torch.Tensor:
    """Map an unconstrained parameter into ``[low, high]`` smoothly."""
    return low + (high - low) * torch.sigmoid(raw)


class ParametricStressRate(nn.Module):
    r"""Arrhenius x DOD x C-rate cycling law, plus an optional cold regime.

    .. math::
        r_{cyc} = k_{ref}\,
          \exp\!\Big[-\tfrac{E_a}{R}\big(\tfrac1T-\tfrac1{T_{ref}}\big)\Big]
          \Big(\tfrac{DOD}{DOD_{ref}}\Big)^{\beta}
          \Big(\tfrac{|C|}{C_{ref}}\Big)^{\alpha}

    All parameters are stored unconstrained and squashed into physical ranges on
    each forward pass, so the optimiser is unconstrained while the physics stays
    admissible -- no clipping, no dead gradients at the bounds.
    """

    def __init__(self, config: StressLawConfig | None = None) -> None:
        super().__init__()
        self.config = config or StressLawConfig()
        # log k_ref, so the rate is positive for any real value.
        self.log_k_ref = nn.Parameter(torch.tensor(-9.0))
        self.raw_activation = nn.Parameter(torch.tensor(0.0))
        self.raw_dod_exponent = nn.Parameter(torch.tensor(0.0))
        self.raw_c_exponent = nn.Parameter(torch.tensor(0.0))
        self.raw_cold_temperature = nn.Parameter(torch.tensor(0.0))
        self.log_cold_width = nn.Parameter(torch.tensor(1.0))
        self.log_cold_scale = nn.Parameter(torch.tensor(-9.0))
        self.log_calendar_scale = nn.Parameter(torch.tensor(-12.0))

    # -- physical parameters -------------------------------------------------
    def activation_energy(self) -> torch.Tensor:
        return _bounded(self.raw_activation, *self.config.activation_energy_bounds_J_mol)

    def dod_exponent(self) -> torch.Tensor:
        return _bounded(self.raw_dod_exponent, *self.config.dod_exponent_bounds)

    def c_rate_exponent(self) -> torch.Tensor:
        if not self.config.fit_c_rate_exponent:
            # Frozen at zero: DOD and C-rate are confounded in this test matrix,
            # and the source publication reports no significant C-rate effect.
            return torch.zeros((), device=self.raw_c_exponent.device)
        return _bounded(self.raw_c_exponent, *self.config.c_rate_exponent_bounds)

    def cold_transition_temperature(self) -> torch.Tensor:
        return _bounded(self.raw_cold_temperature,
                        *self.config.cold_transition_temperature_bounds_K)

    def physical_parameters(self) -> dict[str, float]:
        """Fitted physics, for the manuscript's parameter table."""
        return {
            "k_ref_per_stress": float(torch.exp(self.log_k_ref)),
            "activation_energy_J_per_mol": float(self.activation_energy()),
            "dod_exponent_beta": float(self.dod_exponent()),
            "c_rate_exponent_alpha": float(self.c_rate_exponent()),
            "c_rate_exponent_fitted": bool(self.config.fit_c_rate_exponent),
            "cold_transition_temperature_K": float(self.cold_transition_temperature()),
            "cold_width_K": float(torch.exp(self.log_cold_width)),
            "cold_scale_per_stress": float(torch.exp(self.log_cold_scale)),
        }

    def forward(self, *, temperature_K: torch.Tensor, dod_fraction: torch.Tensor,
                c_rate: torch.Tensor,
                elapsed_time: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(cycling, cold, calendar)`` rate components."""
        config = self.config
        temperature_K = torch.clamp(temperature_K, min=1.0)
        dod = torch.clamp(dod_fraction, min=1e-6) / config.reference_dod_fraction
        c_rate = torch.clamp(torch.abs(c_rate), min=1e-6) / config.reference_c_rate

        arrhenius = torch.exp(
            -(self.activation_energy() / R_GAS)
            * (1.0 / temperature_K - 1.0 / config.reference_temperature_K)
        )
        cycling = (
            torch.exp(self.log_k_ref)
            * arrhenius
            * dod.pow(self.dod_exponent())
            * c_rate.pow(self.c_rate_exponent())
        )

        if config.enable_cold_regime:
            width = torch.clamp(torch.exp(self.log_cold_width), min=1e-3)
            gate = torch.sigmoid(
                (self.cold_transition_temperature() - temperature_K) / width
            )
            cold = gate * torch.exp(self.log_cold_scale) * dod.pow(self.dod_exponent())
        else:
            cold = torch.zeros_like(cycling)

        if config.enable_calendar and elapsed_time is not None:
            calendar = torch.exp(self.log_calendar_scale) * torch.clamp(
                elapsed_time, min=0.0
            )
        else:
            calendar = torch.zeros_like(cycling)
        return cycling, cold, calendar


class DiagnosticRate(nn.Module):
    """Small nonnegative head over measured degradation diagnostics."""

    def __init__(self, feature_dim: int, hidden_dims: Iterable[int] = (8,)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = feature_dim
        for width in hidden_dims:
            layers += [nn.Linear(in_dim, width), nn.Tanh()]
            in_dim = width
        output = nn.Linear(in_dim, 1)
        # Start ~20x below the physics scale (k_ref = exp(-9) = 1.2e-4, and
        # softplus(-12) = 6.1e-6) so the physics explains essentially everything
        # at initialisation and the data has to argue the diagnostics in. A -6.0
        # bias gives softplus = 2.5e-3, i.e. 20x *larger* than k_ref, which would
        # make the model a black box before training even starts.
        nn.init.zeros_(output.weight)
        nn.init.constant_(output.bias, -12.0)
        layers.append(output)
        self.body = nn.Sequential(*layers)
        self.softplus = nn.Softplus()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.softplus(self.body(features)).squeeze(-1)


class HybridRateModel(nn.Module):
    """Parametric physics plus a constrained neural residual.

    Drop-in compatible with ``RateNet``: ``forward(stress, features, u_hat)``
    returns the total rate, while :meth:`components` exposes the decomposition
    for the rate-decomposition figure and the residual-share diagnostic.

    ``column_index`` maps the physical channels the law needs onto positions in
    the auxiliary feature tensor, so the caller stays responsible for feature
    provenance rather than this module guessing by column name.
    """

    def __init__(self, feature_dim: int, *,
                 temperature_index: int,
                 dod_index: int,
                 c_rate_index: int,
                 diagnostic_indices: Iterable[int] = (),
                 elapsed_time_index: int | None = None,
                 config: StressLawConfig | None = None,
                 residual_hidden_dims: Iterable[int] = (8, 8),
                 temperature_is_celsius: bool = True,
                 dod_is_percent: bool = True) -> None:
        super().__init__()
        self.config = config or StressLawConfig()
        self.temperature_index = int(temperature_index)
        self.dod_index = int(dod_index)
        self.c_rate_index = int(c_rate_index)
        self.elapsed_time_index = elapsed_time_index
        self.diagnostic_indices = tuple(int(i) for i in diagnostic_indices)
        self.temperature_is_celsius = bool(temperature_is_celsius)
        self.dod_is_percent = bool(dod_is_percent)

        self.parametric = ParametricStressRate(self.config)
        self.diagnostic = (
            DiagnosticRate(len(self.diagnostic_indices))
            if self.diagnostic_indices else None
        )
        layers: list[nn.Module] = []
        in_dim = feature_dim + 1
        for width in residual_hidden_dims:
            layers += [nn.Linear(in_dim, width), nn.Tanh()]
            in_dim = width
        output = nn.Linear(in_dim, 1)
        # Same reasoning as DiagnosticRate: the residual must start negligible
        # relative to k_ref, or the "hybrid" label is false from epoch zero.
        nn.init.zeros_(output.weight)
        nn.init.constant_(output.bias, -12.0)
        layers.append(output)
        self.residual_body = nn.Sequential(*layers)
        self.softplus = nn.Softplus()

    def _channel(self, features: torch.Tensor, index: int) -> torch.Tensor:
        return features[..., index]

    def components(self, stress: torch.Tensor, features: torch.Tensor,
                   u_hat: torch.Tensor | None = None) -> RateComponents:
        temperature = self._channel(features, self.temperature_index)
        temperature_K = temperature + 273.15 if self.temperature_is_celsius else temperature
        dod = self._channel(features, self.dod_index)
        dod_fraction = dod / 100.0 if self.dod_is_percent else dod
        c_rate = self._channel(features, self.c_rate_index)
        elapsed = (
            self._channel(features, self.elapsed_time_index)
            if self.elapsed_time_index is not None else None
        )

        cycling, cold, calendar = self.parametric(
            temperature_K=temperature_K, dod_fraction=dod_fraction,
            c_rate=c_rate, elapsed_time=elapsed,
        )
        if self.diagnostic is not None:
            diagnostic = self.diagnostic(features[..., list(self.diagnostic_indices)])
        else:
            diagnostic = torch.zeros_like(cycling)

        if stress.dim() == 1:
            stress_column = stress.unsqueeze(-1)
        else:
            stress_column = stress
        residual = self.softplus(
            self.residual_body(torch.cat([stress_column, features], dim=-1))
        ).squeeze(-1)
        return RateComponents(cycling=cycling, cold=cold, calendar=calendar,
                              diagnostic=diagnostic, residual=residual)

    def forward(self, stress: torch.Tensor, features: torch.Tensor,
                u_hat: torch.Tensor | None = None) -> torch.Tensor:
        return self.components(stress, features, u_hat).total

    def physical_parameters(self) -> dict[str, float]:
        return self.parametric.physical_parameters()


def residual_share_penalty(components: RateComponents, *,
                           limit: float | None = None,
                           epsilon: float = 1e-12) -> torch.Tensor:
    """Penalise the neural residual for exceeding its allowed share.

    One-sided by design: a residual *below* the limit is free, so the penalty
    never fights the data when the physics genuinely cannot explain something.
    """
    limit = 0.5 if limit is None else float(limit)
    share = components.residual / (components.total + epsilon)
    return torch.clamp(share - limit, min=0.0).pow(2).mean()


def parameter_prior_penalty(model: ParametricStressRate | HybridRateModel
                            ) -> torch.Tensor:
    """Keep the unconstrained parameters away from the saturated sigmoid tails.

    The bounds are enforced by construction, so this only discourages the
    optimiser from parking a parameter where its gradient has vanished -- which
    would look converged while being unidentified.
    """
    parametric = getattr(model, "parametric", model)
    raw = [
        parametric.raw_activation, parametric.raw_dod_exponent,
        parametric.raw_c_exponent, parametric.raw_cold_temperature,
    ]
    return torch.stack([value.pow(2) for value in raw]).mean() * 1e-3
