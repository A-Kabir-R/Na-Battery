"""Composite loss bundle for NaPINN-Q.

All losses are cell-balanced by default. Every component keeps its raw
row-weighted value alongside the cell-balanced value so that diagnostics
match the plan's reporting requirements.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import torch


def cell_balanced_reduce(values: torch.Tensor, cell_index: torch.Tensor) -> torch.Tensor:
    """Return the mean-over-cells of the per-cell means of ``values``.

    ``cell_index`` must be a 1-D integer tensor of the same length as
    ``values``.
    """
    if values.numel() == 0:
        return values.new_zeros(())
    unique = torch.unique(cell_index)
    per_cell = []
    for c in unique:
        mask = cell_index == c
        if not torch.any(mask):
            continue
        per_cell.append(values[mask].mean())
    if not per_cell:
        return values.mean()
    return torch.stack(per_cell).mean()


def huber(residual: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    """Elementwise Huber loss."""
    abs_r = residual.abs()
    quadratic = 0.5 * residual.pow(2)
    linear = delta * (abs_r - 0.5 * delta)
    return torch.where(abs_r <= delta, quadratic, linear)


def data_loss(pred: torch.Tensor, true: torch.Tensor, cell_index: torch.Tensor,
              *, delta: float = 1.0) -> tuple[torch.Tensor, float]:
    residual = pred - true
    values = huber(residual, delta=delta)
    balanced = cell_balanced_reduce(values, cell_index)
    row_mean = float(values.detach().mean().item()) if values.numel() else float("nan")
    return balanced, row_mean


def pde_loss(residual: torch.Tensor, cell_index: torch.Tensor
             ) -> tuple[torch.Tensor, float]:
    values = residual.pow(2)
    balanced = cell_balanced_reduce(values, cell_index)
    row_mean = float(values.detach().mean().item()) if values.numel() else float("nan")
    return balanced, row_mean


def initial_condition_loss(u_hat_anchor: torch.Tensor, u_true_anchor: torch.Tensor,
                            cell_index: torch.Tensor) -> tuple[torch.Tensor, float]:
    values = (u_hat_anchor - u_true_anchor).pow(2)
    balanced = cell_balanced_reduce(values, cell_index)
    return balanced, float(values.detach().mean().item()) if values.numel() else float("nan")


def integral_consistency_loss(u_hat_next: torch.Tensor, u_hat_integrated: torch.Tensor,
                              cell_index: torch.Tensor) -> tuple[torch.Tensor, float]:
    values = (u_hat_next - u_hat_integrated).pow(2)
    balanced = cell_balanced_reduce(values, cell_index)
    return balanced, float(values.detach().mean().item()) if values.numel() else float("nan")


def monotonicity_loss(u_hat_next: torch.Tensor, u_current: torch.Tensor,
                       cell_index: torch.Tensor, *, epsilon_rec: float
                       ) -> tuple[torch.Tensor, float]:
    excess = torch.relu(u_hat_next - u_current - epsilon_rec)
    values = excess.pow(2)
    balanced = cell_balanced_reduce(values, cell_index)
    return balanced, float(values.detach().mean().item()) if values.numel() else float("nan")


def bounds_loss(u_hat: torch.Tensor, cell_index: torch.Tensor, *,
                u_min: float, u_max: float) -> tuple[torch.Tensor, float]:
    lower = torch.relu(u_min - u_hat).pow(2)
    upper = torch.relu(u_hat - u_max).pow(2)
    values = lower + upper
    balanced = cell_balanced_reduce(values, cell_index)
    return balanced, float(values.detach().mean().item()) if values.numel() else float("nan")


def rate_regularization_loss(r_hat: torch.Tensor, cell_index: torch.Tensor
                             ) -> tuple[torch.Tensor, float]:
    values = r_hat.pow(2)
    balanced = cell_balanced_reduce(values, cell_index)
    return balanced, float(values.detach().mean().item()) if values.numel() else float("nan")


def discrete_state_transition_loss(residual: torch.Tensor, cell_index: torch.Tensor
                                   ) -> tuple[torch.Tensor, float]:
    values = residual.pow(2)
    balanced = cell_balanced_reduce(values, cell_index)
    return balanced, float(values.detach().mean().item()) if values.numel() else float("nan")


@dataclass
class CurriculumSchedule:
    """Warm-up + linear ramp for the physics-loss weights."""

    max_epochs: int
    warmup_fraction: float = 0.10
    ramp_end_fraction: float = 0.30

    def factor(self, epoch: int) -> float:
        if self.max_epochs <= 0:
            return 1.0
        tau = epoch / self.max_epochs
        if tau < self.warmup_fraction:
            return 0.0
        if tau >= self.ramp_end_fraction:
            return 1.0
        span = max(self.ramp_end_fraction - self.warmup_fraction, 1e-9)
        return (tau - self.warmup_fraction) / span


@dataclass
class LossWeights:
    data: float = 1.0
    pde: float = 0.5
    initial_condition: float = 0.5
    integral: float = 0.25
    monotonicity: float = 0.1
    bounds: float = 0.1
    rate_regularization: float = 1.0e-4
    discrete_state_transition: float = 0.0

    def effective(self, curriculum_factor: float) -> "LossWeights":
        return LossWeights(
            data=self.data,
            pde=self.pde * curriculum_factor,
            initial_condition=self.initial_condition,
            integral=self.integral * curriculum_factor,
            monotonicity=self.monotonicity * curriculum_factor,
            bounds=self.bounds * curriculum_factor,
            rate_regularization=self.rate_regularization,
            discrete_state_transition=self.discrete_state_transition * curriculum_factor,
        )

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, float]) -> "LossWeights":
        return cls(
            data=float(mapping.get("data", 1.0)),
            pde=float(mapping.get("pde", 0.5)),
            initial_condition=float(mapping.get("initial_condition", 0.5)),
            integral=float(mapping.get("integral", 0.25)),
            monotonicity=float(mapping.get("monotonicity", 0.1)),
            bounds=float(mapping.get("bounds", 0.1)),
            rate_regularization=float(mapping.get("rate_regularization", 1.0e-4)),
            discrete_state_transition=float(mapping.get("discrete_state_transition", 0.0)),
        )


@dataclass
class LossReport:
    """Container returned by :func:`compose_total_loss`."""

    total: torch.Tensor
    components: dict[str, torch.Tensor]
    row_means: dict[str, float] = field(default_factory=dict)
    effective_weights: dict[str, float] = field(default_factory=dict)
