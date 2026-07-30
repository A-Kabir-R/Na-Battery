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


def observed_degradation_rate(u_current: torch.Tensor, u_true_next: torch.Tensor,
                              stress_delta: torch.Tensor, *,
                              epsilon: float = 1.0e-6,
                              regeneration_tolerance: float = 0.0
                              ) -> tuple[torch.Tensor, torch.Tensor]:
    """Observed degradation rate and its validity mask.

        observed_rate = max((u_current - u_true_next) / max(delta_s, eps), 0)

    Small apparent capacity *regeneration* (u_next slightly above u_current, a
    measurement artefact rather than physical capacity gain) would otherwise
    produce a negative target for a Softplus-constrained rate head, so the rate
    is clamped at zero. ``regeneration_tolerance`` is the fold-derived tolerance
    already used by the monotonicity loss; transitions whose apparent gain
    exceeds it are masked out entirely instead of being clamped, because those
    are not merely noisy.

    Returns
    -------
    (rate, mask) : both (N,) tensors; ``mask`` is boolean.
    """
    safe_delta = torch.clamp(stress_delta, min=epsilon)
    raw_rate = (u_current - u_true_next) / safe_delta
    rate = torch.clamp(raw_rate, min=0.0)
    mask = (
        (stress_delta > epsilon)
        & torch.isfinite(raw_rate)
        & torch.isfinite(u_true_next)
        & ((u_true_next - u_current) <= regeneration_tolerance)
    )
    return rate, mask


def rate_data_loss(r_hat: torch.Tensor, observed_rate: torch.Tensor,
                   cell_index: torch.Tensor, mask: torch.Tensor, *,
                   delta: float = 1.0) -> tuple[torch.Tensor, float, float]:
    """Direct Huber supervision of the rate head on the observed rate.

    Without this the rate network only ever learns through the PDE coupling,
    which on a 140-row training fold leaves it under-determined.

    Returns ``(balanced_loss, row_mean, coverage_fraction)``.
    """
    total = int(mask.numel())
    if total == 0 or not bool(mask.any()):
        zero = r_hat.new_zeros(())
        return zero, float("nan"), 0.0
    values = huber(r_hat[mask] - observed_rate[mask], delta=delta)
    balanced = cell_balanced_reduce(values, cell_index[mask])
    row_mean = float(values.detach().mean().item())
    return balanced, row_mean, float(int(mask.sum()) / total)


def pairwise_delta_loss(u_hat_next: torch.Tensor, u_current: torch.Tensor,
                        u_true_next: torch.Tensor, cell_index: torch.Tensor, *,
                        generator: torch.Generator | None = None,
                        max_pairs: int = 256,
                        delta: float = 1.0
                        ) -> tuple[torch.Tensor, float, int]:
    """Cross-cell relative-degradation loss.

    For anchors i and j drawn from *different* cells, match the difference of
    predicted capacity changes to the difference of true changes::

        L = Huber((pred_delta_i - pred_delta_j) - (true_delta_i - true_delta_j))

    This is a training-time regulariser on ordering between cells, not a new
    model. Pairs are formed only from the rows passed in — the caller supplies
    training rows exclusively, so no validation or holdout row can enter.

    Returns ``(loss, row_mean, n_pairs)``.
    """
    n = int(u_hat_next.numel())
    zero = u_hat_next.new_zeros(())
    if n < 2:
        return zero, float("nan"), 0
    device = u_hat_next.device
    # Random candidate pairs, then keep only the cross-cell ones. Sampling is
    # cheaper and better conditioned than materialising all n^2 pairs.
    n_candidates = min(max_pairs, n * (n - 1))
    if n_candidates <= 0:
        return zero, float("nan"), 0
    if generator is not None and generator.device.type == device.type:
        i = torch.randint(0, n, (n_candidates,), device=device, generator=generator)
        j = torch.randint(0, n, (n_candidates,), device=device, generator=generator)
    else:
        i = torch.randint(0, n, (n_candidates,), device=device)
        j = torch.randint(0, n, (n_candidates,), device=device)
    keep = cell_index[i] != cell_index[j]
    if not bool(keep.any()):
        return zero, float("nan"), 0
    i, j = i[keep], j[keep]
    pred_delta = u_hat_next - u_current
    true_delta = u_true_next - u_current
    residual = (pred_delta[i] - pred_delta[j]) - (true_delta[i] - true_delta[j])
    values = huber(residual, delta=delta)
    return values.mean(), float(values.detach().mean().item()), int(i.numel())


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
    """Warm-up + linear ramp for the physics-loss weights.

    ``max_epochs`` must be the number of epochs the model will *actually* train
    for. The two-phase refit therefore constructs its own schedule from
    ``refit_epochs`` rather than reusing the inner-training budget; reusing the
    larger budget would leave a short refit permanently inside the ramp.
    """

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

    @property
    def warmup_epochs(self) -> int:
        """First epoch at which any physics weight becomes non-zero."""
        import math

        return int(math.ceil(self.warmup_fraction * self.max_epochs))

    @property
    def physics_full_epoch(self) -> int:
        """First epoch at which the curriculum factor is exactly 1.0.

        Checkpoint eligibility starts here: a checkpoint selected earlier was
        chosen under a loss that is not the loss being reported.
        """
        import math

        return int(math.ceil(self.ramp_end_fraction * self.max_epochs))


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
    #: Direct supervision of the rate head. Data-like, so it is NOT ramped.
    rate_data: float = 0.0
    #: Cross-cell relative-degradation regulariser. Data-like, not ramped.
    pairwise: float = 0.0

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
            rate_data=self.rate_data,
            pairwise=self.pairwise,
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
            rate_data=float(mapping.get("rate_data", 0.0)),
            pairwise=float(mapping.get("pairwise", 0.0)),
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "data": self.data,
            "pde": self.pde,
            "initial_condition": self.initial_condition,
            "integral": self.integral,
            "monotonicity": self.monotonicity,
            "bounds": self.bounds,
            "rate_regularization": self.rate_regularization,
            "discrete_state_transition": self.discrete_state_transition,
            "rate_data": self.rate_data,
            "pairwise": self.pairwise,
        }


#: Loss components eligible for gradient-norm balancing. ``data`` is the
#: reference component and is never rescaled.
BALANCED_COMPONENTS: tuple[str, ...] = (
    "pde", "integral", "rate_data", "pairwise", "monotonicity", "bounds",
)


@dataclass
class GradientBalancer:
    """EMA gradient-norm balancing across active loss components.

    Each component's multiplier is ``ema_norm(data) / ema_norm(component)``,
    clipped to ``[min_multiplier, max_multiplier]``, so no single term dominates
    purely because of its numerical scale. The ``data`` component is the
    reference and is never rescaled; its own EMA is tracked only as the numerator.

    Zero, missing and non-finite gradients leave the previous multiplier in
    place rather than producing an inf/NaN update.
    """

    ema: float = 0.95
    min_multiplier: float = 0.1
    max_multiplier: float = 10.0
    _norms: dict[str, float] = field(default_factory=dict)
    _multipliers: dict[str, float] = field(default_factory=dict)

    def observe(self, component: str, norm: float) -> None:
        if norm is None or not isinstance(norm, (int, float)):
            return
        value = float(norm)
        if not (value > 0.0) or not torch.isfinite(torch.tensor(value)):
            return
        previous = self._norms.get(component)
        self._norms[component] = (
            value if previous is None
            else self.ema * previous + (1.0 - self.ema) * value
        )

    def multiplier(self, component: str) -> float:
        """Current multiplier for ``component`` (1.0 when unbalanceable)."""
        if component not in BALANCED_COMPONENTS:
            return 1.0
        reference = self._norms.get("data")
        own = self._norms.get(component)
        if not reference or not own or own <= 0.0:
            return self._multipliers.get(component, 1.0)
        raw = reference / own
        if not torch.isfinite(torch.tensor(raw)):
            return self._multipliers.get(component, 1.0)
        value = float(min(max(raw, self.min_multiplier), self.max_multiplier))
        self._multipliers[component] = value
        return value

    def state(self) -> dict[str, float]:
        out: dict[str, float] = {
            f"gradient_norm_ema_{name}": value for name, value in self._norms.items()
        }
        out.update({
            f"adaptive_multiplier_{name}": self.multiplier(name)
            for name in BALANCED_COMPONENTS
        })
        return out


@dataclass
class LossReport:
    """Container returned by :func:`compose_total_loss`."""

    total: torch.Tensor
    components: dict[str, torch.Tensor]
    row_means: dict[str, float] = field(default_factory=dict)
    effective_weights: dict[str, float] = field(default_factory=dict)
