"""Loss component behavior: monotonicity tolerance, bounds, cell balance."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.pinn.losses import (
    CurriculumSchedule, bounds_loss, cell_balanced_reduce, huber,
    monotonicity_loss,
)


def test_monotonicity_tolerance_no_penalty_below() -> None:
    u_next = torch.tensor([1.0, 1.01])
    u_current = torch.tensor([1.0, 1.0])
    cells = torch.tensor([0, 1])
    loss, _ = monotonicity_loss(u_next, u_current, cells, epsilon_rec=0.02)
    assert float(loss.item()) == 0.0


def test_monotonicity_penalty_above_tolerance() -> None:
    u_next = torch.tensor([1.1, 1.5])
    u_current = torch.tensor([1.0, 1.0])
    cells = torch.tensor([0, 1])
    loss, _ = monotonicity_loss(u_next, u_current, cells, epsilon_rec=0.02)
    assert float(loss.item()) > 0.0


def test_bounds_loss_inside_bounds_is_zero() -> None:
    u = torch.tensor([0.5, 0.9])
    cells = torch.tensor([0, 1])
    loss, _ = bounds_loss(u, cells, u_min=0.0, u_max=1.0)
    assert float(loss.item()) == 0.0


def test_bounds_loss_lower_and_upper() -> None:
    u = torch.tensor([-0.1, 1.2])
    cells = torch.tensor([0, 1])
    loss, _ = bounds_loss(u, cells, u_min=0.0, u_max=1.0)
    assert float(loss.item()) > 0.0


def test_cell_balanced_reduce_matches_manual_mean() -> None:
    values = torch.tensor([1.0, 3.0, 5.0, 7.0])
    cells = torch.tensor([0, 0, 1, 1])
    balanced = cell_balanced_reduce(values, cells).item()
    # Cell 0 mean = 2.0; cell 1 mean = 6.0; mean of means = 4.0
    assert balanced == pytest.approx(4.0)


def test_curriculum_schedule_ramps() -> None:
    curriculum = CurriculumSchedule(max_epochs=100, warmup_fraction=0.1,
                                    ramp_end_fraction=0.3)
    assert curriculum.factor(5) == 0.0
    assert 0 < curriculum.factor(20) < 1
    assert curriculum.factor(50) == 1.0


def test_huber_is_quadratic_below_delta_linear_above() -> None:
    r = torch.tensor([0.5, 2.0])
    values = huber(r, delta=1.0)
    # Below delta: 0.5 * 0.25 = 0.125
    # Above delta: 1.0 * (2.0 - 0.5) = 1.5
    assert values[0].item() == pytest.approx(0.125)
    assert values[1].item() == pytest.approx(1.5)
