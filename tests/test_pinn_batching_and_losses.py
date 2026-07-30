"""Cell-balanced batching, rate supervision and the pairwise loss."""
from __future__ import annotations

import collections

import pytest

torch = pytest.importorskip("torch")

from src.pinn.batching import CellBalancedSampler
from src.pinn.losses import (
    observed_degradation_rate, pairwise_delta_loss, rate_data_loss,
)


def _imbalanced_cells() -> torch.Tensor:
    """One dominant cell with 60 anchors, five small cells with 4 each."""
    return torch.tensor([0] * 60 + [1] * 4 + [2] * 4 + [3] * 4 + [4] * 4 + [5] * 4)


def _sampler(mode: str, batch_size: int = 32) -> CellBalancedSampler:
    cells = _imbalanced_cells()
    return CellBalancedSampler(
        row_positions=torch.arange(cells.numel()),
        cell_index=cells,
        batch_size=batch_size,
        mode=mode,
    )


def _cell_frequencies(mode: str, epochs: int = 200) -> dict[int, float]:
    cells = _imbalanced_cells()
    sampler = _sampler(mode)
    generator = torch.Generator()
    generator.manual_seed(42)
    counter: collections.Counter = collections.Counter()
    for _ in range(epochs):
        for batch in sampler.epoch_batches(generator):
            counter.update(cells[batch].tolist())
    total = sum(counter.values())
    return {cell: count / total for cell, count in counter.items()}


def test_cell_balanced_sampling_is_approximately_uniform_over_cells():
    frequencies = _cell_frequencies("cell_balanced")
    assert len(frequencies) == 6
    for share in frequencies.values():
        # Uniform would be 1/6 = 0.167; allow generous sampling slack.
        assert 0.10 < share < 0.25


def test_sequential_sampling_is_dominated_by_the_largest_cell():
    """Confirms the balanced mode is doing something the old path did not."""
    frequencies = _cell_frequencies("sequential")
    assert frequencies[0] > 0.60


def test_batching_is_deterministic_under_a_fixed_seed():
    sampler = _sampler("cell_balanced")
    first = torch.Generator(); first.manual_seed(42)
    second = torch.Generator(); second.manual_seed(42)
    a = [b.tolist() for b in sampler.epoch_batches(first)]
    b = [b.tolist() for b in sampler.epoch_batches(second)]
    assert a == b


def test_batches_are_resampled_between_epochs():
    sampler = _sampler("cell_balanced")
    generator = torch.Generator(); generator.manual_seed(42)
    first = [b.tolist() for b in sampler.epoch_batches(generator)]
    second = [b.tolist() for b in sampler.epoch_batches(generator)]
    assert first != second


def test_batches_contain_many_distinct_cells_for_the_pairwise_loss():
    cells = _imbalanced_cells()
    sampler = _sampler("cell_balanced")
    generator = torch.Generator(); generator.manual_seed(42)
    for batch in sampler.epoch_batches(generator):
        assert len(set(cells[batch].tolist())) >= 3


def test_only_training_rows_can_appear_in_batches():
    """Rows outside row_positions (i.e. validation rows) must never be drawn."""
    cells = _imbalanced_cells()
    training = torch.arange(0, 40)
    sampler = CellBalancedSampler(
        row_positions=training, cell_index=cells, batch_size=16,
        mode="cell_balanced",
    )
    generator = torch.Generator(); generator.manual_seed(0)
    allowed = set(training.tolist())
    for batch in sampler.epoch_batches(generator):
        assert set(batch.tolist()) <= allowed


def test_full_batch_when_batch_size_is_non_positive():
    sampler = _sampler("sequential", batch_size=0)
    generator = torch.Generator(); generator.manual_seed(0)
    batches = sampler.epoch_batches(generator)
    assert len(batches) == 1
    assert batches[0].numel() == 80


def test_unknown_batch_mode_raises():
    with pytest.raises(ValueError):
        _sampler("round_robin")


# ---------------------------------------------------------------------------
# Rate supervision
# ---------------------------------------------------------------------------
def test_observed_rate_matches_the_finite_difference():
    u_current = torch.tensor([1.0, 1.0])
    u_next = torch.tensor([0.9, 0.8])
    delta = torch.tensor([1.0, 2.0])
    rate, mask = observed_degradation_rate(u_current, u_next, delta)
    assert torch.allclose(rate, torch.tensor([0.1, 0.1]), atol=1e-6)
    assert mask.all()


def test_regeneration_does_not_produce_a_negative_rate_target():
    u_current = torch.tensor([1.0])
    u_next = torch.tensor([1.005])          # small apparent capacity gain
    delta = torch.tensor([1.0])
    rate, mask = observed_degradation_rate(
        u_current, u_next, delta, regeneration_tolerance=0.02,
    )
    assert rate.item() >= 0.0
    assert mask.all()                        # within tolerance: clamped, kept


def test_regeneration_beyond_tolerance_is_masked_out():
    u_current = torch.tensor([1.0])
    u_next = torch.tensor([1.20])            # not plausibly measurement noise
    delta = torch.tensor([1.0])
    _, mask = observed_degradation_rate(
        u_current, u_next, delta, regeneration_tolerance=0.02,
    )
    assert not mask.any()


def test_tiny_or_missing_horizon_is_masked_out():
    u_current = torch.tensor([1.0, 1.0])
    u_next = torch.tensor([0.9, float("nan")])
    delta = torch.tensor([0.0, 1.0])
    _, mask = observed_degradation_rate(u_current, u_next, delta)
    assert not mask[0]
    assert not mask[1]


def test_rate_data_loss_is_zero_for_a_perfect_rate_head():
    observed = torch.tensor([0.1, 0.2, 0.3])
    cells = torch.tensor([0, 1, 2])
    mask = torch.ones(3, dtype=torch.bool)
    loss, row_mean, coverage = rate_data_loss(observed.clone(), observed, cells, mask)
    assert loss.item() == pytest.approx(0.0, abs=1e-9)
    assert coverage == 1.0


def test_rate_data_loss_reports_partial_coverage():
    observed = torch.tensor([0.1, 0.2, 0.3, 0.4])
    predicted = torch.zeros(4)
    cells = torch.tensor([0, 0, 1, 1])
    mask = torch.tensor([True, False, True, False])
    _, _, coverage = rate_data_loss(predicted, observed, cells, mask)
    assert coverage == 0.5


def test_rate_data_loss_is_safe_with_no_valid_rows():
    loss, row_mean, coverage = rate_data_loss(
        torch.zeros(3), torch.zeros(3), torch.tensor([0, 1, 2]),
        torch.zeros(3, dtype=torch.bool),
    )
    assert loss.item() == 0.0
    assert coverage == 0.0


# ---------------------------------------------------------------------------
# Pairwise loss
# ---------------------------------------------------------------------------
def test_pairwise_loss_is_zero_when_deltas_are_exact():
    u_current = torch.tensor([1.0, 1.0, 1.0, 1.0])
    u_true = torch.tensor([0.95, 0.90, 0.85, 0.80])
    cells = torch.tensor([0, 1, 2, 3])
    generator = torch.Generator(); generator.manual_seed(0)
    loss, _, n_pairs = pairwise_delta_loss(
        u_true.clone(), u_current, u_true, cells, generator=generator,
    )
    assert n_pairs > 0
    assert loss.item() == pytest.approx(0.0, abs=1e-9)


def test_pairwise_loss_penalises_a_wrong_ordering():
    u_current = torch.tensor([1.0, 1.0])
    u_true = torch.tensor([0.95, 0.85])
    predicted = torch.tensor([0.85, 0.95])       # ordering inverted
    cells = torch.tensor([0, 1])
    generator = torch.Generator(); generator.manual_seed(0)
    loss, _, n_pairs = pairwise_delta_loss(
        predicted, u_current, u_true, cells, generator=generator,
    )
    assert n_pairs > 0
    assert loss.item() > 0.0


def test_pairwise_loss_only_uses_cross_cell_pairs():
    """All rows from one cell means no valid pair, hence no contribution."""
    u_current = torch.ones(6)
    u_true = torch.linspace(0.9, 0.8, 6)
    cells = torch.zeros(6, dtype=torch.long)
    generator = torch.Generator(); generator.manual_seed(0)
    loss, _, n_pairs = pairwise_delta_loss(
        torch.zeros(6), u_current, u_true, cells, generator=generator,
    )
    assert n_pairs == 0
    assert loss.item() == 0.0


def test_pairwise_loss_handles_a_single_row():
    generator = torch.Generator(); generator.manual_seed(0)
    loss, _, n_pairs = pairwise_delta_loss(
        torch.zeros(1), torch.ones(1), torch.ones(1), torch.zeros(1, dtype=torch.long),
        generator=generator,
    )
    assert n_pairs == 0
    assert loss.item() == 0.0
