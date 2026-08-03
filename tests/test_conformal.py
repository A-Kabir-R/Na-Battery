"""Conformal intervals must calibrate on whole cells and cover at nominal."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.conformal import (
    ConformalError,
    GroupedConformal,
    conformal_quantile,
    coverage_by_group,
    ensemble_intervals,
    interval_metrics,
    max_supported_level,
    min_calibration_units,
    split_calibration_cells,
)


def _predictions(n_cells: int = 40, per_cell: int = 4, noise: float = 0.01,
                 seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_cells):
        for _ in range(per_cell):
            truth = rng.normal()
            rows.append({
                "cell": f"C{c}",
                "condition": f"cond{c % 4}",
                "y_true": truth,
                "y_pred": truth + rng.normal(scale=noise),
            })
    return pd.DataFrame(rows)


def test_min_calibration_units_matches_the_split_conformal_bound():
    # ceil((n+1)(1-alpha)) <= n  <=>  n >= 1/alpha - 1
    assert min_calibration_units(0.1) == 9
    assert min_calibration_units(0.2) == 4
    assert min_calibration_units(0.05) == 19
    for alpha in (0.05, 0.1, 0.2, 0.5):
        n = min_calibration_units(alpha)
        assert np.ceil((n + 1) * (1 - alpha)) <= n
        assert np.ceil(((n - 1) + 1) * (1 - alpha)) > n - 1


def test_max_supported_level_is_the_inverse():
    assert max_supported_level(9) == pytest.approx(0.9)
    assert max_supported_level(19) == pytest.approx(0.95)


def test_too_few_calibration_cells_raises_rather_than_mislabelling():
    frame = _predictions(n_cells=5)
    with pytest.raises(ConformalError, match="cannot support"):
        GroupedConformal(alpha=0.1).fit(frame)


def test_conformal_quantile_uses_the_finite_sample_order_statistic():
    scores = np.arange(1.0, 11.0)          # n = 10
    # ceil((10+1)*0.9)/10 = 1.0 -> the largest score.
    assert conformal_quantile(scores, alpha=0.1) == 10.0


def test_marginal_coverage_reaches_nominal_on_exchangeable_cells():
    frame = _predictions(n_cells=60, seed=1)
    cells = frame["cell"].unique()
    calibration = frame[frame["cell"].isin(cells[:30])]
    test = frame[frame["cell"].isin(cells[30:])]
    model = GroupedConformal(alpha=0.1, score_reduction="mean").fit(calibration)
    intervals = model.predict_interval(test)
    metrics = interval_metrics(pd.concat([test, intervals], axis=1), alpha=0.1)
    assert metrics["PICP"] >= 0.85, "conformal must not under-cover on exchangeable data"


def test_normalized_intervals_still_vary_on_cells_never_calibrated_on():
    # The regression this guards: normalising by each calibration *cell's* own
    # residual spread is undefined for a test cell, so every interval silently
    # collapsed to one constant and the "locally adaptive" variant was a global
    # rescale. Under a cell-disjoint protocol every test cell is unseen, which
    # is exactly the case that was broken.
    calibration = _predictions(n_cells=30, seed=1)
    model = GroupedConformal(alpha=0.2, normalize=True).fit(calibration)

    unseen = pd.DataFrame({
        "cell": ["never_seen_a", "never_seen_b"],
        "y_pred": [0.05, 5.0],
    })
    half_width = model.predict_interval(unseen)["half_width"].to_numpy()
    assert half_width[1] > half_width[0], (
        "a much larger predicted change must carry a wider interval"
    )
    assert (half_width > 0).all()


def test_normalizer_floor_keeps_a_near_zero_prediction_from_collapsing():
    calibration = _predictions(n_cells=30, seed=2)
    model = GroupedConformal(alpha=0.2, normalize=True).fit(calibration)
    zero = pd.DataFrame({"cell": ["unseen"], "y_pred": [0.0]})
    assert model.predict_interval(zero)["half_width"].iloc[0] > 0.0


def test_max_reduction_is_conservative_relative_to_mean():
    frame = _predictions(n_cells=60, seed=2)
    cells = frame["cell"].unique()
    calibration = frame[frame["cell"].isin(cells[:30])]
    wide = GroupedConformal(alpha=0.1, score_reduction="max").fit(calibration)
    tight = GroupedConformal(alpha=0.1, score_reduction="mean").fit(calibration)
    assert wide.quantile_ >= tight.quantile_


def test_calibration_split_is_by_whole_cells():
    frame = _predictions(n_cells=20)
    train_idx, calibration_idx = split_calibration_cells(frame, seed=0)
    train_cells = set(frame.iloc[train_idx]["cell"])
    calibration_cells = set(frame.iloc[calibration_idx]["cell"])
    assert not (train_cells & calibration_cells), (
        "an anchor-level split would leak a cell's residual scale into its own "
        "interval"
    )


def test_calibration_split_excludes_the_omitted_condition():
    frame = _predictions(n_cells=20)
    _, calibration_idx = split_calibration_cells(
        frame, seed=0, exclude_conditions=("cond0",),
    )
    assert "cond0" not in set(frame.iloc[calibration_idx]["condition"]), (
        "under LOCO the omitted condition must not calibrate the intervals that "
        "claim not to have seen it"
    )


def test_calibration_split_never_consumes_an_entire_condition():
    frame = _predictions(n_cells=20)
    train_idx, _ = split_calibration_cells(
        frame, calibration_fraction=0.9, seed=0,
    )
    assert frame.iloc[train_idx]["condition"].nunique() == frame["condition"].nunique()


def test_interval_score_penalises_both_width_and_misses():
    base = pd.DataFrame({"y_true": [0.0], "lower": [-1.0], "upper": [1.0]})
    wide = pd.DataFrame({"y_true": [0.0], "lower": [-5.0], "upper": [5.0]})
    missed = pd.DataFrame({"y_true": [9.0], "lower": [-1.0], "upper": [1.0]})
    s_base = interval_metrics(base, alpha=0.1)["interval_score"]
    assert interval_metrics(wide, alpha=0.1)["interval_score"] > s_base
    assert interval_metrics(missed, alpha=0.1)["interval_score"] > s_base


def test_coverage_by_group_exposes_a_hidden_failure():
    """Marginal coverage can look fine while one condition is fully uncovered."""
    good = pd.DataFrame({
        "condition": ["ok"] * 90, "y_true": np.zeros(90),
        "lower": -np.ones(90), "upper": np.ones(90),
    })
    bad = pd.DataFrame({
        "condition": ["bad"] * 10, "y_true": np.full(10, 5.0),
        "lower": -np.ones(10), "upper": np.ones(10),
    })
    frame = pd.concat([good, bad], ignore_index=True)
    assert interval_metrics(frame, alpha=0.1)["PICP"] == pytest.approx(0.9)
    per_condition = coverage_by_group(frame, ("condition",), alpha=0.1)
    assert per_condition.set_index("condition").loc["bad", "PICP"] == 0.0


def test_ensemble_collapses_seeds_to_mean_and_spread():
    frame = pd.DataFrame({
        "anchor_id": ["a", "a", "a", "b", "b", "b"],
        "seed": [1, 2, 3, 1, 2, 3],
        "y_pred": [1.0, 2.0, 3.0, 10.0, 10.0, 10.0],
    })
    out = ensemble_intervals(frame).set_index("anchor_id")
    assert out.loc["a", "ensemble_mean"] == pytest.approx(2.0)
    assert out.loc["b", "ensemble_std"] == pytest.approx(0.0)
    assert int(out.loc["a", "n_seeds"]) == 3
