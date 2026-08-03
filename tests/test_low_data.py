"""Learning curves must subsample whole cells, stratified, and never leak."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.low_data import (
    LowDataError,
    LowDataStudy,
    cells_to_reach,
    sample_training_cells,
    summarize_curve,
)


def _pool(n_conditions: int = 4, cells_per_condition: int = 3,
          per_cell: int = 3) -> pd.DataFrame:
    rows = []
    for c in range(n_conditions):
        for k in range(cells_per_condition):
            for _ in range(per_cell):
                rows.append({
                    "cell": f"cond{c}_cell{k}",
                    "condition": f"cond{c}",
                    "y": float(c),
                })
    return pd.DataFrame(rows)


def test_sampling_returns_whole_cells():
    pool = _pool()
    cells = sample_training_cells(pool, 0.5, seed=0)
    assert set(cells) <= set(pool["cell"])
    assert len(cells) == len(set(cells))


def test_small_budget_spreads_across_conditions():
    pool = _pool(n_conditions=4, cells_per_condition=3)
    cells = sample_training_cells(pool, 4 / 12, seed=0)   # budget of 4 cells
    conditions = {cell.split("_")[0] for cell in cells}
    assert len(conditions) == 4, (
        "a small budget drawn from one corner of the test matrix measures the "
        "corner, not the budget"
    )


def test_full_fraction_takes_every_cell():
    pool = _pool()
    assert set(sample_training_cells(pool, 1.0, seed=0)) == set(pool["cell"])


def test_sampling_is_deterministic_for_a_seed():
    pool = _pool()
    assert sample_training_cells(pool, 0.5, seed=3) == sample_training_cells(
        pool, 0.5, seed=3,
    )


def test_different_seeds_give_different_draws():
    pool = _pool(n_conditions=4, cells_per_condition=4)
    draws = {tuple(sample_training_cells(pool, 0.5, seed=s)) for s in range(8)}
    assert len(draws) > 1, "repeats must actually vary the training set"


def test_invalid_fraction_rejected():
    with pytest.raises(LowDataError):
        sample_training_cells(_pool(), 0.0)
    with pytest.raises(LowDataError):
        sample_training_cells(_pool(), 1.5)


def test_study_never_trains_on_a_test_cell():
    pool, test = _pool(), _pool()
    test["cell"] = test["cell"] + "_TEST"
    seen: list[set[str]] = []

    def fit_predict(train, test_frame):
        seen.append(set(train["cell"]))
        return np.zeros(len(test_frame))

    LowDataStudy(fractions=(0.5, 1.0), repeats=2).run(
        pool, test, "y", fit_predict,
    )
    test_cells = set(test["cell"])
    for train_cells in seen:
        assert not (train_cells & test_cells)


def test_full_fraction_is_not_repeated():
    pool, test = _pool(), _pool()
    records = LowDataStudy(fractions=(1.0,), repeats=5).run(
        pool, test, "y", lambda tr, te: np.zeros(len(te)),
    )
    assert len(records) == 1, (
        "repeating a deterministic full-data fit would shrink the apparent "
        "variance for free"
    )


def test_budget_grows_monotonically_with_fraction():
    pool, test = _pool(), _pool()
    records = LowDataStudy(fractions=(0.2, 0.6, 1.0), repeats=3).run(
        pool, test, "y", lambda tr, te: np.zeros(len(te)),
    )
    medians = records.groupby("fraction")["n_train_cells"].median()
    assert medians.is_monotonic_increasing


def test_failed_fit_is_recorded_not_swallowed():
    pool, test = _pool(), _pool()

    def explode(train, test_frame):
        raise ValueError("boom")

    records = LowDataStudy(fractions=(0.5,), repeats=2).run(
        pool, test, "y", explode,
    )
    assert (records["status"] == "failed").all()
    assert records["error"].str.contains("boom").all()


def test_summarize_reports_a_band_and_cells_to_reach():
    records = pd.DataFrame({
        "model": ["m"] * 6,
        "fraction": [0.5] * 3 + [1.0] * 3,
        "status": ["ok"] * 6,
        "cell_macro_MAE": [0.3, 0.2, 0.25, 0.1, 0.11, 0.09],
        "n_train_cells": [5, 5, 5, 10, 10, 10],
    })
    curve = summarize_curve(records)
    assert set(curve.columns) >= {"median", "q25", "q75", "median_train_cells"}
    assert cells_to_reach(curve, 0.15) == 10.0
    assert np.isnan(cells_to_reach(curve, 0.01)), (
        "a threshold the curve never reaches must be NaN, not extrapolated"
    )
