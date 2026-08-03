"""Epoch selection must use every inner fold, over their common epoch range."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.pinn.nested_selection import (
    NestedSelectionError,
    select_epoch,
    selection_report,
)


def _curve(values, start: int = 0) -> pd.DataFrame:
    return pd.DataFrame({
        "epoch": np.arange(start, start + len(values)),
        "val_MAE": np.asarray(values, dtype=float),
    })


def test_selects_the_argmin_of_the_mean_curve():
    # Fold 0 favours epoch 1, fold 1 favours epoch 3; the mean favours epoch 2.
    logs = {
        0: _curve([1.0, 0.1, 0.2, 0.9]),
        1: _curve([1.0, 0.9, 0.2, 0.1]),
    }
    selection = select_epoch(logs)
    assert selection.best_epoch == 2
    assert selection.refit_epochs == 3
    assert selection.n_folds == 2


def test_per_fold_argmins_are_reported_for_stability():
    logs = {0: _curve([1.0, 0.1, 0.5]), 1: _curve([1.0, 0.5, 0.1])}
    selection = select_epoch(logs)
    assert selection.per_fold_best_epoch == {0: 1, 1: 2}
    assert selection.spread() == 1


def test_mean_is_taken_only_over_the_common_epoch_range():
    """A fold that stopped early must not drop out of the denominator.

    Fold 1 stops at epoch 2. Epoch 5 looks excellent in fold 0 alone, but no
    mean may be formed there, or late epochs win purely by attrition.
    """
    logs = {
        0: _curve([1.0, 0.9, 0.8, 0.7, 0.6, 0.001]),
        1: _curve([1.0, 0.5, 0.4]),
    }
    selection = select_epoch(logs)
    assert selection.max_common_epoch == 2
    assert selection.best_epoch <= 2, (
        "selection escaped the range where every fold contributed"
    )


def test_min_epoch_gate_blocks_pre_curriculum_epochs():
    """Epochs before physics reaches full strength must not be selectable."""
    logs = {0: _curve([0.01, 0.02, 0.5, 0.4]), 1: _curve([0.01, 0.02, 0.5, 0.4])}
    ungated = select_epoch(logs)
    assert ungated.best_epoch == 0
    gated = select_epoch(logs, min_epoch=2)
    assert gated.best_epoch == 3


def test_falls_back_rather_than_crashing_when_the_gate_is_unreachable():
    logs = {0: _curve([0.3, 0.1]), 1: _curve([0.3, 0.2])}
    selection = select_epoch(logs, min_epoch=99)
    assert selection.best_epoch == 1


def test_disjoint_epoch_ranges_raise_a_useful_error():
    logs = {0: _curve([0.5, 0.4], start=0), 1: _curve([0.3, 0.2], start=10)}
    with pytest.raises(NestedSelectionError, match="common epoch range"):
        select_epoch(logs)


def test_missing_validation_column_is_rejected():
    logs = {0: pd.DataFrame({"epoch": [0, 1], "other": [1.0, 2.0]})}
    with pytest.raises(NestedSelectionError, match="missing"):
        select_epoch(logs)


def test_empty_input_is_rejected():
    with pytest.raises(NestedSelectionError):
        select_epoch({})


def test_nan_rows_are_ignored_not_propagated():
    logs = {
        0: pd.DataFrame({"epoch": [0, 1, 2], "val_MAE": [1.0, np.nan, 0.2]}),
        1: pd.DataFrame({"epoch": [0, 1, 2], "val_MAE": [1.0, 0.5, 0.3]}),
    }
    selection = select_epoch(logs)
    assert selection.best_epoch == 2
    assert np.isfinite(selection.best_mean_validation)


def test_single_fold_still_works_and_says_so():
    selection = select_epoch({0: _curve([1.0, 0.2, 0.9])})
    assert selection.best_epoch == 1
    assert selection.n_folds == 1
    assert selection.spread() == 0


def test_report_exposes_the_selection_spread():
    logs = {0: _curve([1.0, 0.1, 0.5]), 1: _curve([1.0, 0.5, 0.1])}
    report = selection_report(select_epoch(logs))
    assert report.loc[0, "n_inner_folds"] == 2
    assert report.loc[0, "per_fold_spread"] == 1
    assert "0:1" in report.loc[0, "per_fold_best_epoch"]
