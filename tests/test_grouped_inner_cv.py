"""Hyperparameter selection must never see the same cell on both sides.

Cells contribute several anchors each, so an ungrouped inner CV puts the same
cell in inner-train and inner-validation and selects hyperparameters that
exploit cell identity. On the real split this affected 1-2 boundary cells in
every inner fold of every outer fold.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold

from src.models.regressors import build_tuned_regressors
from src.pipeline.run_experiment import GROUP_COLUMN, _fit_partition


def test_tuned_regressors_use_grouped_inner_cv_and_nan_error_score():
    for name, estimator in build_tuned_regressors().items():
        if not isinstance(estimator, GridSearchCV):
            continue  # the Dummy/PreviousRPT baselines are not searched
        assert isinstance(estimator.cv, GroupKFold), (
            f"{name} inner CV is {estimator.cv!r}; an integer resolves to plain "
            "KFold and leaks cell identity into hyperparameter selection"
        )
        assert np.isnan(estimator.error_score), (
            f"{name} uses error_score={estimator.error_score!r}. With scoring='r2', "
            "0.0 is the score of 'predict the training mean', so failed candidates "
            "would outrank legitimately negative ones."
        )


class _GroupSpy(GroupKFold):
    """GroupKFold that records the groups it was handed."""

    received: list = []

    def split(self, X, y=None, groups=None):
        _GroupSpy.received.append(None if groups is None else np.asarray(groups))
        return super().split(X, y, groups)


def _frame(n_cells: int = 6, per_cell: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for c in range(n_cells):
        for _ in range(per_cell):
            x = rng.normal()
            rows.append({
                GROUP_COLUMN: f"CELL{c}",
                "f1": x,
                "f2": rng.normal(),
                "target": 0.3 * x + 0.01 * c,
            })
    return pd.DataFrame(rows)


def test_fit_partition_passes_cell_groups_to_the_search():
    """Without groups, GroupKFold.split raises; the spy proves they arrive."""
    _GroupSpy.received = []
    frame = _frame()
    search = GridSearchCV(
        Ridge(),
        {"alpha": [0.1, 1.0]},
        cv=_GroupSpy(n_splits=3),
        scoring="r2",
        error_score=np.nan,
    )
    _fit_partition(
        frame, frame, target="target", model_name="Ridge",
        model=search, kind="regression", preprocessing="",
    )

    assert _GroupSpy.received, "the inner splitter was never called"
    for groups in _GroupSpy.received:
        assert groups is not None, "fit() was called without groups"
        assert len(groups) == len(frame)
        assert set(groups) == set(frame[GROUP_COLUMN])


def test_grouped_inner_split_has_no_cell_overlap():
    frame = _frame()
    cells = frame[GROUP_COLUMN].to_numpy()
    for train_idx, val_idx in GroupKFold(n_splits=3).split(frame, groups=cells):
        overlap = set(cells[train_idx]) & set(cells[val_idx])
        assert not overlap, f"cell(s) on both sides of an inner split: {overlap}"
