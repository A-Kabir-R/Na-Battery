"""The inner selection split must be a cell-grouped partition, not a shuffle.

Choosing the epoch count on one split whose validation half holds four or five
cells is unstable; a proper k-fold partition is the primitive that makes nested
selection expressible. Whatever the caller does with the folds, no cell may
appear on both sides of any of them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.pinn.dataset import inner_fold_indices, inner_split_indices


def _frame(n_cells: int = 9, per_cell: int = 4) -> pd.DataFrame:
    return pd.DataFrame({
        "cell": [f"C{c}" for c in range(n_cells) for _ in range(per_cell)],
        "value": np.arange(n_cells * per_cell, dtype=float),
    })


def test_inner_folds_are_cell_disjoint():
    frame = _frame()
    cells = frame["cell"].to_numpy()
    for train_idx, val_idx in inner_fold_indices(frame, n_inner=3, seed=0):
        overlap = set(cells[train_idx]) & set(cells[val_idx])
        assert not overlap, f"cell(s) on both sides of an inner fold: {overlap}"


def test_validation_halves_partition_every_cell_exactly_once():
    frame = _frame()
    cells = frame["cell"].to_numpy()
    folds = inner_fold_indices(frame, n_inner=3, seed=0)
    assert len(folds) == 3
    seen: list[str] = []
    for _, val_idx in folds:
        seen.extend(sorted(set(cells[val_idx])))
    assert sorted(seen) == sorted(set(cells)), (
        "every cell must be validated exactly once across the inner folds"
    )
    assert len(seen) == len(set(seen)), "a cell was validated in two folds"


def test_fold_sizes_are_balanced():
    frame = _frame(n_cells=9)
    cells = frame["cell"].to_numpy()
    sizes = [
        len(set(cells[val_idx]))
        for _, val_idx in inner_fold_indices(frame, n_inner=3, seed=0)
    ]
    assert max(sizes) - min(sizes) <= 1, f"unbalanced inner folds: {sizes}"


def test_partition_is_deterministic_for_a_given_seed():
    frame = _frame()
    a = inner_fold_indices(frame, n_inner=3, seed=7)
    b = inner_fold_indices(frame, n_inner=3, seed=7)
    for (a_tr, a_va), (b_tr, b_va) in zip(a, b):
        assert np.array_equal(a_tr, b_tr)
        assert np.array_equal(a_va, b_va)


def test_inner_split_indices_selects_the_requested_fold():
    frame = _frame()
    folds = inner_fold_indices(frame, n_inner=3, seed=0)
    for k, (expected_train, expected_val) in enumerate(folds):
        train_idx, val_idx = inner_split_indices(
            frame, n_inner=3, seed=0, inner_fold=k,
        )
        assert np.array_equal(train_idx, expected_train)
        assert np.array_equal(val_idx, expected_val)


def test_more_folds_than_cells_is_clamped_not_crashed():
    frame = _frame(n_cells=2, per_cell=3)
    folds = inner_fold_indices(frame, n_inner=5, seed=0)
    assert 2 <= len(folds) <= 2
    for train_idx, val_idx in folds:
        assert train_idx.size and val_idx.size


def test_single_cell_is_rejected():
    with pytest.raises(ValueError):
        inner_fold_indices(_frame(n_cells=1), n_inner=3, seed=0)
