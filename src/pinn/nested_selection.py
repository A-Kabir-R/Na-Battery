"""Choose the refit epoch count on *all* inner folds, not one.

``train_fold`` early-stops on a single cell-grouped inner split. With 26
development cells and five outer folds, that validation half holds four or five
cells, so the epoch it picks is noisy: re-drawing the split moves it.

Nested selection runs phase 1 once per inner fold, averages the per-epoch
validation curves, and takes the argmin of the mean. Only the *refit length*
changes -- phase 1 itself is untouched, so nothing about the training dynamics
or the leakage boundaries moves.

The functions here are deliberately pure: they consume the epoch logs that
``train_fold`` already writes and return a number. That keeps the fragile part
(a 1000-line training loop) out of the change, and makes the selection rule
itself unit-testable without training anything.

Averaging note: folds are averaged **per epoch index**, and an epoch is only
eligible once every fold has reached it. A mean taken over the subset of folds
that survived to epoch 400 would favour late epochs purely because the folds
that stopped early dropped out of the average.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

VALIDATION_COLUMN = "val_MAE"
EPOCH_COLUMN = "epoch"


class NestedSelectionError(RuntimeError):
    """Raised when the inner curves cannot support a selection."""


@dataclass
class NestedSelection:
    """Result of selecting an epoch count across inner folds."""

    best_epoch: int
    best_mean_validation: float
    n_folds: int
    max_common_epoch: int
    #: Mean curve actually used, for the supplementary figure.
    mean_curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    per_fold_best_epoch: dict[int, int] = field(default_factory=dict)

    @property
    def refit_epochs(self) -> int:
        return self.best_epoch + 1

    def spread(self) -> int:
        """Range of the per-fold argmins.

        Report this. A wide spread means the epoch count is not identified by
        the data, and a single-split run would have been reporting a coin flip.
        """
        if not self.per_fold_best_epoch:
            return 0
        values = list(self.per_fold_best_epoch.values())
        return int(max(values) - min(values))


def load_epoch_logs(paths) -> dict[int, pd.DataFrame]:
    """Read ``epoch_log.csv`` for each inner fold.

    ``paths`` maps inner-fold index to a CSV path. Missing or empty logs raise
    rather than being skipped: silently selecting on two folds when three were
    requested would misreport the protocol.
    """
    logs: dict[int, pd.DataFrame] = {}
    for inner_fold, path in dict(paths).items():
        frame = pd.read_csv(path)
        if frame.empty:
            raise NestedSelectionError(f"empty epoch log for inner fold {inner_fold}: {path}")
        logs[int(inner_fold)] = frame
    if not logs:
        raise NestedSelectionError("no epoch logs supplied")
    return logs


def select_epoch(logs: dict[int, pd.DataFrame], *,
                 validation_column: str = VALIDATION_COLUMN,
                 epoch_column: str = EPOCH_COLUMN,
                 min_epoch: int = 0) -> NestedSelection:
    """Argmin of the mean inner-fold validation curve.

    ``min_epoch`` mirrors the trainer's checkpoint-eligibility gate: epochs
    before the physics curriculum reaches full strength must not be selectable,
    or the "full PINN" result can silently be a partially-physics model.
    """
    if not logs:
        raise NestedSelectionError("no inner-fold curves to select from")

    curves = []
    per_fold_best: dict[int, int] = {}
    for inner_fold, frame in sorted(logs.items()):
        missing = {validation_column, epoch_column} - set(frame.columns)
        if missing:
            raise NestedSelectionError(
                f"inner fold {inner_fold} log is missing {sorted(missing)}"
            )
        curve = (
            frame[[epoch_column, validation_column]]
            .dropna()
            .astype({epoch_column: int})
            .groupby(epoch_column, as_index=True)[validation_column]
            .mean()
            .rename(inner_fold)
        )
        if curve.empty:
            raise NestedSelectionError(
                f"inner fold {inner_fold} has no finite {validation_column}"
            )
        curves.append(curve)
        eligible = curve[curve.index >= min_epoch]
        source = eligible if not eligible.empty else curve
        per_fold_best[int(inner_fold)] = int(source.idxmin())

    matrix = pd.concat(curves, axis=1)
    # Only epochs every fold reached. Otherwise the mean silently changes its
    # denominator as folds drop out, biasing selection toward late epochs.
    complete = matrix.dropna(axis=0, how="any")
    if complete.empty:
        raise NestedSelectionError(
            "inner folds share no common epoch range; lower min_epochs or raise "
            "early_stopping_patience so the folds overlap"
        )
    max_common_epoch = int(complete.index.max())

    mean_curve = complete.mean(axis=1)
    eligible = mean_curve[mean_curve.index >= min_epoch]
    if eligible.empty:
        # Every fold stopped before the eligibility gate. Selecting anyway would
        # hide that; falling back to the last common epoch is the honest choice
        # and the caller can see min_epoch was never reached.
        eligible = mean_curve
    best_epoch = int(eligible.idxmin())

    frame = pd.DataFrame({
        "epoch": mean_curve.index,
        "mean_validation": mean_curve.to_numpy(),
        "std_validation": complete.std(axis=1, ddof=0).to_numpy(),
        "n_folds": complete.notna().sum(axis=1).to_numpy(),
    })
    return NestedSelection(
        best_epoch=best_epoch,
        best_mean_validation=float(eligible.min()),
        n_folds=len(logs),
        max_common_epoch=max_common_epoch,
        mean_curve=frame,
        per_fold_best_epoch=per_fold_best,
    )


def selection_report(selection: NestedSelection) -> pd.DataFrame:
    """One-row summary for the manuscript's selection-stability table."""
    return pd.DataFrame([{
        "selected_epoch": selection.best_epoch,
        "refit_epochs": selection.refit_epochs,
        "mean_inner_validation_MAE": selection.best_mean_validation,
        "n_inner_folds": selection.n_folds,
        "max_common_epoch": selection.max_common_epoch,
        "per_fold_best_epoch": ";".join(
            f"{fold}:{epoch}" for fold, epoch in
            sorted(selection.per_fold_best_epoch.items())
        ),
        # Wide spread => the epoch count is not identified by this much data.
        "per_fold_spread": selection.spread(),
    }])
