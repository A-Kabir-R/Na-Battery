"""Learning curves in units of *training cells*, not training rows.

The claim the journal cares about is "physics constraints reduce the aging-test
burden", and the currency of that burden is cells: a cell is weeks of cycler
time, an anchor is free once the cell exists. Subsampling anchors would therefore
measure nothing anyone pays for, and would also leak -- the remaining anchors of
a dropped cell would still carry its trajectory.

Every fraction here subsamples **whole cells**, stratified by condition so a
small budget does not accidentally consist entirely of one corner of the test
matrix, and repeats the draw so the curve carries an uncertainty band rather
than a single lucky sample.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import pandas as pd

#: Cell-level training fractions for the learning curve.
DEFAULT_FRACTIONS: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8, 1.0)

#: Repeats per fraction. Twenty is the point where the band stops moving on a
#: 26-cell development set; fewer makes the curve a story about which cells were
#: drawn rather than about how many.
DEFAULT_REPEATS: int = 20


class LowDataError(RuntimeError):
    """Raised when a requested budget cannot produce a usable training set."""


def sample_training_cells(frame: pd.DataFrame, fraction: float, *,
                          seed: int = 0,
                          group_column: str = "cell",
                          condition_column: str = "condition") -> list[str]:
    """Draw a condition-stratified subset of whole cells.

    At least one cell is kept per condition until the budget runs out, so the
    smallest budgets stay spread across the test matrix. Conditions are visited
    in a shuffled order so the same condition is not systematically favoured
    when the budget cannot cover all of them.
    """
    if not 0.0 < fraction <= 1.0:
        raise LowDataError(f"fraction must be in (0, 1], got {fraction}")
    table = frame[[group_column, condition_column]].drop_duplicates()
    all_cells = table[group_column].astype(str).unique()
    target = int(max(1, round(len(all_cells) * fraction)))
    rng = np.random.default_rng(seed)

    by_condition: dict[str, list[str]] = {}
    for condition, group in table.groupby(condition_column, sort=True):
        cells = group[group_column].astype(str).to_numpy()
        rng.shuffle(cells)
        by_condition[str(condition)] = list(cells)

    order = list(by_condition)
    rng.shuffle(order)
    chosen: list[str] = []
    # Round-robin across conditions: one cell each per pass, until the budget is
    # met or every condition is exhausted.
    while len(chosen) < target and any(by_condition.values()):
        for condition in order:
            if len(chosen) >= target:
                break
            if by_condition[condition]:
                chosen.append(by_condition[condition].pop())
    if not chosen:
        raise LowDataError("no cells selected")
    return sorted(chosen)


@dataclass
class LowDataStudy:
    """Run a fit/predict callable over a grid of cell budgets.

    ``fit_predict`` receives ``(train_frame, test_frame)`` and returns
    predictions aligned to ``test_frame``. Keeping the model opaque means the
    same study runs for a classical baseline, a physics baseline or the PINN
    without special-casing.
    """

    fractions: Sequence[float] = DEFAULT_FRACTIONS
    repeats: int = DEFAULT_REPEATS
    group_column: str = "cell"
    condition_column: str = "condition"
    seed: int = 0
    #: Every repeat of every ``run`` call, accumulated. Callers run one study
    #: instance across folds, so replacing this per call would silently keep
    #: only the last fold.
    records: list[dict] = field(default_factory=list)

    def run(self, train_pool: pd.DataFrame, test_frame: pd.DataFrame,
            target: str,
            fit_predict: Callable[[pd.DataFrame, pd.DataFrame], np.ndarray],
            *, model_name: str = "model",
            fold: int | str = 0) -> pd.DataFrame:
        y_test = test_frame[target].to_numpy(dtype=float)
        test_cells = test_frame[self.group_column].astype(str).to_numpy()
        test_conditions = test_frame[self.condition_column].astype(str).to_numpy()
        rows: list[dict] = []

        for fraction in self.fractions:
            for repeat in range(self.repeats):
                # Full budget is deterministic; repeating it would only re-run
                # an identical fit and shrink the apparent variance for free.
                if fraction >= 1.0 and repeat > 0:
                    continue
                seed = self.seed + 1000 * int(round(fraction * 100)) + repeat
                cells = sample_training_cells(
                    train_pool, fraction, seed=seed,
                    group_column=self.group_column,
                    condition_column=self.condition_column,
                )
                subset = train_pool[
                    train_pool[self.group_column].astype(str).isin(cells)
                ]
                if subset.empty:
                    continue
                try:
                    prediction = np.asarray(
                        fit_predict(subset, test_frame), dtype=float
                    )
                    if prediction.shape != y_test.shape:
                        # A length-1 return would broadcast and produce a
                        # plausible-looking MAE against the wrong thing.
                        raise LowDataError(
                            f"fit_predict returned {prediction.shape}, "
                            f"expected {y_test.shape}"
                        )
                except Exception as error:  # noqa: BLE001 - recorded, not swallowed
                    rows.append({
                        "model": model_name, "fold": fold, "fraction": fraction,
                        "repeat": repeat, "n_train_cells": len(cells),
                        "n_train_rows": len(subset), "status": "failed",
                        "error": str(error), "MAE": np.nan,
                        "cell_macro_MAE": np.nan, "condition_macro_MAE": np.nan,
                    })
                    continue
                error_abs = np.abs(y_test - prediction)
                rows.append({
                    "model": model_name,
                    "fold": fold,
                    "fraction": fraction,
                    "repeat": repeat,
                    "n_train_cells": len(cells),
                    "n_train_rows": int(len(subset)),
                    "n_train_conditions": int(
                        subset[self.condition_column].nunique()
                    ),
                    "status": "ok",
                    "error": "",
                    "MAE": float(np.nanmean(error_abs)),
                    "cell_macro_MAE": float(
                        pd.Series(error_abs).groupby(test_cells).mean().mean()
                    ),
                    "condition_macro_MAE": float(
                        pd.Series(error_abs).groupby(test_conditions).mean().mean()
                    ),
                })
        self.records.extend(rows)
        return pd.DataFrame(rows)


def summarize_curve(records: pd.DataFrame, *,
                    metric: str = "cell_macro_MAE") -> pd.DataFrame:
    """Median and inter-quartile band per budget.

    Median rather than mean: a single unlucky draw at a small budget can produce
    an enormous error, and a mean would let that one draw define the curve.
    """
    usable = records[records["status"].eq("ok")]
    if usable.empty:
        return pd.DataFrame()
    grouped = usable.groupby(["model", "fraction"], sort=True)[metric]
    summary = grouped.agg(
        median="median",
        q25=lambda s: s.quantile(0.25),
        q75=lambda s: s.quantile(0.75),
        n_repeats="size",
    ).reset_index()
    cells = (
        usable.groupby(["model", "fraction"])["n_train_cells"].median()
        .rename("median_train_cells").reset_index()
    )
    return summary.merge(cells, on=["model", "fraction"], how="left")


def cells_to_reach(curve: pd.DataFrame, threshold: float, *,
                   value_column: str = "median") -> float:
    """Smallest median training-cell count whose error is at or below ``threshold``.

    This is the number the data-efficiency claim reduces to: *"the proposed model
    needs N fewer cells than the baseline to reach the same error"*. Returns NaN
    when the curve never gets there, which must be reported as such rather than
    extrapolated.
    """
    reached = curve[curve[value_column] <= threshold]
    if reached.empty:
        return float("nan")
    return float(reached["median_train_cells"].min())
