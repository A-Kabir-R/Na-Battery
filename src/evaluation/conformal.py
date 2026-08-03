"""Cell-grouped split-conformal prediction intervals.

Accuracy differences between the top model families on this dataset are not
statistically resolvable -- a paired cell-clustered bootstrap puts ExtraTrees vs
Ridge at [-0.00041, 0.00083] Ah, crossing zero. Calibrated *uncertainty* is
therefore where the methodological contribution lands, and it has to be done at
the right grouping level to mean anything.

The unit of exchangeability is the **cell**, not the anchor. Anchors within a
cell share a trajectory, an assembly and a tester slot, so calibrating on a
random subset of anchors leaks the cell's residual scale into its own interval
and produces coverage that looks correct in-sample and fails on a new cell.
Every function here therefore splits, calibrates and evaluates on whole cells.

Under leave-one-condition-out the calibration set must also exclude the omitted
condition, otherwise the "unseen condition" claim is false: the intervals would
have been tuned on exactly the distribution they claim not to have seen. Split
conformal guarantees marginal coverage only under exchangeability, which
condition shift breaks by construction -- so the honest deliverable is to
*report* the coverage drop, not to hide it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


class ConformalError(RuntimeError):
    """Raised when a calibration set cannot support the requested level."""


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample split-conformal quantile of nonconformity ``scores``.

    Uses the ``ceil((n+1)(1-alpha))/n`` order statistic, not the plain empirical
    quantile: the ``+1`` is what buys the finite-sample guarantee. With too few
    calibration units for the requested level the quantile is unbounded, which is
    reported rather than silently truncated to the maximum score.
    """
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    n = scores.size
    if n == 0:
        raise ConformalError("no finite nonconformity scores")
    level = np.ceil((n + 1) * (1.0 - alpha)) / n
    if level > 1.0:
        raise ConformalError(
            f"{n} calibration units cannot support alpha={alpha:g}; "
            f"need at least {int(np.ceil(1.0 / alpha)) - 1}"
        )
    return float(np.quantile(scores, level, method="higher"))


def min_calibration_units(alpha: float) -> int:
    """Smallest calibration set that can support a ``1 - alpha`` interval.

    Split conformal needs ``ceil((n+1)(1-alpha)) <= n``, i.e. ``n >= 1/alpha - 1``.
    This binds hard when the calibration units are *cells*: 90% intervals need 9
    calibration cells, and this dataset has only ~21 cells in an outer-training
    fold, so a 30% calibration split leaves 7 and cannot support 90%.

    Report the achievable level rather than silently degrading it -- an interval
    labelled 90% that was calibrated on too few units is simply mislabelled.
    """
    if not 0.0 < alpha < 1.0:
        raise ConformalError(f"alpha must be in (0, 1), got {alpha}")
    return int(np.ceil(1.0 / alpha)) - 1


def max_supported_level(n_units: int) -> float:
    """Highest nominal coverage ``n_units`` calibration units can certify."""
    if n_units < 1:
        return float("nan")
    return float(n_units / (n_units + 1.0))


@dataclass
class GroupedConformal:
    """Split conformal calibrated on whole cells.

    ``normalize`` switches from absolute residuals to residuals scaled by a
    *difficulty estimate* -- the locally adaptive variant, which widens the
    interval for anchors the model expects to find hard.

    The difficulty estimate has to be computable for a row the model has never
    seen. An earlier version normalised by each calibration **cell's** own
    residual spread, which cannot work: a test cell is by construction not a
    calibration cell, so the lookup missed on every row and every interval fell
    back to one constant. That is a global rescale wearing a local label. The
    normaliser here is the predicted magnitude, floored at a calibration
    quantile so a near-zero prediction cannot collapse the interval to nothing.
    """

    alpha: float = 0.1
    normalize: bool = False
    group_column: str = "cell"
    #: How each calibration cell's anchors collapse to one score. ``"max"`` is
    #: conservative and is the safe default -- it certifies the whole cell -- but
    #: it over-covers noticeably here (PICP 0.98 at a nominal 0.80). ``"mean"``
    #: and ``"median"`` are tighter and usually closer to nominal; state which
    #: was used, because they are not the same guarantee.
    score_reduction: str = "max"
    #: The normaliser is floored at this quantile of the calibration
    #: predictions' magnitude. Without a floor an anchor predicted at ~0 gets a
    #: ~0-width interval and coverage collapses on exactly the low-signal rows.
    normalizer_floor_quantile: float = 0.25

    def _normalizer(self, prediction: np.ndarray) -> np.ndarray:
        """Difficulty estimate for rows the model has never seen."""
        if not self.normalize:
            return np.ones(len(prediction), dtype=float)
        return np.maximum(np.abs(np.asarray(prediction, dtype=float)),
                          self.scale_floor_)

    def fit(self, calibration: pd.DataFrame, *,
            true_column: str = "y_true",
            pred_column: str = "y_pred") -> "GroupedConformal":
        frame = calibration[[self.group_column, true_column, pred_column]].copy()
        residual = (frame[true_column] - frame[pred_column]).abs()
        frame["residual"] = residual
        finite = frame["residual"].to_numpy(dtype=float)
        if not np.isfinite(finite).any():
            raise ConformalError("calibration residuals are all non-finite")

        self.n_calibration_cells_ = int(frame[self.group_column].nunique())
        prediction = frame[pred_column].to_numpy(dtype=float)
        if self.normalize:
            magnitude = np.abs(prediction[np.isfinite(prediction)])
            floor = (
                float(np.quantile(magnitude, self.normalizer_floor_quantile))
                if magnitude.size else 0.0
            )
            # A degenerate calibration set (every prediction identically zero)
            # would otherwise divide by zero; fall back to an unnormalised fit.
            self.scale_floor_ = max(floor, 1e-12)
            frame["score"] = frame["residual"] / self._normalizer(prediction)
        else:
            self.scale_floor_ = 1.0
            frame["score"] = frame["residual"]

        # One score per cell. Pooling anchors would let a cell with many anchors
        # dominate the quantile and would break the exchangeability argument,
        # which holds over cells rather than rows.
        if self.score_reduction not in {"max", "mean", "median"}:
            raise ConformalError(
                f"unknown score_reduction: {self.score_reduction!r}"
            )
        per_cell = getattr(
            frame.groupby(self.group_column)["score"], self.score_reduction
        )()
        self.quantile_ = conformal_quantile(per_cell.to_numpy(), self.alpha)
        self.per_cell_scores_ = per_cell
        return self

    def predict_interval(self, frame: pd.DataFrame, *,
                         pred_column: str = "y_pred") -> pd.DataFrame:
        """Return lower/upper bounds for each row of ``frame``.

        Depends only on the prediction, so it is defined for a cell that was
        never calibrated on -- which is every test cell under a cell-disjoint
        protocol.
        """
        prediction = frame[pred_column].to_numpy(dtype=float)
        half_width = self.quantile_ * self._normalizer(prediction)
        return pd.DataFrame({
            "lower": prediction - half_width,
            "upper": prediction + half_width,
            "half_width": half_width,
        }, index=frame.index)


def split_calibration_cells(frame: pd.DataFrame, *,
                            calibration_fraction: float = 0.3,
                            group_column: str = "cell",
                            condition_column: str = "condition",
                            seed: int = 0,
                            exclude_conditions: tuple[str, ...] = ()) -> tuple[np.ndarray, np.ndarray]:
    """Split training rows into (proper-train, calibration) by whole cells.

    Stratified by condition where possible so the calibration set is not drawn
    from one corner of the test matrix. ``exclude_conditions`` removes conditions
    from the *calibration* side: under LOCO the omitted condition must never
    calibrate the intervals that claim not to have seen it.
    """
    table = frame[[group_column, condition_column]].drop_duplicates()
    eligible = table[~table[condition_column].astype(str).isin(exclude_conditions)]
    if eligible.empty:
        raise ConformalError("no cells eligible for calibration after exclusions")
    rng = np.random.default_rng(seed)
    calibration_cells: list[str] = []
    for _, group in eligible.groupby(condition_column, sort=True):
        cells = group[group_column].astype(str).to_numpy()
        rng.shuffle(cells)
        take = int(max(1, round(len(cells) * calibration_fraction)))
        # Never consume an entire condition: a condition with one cell would
        # otherwise vanish from the proper-training set.
        take = min(take, max(len(cells) - 1, 0))
        calibration_cells.extend(cells[:take])
    if not calibration_cells:
        raise ConformalError(
            "every condition has a single cell; cannot hold out calibration cells"
        )
    cells = frame[group_column].astype(str).to_numpy()
    is_calibration = np.isin(cells, calibration_cells)
    return np.flatnonzero(~is_calibration), np.flatnonzero(is_calibration)


def interval_metrics(frame: pd.DataFrame, *, alpha: float = 0.1,
                     true_column: str = "y_true",
                     lower_column: str = "lower",
                     upper_column: str = "upper") -> dict[str, float]:
    """PICP, MPIW and the mean interval score.

    The interval score (Gneiting & Raftery) is the one to lead with: coverage
    alone is gameable by widening intervals without limit, and width alone is
    gameable by shrinking them. The score penalises both.
    """
    y = frame[true_column].to_numpy(dtype=float)
    lower = frame[lower_column].to_numpy(dtype=float)
    upper = frame[upper_column].to_numpy(dtype=float)
    mask = np.isfinite(y) & np.isfinite(lower) & np.isfinite(upper)
    # Same keys on every path. A short dict here makes coverage_by_group emit
    # ragged rows, which pandas fills with NaN columns that read as "measured
    # and missing" rather than "never measured".
    if not mask.any():
        return {"PICP": float("nan"), "MPIW": float("nan"),
                "interval_score": float("nan"), "n": 0,
                "nominal_coverage": float(1.0 - alpha),
                "coverage_gap": float("nan")}
    y, lower, upper = y[mask], lower[mask], upper[mask]
    width = upper - lower
    covered = (y >= lower) & (y <= upper)
    penalty = (
        (2.0 / alpha) * (lower - y) * (y < lower)
        + (2.0 / alpha) * (y - upper) * (y > upper)
    )
    return {
        "PICP": float(np.mean(covered)),
        "MPIW": float(np.mean(width)),
        "interval_score": float(np.mean(width + penalty)),
        "n": int(mask.sum()),
        "nominal_coverage": float(1.0 - alpha),
        "coverage_gap": float(np.mean(covered) - (1.0 - alpha)),
    }


def coverage_by_group(frame: pd.DataFrame, group_columns: tuple[str, ...],
                      *, alpha: float = 0.1, **kwargs) -> pd.DataFrame:
    """Interval metrics broken out per group.

    Marginal coverage can sit exactly on nominal while a whole condition is
    badly under-covered, which is precisely the failure mode that matters for a
    deployment claim. Always report this alongside the pooled number.
    """
    rows = []
    for keys, group in frame.groupby(list(group_columns), dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        metrics = interval_metrics(group, alpha=alpha, **kwargs)
        rows.append({**dict(zip(group_columns, keys)), **metrics})
    return pd.DataFrame(rows)


def ensemble_intervals(predictions: pd.DataFrame, *,
                       seed_column: str = "seed",
                       key_columns: tuple[str, ...] = ("anchor_id",),
                       pred_column: str = "y_pred") -> pd.DataFrame:
    """Collapse per-seed predictions into a deep-ensemble mean and spread.

    Seeds are currently trained and reported separately, so the reported figure
    is one arbitrarily-designated seed. The ensemble mean is both a better
    predictor and an honest one: it removes the temptation to report the seed
    that happened to win.
    """
    if seed_column not in predictions.columns:
        raise ConformalError(f"missing seed column: {seed_column}")
    grouped = predictions.groupby(list(key_columns), dropna=False)[pred_column]
    out = grouped.agg(["mean", "std", "count"]).reset_index()
    return out.rename(columns={
        "mean": "ensemble_mean", "std": "ensemble_std", "count": "n_seeds",
    })
