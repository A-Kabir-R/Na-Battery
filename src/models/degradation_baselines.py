"""Physics baselines for next-RPT capacity prediction.

The model under test integrates ``du/ds = -r_theta`` over a completed cycling
block. Its degenerate case -- a *constant* rate -- is therefore the baseline the
paper cannot omit: if a learned rate cannot beat ``r = k``, the neural machinery
is unjustified by construction. That baseline did not previously exist anywhere
in the repository.

All estimators here predict the **delta** target (capacity change over the
block), because the level target is autoregressive: ``previous_rpt_Q_Ah``
correlates 0.910 with ``next_rpt_Q_Ah``, so persistence alone scores R2 = 0.738
on the level view and any level-target R2 must be read against that floor.

Every estimator follows the scikit-learn ``fit``/``predict`` protocol on a
DataFrame so it can be dropped into the same fold runner as the classical
models, and each exposes ``fitted_parameters_`` so the fitted physics can be
reported rather than hidden inside a black box.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import optimize

#: Universal gas constant, J/(mol*K).
R_GAS = 8.314462618

#: Stress coordinate: cumulative equivalent full cycles at the anchor.
STRESS_COLUMN = "EFC_cum"
GROUP_COLUMN = "cell"
CONDITION_COLUMN = "condition"


def stress_interval(frame: pd.DataFrame, *,
                    stress_column: str = STRESS_COLUMN,
                    group_column: str = GROUP_COLUMN) -> np.ndarray:
    """Throughput accumulated during the block between previous and next RPT.

    ``EFC_cum`` is cumulative from birth and is measured at the *end* of the
    anchor's block, i.e. immediately before the target RPT. The interval for
    anchor ``k`` is therefore ``EFC(k) - EFC(k-1)`` within the same cell, and the
    first anchor of a cell spans birth to its own block end.

    Deliberately backward-looking: differencing forward would use the *next*
    block's throughput, which is not yet known at the anchor.
    """
    ordered = frame.sort_values([group_column, stress_column], kind="mergesort")
    delta = ordered.groupby(group_column, sort=False)[stress_column].diff()
    delta = delta.fillna(ordered[stress_column])
    return delta.reindex(frame.index).to_numpy(dtype=float)


class _DeltaBaseline:
    """Shared plumbing: predict a capacity *decrement* over the block."""

    fitted_parameters_: dict[str, float]

    def _interval(self, frame: pd.DataFrame) -> np.ndarray:
        interval = stress_interval(frame)
        # A non-positive interval means the anchor pairing is broken upstream;
        # a zero-length block cannot degrade, so contribute nothing rather than
        # silently producing infinities in the rate.
        return np.where(np.isfinite(interval) & (interval > 0), interval, 0.0)

    def fit(self, frame: pd.DataFrame, y: np.ndarray):  # pragma: no cover - interface
        raise NotImplementedError

    def predict(self, frame: pd.DataFrame) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def get_params(self, deep: bool = True) -> dict:
        return {}

    def set_params(self, **params):
        return self


@dataclass
class ConstantRate(_DeltaBaseline):
    """``delta = -k * ds`` with one global degradation rate.

    The degenerate case of the neural ODE. ``k`` is fitted as the least-squares
    slope through the origin, which is the maximum-likelihood estimate under
    homoscedastic noise and is not the same as the mean of the per-row rates
    (that would weight short blocks equally with long ones).
    """

    fitted_parameters_: dict[str, float] = field(default_factory=dict)

    def fit(self, frame: pd.DataFrame, y: np.ndarray) -> "ConstantRate":
        interval = self._interval(frame)
        y = np.asarray(y, dtype=float)
        usable = np.isfinite(y) & (interval > 0)
        denominator = float(np.sum(interval[usable] ** 2))
        numerator = float(np.sum(-y[usable] * interval[usable]))
        self.k_ = numerator / denominator if denominator > 0 else 0.0
        self.fitted_parameters_ = {"k_per_EFC": self.k_}
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return -self.k_ * self._interval(frame)


@dataclass
class PerConditionRate(_DeltaBaseline):
    """``delta = -k(condition) * ds``: a non-parametric condition lookup.

    Only defined for conditions seen in training; unseen conditions fall back to
    the pooled rate. That fallback is exactly why this cannot be the headline
    baseline for leave-one-condition-out -- report it as an *interpolation*
    reference and use :class:`ArrheniusDodRate` for extrapolation.
    """

    fitted_parameters_: dict[str, float] = field(default_factory=dict)

    def fit(self, frame: pd.DataFrame, y: np.ndarray) -> "PerConditionRate":
        interval = self._interval(frame)
        y = np.asarray(y, dtype=float)
        self.rates_: dict[str, float] = {}
        for condition, index in frame.groupby(CONDITION_COLUMN).groups.items():
            mask = frame.index.isin(index)
            usable = mask & np.isfinite(y) & (interval > 0)
            denominator = float(np.sum(interval[usable] ** 2))
            if denominator > 0:
                self.rates_[str(condition)] = float(
                    np.sum(-y[usable] * interval[usable]) / denominator
                )
        pooled = ConstantRate().fit(frame, y)
        self.fallback_ = pooled.k_
        self.fitted_parameters_ = {**self.rates_, "fallback_k": self.fallback_}
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        k = frame[CONDITION_COLUMN].astype(str).map(self.rates_).fillna(self.fallback_)
        return -k.to_numpy(dtype=float) * self._interval(frame)


@dataclass
class ArrheniusDodRate(_DeltaBaseline):
    r"""Interpretable stress law integrated over the block.

    .. math::
        r = k_{ref}\,
            \exp\!\left[-\frac{E_a}{R}\left(\frac1T-\frac1{T_{ref}}\right)\right]
            \left(\frac{DOD}{DOD_{ref}}\right)^{\beta}
            \left(\frac{|C|}{C_{ref}}\right)^{\alpha}

    Fitted by nonlinear least squares **on the delta target**, using every usable
    row. A log-space linear fit is convex and tempting, but ``log`` is defined
    only where the observed rate is positive, so it silently discards the 29.7%
    of transitions that show a capacity *gain*. Those gains are measurement
    noise, not a different physics -- dropping them conditions the fit on the
    upper tail of the noise and biases :math:`k_{ref}` **upward**, worst exactly
    in the low-DOD conditions where the signal-to-noise analysis says the target
    is noise-dominated. The log fit on positive rows is retained only as the
    starting point for the refinement, which is what it is good for.

    On this dataset the C-rate exponent :math:`\alpha` is a **test of a published
    result**, not a discovery: Klick et al. (Batteries & Supercaps 2025) report
    that cycling rate does not significantly affect the ageing trajectory while
    smaller DOD markedly reduces degradation. An :math:`\alpha` whose interval
    spans zero confirms them.
    """

    reference_temperature_K: float = 298.15
    reference_dod_fraction: float = 1.0
    reference_c_rate: float = 1.0
    fitted_parameters_: dict[str, float] = field(default_factory=dict)

    def _design(self, frame: pd.DataFrame) -> np.ndarray:
        temperature_K = frame["T_degC"].to_numpy(dtype=float) + 273.15
        dod = frame["DOD_pct"].to_numpy(dtype=float) / 100.0
        c_rate = np.abs(frame["C_dis"].to_numpy(dtype=float))
        return np.column_stack([
            np.ones(len(frame)),
            -(1.0 / temperature_K - 1.0 / self.reference_temperature_K),
            np.log(np.maximum(dod, 1e-6) / self.reference_dod_fraction),
            np.log(np.maximum(c_rate, 1e-6) / self.reference_c_rate),
        ])

    def fit(self, frame: pd.DataFrame, y: np.ndarray) -> "ArrheniusDodRate":
        interval = self._interval(frame)
        y = np.asarray(y, dtype=float)
        design = self._design(frame)
        n_parameters = design.shape[1]
        rate = np.divide(-y, interval, out=np.zeros_like(y), where=interval > 0)

        # Every row with a real block length is usable -- including the ones
        # whose observed delta is a capacity gain.
        usable = np.isfinite(y) & (interval > 0) & np.isfinite(design).all(axis=1)
        positive = usable & (rate > 0)
        self.n_rows_gain_ = int((usable & (rate <= 0)).sum())

        if usable.sum() < n_parameters:
            # Not enough rows to identify four coefficients. Report a zero rate
            # rather than a fit nobody should read.
            self.coefficients_ = np.zeros(n_parameters)
            self.coefficients_[0] = (
                np.log(max(float(np.mean(rate[positive])), 1e-12))
                if positive.any() else -np.inf
            )
        else:
            # Starting point: the closed-form log fit on the positive-rate rows.
            if positive.sum() >= n_parameters:
                start, *_ = np.linalg.lstsq(
                    design[positive], np.log(rate[positive]), rcond=None,
                )
            else:
                start = np.zeros(n_parameters)
                start[0] = np.log(
                    max(float(np.mean(rate[positive])), 1e-12)
                    if positive.any() else 1e-6
                )

            matrix, block, target = design[usable], interval[usable], y[usable]

            def residual(theta: np.ndarray) -> np.ndarray:
                # delta_hat = -exp(X theta) * ds; residual = y - delta_hat.
                # The clip keeps exp finite while the optimiser explores.
                return target + np.exp(np.clip(matrix @ theta, -50.0, 50.0)) * block

            try:
                solution = optimize.least_squares(residual, start, method="trf")
                self.coefficients_ = (
                    solution.x if np.all(np.isfinite(solution.x)) else start
                )
            except Exception:  # noqa: BLE001 - a failed refinement keeps the seed
                self.coefficients_ = start

        intercept, activation, beta, alpha = self.coefficients_
        self.fitted_parameters_ = {
            "k_ref_per_EFC": float(np.exp(intercept)),
            "activation_energy_J_per_mol": float(activation * R_GAS),
            "dod_exponent_beta": float(beta),
            "c_rate_exponent_alpha": float(alpha),
            "n_rows_used": int(usable.sum()),
            "n_rows_capacity_gain": self.n_rows_gain_,
        }
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        rate = np.exp(self._design(frame) @ self.coefficients_)
        return -rate * self._interval(frame)


@dataclass
class CellRandomInterceptRate(_DeltaBaseline):
    """Mixed-effects style rate: a shared slope plus a per-cell offset.

    Fitted by shrinking each cell's own rate toward the pooled rate, with the
    shrinkage weight set by how much stress that cell actually accumulated. A
    cell unseen at prediction time gets the pooled rate, which is the correct
    behaviour for a *new* cell -- the random effect is not identifiable for it.

    Consequence worth stating in the paper rather than discovering in review:
    under a strictly cell-disjoint protocol every test cell is unseen, so this
    estimator is **identical to** :class:`ConstantRate` by construction. It earns
    its place only in within-cell or partially-observed-cell protocols. Reporting
    it as a distinct competitor under cell-disjoint CV would be padding.
    """

    #: Prior weight, in the same units as the accumulated squared stress. Larger
    #: values pull individual cells harder toward the pooled slope.
    shrinkage_strength: float = 1.0e4
    fitted_parameters_: dict[str, float] = field(default_factory=dict)

    def fit(self, frame: pd.DataFrame, y: np.ndarray) -> "CellRandomInterceptRate":
        interval = self._interval(frame)
        y = np.asarray(y, dtype=float)
        pooled = ConstantRate().fit(frame, y)
        self.pooled_k_ = pooled.k_
        self.cell_k_: dict[str, float] = {}
        for cell, index in frame.groupby(GROUP_COLUMN).groups.items():
            mask = frame.index.isin(index)
            usable = mask & np.isfinite(y) & (interval > 0)
            weight = float(np.sum(interval[usable] ** 2))
            if weight <= 0:
                continue
            own = float(np.sum(-y[usable] * interval[usable]) / weight)
            # Empirical-Bayes style shrinkage: cells with little accumulated
            # stress are pulled hard toward the pooled slope. The prior weight is
            # a constant, not a multiple of `weight` -- making it proportional
            # would give a fixed 50% shrinkage for every cell regardless of
            # evidence, which defeats the purpose.
            lam = weight / (weight + self.shrinkage_strength)
            self.cell_k_[str(cell)] = lam * own + (1.0 - lam) * self.pooled_k_
        self.fitted_parameters_ = {"pooled_k_per_EFC": self.pooled_k_,
                                   "n_cells": float(len(self.cell_k_))}
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        k = frame[GROUP_COLUMN].astype(str).map(self.cell_k_).fillna(self.pooled_k_)
        return -k.to_numpy(dtype=float) * self._interval(frame)


class ZeroChange(_DeltaBaseline):
    """Persistence on the delta view: predict no capacity change.

    MAE 0.00871 Ah on this dataset. It wins outright in the noise-dominated
    conditions (DOD20/DOD80 at 25 C), where the within-cell lag-1
    autocorrelation of delta approaches -0.5 -- the signature of pure RPT
    measurement noise, which is by definition unpredictable.
    """

    def fit(self, frame: pd.DataFrame, y: np.ndarray) -> "ZeroChange":
        self.fitted_parameters_ = {}
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(frame), dtype=float)


def stress_design_diagnostics(frame: pd.DataFrame) -> dict[str, float]:
    """Report whether the test matrix can identify the stress-law exponents.

    On this dataset it cannot, and the paper must say so. DOD20 and DOD80 are
    tested **only** at 2C, so C-rate varies independently of DOD only inside
    DOD100; and ``C_ch`` equals ``C_dis`` on every row, so charge and discharge
    rate are the same variable. The consequence is that a freely fitted C-rate
    exponent absorbs DOD structure and cannot be interpreted on its own -- which
    matters because Klick et al. report C-rate has no significant effect, and a
    spurious non-zero exponent would appear to contradict them.

    Returns the Belsley condition index (values above ~30 indicate serious
    collinearity), the DOD/C-rate correlation, and the number of distinct C-rates
    observed within each DOD level.

    The columns are scaled to unit norm before the condition number is taken.
    That is not cosmetic: ``1/T`` is O(3e-3) while the intercept is O(1), so the
    raw condition number is inflated by roughly the ratio of the column norms and
    reports catastrophic collinearity for a perfectly identifiable design. The
    ~30 threshold is defined on the scaled matrix.
    """
    temperature_K = frame["T_degC"].to_numpy(dtype=float) + 273.15
    log_dod = np.log(np.maximum(frame["DOD_pct"].to_numpy(dtype=float) / 100.0, 1e-6))
    log_c = np.log(np.maximum(np.abs(frame["C_dis"].to_numpy(dtype=float)), 1e-6))
    design = np.column_stack([
        np.ones(len(frame)), -1.0 / temperature_K, log_dod, log_c,
    ])
    norms = np.linalg.norm(design, axis=0)
    scaled = design / np.where(norms > 0, norms, 1.0)
    within_dod = frame.groupby("DOD_pct")["C_dis"].nunique()
    correlation = (
        float(np.corrcoef(log_dod, log_c)[0, 1])
        if np.std(log_dod) > 0 and np.std(log_c) > 0 else float("nan")
    )
    return {
        "condition_number": float(np.linalg.cond(scaled)),
        "condition_number_unscaled": float(np.linalg.cond(design)),
        "corr_log_dod_log_c_rate": correlation,
        "charge_equals_discharge_rate": bool(
            (frame["C_ch"] == frame["C_dis"]).all()
        ),
        "dod_levels_with_c_rate_variation": int((within_dod > 1).sum()),
        "n_dod_levels": int(len(within_dod)),
        "c_rate_exponent_identifiable": bool((within_dod > 1).sum() >= 2),
    }


def build_degradation_baselines() -> dict[str, _DeltaBaseline]:
    """Registry of physics baselines, keyed by report name."""
    return {
        "ZeroChange": ZeroChange(),
        "ConstantRate": ConstantRate(),
        "PerConditionRate": PerConditionRate(),
        "ArrheniusDodRate": ArrheniusDodRate(),
        "CellRandomInterceptRate": CellRandomInterceptRate(),
    }
