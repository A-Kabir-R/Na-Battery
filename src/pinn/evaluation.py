"""Aggregation, physics metrics and bootstrap comparisons for NaPINN-Q."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

TARGET_VIEWS = (
    "next_rpt_Q_Ah",
    "next_rpt_SOH_pct",
    "delta_next_rpt_Q_Ah",
    "delta_next_rpt_SOH_pct",
)

# Targets for which MAPE is meaningless because typical magnitudes are
# small enough that any small y_true blows the ratio up.
_MAPE_DISABLED_TARGETS = frozenset({
    "delta_next_rpt_Q_Ah",
    "delta_next_rpt_SOH_pct",
})
_MAPE_MIN_ABS = 1e-6  # y_true below this magnitude is masked out of MAPE


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, *,
                        target: str | None = None) -> dict[str, float]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not mask.any():
        return {"MAE": float("nan"), "RMSE": float("nan"), "R2": float("nan"),
                "MaxError": float("nan"), "MAPE": float("nan"), "n": 0}
    y_true = y_true[mask]; y_pred = y_pred[mask]
    residual = y_pred - y_true
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    max_err = float(np.max(np.abs(residual)))
    ss_res = float(np.sum(residual ** 2))
    mean_true = float(np.mean(y_true))
    ss_tot = float(np.sum((y_true - mean_true) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    if target in _MAPE_DISABLED_TARGETS:
        mape = float("nan")
    else:
        stable = np.abs(y_true) >= _MAPE_MIN_ABS
        if not stable.any():
            mape = float("nan")
        else:
            mape = float(100.0 * np.mean(np.abs(residual[stable] / y_true[stable])))
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MaxError": max_err,
            "MAPE": mape, "n": int(y_true.size)}


def target_metric_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    """Compute per-target metrics for pooled, cell-macro and condition-macro."""
    rows: list[dict[str, object]] = []
    for target in TARGET_VIEWS:
        pred_col = f"predicted_{target.replace('next_rpt_', 'next_').replace('delta_next_rpt_', 'delta_')}"
        # Direct column name lookup
        target_to_columns = {
            "next_rpt_Q_Ah": ("true_next_Q_Ah", "predicted_next_Q_Ah"),
            "next_rpt_SOH_pct": ("true_next_SOH_pct", "predicted_next_SOH_pct"),
            "delta_next_rpt_Q_Ah": ("true_delta_Q_Ah", "predicted_delta_Q_Ah"),
            "delta_next_rpt_SOH_pct": ("true_delta_SOH_pct", "predicted_delta_SOH_pct"),
        }
        true_col, pred_col = target_to_columns[target]
        if true_col not in predictions.columns or pred_col not in predictions.columns:
            continue

        for aggregation in ("pooled", "cell_macro", "condition_macro"):
            if aggregation == "pooled":
                metrics = _regression_metrics(
                    predictions[true_col].to_numpy(dtype=float),
                    predictions[pred_col].to_numpy(dtype=float),
                    target=target,
                )
            else:
                group_col = "cell_id" if aggregation == "cell_macro" else "condition_id"
                per_group = []
                for _, group in predictions.groupby(group_col, dropna=False):
                    per_group.append(_regression_metrics(
                        group[true_col].to_numpy(dtype=float),
                        group[pred_col].to_numpy(dtype=float),
                        target=target,
                    ))
                if not per_group:
                    metrics = {"MAE": float("nan"), "RMSE": float("nan"),
                               "R2": float("nan"), "MaxError": float("nan"),
                               "MAPE": float("nan"), "n": 0}
                else:
                    metrics = {
                        key: float(np.nanmean([row[key] for row in per_group]))
                        if key != "n" else int(sum(row["n"] for row in per_group))
                        for key in ("MAE", "RMSE", "R2", "MaxError", "MAPE", "n")
                    }
            row = {
                "target": target,
                "aggregation": aggregation,
                **metrics,
            }
            for column in ("architecture", "preprocessing", "fold", "seed",
                           "evaluation_role", "ablation"):
                if column in predictions.columns:
                    values = predictions[column].astype(str).unique()
                    row[column] = values[0] if len(values) == 1 else ";".join(values)
            rows.append(row)
    return pd.DataFrame(rows)


def horizon_metric_rows(predictions: pd.DataFrame,
                         horizon_bins: Iterable[float] = (0.0, 1.0, 5.0, 15.0, 1e6),
                         horizon_labels: Iterable[str] = ("<1d", "1-5d", "5-15d", ">15d"),
                         ) -> pd.DataFrame:
    if "horizon_days" not in predictions.columns:
        return pd.DataFrame()
    bins = list(horizon_bins)
    labels = list(horizon_labels)
    binned = pd.cut(predictions["horizon_days"].astype(float),
                    bins=bins, labels=labels, include_lowest=True)
    rows = []
    _metadata_columns = ["architecture", "preprocessing", "fold", "seed",
                         "evaluation_role", "ablation"]
    for bin_label, group in predictions.groupby(binned, dropna=False, observed=False):
        metrics = _regression_metrics(
            group["true_next_Q_Ah"].to_numpy(dtype=float),
            group["predicted_next_Q_Ah"].to_numpy(dtype=float),
            target="next_rpt_Q_Ah",
        )
        metadata: dict[str, object] = {}
        for col in _metadata_columns:
            if col in predictions.columns:
                values = predictions[col].dropna().astype(str).unique()
                metadata[col] = values[0] if len(values) == 1 else ";".join(values)
        rows.append({"horizon_bin": str(bin_label), **metadata, **metrics})
    return pd.DataFrame(rows)


def _nan_safe_abs_mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    return float(np.mean(np.abs(values))) if values.size else float("nan")


def _nan_safe_rms(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    return float(np.sqrt(np.mean(values ** 2))) if values.size else float("nan")


def _nan_safe_frac(series: pd.Series, predicate) -> float:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return float("nan")
    return float(predicate(values[finite]).mean())


def physics_metric_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    """Compute physics diagnostics per run, reporting normalized AND physical
    residual/derivative magnitudes side by side (Stage 2 fix #9).

    DNN-Q rows have ``NaN`` PDE/integral/rate columns and are handled with
    NaN-safe reducers so a family comparison plot doesn't crash (Stage 2
    fix #10).
    """
    if predictions.empty:
        return pd.DataFrame()
    # Prefer the new normalized/physical column names, fall back to the
    # legacy single columns if a downstream user reads an older parquet.
    def _col(*candidates: str) -> pd.Series:
        for name in candidates:
            if name in predictions.columns:
                return predictions[name]
        return pd.Series(dtype=float, index=predictions.index)

    pde_norm = _col("pde_residual_normalized", "pde_residual")
    pde_phys = _col("pde_residual_physical", "pde_residual")
    du_norm = _col("du_dstress_normalized", "du_ds")
    du_phys = _col("du_dstress_physical", "du_ds")
    rate_norm = _col("predicted_degradation_rate_normalized",
                     "predicted_degradation_rate")
    rate_phys = _col("predicted_degradation_rate_physical",
                     "predicted_degradation_rate")

    physics = {
        "pde_residual_MAE_normalized": _nan_safe_abs_mean(pde_norm),
        "pde_residual_RMSE_normalized": _nan_safe_rms(pde_norm),
        "pde_residual_MAE_physical": _nan_safe_abs_mean(pde_phys),
        "pde_residual_RMSE_physical": _nan_safe_rms(pde_phys),
        "positive_derivative_fraction":
            _nan_safe_frac(du_norm, lambda a: a > 0),
        "mean_du_dstress_normalized": _nan_safe_abs_mean(du_norm) if du_norm.size else float("nan"),
        "mean_du_dstress_physical": _nan_safe_abs_mean(du_phys) if du_phys.size else float("nan"),
        "monotonicity_violation_fraction":
            _nan_safe_frac(predictions.get("monotonicity_violation", pd.Series(dtype=float)),
                           lambda a: a > 0),
        "mean_monotonicity_violation":
            float(pd.to_numeric(predictions.get("monotonicity_violation",
                                                 pd.Series(dtype=float)),
                                 errors="coerce").mean(skipna=True)),
        "max_monotonicity_violation":
            float(pd.to_numeric(predictions.get("monotonicity_violation",
                                                 pd.Series(dtype=float)),
                                 errors="coerce").max(skipna=True)),
        "initial_condition_MAE":
            _nan_safe_abs_mean(predictions.get("initial_condition_error", pd.Series(dtype=float))),
        "integral_consistency_MAE":
            _nan_safe_abs_mean(predictions.get("integral_consistency_error", pd.Series(dtype=float))),
        "lower_bound_violation_fraction":
            _nan_safe_frac(predictions.get("lower_bound_violation", pd.Series(dtype=float)),
                           lambda a: a > 0),
        "upper_bound_violation_fraction":
            _nan_safe_frac(predictions.get("upper_bound_violation", pd.Series(dtype=float)),
                           lambda a: a > 0),
        "negative_rate_fraction":
            _nan_safe_frac(rate_norm, lambda a: a < 0),
        "mean_predicted_rate_normalized":
            float(pd.to_numeric(rate_norm, errors="coerce").mean(skipna=True)),
        "std_predicted_rate_normalized":
            float(pd.to_numeric(rate_norm, errors="coerce").std(skipna=True)),
        "mean_predicted_rate_physical":
            float(pd.to_numeric(rate_phys, errors="coerce").mean(skipna=True)),
        "std_predicted_rate_physical":
            float(pd.to_numeric(rate_phys, errors="coerce").std(skipna=True)),
    }
    row = {**physics}
    for column in ("architecture", "preprocessing", "fold", "seed",
                   "evaluation_role", "ablation"):
        if column in predictions.columns:
            values = predictions[column].astype(str).unique()
            row[column] = values[0] if len(values) == 1 else ";".join(values)
    return pd.DataFrame([row])


def cell_paired_bootstrap(a: pd.DataFrame, b: pd.DataFrame, *, metric: str = "MAE",
                          target: str = "next_rpt_Q_Ah",
                          replicates: int = 1000, seed: int = 42) -> dict[str, float]:
    """Cell-level paired bootstrap of ``metric(a) - metric(b)``.

    The correct paired procedure is to compute one per-cell metric difference
    ``d_cell = metric(a|cell) - metric(b|cell)`` and then resample the
    difference vector with replacement. Sampling cells and then row-filtering
    with ``isin`` silently deduplicates: cell drawn three times contributes
    the same rows once, so the "bootstrap" no longer resamples with proper
    multiplicity.
    """
    target_to_columns = {
        "next_rpt_Q_Ah": ("true_next_Q_Ah", "predicted_next_Q_Ah"),
        "next_rpt_SOH_pct": ("true_next_SOH_pct", "predicted_next_SOH_pct"),
        "delta_next_rpt_Q_Ah": ("true_delta_Q_Ah", "predicted_delta_Q_Ah"),
        "delta_next_rpt_SOH_pct": ("true_delta_SOH_pct", "predicted_delta_SOH_pct"),
    }
    true_col, pred_col = target_to_columns[target]
    common_cells = sorted(set(a["cell_id"].astype(str)) & set(b["cell_id"].astype(str)))
    if not common_cells:
        return {"mean": float("nan"), "median": float("nan"),
                "ci_low": float("nan"), "ci_high": float("nan"),
                "win_fraction": float("nan"),
                "n_cells": 0, "replicates": int(replicates)}

    # Precompute one metric difference per cell.
    a_by_cell = {str(cid): grp for cid, grp in a.groupby(a["cell_id"].astype(str))}
    b_by_cell = {str(cid): grp for cid, grp in b.groupby(b["cell_id"].astype(str))}
    cell_diffs: list[float] = []
    for cell in common_cells:
        grp_a = a_by_cell[cell]
        grp_b = b_by_cell[cell]
        m_a = _regression_metrics(
            grp_a[true_col].to_numpy(dtype=float),
            grp_a[pred_col].to_numpy(dtype=float),
            target=target,
        )[metric]
        m_b = _regression_metrics(
            grp_b[true_col].to_numpy(dtype=float),
            grp_b[pred_col].to_numpy(dtype=float),
            target=target,
        )[metric]
        cell_diffs.append(m_a - m_b)
    cell_diffs_arr = np.asarray(cell_diffs, dtype=float)

    rng = np.random.default_rng(seed)
    n_cells = cell_diffs_arr.size
    replicate_means = np.empty(int(replicates), dtype=float)
    for i in range(int(replicates)):
        idx = rng.integers(0, n_cells, size=n_cells)  # sampling with replacement
        replicate_means[i] = float(np.nanmean(cell_diffs_arr[idx]))
    return {
        "mean": float(np.nanmean(replicate_means)),
        "median": float(np.nanmedian(replicate_means)),
        "ci_low": float(np.nanpercentile(replicate_means, 2.5)),
        "ci_high": float(np.nanpercentile(replicate_means, 97.5)),
        "win_fraction": float(np.mean(replicate_means < 0)),
        "n_cells": int(n_cells),
        "replicates": int(replicates),
    }


def aggregate_predictions(prediction_paths: Iterable[Path]) -> pd.DataFrame:
    frames = []
    for path in prediction_paths:
        path = Path(path)
        if not path.exists():
            continue
        frames.append(pd.read_parquet(path))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
