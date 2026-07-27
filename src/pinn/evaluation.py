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


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
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
    if (y_true == 0).any():
        mape = float("nan")
    else:
        mape = float(100.0 * np.mean(np.abs(residual / y_true)))
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
                )
            else:
                group_col = "cell_id" if aggregation == "cell_macro" else "condition_id"
                per_group = []
                for _, group in predictions.groupby(group_col, dropna=False):
                    per_group.append(_regression_metrics(
                        group[true_col].to_numpy(dtype=float),
                        group[pred_col].to_numpy(dtype=float),
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
    for bin_label, group in predictions.groupby(binned, dropna=False, observed=False):
        metrics = _regression_metrics(
            group["true_next_Q_Ah"].to_numpy(dtype=float),
            group["predicted_next_Q_Ah"].to_numpy(dtype=float),
        )
        rows.append({"horizon_bin": str(bin_label), **metrics,
                     "architecture": predictions["architecture"].iloc[0] if "architecture" in predictions.columns else "",
                     "preprocessing": predictions["preprocessing"].iloc[0] if "preprocessing" in predictions.columns else ""})
    return pd.DataFrame(rows)


def physics_metric_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    physics = {
        "pde_residual_MAE": float(np.mean(np.abs(predictions["pde_residual"]))),
        "pde_residual_RMSE": float(np.sqrt(np.mean(predictions["pde_residual"].astype(float) ** 2))),
        "positive_derivative_fraction": float((predictions["du_ds"] > 0).mean()),
        "monotonicity_violation_fraction": float((predictions["monotonicity_violation"] > 0).mean()),
        "mean_monotonicity_violation": float(predictions["monotonicity_violation"].mean()),
        "max_monotonicity_violation": float(predictions["monotonicity_violation"].max()),
        "initial_condition_MAE": float(np.mean(np.abs(predictions["initial_condition_error"]))),
        "integral_consistency_MAE": float(np.mean(np.abs(predictions["integral_consistency_error"]))),
        "lower_bound_violation_fraction": float((predictions["lower_bound_violation"] > 0).mean()),
        "upper_bound_violation_fraction": float((predictions["upper_bound_violation"] > 0).mean()),
        "negative_rate_fraction": float((predictions["predicted_degradation_rate"] < 0).mean()),
        "mean_predicted_rate": float(predictions["predicted_degradation_rate"].mean()),
        "std_predicted_rate": float(predictions["predicted_degradation_rate"].std()),
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

    Both frames must share the ``cell_id`` axis and refer to the same target.
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
                "win_fraction": float("nan")}
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(int(replicates)):
        sample = rng.choice(common_cells, size=len(common_cells), replace=True)
        subset_a = a[a["cell_id"].astype(str).isin(sample)]
        subset_b = b[b["cell_id"].astype(str).isin(sample)]
        metric_a = _regression_metrics(
            subset_a[true_col].to_numpy(dtype=float),
            subset_a[pred_col].to_numpy(dtype=float),
        )[metric]
        metric_b = _regression_metrics(
            subset_b[true_col].to_numpy(dtype=float),
            subset_b[pred_col].to_numpy(dtype=float),
        )[metric]
        diffs.append(metric_a - metric_b)
    diffs_arr = np.asarray(diffs, dtype=float)
    return {
        "mean": float(np.nanmean(diffs_arr)),
        "median": float(np.nanmedian(diffs_arr)),
        "ci_low": float(np.nanpercentile(diffs_arr, 2.5)),
        "ci_high": float(np.nanpercentile(diffs_arr, 97.5)),
        "win_fraction": float(np.mean(diffs_arr < 0)),
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
