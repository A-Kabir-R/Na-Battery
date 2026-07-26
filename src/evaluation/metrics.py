"""Regression and classification metric bundles."""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             max_error, mean_absolute_error,
                             mean_absolute_percentage_error, precision_score,
                             r2_score, recall_score, roc_auc_score,
                             root_mean_squared_error)

REGRESSION_METRIC_NAMES = ("MAE", "RMSE", "MAPE", "R2", "MaxError")


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return {k: np.nan for k in REGRESSION_METRIC_NAMES}
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    return {
        "MAE":      float(mean_absolute_error(y_true, y_pred)),
        "RMSE":     float(root_mean_squared_error(y_true, y_pred)),
        "MAPE":     float(mean_absolute_percentage_error(
                        y_true, y_pred)) if (y_true != 0).all() else float("nan"),
        "R2":       float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else float("nan"),
        "MaxError": float(max_error(y_true, y_pred)),
    }


def aggregated_regression_metrics(predictions, aggregation: str) -> Dict[str, float]:
    """Calculate pooled or equal-weighted cell/condition regression metrics."""
    if aggregation == "pooled":
        return regression_metrics(
            predictions["y_true"].to_numpy(dtype=float),
            predictions["y_pred"].to_numpy(dtype=float),
        )
    group_column = {"cell_macro": "cell", "condition_macro": "condition"}.get(aggregation)
    if group_column is None:
        raise ValueError(f"unknown aggregation: {aggregation}")
    values = []
    for _, group in predictions.groupby(group_column, dropna=False):
        values.append(regression_metrics(
            group["y_true"].to_numpy(dtype=float),
            group["y_pred"].to_numpy(dtype=float),
        ))
    if not values:
        return {metric: np.nan for metric in REGRESSION_METRIC_NAMES}
    return {
        metric: float(np.nanmean([row[metric] for row in values]))
        if np.isfinite([row[metric] for row in values]).any() else np.nan
        for metric in REGRESSION_METRIC_NAMES
    }


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                            y_score: np.ndarray | None = None) -> Dict[str, float]:
    m = {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "F1":        float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_score is not None and len(np.unique(y_true)) > 1:
        try:
            m["ROC_AUC"] = float(roc_auc_score(y_true, y_score))
            m["PR_AUC"]  = float(average_precision_score(y_true, y_score))
        except Exception:
            m["ROC_AUC"] = m["PR_AUC"] = np.nan
    else:
        m["ROC_AUC"] = m["PR_AUC"] = np.nan
    return m
