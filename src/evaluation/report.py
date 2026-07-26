"""Build the comparison table from raw long-format results.

Handles the `split` column (train/test) that was added when we started logging
both training and validation metrics. If a raw_results.csv predates that change,
we default `split="test"` so old files still aggregate.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .metrics import REGRESSION_METRIC_NAMES, aggregated_regression_metrics, regression_metrics


def _ensure_split(df: pd.DataFrame) -> pd.DataFrame:
    if "split" not in df.columns:
        df = df.copy()
        df["split"] = "test"
    return df


def aggregate(raw_csv: Path | str) -> pd.DataFrame:
    df = _ensure_split(pd.read_csv(raw_csv))
    agg = (df.groupby(["preprocessing", "target", "model", "split", "metric"])["value"]
             .agg(["mean", "std", "count"])
             .reset_index())
    return agg


def pivot_comparison(agg: pd.DataFrame) -> pd.DataFrame:
    """Wide comparison: rows=(preprocessing, model), columns={target}_{split}_{metric}."""
    agg = _ensure_split(agg.copy())
    agg["target_metric"] = (agg["target"] + "_" + agg["split"].astype(str)
                             + "_" + agg["metric"])
    wide = agg.pivot_table(index=["preprocessing", "model"],
                            columns="target_metric",
                            values="mean").reset_index()
    return wide


def overfitting_gap(agg: pd.DataFrame) -> pd.DataFrame:
    """Per (preprocessing, target, model, metric): mean_test - mean_train.

    Interpretation for common metrics:
      - MAE / RMSE / MAPE: positive gap = worse on test = classic overfit.
      - R2 / F1 / accuracy / ROC_AUC: negative gap = worse on test = overfit.
    """
    a = _ensure_split(agg.copy())
    rows = []
    for fit_split, evaluation_split, label in (
        ("cv_train", "cv_validation", "cv"),
        ("development_fit", "holdout", "holdout"),
        ("train", "test", "legacy"),
    ):
        part = a[a["split"].isin([fit_split, evaluation_split])]
        if part.empty:
            continue
        piv = part.pivot_table(
            index=["preprocessing", "target", "model", "metric"],
            columns="split", values="mean",
        ).reset_index()
        if fit_split not in piv or evaluation_split not in piv:
            continue
        piv["comparison"] = label
        piv["fit_split"] = fit_split
        piv["evaluation_split"] = evaluation_split
        piv["fit"] = piv[fit_split]
        piv["evaluation"] = piv[evaluation_split]
        piv["gap_evaluation_minus_fit"] = piv["evaluation"] - piv["fit"]
        rows.append(piv[[
            "preprocessing", "target", "model", "metric", "comparison",
            "fit_split", "evaluation_split", "fit", "evaluation",
            "gap_evaluation_minus_fit",
        ]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def collect_predictions(artifacts: Path | str,
                        raw_results: Path | str | pd.DataFrame | None = None) -> pd.DataFrame:
    """Collect one OOF prediction per development row and one holdout prediction."""
    artifacts = Path(artifacts)
    expected = None
    if raw_results is not None:
        expected = raw_results.copy() if isinstance(raw_results, pd.DataFrame) else pd.read_csv(raw_results)

    def completed_current(path: Path) -> bool:
        status_path = path.parent / "fold_status.json"
        if not status_path.exists():
            return False
        try:
            import json
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except Exception:
            return False
        if status.get("status") != "completed" or not status.get("fingerprint"):
            return False
        if expected is None:
            return True
        relative = path.relative_to(artifacts / "folds")
        preprocessing, target, model, partition = relative.parts[:4]
        fold = -1 if partition == "holdout" else int(partition.removeprefix("fold_"))
        rows = expected[
            expected["preprocessing"].astype(str).eq(preprocessing)
            & expected["target"].astype(str).eq(target)
            & expected["model"].astype(str).eq(model)
            & expected["fold"].eq(fold)
            & expected["status"].eq("completed")
        ]
        return not rows.empty and status["fingerprint"] in set(rows["fingerprint"].dropna())

    frames: list[pd.DataFrame] = []
    for path in sorted((artifacts / "folds").glob("*/*/*/fold_*/predictions.parquet")):
        if not completed_current(path):
            continue
        frame = pd.read_parquet(path)
        frame = frame[frame["split"] == "cv_validation"].copy()
        if not frame.empty:
            frame["evaluation_role"] = "cv_oof"
            frames.append(frame)
    for path in sorted((artifacts / "folds").glob("*/*/*/holdout/predictions.parquet")):
        if not completed_current(path):
            continue
        frame = pd.read_parquet(path)
        frame = frame[frame["split"].isin(["development_fit", "holdout"])].copy()
        if not frame.empty:
            frame["evaluation_role"] = frame["split"]
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _nanmean_metric_dicts(dicts: list[dict[str, float]]) -> dict[str, float]:
    if not dicts:
        return {metric: np.nan for metric in REGRESSION_METRIC_NAMES}
    out: dict[str, float] = {}
    for metric in REGRESSION_METRIC_NAMES:
        values = np.fromiter((d[metric] for d in dicts), dtype=float, count=len(dicts))
        finite = values[np.isfinite(values)]
        out[metric] = float(finite.mean()) if finite.size else np.nan
    return out


def _bootstrap_intervals(frame: pd.DataFrame, *, replicates: int,
                         random_state: int) -> dict[tuple[str, str], tuple[float, float]]:
    cell_table = frame[["condition", "cell"]].dropna().drop_duplicates()
    if len(cell_table) < 2 or replicates <= 0:
        return {}
    cell_arrays: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
    cell_metric_cache: dict[tuple, dict[str, float]] = {}
    for (condition, cell), group in frame.groupby(["condition", "cell"], dropna=False):
        y_true = group["y_true"].to_numpy(dtype=float)
        y_pred = group["y_pred"].to_numpy(dtype=float)
        cell_arrays[(condition, cell)] = (y_true, y_pred)
        cell_metric_cache[(condition, cell)] = regression_metrics(y_true, y_pred)
    condition_cells = {
        condition: np.asarray(group["cell"].tolist(), dtype=object)
        for condition, group in cell_table.groupby("condition", dropna=False)
    }
    rng = np.random.default_rng(random_state)
    samples: dict[tuple[str, str], list[float]] = {
        (aggregation, metric): []
        for aggregation in ("pooled", "cell_macro", "condition_macro")
        for metric in REGRESSION_METRIC_NAMES
    }
    for _ in range(replicates):
        selected: list[tuple] = []
        by_condition: dict[object, list[dict[str, float]]] = {}
        y_true_parts: list[np.ndarray] = []
        y_pred_parts: list[np.ndarray] = []
        for condition, cells in condition_cells.items():
            picks = rng.choice(cells, size=len(cells), replace=True)
            for cell in picks:
                key = (condition, cell)
                selected.append(key)
                y_t, y_p = cell_arrays[key]
                y_true_parts.append(y_t)
                y_pred_parts.append(y_p)
                by_condition.setdefault(condition, []).append(cell_metric_cache[key])
        pooled = regression_metrics(
            np.concatenate(y_true_parts), np.concatenate(y_pred_parts)
        )
        cell_macro = _nanmean_metric_dicts(
            [cell_metric_cache[key] for key in selected]
        )
        condition_macro = _nanmean_metric_dicts(
            [_nanmean_metric_dicts(cells) for cells in by_condition.values()]
        )
        for metric in REGRESSION_METRIC_NAMES:
            samples[("pooled", metric)].append(pooled[metric])
            samples[("cell_macro", metric)].append(cell_macro[metric])
            samples[("condition_macro", metric)].append(condition_macro[metric])
    intervals = {}
    for key, values in samples.items():
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if len(finite):
            intervals[key] = tuple(np.percentile(finite, [2.5, 97.5]))
    return intervals


def prediction_metric_tables(predictions: pd.DataFrame, *, bootstrap_replicates: int = 1000,
                             random_state: int = 42,
                             horizon_edges: Iterable[float] = (0, 1, 5, 15, 1e6),
                             horizon_labels: Iterable[str] = ("<1d", "1-5d", "5-15d", ">15d"),
                             ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build publication-level overall, subgroup, horizon, and QC tables."""
    summary_columns = [
        "preprocessing", "target", "model", "evaluation_role", "aggregation", "metric",
        "estimate", "ci_low", "ci_high", "n_rows", "n_valid_rows", "n_cells",
        "n_conditions", "n_groups_total", "n_groups_metric_valid",
    ]
    subgroup_columns = [
        "preprocessing", "target", "model", "evaluation_role", "group_type",
        "group_value", "metric", "estimate", "n_rows", "n_cells",
    ]
    horizon_columns = [
        "preprocessing", "target", "model", "evaluation_role", "horizon_bin",
        "aggregation", "metric", "estimate", "n_rows", "n_cells",
    ]
    plausibility_columns = [
        "preprocessing", "target", "model", "evaluation_role", "n_rows",
        "invalid_prediction_count", "invalid_prediction_fraction", "prediction_min",
        "prediction_max",
    ]
    if predictions.empty:
        return (
            pd.DataFrame(columns=summary_columns), pd.DataFrame(columns=subgroup_columns),
            pd.DataFrame(columns=horizon_columns), pd.DataFrame(columns=plausibility_columns),
        )
    keys = ["preprocessing", "target", "model", "evaluation_role"]
    summary_rows: list[dict] = []
    subgroup_rows: list[dict] = []
    horizon_rows: list[dict] = []
    plausibility_rows: list[dict] = []
    for key, frame in predictions.groupby(keys, dropna=False):
        base = dict(zip(keys, key))
        intervals = _bootstrap_intervals(
            frame, replicates=bootstrap_replicates, random_state=random_state
        )
        per_cell_metrics: dict[object, dict[str, float]] = {}
        per_cell_rows: dict[object, int] = {}
        for cell_value, cell_frame in frame.groupby("cell", dropna=False):
            per_cell_metrics[cell_value] = regression_metrics(
                cell_frame["y_true"].to_numpy(dtype=float),
                cell_frame["y_pred"].to_numpy(dtype=float),
            )
            per_cell_rows[cell_value] = len(cell_frame)
        per_condition_metrics: dict[object, dict[str, float]] = {}
        per_condition_rows: dict[object, int] = {}
        per_condition_cells: dict[object, int] = {}
        for condition_value, condition_frame in frame.groupby("condition", dropna=False):
            per_condition_metrics[condition_value] = regression_metrics(
                condition_frame["y_true"].to_numpy(dtype=float),
                condition_frame["y_pred"].to_numpy(dtype=float),
            )
            per_condition_rows[condition_value] = len(condition_frame)
            per_condition_cells[condition_value] = condition_frame["cell"].nunique()
        pooled_metrics = regression_metrics(
            frame["y_true"].to_numpy(dtype=float),
            frame["y_pred"].to_numpy(dtype=float),
        )
        cell_macro_metrics = _nanmean_metric_dicts(list(per_cell_metrics.values()))
        condition_macro_metrics = _nanmean_metric_dicts(list(per_condition_metrics.values()))
        n_rows_total = len(frame)
        n_valid_rows_total = int((
            np.isfinite(frame["y_true"].to_numpy(dtype=float))
            & np.isfinite(frame["y_pred"].to_numpy(dtype=float))
        ).sum())
        n_cells_total = frame["cell"].nunique()
        n_conditions_total = frame["condition"].nunique()
        for aggregation, aggregate_metrics, group_source in (
            ("pooled", pooled_metrics, None),
            ("cell_macro", cell_macro_metrics, per_cell_metrics),
            ("condition_macro", condition_macro_metrics, per_condition_metrics),
        ):
            for metric, estimate in aggregate_metrics.items():
                if str(base["target"]).startswith("delta_") and metric == "MAPE":
                    continue
                ci = intervals.get((aggregation, metric), (np.nan, np.nan))
                if group_source is None:
                    group_metrics_values = [estimate]
                else:
                    group_metrics_values = [
                        metric_dict[metric] for metric_dict in group_source.values()
                    ]
                summary_rows.append({
                    **base, "aggregation": aggregation, "metric": metric,
                    "estimate": estimate, "ci_low": ci[0], "ci_high": ci[1],
                    "n_rows": n_rows_total,
                    "n_valid_rows": n_valid_rows_total,
                    "n_cells": n_cells_total,
                    "n_conditions": n_conditions_total,
                    "n_groups_total": len(group_metrics_values),
                    "n_groups_metric_valid": int(np.isfinite(group_metrics_values).sum()),
                })
        for group_type, cache, rows_lookup in (
            ("cell", per_cell_metrics, per_cell_rows),
            ("condition", per_condition_metrics, per_condition_rows),
        ):
            for group_value, metric_dict in cache.items():
                n_cells_group = (
                    1 if group_type == "cell" else per_condition_cells[group_value]
                )
                for metric, estimate in metric_dict.items():
                    if str(base["target"]).startswith("delta_") and metric == "MAPE":
                        continue
                    subgroup_rows.append({
                        **base, "group_type": group_type, "group_value": group_value,
                        "metric": metric, "estimate": estimate,
                        "n_rows": rows_lookup[group_value],
                        "n_cells": n_cells_group,
                    })
        if "next_rpt_horizon_days" in frame:
            horizon = frame.copy()
            horizon["horizon_bin"] = pd.cut(
                horizon["next_rpt_horizon_days"], bins=list(horizon_edges),
                labels=list(horizon_labels), include_lowest=True, right=False,
            )
            for horizon_bin, group in horizon.dropna(subset=["horizon_bin"]).groupby(
                "horizon_bin", observed=True
            ):
                for aggregation in ("pooled", "cell_macro", "condition_macro"):
                    metrics = aggregated_regression_metrics(group, aggregation)
                    for metric, estimate in metrics.items():
                        if str(base["target"]).startswith("delta_") and metric == "MAPE":
                            continue
                        horizon_rows.append({
                            **base, "horizon_bin": str(horizon_bin),
                            "aggregation": aggregation, "metric": metric,
                            "estimate": estimate, "n_rows": len(group),
                            "n_cells": group["cell"].nunique(),
                        })
        valid = frame.get("prediction_valid", pd.Series(True, index=frame.index)).fillna(False)
        plausibility_rows.append({
            **base, "n_rows": len(frame), "invalid_prediction_count": int((~valid).sum()),
            "invalid_prediction_fraction": float((~valid).mean()),
            "prediction_min": float(frame["y_pred"].min()),
            "prediction_max": float(frame["y_pred"].max()),
        })
    return (
        pd.DataFrame(summary_rows, columns=summary_columns),
        pd.DataFrame(subgroup_rows, columns=subgroup_columns),
        pd.DataFrame(horizon_rows, columns=horizon_columns),
        pd.DataFrame(plausibility_rows, columns=plausibility_columns),
    )
