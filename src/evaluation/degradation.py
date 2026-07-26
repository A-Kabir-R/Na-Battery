"""Descriptive RPT-SOH trajectories and explicitly projected time-to-threshold."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _linear_fit(x: pd.Series, y: pd.Series) -> tuple[float, float, float]:
    x_all = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    y_all = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    valid = np.isfinite(x_all) & np.isfinite(y_all)
    x_values = x_all[valid]
    y_values = y_all[valid]
    if len(x_values) < 3 or np.ptp(x_values) <= 0:
        return np.nan, np.nan, np.nan
    slope, intercept = np.polyfit(x_values, y_values, 1)
    prediction = slope * x_values + intercept
    denominator = np.sum((y_values - y_values.mean()) ** 2)
    r2 = 1.0 - np.sum((y_values - prediction) ** 2) / denominator if denominator > 0 else np.nan
    return float(slope), float(intercept), float(r2)


def build_rpt_longitudinal(rpt: pd.DataFrame, cycles: pd.DataFrame,
                           *, initial_visit: int = 0,
                           nominal_capacity_Ah: float = 1.2) -> pd.DataFrame:
    frame = rpt[rpt["rpt_qc_status"].eq("valid")].copy()
    frame["file_start_time"] = pd.to_datetime(frame["file_start_time"], errors="coerce")
    frame = frame.sort_values(["condition", "cell", "file_start_time", "test_id"]).reset_index(drop=True)
    baseline = (
        frame[frame["visit"].eq(initial_visit)]
        .groupby(["condition", "cell"])
        .agg(initial_rpt_Q_Ah=("reference_capacity_Ah", "first"),
             initial_rpt_time=("file_start_time", "first"))
    )
    frame = frame.join(baseline, on=["condition", "cell"])
    frame["rpt_SOH_pct"] = 100.0 * frame["reference_capacity_Ah"] / frame["initial_rpt_Q_Ah"]
    frame["elapsed_days"] = (
        frame["file_start_time"] - frame["initial_rpt_time"]
    ).dt.total_seconds() / 86400.0
    frame["rpt_sequence"] = frame.groupby(["condition", "cell"]).cumcount()
    frame["delta_rpt_Q_Ah"] = frame.groupby(["condition", "cell"])[
        "reference_capacity_Ah"
    ].diff()
    frame["delta_rpt_SOH_pct"] = frame.groupby(["condition", "cell"])["rpt_SOH_pct"].diff()

    complete = cycles[cycles["cycle_complete"].fillna(False)].copy()
    complete["file_start_time"] = pd.to_datetime(complete["file_start_time"], errors="coerce")
    complete["cycle_absolute_time"] = complete["file_start_time"] + pd.to_timedelta(
        complete["cycle_end_time_s"], unit="s"
    )
    frame["preceding_global_cycle"] = pd.NA
    frame["preceding_Ah_throughput"] = np.nan
    for index, row in frame.iterrows():
        prior = complete[
            complete["condition"].eq(row["condition"])
            & complete["cell"].eq(row["cell"])
            & complete["cycle_absolute_time"].lt(row["file_start_time"])
        ].sort_values("cycle_absolute_time")
        if prior.empty:
            continue
        latest = prior.iloc[-1]
        frame.at[index, "preceding_global_cycle"] = latest["global_cycle"]
        frame.at[index, "preceding_Ah_throughput"] = latest["cumulative_Ah_throughput"]
    frame["preceding_global_cycle"] = frame["preceding_global_cycle"].astype("Int64")
    frame["preceding_EFC"] = frame["preceding_Ah_throughput"] / nominal_capacity_Ah
    return frame


def cell_degradation_summary(longitudinal: pd.DataFrame,
                             *, threshold_pct: float = 80.0,
                             minimum_projection_r2: float = 0.5,
                             maximum_extrapolation_ratio: float = 3.0) -> pd.DataFrame:
    rows = []
    for (condition, cell), group in longitudinal.groupby(["condition", "cell"]):
        group = group.sort_values("file_start_time")
        slope_day, intercept_day, r2_day = _linear_fit(group["elapsed_days"], group["rpt_SOH_pct"])
        slope_cycle, intercept_cycle, r2_cycle = _linear_fit(
            group["preceding_global_cycle"].astype(float), group["rpt_SOH_pct"]
        )
        slope_efc, intercept_efc, r2_efc = _linear_fit(group["preceding_EFC"], group["rpt_SOH_pct"])

        def projected_crossing(slope: float, intercept: float, fit_r2: float,
                               observed_max: float) -> tuple[float, float, str]:
            if not np.isfinite(slope) or slope >= 0 or not np.isfinite(intercept):
                return np.nan, np.nan, "not_projectable_from_observed_trend"
            crossing = (threshold_pct - intercept) / slope
            if not np.isfinite(observed_max) or crossing <= observed_max:
                return np.nan, np.nan, "crossing_not_beyond_observed_support"
            ratio = crossing / observed_max if observed_max > 0 else np.nan
            quality = np.isfinite(fit_r2) and fit_r2 >= minimum_projection_r2
            quality &= np.isfinite(ratio) and ratio <= maximum_extrapolation_ratio
            status = (
                "exploratory_linear_extrapolation_quality_eligible" if quality
                else "exploratory_linear_extrapolation_low_confidence"
            )
            return float(crossing), float(ratio), status

        projected_day, day_ratio, day_status = projected_crossing(
            slope_day, intercept_day, r2_day, group["elapsed_days"].max()
        )
        projected_cycle, cycle_ratio, cycle_status = projected_crossing(
            slope_cycle, intercept_cycle, r2_cycle,
            group["preceding_global_cycle"].astype(float).max()
        )
        projected_efc, efc_ratio, efc_status = projected_crossing(
            slope_efc, intercept_efc, r2_efc, group["preceding_EFC"].max()
        )
        rows.append({
            "condition": condition, "cell": cell, "n_valid_rpt": len(group),
            "initial_rpt_Q_Ah": group["initial_rpt_Q_Ah"].iloc[0],
            "final_rpt_Q_Ah": group["reference_capacity_Ah"].iloc[-1],
            "final_rpt_SOH_pct": group["rpt_SOH_pct"].iloc[-1],
            "minimum_rpt_SOH_pct": group["rpt_SOH_pct"].min(),
            "observed_80pct_event": bool(group["rpt_SOH_pct"].lt(threshold_pct).any()),
            "soh_slope_pct_per_100_days": 100.0 * slope_day,
            "soh_vs_days_R2": r2_day,
            "soh_slope_pct_per_100_cycles": 100.0 * slope_cycle,
            "soh_vs_cycles_R2": r2_cycle,
            "soh_slope_pct_per_100_EFC": 100.0 * slope_efc,
            "soh_vs_EFC_R2": r2_efc,
            "projected_day_at_80pct": projected_day,
            "projected_day_extrapolation_ratio": day_ratio,
            "projected_day_status": day_status,
            "projected_cycle_at_80pct": projected_cycle,
            "projected_cycle_extrapolation_ratio": cycle_ratio,
            "projected_cycle_status": cycle_status,
            "projected_EFC_at_80pct": projected_efc,
            "projected_EFC_extrapolation_ratio": efc_ratio,
            "projected_EFC_status": efc_status,
            "projection_status": day_status,
            "projection_minimum_fit_R2": minimum_projection_r2,
            "projection_maximum_extrapolation_ratio": maximum_extrapolation_ratio,
        })
    return pd.DataFrame(rows)


def condition_degradation_summary(cells: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        "initial_rpt_Q_Ah", "final_rpt_Q_Ah", "final_rpt_SOH_pct",
        "minimum_rpt_SOH_pct", "soh_slope_pct_per_100_days",
        "soh_slope_pct_per_100_cycles", "soh_slope_pct_per_100_EFC",
        "projected_day_at_80pct", "projected_cycle_at_80pct", "projected_EFC_at_80pct",
    ]
    pieces = []
    for condition, group in cells.groupby("condition"):
        row = {"condition": condition, "n_cells": group["cell"].nunique()}
        for column in numeric:
            values = pd.to_numeric(group[column], errors="coerce")
            if column.startswith("projected_day_"):
                values = values[group["projected_day_status"].eq(
                    "exploratory_linear_extrapolation_quality_eligible"
                )]
            elif column.startswith("projected_cycle_"):
                values = values[group["projected_cycle_status"].eq(
                    "exploratory_linear_extrapolation_quality_eligible"
                )]
            elif column.startswith("projected_EFC_"):
                values = values[group["projected_EFC_status"].eq(
                    "exploratory_linear_extrapolation_quality_eligible"
                )]
            values = values.dropna()
            row[f"{column}_mean"] = values.mean() if len(values) else np.nan
            row[f"{column}_median"] = values.median() if len(values) else np.nan
            row[f"{column}_std"] = values.std() if len(values) else np.nan
        row["observed_80pct_event_count"] = int(group["observed_80pct_event"].sum())
        for axis in ("day", "cycle", "EFC"):
            row[f"n_projectable_{axis}_quality_eligible"] = int(group[
                f"projected_{axis}_status"
            ].eq("exploratory_linear_extrapolation_quality_eligible").sum())
        pieces.append(row)
    return pd.DataFrame(pieces)
