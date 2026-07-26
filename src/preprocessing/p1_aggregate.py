"""P1 per-cycle aggregate features from validated canonical cycles."""
from __future__ import annotations

import pandas as pd

from ..io.loaders import load_cycle_metrics, load_inventory

BASE_NUMERIC = [
    "Q_charge_Ah", "Q_discharge_Ah", "E_charge_Wh", "E_discharge_Wh",
    "coulombic_efficiency", "energy_efficiency", "V_max", "V_min", "T_max", "T_min",
    "T_mean", "Ah_span", "duration_s", "cumulative_Ah_throughput",
    "cycle_start_time_s", "cycle_end_time_s",
]
CONDITION_NUMERIC = ["DOD_pct", "C_ch", "C_dis", "T_degC"]
ID_COLS = [
    "condition", "cell", "visit", "test_type", "path", "relative_path", "file_id",
    "file_start_time", "segment_index", "cycle_in_file", "global_cycle", "cycle_complete",
    "cycle_qc_status", "cycle_qc_flags",
]
TARGET_COLS = ["SOH_pct", "Q_initial_Ah"]


def build_p1(df: pd.DataFrame | None = None) -> pd.DataFrame:
    if df is None:
        df = load_cycle_metrics()
    frame = df.copy()
    if "cycle_complete" not in frame.columns:
        raise ValueError(
            "cycle_complete is missing; rebuild canonical tables with scripts/00_build_canonical.py"
        )
    # Incomplete boundary segments remain in cycles.parquet for QC but cannot
    # enter model features.
    frame = frame[frame["cycle_complete"].fillna(False)].copy()

    missing_condition = [column for column in CONDITION_NUMERIC if column not in frame.columns]
    if missing_condition:
        metadata = load_inventory()[["condition", *missing_condition]].drop_duplicates()
        if metadata["condition"].duplicated().any():
            raise ValueError("condition metadata is not unique")
        before = len(frame)
        frame = frame.merge(metadata, on="condition", how="left", validate="many_to_one")
        if len(frame) != before:
            raise AssertionError("condition metadata merge changed row count")

    ce_low, ce_high = 0.4, 1.05
    frame["ce_outlier"] = (
        (frame["coulombic_efficiency"] < ce_low) |
        (frame["coulombic_efficiency"] > ce_high)
    )
    if "SOH_pct" in frame.columns:
        frame["soh_outlier"] = (frame["SOH_pct"] < 0) | (frame["SOH_pct"] > 150)

    keep_id = [column for column in ID_COLS if column in frame.columns]
    keep_numeric = [column for column in BASE_NUMERIC + CONDITION_NUMERIC if column in frame.columns]
    keep_flags = [column for column in ["ce_outlier", "soh_outlier"] if column in frame.columns]
    keep_targets = [column for column in TARGET_COLS if column in frame.columns]
    return frame[keep_id + keep_numeric + keep_flags + keep_targets].copy()


if __name__ == "__main__":
    p1 = build_p1()
    print("P1 shape:", p1.shape)
    print(p1.head(3))
