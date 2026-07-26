"""Observed CID/gassing event target.

Cycling-derived CE, SOH, and temperature proxy labels were removed because
they leak predictor trajectories into the outcome and are not observed gassing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "failure_label"


def add_target(df: pd.DataFrame, events: pd.DataFrame | None = None) -> pd.DataFrame:
    out = df.copy()
    out[TARGET] = np.nan
    out["failure_label_status"] = "disabled_missing_observed_cid_events"
    if events is None or events.empty:
        return out
    required = {"condition", "cell", "event_type", "event_observed"}
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"events table is missing columns: {sorted(missing)}")
    if events.duplicated(["condition", "cell"]).any():
        raise ValueError("events must contain at most one terminal outcome per cell-condition")

    outcomes = events.set_index(["condition", "cell"])
    for (condition, cell), event in outcomes.iterrows():
        mask = (out["condition"] == condition) & (out["cell"] == cell)
        if bool(event["event_observed"]) and event["event_type"] == "CID":
            out.loc[mask, TARGET] = 1.0
            out.loc[mask, "failure_label_status"] = "observed_cid"
        elif bool(event["event_observed"]) and event["event_type"] == "NO_CID":
            out.loc[mask, TARGET] = 0.0
            out.loc[mask, "failure_label_status"] = "observed_no_cid"
        else:
            out.loc[mask, "failure_label_status"] = "censored"
    return out
