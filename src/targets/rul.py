"""Capacity-EOL event metadata and secondary uncensored RUL target."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..io.loaders import load_config

TARGET = "RUL_cycles"


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    cfg = load_config()
    threshold = float(cfg["targets"]["capacity_eol"]["soh_threshold_pct"])
    persistence = int(cfg["targets"]["capacity_eol"]["minimum_consecutive_cycles"])
    out = df.copy()
    out[TARGET] = np.nan
    out["capacity_eol_event"] = 0
    out["capacity_eol_cycle"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    out["capacity_censor_cycle"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    if "SOH_pct" not in out.columns:
        out["capacity_eol_status"] = "disabled_missing_validated_soh"
        return out

    out["capacity_eol_status"] = "right_censored"
    for _, group in out.groupby(["condition", "cell"]):
        ordered = group.sort_values("global_cycle")
        censor_cycle = int(ordered["global_cycle"].max())
        out.loc[ordered.index, "capacity_censor_cycle"] = censor_cycle
        below = ordered["SOH_pct"].lt(threshold).fillna(False)
        sustained_end = below.rolling(persistence, min_periods=persistence).sum().eq(persistence)
        if not sustained_end.any():
            continue
        end_position = int(np.flatnonzero(sustained_end.to_numpy())[0])
        start_position = end_position - persistence + 1
        eol_cycle = int(ordered.iloc[start_position]["global_cycle"])
        eligible = ordered["global_cycle"] <= eol_cycle
        rul = (eol_cycle - ordered.loc[eligible, "global_cycle"]).astype(float)
        out.loc[ordered.index, "capacity_eol_event"] = 1
        out.loc[ordered.index, "capacity_eol_cycle"] = eol_cycle
        out.loc[ordered.index, "capacity_eol_status"] = "observed_sustained_crossing"
        out.loc[rul.index, TARGET] = rul
    return out
