"""Handle activation cycles (SOH > 100%) by clipping the target SOH."""
from __future__ import annotations

import pandas as pd


def clip_soh(df: pd.DataFrame, soh_max: float = 100.0, col: str = "SOH_pct") -> pd.DataFrame:
    out = df.copy()
    out[col] = out[col].clip(upper=soh_max)
    return out
