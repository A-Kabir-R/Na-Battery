"""Sodium-ion gassing indicators derived from per-cycle metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_gassing_markers(df: pd.DataFrame, ce_col: str = "coulombic_efficiency",
                        t_col: str = "T_max", window: int = 20) -> pd.DataFrame:
    def per_cell(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("global_cycle").reset_index(drop=True)
        ce = g[ce_col]
        t  = g[t_col] if t_col in g.columns else None
        g["gm_ce_var_r20"] = ce.rolling(window, min_periods=5).var()
        g["gm_ce_low_flag"] = (ce < 0.98).astype(int)
        g["gm_ce_low_rate_r20"] = g["gm_ce_low_flag"].rolling(window, min_periods=5).mean()
        if t is not None:
            g["gm_tmax_slope_r20"] = t.rolling(window, min_periods=5).apply(
                lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) >= 2 else np.nan,
                raw=True,
            )
        return g

    pieces = [per_cell(g) for _, g in df.groupby(["condition", "cell"])]
    return pd.concat(pieces, ignore_index=True)
