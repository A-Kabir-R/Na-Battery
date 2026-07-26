"""P2 — rolling window + Severson-style delta + gassing markers.

Extends P1 with per-cell temporal aggregates.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..io.loaders import load_config
from .p1_aggregate import build_p1


def _rolling_stats(series: pd.Series, window: int) -> pd.DataFrame:
    r = series.rolling(window, min_periods=max(2, window // 2))
    return pd.DataFrame({
        f"{series.name}_r{window}_mean": r.mean(),
        f"{series.name}_r{window}_std":  r.std(),
        f"{series.name}_r{window}_slope": r.apply(
            lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) >= 2 else np.nan,
            raw=True,
        ),
    })


def _delta_features(series: pd.Series, lag: int) -> pd.Series:
    return (series - series.shift(lag)).rename(f"{series.name}_delta{lag}")


def build_p2(df: pd.DataFrame | None = None) -> pd.DataFrame:
    cfg = load_config()
    p1 = build_p1(df)
    win = cfg["preprocessing"]["rolling_window"]
    slag = cfg["preprocessing"]["severson_window"][1] - cfg["preprocessing"]["severson_window"][0]

    def per_cell(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("global_cycle").reset_index(drop=True)
        pieces = [g]
        for col in ["Q_discharge_Ah", "coulombic_efficiency",
                    "energy_efficiency", "T_max"]:
            if col in g.columns:
                pieces.append(_rolling_stats(g[col], win))
                pieces.append(_delta_features(g[col], slag).to_frame())
        # Scalar capacity-difference variability. This is deliberately not
        # named as the voltage-resolved Severson Delta-Q(V) feature.
        if "Q_discharge_Ah" in g.columns:
            dq = g["Q_discharge_Ah"].diff()
            var_dq = dq.rolling(win, min_periods=max(2, win // 2)).var()
            log_var = np.log(var_dq.replace(0, np.nan))
            pieces.append(log_var.rename(f"Q_discharge_diff_logvar_r{win}").to_frame())
            nominal_capacity = float(cfg["protocol"]["nominal_capacity_Ah"])
            efc = g["Q_discharge_Ah"].fillna(0).cumsum() / nominal_capacity
            pieces.append(efc.rename("EFC_cum").to_frame())
        return pd.concat(pieces, axis=1)

    pieces = [per_cell(g) for _, g in p1.groupby(["condition", "cell"])]
    return pd.concat(pieces, ignore_index=True).reset_index(drop=True)


if __name__ == "__main__":
    p2 = build_p2()
    print("P2 shape:", p2.shape)
    print(list(p2.columns))
