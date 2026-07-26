"""Drop cycles whose Q_ch or Q_dch is < frac * that cell's median (interrupted cycles)."""
from __future__ import annotations

import pandas as pd


def drop_incomplete(df: pd.DataFrame, frac: float = 0.5,
                    q_ch_col: str = "Q_charge_Ah",
                    q_dch_col: str = "Q_discharge_Ah") -> pd.DataFrame:
    med = df.groupby(["condition", "cell"])[[q_ch_col, q_dch_col]].transform("median")
    keep = (df[q_ch_col] >= frac * med[q_ch_col]) & (df[q_dch_col] >= frac * med[q_dch_col])
    return df.loc[keep].reset_index(drop=True)
