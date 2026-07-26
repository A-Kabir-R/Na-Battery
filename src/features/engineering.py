"""General feature-engineering helpers reused across pipelines."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..io.loaders import load_inventory


def add_cycles_since_last_cu(df: pd.DataFrame) -> pd.DataFrame:
    """For each CYC row, count cycles since the most recent CU (RPT) visit."""
    def per_cell(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values(["visit", "cycle_in_file"]).reset_index(drop=True)
        g["cycles_since_last_cu"] = g.groupby("visit").cumcount() + 1
        return g

    pieces = [per_cell(g) for _, g in df.groupby(["condition", "cell"])]
    return pd.concat(pieces, ignore_index=True)


def add_delta_capacity_since_last_cu(df: pd.DataFrame) -> pd.DataFrame:
    """For each CYC row, compute Q_discharge_Ah minus that cell's Q at start
    of the current visit (proxy for capacity delta since last RPT)."""
    def per_cell(g: pd.DataFrame) -> pd.DataFrame:
        first_q = g.groupby("visit")["Q_discharge_Ah"].transform("first")
        g = g.copy()
        g["delta_Q_since_last_cu"] = g["Q_discharge_Ah"] - first_q
        return g

    pieces = [per_cell(g) for _, g in df.groupby(["condition", "cell"])]
    return pd.concat(pieces, ignore_index=True)
