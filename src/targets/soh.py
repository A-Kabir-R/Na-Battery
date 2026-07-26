"""SOH target — already computed as SOH_pct in cycle_metrics.csv."""
from __future__ import annotations

import pandas as pd

TARGET = "SOH_pct"


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    if TARGET not in df.columns:
        raise ValueError(f"expected {TARGET} column already present from cycle_metrics")
    return df
