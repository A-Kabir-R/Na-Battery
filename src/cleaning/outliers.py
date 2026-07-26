"""Outlier suppression for raw time-series signals and per-cycle CE."""
from __future__ import annotations

import numpy as np
import pandas as pd


def hampel(x: np.ndarray, window: int = 5, k: float = 3.0) -> np.ndarray:
    """Vectorized Hampel filter. Replaces outliers with rolling median.

    x        : 1-D array
    window   : half-window; total window = 2*window+1
    k        : threshold in units of 1.4826*MAD
    """
    x = np.asarray(x, dtype=float).copy()
    n = len(x)
    if n < 2 * window + 1:
        return x
    # Rolling median + MAD via pandas for O(n log n).
    s = pd.Series(x)
    med = s.rolling(2 * window + 1, center=True, min_periods=1).median()
    mad = (s - med).abs().rolling(2 * window + 1, center=True, min_periods=1).median()
    threshold = k * 1.4826 * mad
    mask = (s - med).abs() > threshold
    x[mask.to_numpy()] = med[mask].to_numpy()
    return x


def clip_ce(df: pd.DataFrame, lo: float = 0.4, hi: float = 1.05,
            col: str = "coulombic_efficiency", flag_col: str = "ce_clipped") -> pd.DataFrame:
    out = df.copy()
    ce = out[col]
    out[flag_col] = (ce < lo) | (ce > hi)
    out[col] = ce.clip(lo, hi)
    return out
