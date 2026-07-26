"""Fold-fitted numeric feature selection, imputation, and scaling."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

TARGET_COLUMNS = {
    "SOH_pct", "RUL_cycles", "next_rpt_Q_Ah", "next_rpt_SOH_pct",
    "delta_next_rpt_Q_Ah", "delta_next_rpt_SOH_pct", "failure_label",
    "capacity_eol_event", "capacity_eol_cycle", "capacity_censor_cycle",
}
NUMERIC_IDENTIFIERS = {
    "visit", "test_id", "segment_index", "cycle_in_file", "global_cycle", "cv_fold",
    "start_row", "stop_row",
}
TARGET_CONSTRUCTION_METADATA = {
    "initial_rpt_visit", "previous_rpt_visit", "next_rpt_visit",
    "next_rpt_horizon_days", "is_prediction_anchor",
}
TARGET_DEFINING_FEATURES = {
    # Same-cycle capacity directly defines legacy online SOH and capacity EOL.
    "SOH_pct": {"Q_discharge_Ah", "Q_initial_Ah"},
    "RUL_cycles": {"SOH_pct", "Q_discharge_Ah", "Q_initial_Ah"},
}


def numeric_feature_cols(df: pd.DataFrame,
                         drop: Sequence[str] = ("SOH_pct", "Q_initial_Ah")) -> List[str]:
    exclude = set(drop) | TARGET_COLUMNS | NUMERIC_IDENTIFIERS | TARGET_CONSTRUCTION_METADATA | {
        "cell", "condition", "path", "relative_path", "file_id", "test_type",
    }
    return [column for column in df.select_dtypes(include=[np.number]).columns if column not in exclude]


def drop_low_variance(df: pd.DataFrame, cols: Iterable[str],
                      threshold: float = 1e-8) -> List[str]:
    return [column for column in cols if df[column].var(skipna=True) > threshold]


def drop_high_correlation(df: pd.DataFrame, cols: Iterable[str],
                          threshold: float = 0.99) -> List[str]:
    columns = list(cols)
    if len(columns) < 2:
        return columns
    corr = df[columns].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    to_drop = {column for column in upper.columns if any(upper[column] > threshold)}
    return [column for column in columns if column not in to_drop]


@dataclass
class FoldPreprocessor:
    """Preprocessor whose learned state is derived from training rows only."""

    scale: bool = False
    variance_threshold: float = 1e-8
    correlation_threshold: float = 0.99
    feature_names_: list[str] | None = None
    medians_: pd.Series | None = None
    scaler_: StandardScaler | None = None

    def fit(self, df: pd.DataFrame, *, target: str) -> "FoldPreprocessor":
        drop = set(TARGET_DEFINING_FEATURES.get(target, set())) | {target, "Q_initial_Ah"}
        candidates = numeric_feature_cols(df, drop=tuple(drop))
        numeric = df[candidates].replace([np.inf, -np.inf], np.nan)
        candidates = [column for column in candidates if numeric[column].notna().any()]
        candidates = drop_low_variance(numeric, candidates, self.variance_threshold)
        candidates = drop_high_correlation(numeric, candidates, self.correlation_threshold)
        if not candidates:
            raise ValueError("no usable training features after fold-local selection")
        medians = numeric[candidates].median(axis=0, skipna=True)
        candidates = [column for column in candidates if pd.notna(medians[column])]
        if not candidates:
            raise ValueError("all training features have undefined medians")
        self.feature_names_ = candidates
        self.medians_ = medians[candidates].astype(float)
        transformed = numeric[candidates].fillna(self.medians_).to_numpy(dtype=float)
        self.scaler_ = StandardScaler().fit(transformed) if self.scale else None
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.feature_names_ is None or self.medians_ is None:
            raise RuntimeError("FoldPreprocessor must be fitted before transform")
        missing = set(self.feature_names_) - set(df.columns)
        if missing:
            raise ValueError(f"missing fitted feature columns: {sorted(missing)}")
        numeric = df[self.feature_names_].replace([np.inf, -np.inf], np.nan)
        transformed = numeric.fillna(self.medians_).to_numpy(dtype=float)
        if self.scaler_ is not None:
            transformed = self.scaler_.transform(transformed)
        return transformed

    def fit_transform(self, df: pd.DataFrame, *, target: str) -> np.ndarray:
        return self.fit(df, target=target).transform(df)
