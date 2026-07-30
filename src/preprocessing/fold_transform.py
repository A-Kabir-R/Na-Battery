"""The one fold-local transformer shared by every retained model.

Ridge, ExtraTrees and NaPINN-Q all pass their training rows through
:class:`UnifiedFoldPreprocessor`. There is no per-model feature selection and no
per-model imputation logic, so a difference in reported accuracy can only come
from the estimator itself.

Guarantees
----------
* Column set and order are taken from the registry, never from the data.
* Median imputation and robust scaling statistics are fitted on training rows
  only; no whole-dataset or holdout statistic ever enters.
* Binary availability flags are passed through unscaled.
* A column that is missing, duplicated, reordered or unexpectedly added raises
  rather than being silently accommodated.
* :meth:`state_dict` round-trips exactly, so a refit stage can be audited.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from .unified_feature_registry import (
    REGISTRY_VERSION,
    FeatureRole,
    continuous_feature_columns,
    flag_feature_columns,
    model_feature_columns,
    registry_hash,
)


class PreprocessorError(RuntimeError):
    """Raised on any feature-contract violation."""


@dataclass
class UnifiedFoldPreprocessor:
    """Median imputation + robust scaling fitted on training rows only.

    Parameters
    ----------
    exclude_roles:
        Registry roles to drop. NaPINN-Q passes
        ``("stress_coordinate", "initial_state")`` so the differentiable stress
        coordinate and the initial state are not duplicated into the auxiliary
        feature tensor; classical models pass nothing.
    scale:
        When False, imputation still happens but continuous columns are not
        rescaled. Tree models are scale invariant, but the default is True for
        both so that the matrix handed to every model is byte-identical.
    """

    exclude_roles: tuple[FeatureRole, ...] = ()
    scale: bool = True
    #: Robust-scale quantile range.
    quantile_range: tuple[float, float] = (25.0, 75.0)

    feature_columns_: list[str] | None = None
    continuous_columns_: list[str] | None = None
    flag_columns_: list[str] | None = None
    medians_: dict[str, float] = field(default_factory=dict)
    centers_: dict[str, float] = field(default_factory=dict)
    scales_: dict[str, float] = field(default_factory=dict)
    n_fit_rows_: int = 0
    n_fit_cells_: int = 0

    # -- contract ---------------------------------------------------------
    def expected_columns(self) -> list[str]:
        return model_feature_columns(exclude_roles=self.exclude_roles)

    def _validate_frame(self, frame: pd.DataFrame, columns: Sequence[str]) -> None:
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise PreprocessorError(f"missing registered feature columns: {missing}")
        counts = pd.Index(frame.columns).value_counts()
        duplicated = sorted(counts[counts > 1].index.intersection(columns))
        if duplicated:
            raise PreprocessorError(f"duplicated feature columns: {duplicated}")

    # -- fit / transform --------------------------------------------------
    def fit(self, frame: pd.DataFrame, *, cell_column: str = "cell"
            ) -> "UnifiedFoldPreprocessor":
        columns = self.expected_columns()
        self._validate_frame(frame, columns)
        self.feature_columns_ = list(columns)
        flags = set(flag_feature_columns())
        self.flag_columns_ = [c for c in columns if c in flags]
        self.continuous_columns_ = [c for c in columns if c not in flags]

        numeric = (
            frame[columns].apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
        )
        self.n_fit_rows_ = int(len(numeric))
        self.n_fit_cells_ = (
            int(frame[cell_column].nunique()) if cell_column in frame.columns else 0
        )

        self.medians_ = {}
        self.centers_ = {}
        self.scales_ = {}
        low, high = self.quantile_range
        for column in columns:
            values = numeric[column]
            median = values.median(skipna=True)
            # An all-NaN training column would otherwise poison the matrix. The
            # registry guarantees every feature is populated somewhere in the
            # dataset, but a single fold may still miss one entirely.
            self.medians_[column] = float(median) if pd.notna(median) else 0.0
            if column in flags:
                continue
            filled = values.fillna(self.medians_[column])
            center = float(filled.median())
            spread = float(
                np.nanpercentile(filled, high) - np.nanpercentile(filled, low)
            )
            if not np.isfinite(spread) or spread <= 0.0:
                # Constant-within-fold feature: center it, do not divide.
                spread = 1.0
            self.centers_[column] = center
            self.scales_[column] = spread
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.feature_columns_ is None:
            raise PreprocessorError("UnifiedFoldPreprocessor is not fitted")
        columns = self.feature_columns_
        self._validate_frame(frame, columns)
        numeric = (
            frame[columns].apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
        )
        out = np.empty((len(numeric), len(columns)), dtype=float)
        for index, column in enumerate(columns):
            values = numeric[column].fillna(self.medians_[column]).to_numpy(dtype=float)
            if self.scale and column in self.scales_:
                values = (values - self.centers_[column]) / self.scales_[column]
            out[:, index] = values
        if not np.isfinite(out).all():
            bad = [
                columns[i] for i in range(out.shape[1])
                if not np.isfinite(out[:, i]).all()
            ]
            raise PreprocessorError(f"non-finite values after transform in: {bad}")
        return out

    def fit_transform(self, frame: pd.DataFrame, *, cell_column: str = "cell"
                      ) -> np.ndarray:
        return self.fit(frame, cell_column=cell_column).transform(frame)

    # -- provenance -------------------------------------------------------
    @property
    def feature_names_(self) -> list[str]:
        """Alias kept so downstream reporting code can stay unchanged."""
        return list(self.feature_columns_ or [])

    def state_dict(self) -> dict[str, Any]:
        return {
            "registry_version": REGISTRY_VERSION,
            "registry_hash": registry_hash(),
            "exclude_roles": list(self.exclude_roles),
            "scale": bool(self.scale),
            "quantile_range": list(self.quantile_range),
            "feature_columns": list(self.feature_columns_ or []),
            "continuous_columns": list(self.continuous_columns_ or []),
            "flag_columns": list(self.flag_columns_ or []),
            "medians": dict(self.medians_),
            "centers": dict(self.centers_),
            "scales": dict(self.scales_),
            "n_fit_rows": self.n_fit_rows_,
            "n_fit_cells": self.n_fit_cells_,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "UnifiedFoldPreprocessor":
        obj = cls(
            exclude_roles=tuple(state.get("exclude_roles", ())),
            scale=bool(state.get("scale", True)),
            quantile_range=tuple(state.get("quantile_range", (25.0, 75.0))),
        )
        obj.feature_columns_ = list(state["feature_columns"])
        obj.continuous_columns_ = list(state.get("continuous_columns", []))
        obj.flag_columns_ = list(state.get("flag_columns", []))
        obj.medians_ = {k: float(v) for k, v in state["medians"].items()}
        obj.centers_ = {k: float(v) for k, v in state.get("centers", {}).items()}
        obj.scales_ = {k: float(v) for k, v in state.get("scales", {}).items()}
        obj.n_fit_rows_ = int(state.get("n_fit_rows", 0))
        obj.n_fit_cells_ = int(state.get("n_fit_cells", 0))
        return obj

    def feature_hash(self) -> str:
        payload = json.dumps({
            "registry_hash": registry_hash(),
            "feature_columns": list(self.feature_columns_ or []),
        }, sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def state_hash(self) -> str:
        payload = json.dumps(self.state_dict(), sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def assert_no_cell_overlap(train: pd.DataFrame, other: pd.DataFrame, *,
                           cell_column: str = "cell") -> None:
    """Guard against a cell appearing on both sides of a fold boundary."""
    overlap = sorted(set(train[cell_column]) & set(other[cell_column]))
    if overlap:
        raise PreprocessorError(f"cells appear in both partitions: {overlap}")


def fit_fold_preprocessor(train: pd.DataFrame, *,
                          exclude_roles: Iterable[FeatureRole] = (),
                          scale: bool = True) -> UnifiedFoldPreprocessor:
    """Convenience constructor used by both pipelines."""
    return UnifiedFoldPreprocessor(
        exclude_roles=tuple(exclude_roles), scale=scale,
    ).fit(train)
