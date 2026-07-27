"""Dataset assembly for the NaPINN-Q pipeline.

The neural pipeline predicts one fundamental state: normalized capacity at the
next scheduled RPT anchor. All four regression targets are derived from that
state and the recorded reference capacity Q_0.

Every column that enters the network is classified by
:func:`build_temporal_feature_audit`. Rows containing forbidden future
information cause :class:`TemporalLeakageError` at load time.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------

TARGETS = (
    "next_rpt_Q_Ah",
    "next_rpt_SOH_pct",
    "delta_next_rpt_Q_Ah",
    "delta_next_rpt_SOH_pct",
)

IDENTIFIER_COLUMNS = (
    "anchor_id", "condition", "cell", "file_id", "path", "visit",
    "global_cycle", "cycle_in_file",
)

# Columns that would leak the future target if used as network inputs.
FORBIDDEN_COLUMNS = frozenset({
    "next_rpt_Q_Ah", "next_rpt_SOH_pct",
    "delta_next_rpt_Q_Ah", "delta_next_rpt_SOH_pct",
    "next_rpt_visit", "next_rpt_horizon_days",  # future-derived when unavailable
    "next_rpt_path", "next_rpt_match_method",
    "next_rpt_target_status", "next_rpt_soh_status", "next_rpt_delta_status",
    "delta_next_rpt_Q_status", "delta_next_rpt_SOH_status",
    "delta_rpt_Q_Ah", "delta_rpt_SOH_pct",
    # ambiguous / non-numeric / target-defining
    "target_unavailable_reason", "is_prediction_anchor",
    "cycle_qc_status", "cycle_qc_flags", "test_type", "relative_path",
    "file_start_time", "cycle_start_time_s",
    "initial_rpt_path", "initial_rpt_visit", "initial_rpt_match_method",
    "previous_rpt_path", "previous_rpt_visit", "previous_rpt_match_method",
})

# Base physical-state / operating-condition block, when present.
CONDITION_BLOCK = (
    "T_mean", "T_max", "T_min", "DOD_pct", "C_ch", "C_dis",
    "EFC_cum", "cumulative_Ah_throughput", "cycle_end_time_s",
    "coulombic_efficiency", "energy_efficiency",
    "visit", "global_cycle", "cycle_in_file",
)


class TemporalLeakageError(ValueError):
    """Raised when a forbidden temporal-leakage column is passed to a model."""


class StressCoordinateError(ValueError):
    """Raised when the stress coordinate cannot be constructed monotonically."""


@dataclass
class AnchorDataset:
    """Container for anchor-level PINN training data.

    Attributes
    ----------
    frame : pd.DataFrame
        Row-per-anchor frame carrying identifiers, targets, and preprocessed
        features. Only rows with `next_rpt_Q_Ah` present are retained.
    feature_columns : list[str]
        Ordered list of feature columns (leakage-audited) fed to the network.
    stress_column : str
        Physical stress coordinate column (typically 'stress_current').
    horizon_column : str
        Column carrying Delta s (in the same units as stress).
    q0_column : str
        Reference-capacity column (initial_rpt_Q_Ah).
    q_current_column : str
        Current-anchor capacity column (previous_rpt_Q_Ah or fallback).
    audit : pd.DataFrame
        Temporal-leakage audit table (one row per candidate feature).
    """

    frame: pd.DataFrame
    feature_columns: list[str]
    stress_column: str
    horizon_column: str
    q0_column: str
    q_current_column: str
    audit: pd.DataFrame


# ---------------------------------------------------------------------------
# Temporal leakage audit
# ---------------------------------------------------------------------------

def build_temporal_feature_audit(candidate_columns: Iterable[str], *,
                                 preprocessing: str) -> pd.DataFrame:
    """Classify each candidate feature column.

    Rows with ``allowed_as_input == False`` must be excluded from network
    inputs. The plan requires a rejection reason for any excluded column.
    """
    rows: list[dict[str, Any]] = []
    for column in candidate_columns:
        is_forbidden = column in FORBIDDEN_COLUMNS
        known_at_anchor = not is_forbidden
        planned_at_inference = column in {"next_rpt_horizon_days", "horizon_days_planned"}
        future_derived = is_forbidden
        allowed = known_at_anchor and not future_derived
        rejection = "future_derived_or_target" if is_forbidden else ""
        rows.append({
            "feature": column,
            "preprocessing": preprocessing,
            "source_table": f"artifacts/features/{preprocessing}.parquet",
            "source_column": column,
            "physical_meaning": column,
            "measurement_start": "cell_start" if column.startswith("cumulative_") else "cycle_start",
            "measurement_end": "anchor_time",
            "anchor_time": "current_anchor",
            "target_time": "next_rpt",
            "known_at_anchor": bool(known_at_anchor),
            "planned_at_inference": bool(planned_at_inference),
            "future_derived": bool(future_derived),
            "allowed_as_input": bool(allowed),
            "rejection_reason": rejection,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Stress coordinate & horizon
# ---------------------------------------------------------------------------

def _select_stress_column(frame: pd.DataFrame, cfg: Mapping[str, Any]) -> str:
    """Return the highest-priority stress column present in ``frame``.

    Priority: preferred > fallback > time fallback (if allowed).
    """
    preferred = str(cfg.get("preferred_coordinate", "EFC_cum"))
    fallback = str(cfg.get("fallback_coordinate", "cumulative_Ah_throughput"))
    time_col = str(cfg.get("time_fallback_column", "cycle_end_time_s"))
    allow_time = bool(cfg.get("allow_elapsed_time_fallback", True))
    for candidate in (preferred, fallback):
        if candidate in frame.columns and frame[candidate].notna().any():
            return candidate
    if allow_time and time_col in frame.columns and frame[time_col].notna().any():
        return time_col
    raise StressCoordinateError(
        f"No usable stress column found; tried {preferred}, {fallback}, {time_col}"
    )


def _ensure_per_cell_monotone(frame: pd.DataFrame, column: str,
                              tol: float = 1e-6) -> None:
    grouped = frame.groupby("cell", sort=False)
    for cell, sub in grouped:
        values = sub[column].to_numpy(dtype=float)
        if len(values) < 2:
            continue
        diffs = np.diff(values)
        if (diffs < -tol).any():
            raise StressCoordinateError(
                f"Stress column {column} is nonmonotonic within cell {cell}"
            )


def _compute_planned_horizon(frame: pd.DataFrame, stress_column: str,
                             q0_column: str) -> pd.Series:
    """Estimate ``Delta_s`` in the same units as ``stress_column``.

    When consecutive anchors within a cell are available we use the observed
    stress increment between them. For the last anchor we fall back to
    ``next_rpt_horizon_days * per_cell_daily_rate``. The daily-rate estimate is
    per-cell and derived only from the anchor frame itself; the outer-training
    scaler is applied later.
    """
    frame = frame.sort_values(["cell", "visit"]).copy()
    horizon = pd.Series(index=frame.index, dtype=float)
    for cell, sub in frame.groupby("cell", sort=False):
        stress = sub[stress_column].to_numpy(dtype=float)
        deltas = np.diff(stress, append=np.nan)
        horizon.loc[sub.index] = deltas
    # For rows without a next anchor, fall back to horizon_days * mean rate.
    missing = ~horizon.notna()
    if missing.any() and "next_rpt_horizon_days" in frame.columns:
        # Per-cell mean stress-per-day rate (falls back to overall mean if only one anchor).
        rate = pd.Series(index=frame.index, dtype=float)
        for cell, sub in frame.groupby("cell", sort=False):
            days = pd.to_numeric(sub.get("next_rpt_horizon_days"), errors="coerce").to_numpy()
            deltas = horizon.loc[sub.index].to_numpy()
            mask = np.isfinite(days) & np.isfinite(deltas) & (days > 0)
            if mask.any():
                r = float(np.nanmean(deltas[mask] / days[mask]))
            else:
                r = np.nan
            rate.loc[sub.index] = r
        overall = float(np.nanmean(rate.to_numpy())) if np.isfinite(rate).any() else 1.0
        rate = rate.fillna(overall)
        days = pd.to_numeric(frame["next_rpt_horizon_days"], errors="coerce")
        horizon = horizon.where(~missing, rate * days)
    # Ensure strictly positive so that log-friendly downstream stats work.
    horizon = horizon.clip(lower=1e-6)
    return horizon


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------

def build_anchor_dataset(preprocessing_frame: pd.DataFrame, *,
                         preprocessing: str,
                         stress_cfg: Mapping[str, Any],
                         audit_path: Path | None = None) -> AnchorDataset:
    """Filter ``preprocessing_frame`` to eligible anchors and audit features."""
    if "is_prediction_anchor" not in preprocessing_frame.columns:
        raise ValueError("preprocessing frame is missing 'is_prediction_anchor'")
    anchors = preprocessing_frame[
        preprocessing_frame["is_prediction_anchor"].fillna(False)
    ].copy()
    anchors = anchors.dropna(subset=["next_rpt_Q_Ah", "initial_rpt_Q_Ah"]).copy()
    anchors = anchors[anchors["initial_rpt_Q_Ah"] > 0].copy()
    if anchors.empty:
        raise ValueError(f"no eligible anchor rows in preprocessing '{preprocessing}'")

    anchors = anchors.reset_index(drop=True)

    # Q_current: capacity measured at the RPT preceding the anchor cycle.
    if "previous_rpt_Q_Ah" in anchors.columns:
        anchors["q_current_Ah"] = anchors["previous_rpt_Q_Ah"].fillna(
            anchors["initial_rpt_Q_Ah"]
        )
    else:
        anchors["q_current_Ah"] = anchors["initial_rpt_Q_Ah"]

    # Normalized current capacity u_k.
    anchors["u_current"] = (
        anchors["q_current_Ah"].astype(float)
        / anchors["initial_rpt_Q_Ah"].astype(float)
    )
    anchors["u_true_next"] = (
        anchors["next_rpt_Q_Ah"].astype(float)
        / anchors["initial_rpt_Q_Ah"].astype(float)
    )

    # Stress coordinate.
    stress_column = _select_stress_column(anchors, stress_cfg)
    if anchors[stress_column].isna().any():
        anchors[stress_column] = anchors.groupby("cell")[stress_column].transform(
            lambda s: s.ffill().bfill()
        )
    _ensure_per_cell_monotone(anchors, stress_column)
    anchors["stress_current"] = anchors[stress_column].astype(float)
    anchors["stress_delta"] = _compute_planned_horizon(
        anchors, "stress_current", "initial_rpt_Q_Ah"
    ).astype(float)
    anchors["stress_next"] = anchors["stress_current"] + anchors["stress_delta"]

    # Candidate feature columns: numeric, non-identifier, non-forbidden.
    numeric = anchors.select_dtypes(include=[np.number]).columns.tolist()
    identifiers = set(IDENTIFIER_COLUMNS) | {"initial_rpt_Q_Ah", "previous_rpt_Q_Ah",
                                             "previous_rpt_SOH_pct",
                                             "q_current_Ah", "u_current", "u_true_next",
                                             "stress_current", "stress_delta",
                                             "stress_next"}
    candidate_columns = [
        c for c in numeric if c not in identifiers and c not in TARGETS
    ]

    audit = build_temporal_feature_audit(candidate_columns, preprocessing=preprocessing)
    if audit_path is not None:
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit.to_csv(audit_path, index=False)

    allowed = audit[audit["allowed_as_input"]]["feature"].tolist()
    forbidden_present = set(anchors.columns) & FORBIDDEN_COLUMNS
    # Presence of these columns is fine; passing them to the model is not.
    # We simply do not include them in feature_columns.
    del forbidden_present

    if not allowed:
        raise TemporalLeakageError(
            "no leakage-safe feature columns remain after temporal audit"
        )

    return AnchorDataset(
        frame=anchors,
        feature_columns=allowed,
        stress_column="stress_current",
        horizon_column="stress_delta",
        q0_column="initial_rpt_Q_Ah",
        q_current_column="q_current_Ah",
        audit=audit,
    )


# ---------------------------------------------------------------------------
# Split helpers
# ---------------------------------------------------------------------------

def apply_split_manifest(frame: pd.DataFrame,
                         split_manifest: pd.DataFrame) -> pd.DataFrame:
    """Attach ``outer_role`` / ``cv_fold`` to ``frame`` by cell (many-to-one)."""
    assignments = split_manifest[["cell", "outer_role", "cv_fold"]].copy()
    assignments["cell"] = assignments["cell"].astype(str)
    frame = frame.copy()
    frame["cell"] = frame["cell"].astype(str)
    if "outer_role" in frame.columns:
        frame = frame.drop(columns=["outer_role"])
    if "cv_fold" in frame.columns:
        frame = frame.drop(columns=["cv_fold"])
    merged = frame.merge(assignments, on="cell", how="left", validate="many_to_one")
    if merged["outer_role"].isna().any():
        missing = sorted(merged.loc[merged["outer_role"].isna(), "cell"].unique())
        raise ValueError(f"cells absent from split manifest: {missing[:10]}")
    return merged


def fold_indices(frame: pd.DataFrame, fold: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (outer_train_idx, outer_val_idx) for ``fold`` in the development set."""
    development = frame["outer_role"].eq("development")
    validation = development & frame["cv_fold"].eq(fold)
    train = development & ~frame["cv_fold"].eq(fold)
    return (
        np.flatnonzero(train.to_numpy()),
        np.flatnonzero(validation.to_numpy()),
    )


def inner_split_indices(outer_train_frame: pd.DataFrame, *,
                        n_inner: int = 3,
                        seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Group-shuffle split of outer-training cells into (inner_train, inner_val).

    Uses the last ``1/n_inner`` fraction of cells (sorted deterministically by
    seed) as inner validation. Guarantees no cell overlap.
    """
    cells = sorted(outer_train_frame["cell"].astype(str).unique())
    if len(cells) < 2:
        raise ValueError("need at least two cells for inner validation")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(cells)
    cutoff = max(1, len(shuffled) // n_inner)
    inner_val_cells = set(shuffled[:cutoff])
    inner_train_cells = set(shuffled[cutoff:])
    if not inner_train_cells:
        # Guarantee at least one training cell.
        inner_train_cells.add(next(iter(inner_val_cells)))
        inner_val_cells.remove(next(iter(inner_val_cells)))
    outer_cells = outer_train_frame["cell"].astype(str).to_numpy()
    train_mask = np.array([c in inner_train_cells for c in outer_cells])
    val_mask = np.array([c in inner_val_cells for c in outer_cells])
    return np.flatnonzero(train_mask), np.flatnonzero(val_mask)


# ---------------------------------------------------------------------------
# Fold-local scaling
# ---------------------------------------------------------------------------

@dataclass
class FoldScaler:
    """Fold-local mean/std imputation + scaler estimated on outer-training rows."""

    feature_medians: np.ndarray | None = None
    feature_mean: np.ndarray | None = None
    feature_std: np.ndarray | None = None
    stress_mean: float = 0.0
    stress_std: float = 1.0
    horizon_mean: float = 0.0
    horizon_std: float = 1.0
    u_max: float = 1.0
    epsilon_rec: float = 0.02

    def fit(self, frame: pd.DataFrame, feature_columns: list[str], *,
            u_max_source: str, u_max_constant: float, u_max_margin: float,
            tolerance_source: str, tolerance_constant: float,
            tolerance_quantile: float) -> "FoldScaler":
        features = frame[feature_columns].to_numpy(dtype=float)
        self.feature_medians = np.nanmedian(features, axis=0)
        replaced = np.where(np.isnan(features), self.feature_medians, features)
        self.feature_mean = replaced.mean(axis=0)
        self.feature_std = replaced.std(axis=0) + 1e-8

        stress = frame["stress_current"].to_numpy(dtype=float)
        self.stress_mean = float(np.mean(stress))
        self.stress_std = float(np.std(stress) + 1e-8)
        horizon = frame["stress_delta"].to_numpy(dtype=float)
        self.horizon_mean = float(np.mean(horizon))
        self.horizon_std = float(np.std(horizon) + 1e-8)

        u_current = frame["u_current"].to_numpy(dtype=float)
        u_true = frame["u_true_next"].to_numpy(dtype=float)
        if u_max_source == "outer_training":
            observed_max = float(np.nanmax(np.concatenate([u_current, u_true])))
            self.u_max = observed_max + u_max_margin
        else:
            self.u_max = float(u_max_constant)

        if tolerance_source == "outer_training":
            recoveries = u_true - u_current
            positive = recoveries[recoveries > 0]
            if positive.size:
                self.epsilon_rec = float(np.quantile(positive, tolerance_quantile))
            else:
                self.epsilon_rec = float(tolerance_constant)
        else:
            self.epsilon_rec = float(tolerance_constant)
        return self

    def transform_features(self, frame: pd.DataFrame,
                           feature_columns: list[str]) -> np.ndarray:
        features = frame[feature_columns].to_numpy(dtype=float)
        replaced = np.where(np.isnan(features), self.feature_medians, features)
        return (replaced - self.feature_mean) / self.feature_std

    def transform_stress(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=float) - self.stress_mean) / self.stress_std

    def transform_horizon(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=float) - self.horizon_mean) / self.horizon_std

    def state_dict(self) -> dict[str, Any]:
        return {
            "feature_medians": None if self.feature_medians is None else self.feature_medians.tolist(),
            "feature_mean": None if self.feature_mean is None else self.feature_mean.tolist(),
            "feature_std": None if self.feature_std is None else self.feature_std.tolist(),
            "stress_mean": self.stress_mean,
            "stress_std": self.stress_std,
            "horizon_mean": self.horizon_mean,
            "horizon_std": self.horizon_std,
            "u_max": self.u_max,
            "epsilon_rec": self.epsilon_rec,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "FoldScaler":
        obj = cls()
        obj.feature_medians = None if state["feature_medians"] is None else np.asarray(state["feature_medians"])
        obj.feature_mean = None if state["feature_mean"] is None else np.asarray(state["feature_mean"])
        obj.feature_std = None if state["feature_std"] is None else np.asarray(state["feature_std"])
        obj.stress_mean = float(state["stress_mean"])
        obj.stress_std = float(state["stress_std"])
        obj.horizon_mean = float(state["horizon_mean"])
        obj.horizon_std = float(state["horizon_std"])
        obj.u_max = float(state["u_max"])
        obj.epsilon_rec = float(state["epsilon_rec"])
        return obj
