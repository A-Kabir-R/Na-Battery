"""Dataset assembly for the NaPINN-Q pipeline.

The neural pipeline predicts one fundamental state: normalized capacity at the
next scheduled RPT anchor. All four regression targets are derived from that
state and the recorded reference capacity Q_0.

Physical time alignment (see the Stage-1 diagnosis fixes)
---------------------------------------------------------
For every anchor (the final complete cycle of one CYC file):

    Previous RPT  ->  CYC block  ->  Next RPT
    time = s_prev     duration Δs    time = s_next

we associate:

    u_current       = Q_previous_rpt / Q_0   at time s_prev
    u_true_next     = Q_next_rpt     / Q_0   at time s_next
    stress_current  = stress accumulated up to the previous RPT
    stress_next     = stress accumulated up to the current CYC block end
    stress_delta    = stress_next - stress_current  (exposure of the CYC block)

``stress_current`` is derived as the previous anchor's ``stress_next`` within a
cell (chronologically), falling back to 0 for the first anchor (initial RPT is
at the cell start). No future information is ever consulted.

Feature-column audit
--------------------
:func:`build_temporal_feature_audit` classifies every candidate column into
``allowed``, ``rejected`` or ``unaudited``. Unaudited columns are rejected by
default; the ``pinn.audit.allow_unaudited`` config toggle permits them (with an
audit-row warning) only when the researcher explicitly opts in.
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

#: Name of the single preprocessing used for all final training. When the
#: dataset is built for this level, features come from the fixed registry
#: instead of dtype-based column discovery.
UNIFIED_PREPROCESSING = "unified"

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
    "next_rpt_visit", "next_rpt_horizon_days",
    "next_rpt_path", "next_rpt_match_method",
    "next_rpt_target_status", "next_rpt_soh_status", "next_rpt_delta_status",
    "delta_next_rpt_Q_status", "delta_next_rpt_SOH_status",
    "delta_rpt_Q_Ah", "delta_rpt_SOH_pct",
    "target_unavailable_reason", "is_prediction_anchor",
    "cycle_qc_status", "cycle_qc_flags", "test_type", "relative_path",
    "file_start_time", "cycle_start_time_s",
    "initial_rpt_path", "initial_rpt_visit", "initial_rpt_match_method",
    "previous_rpt_path", "previous_rpt_visit", "previous_rpt_match_method",
})

# Explicit allow-list of causal, known-safe columns. Any column outside this
# set is treated as unaudited and rejected unless
# ``pinn.audit.allow_unaudited`` is true.
CAUSAL_ALLOW_LIST = frozenset({
    # Operating conditions (measured within the current cycle / block).
    "T_mean", "T_max", "T_min", "DOD_pct", "C_ch", "C_dis",
    "coulombic_efficiency", "energy_efficiency",
    # Cumulative stress measured up to and including the anchor.
    "EFC_cum", "cumulative_Ah_throughput", "cycle_end_time_s",
    # State observed at the *previous* RPT (causal by construction).
    "previous_rpt_Q_Ah", "previous_rpt_SOH_pct",
    # Reference capacity is a per-cell constant known at the initial RPT.
    "initial_rpt_Q_Ah",
})

CAUSAL_ALLOW_PREFIXES: tuple[str, ...] = (
    "T_", "V_", "I_", "Q_", "capacity_",
    "cyc_", "step_", "protocol_",
    "cumulative_", "efc_",
    "prev_", "since_start_", "backward_", "rolling_backward_",
    "p1_", "p2_", "p3_",
)

# Explicit reject patterns even if not in FORBIDDEN_COLUMNS.
LEAKAGE_REJECT_PREFIXES: tuple[str, ...] = (
    "next_", "future_", "planned_", "target_", "delta_next_",
)
LEAKAGE_REJECT_SUFFIXES: tuple[str, ...] = (
    "_ahead", "_lookahead", "_forward", "_centered", "_future",
)
LEAKAGE_REJECT_SUBSTRINGS: tuple[str, ...] = (
    "centered_roll", "roll_centered",
)


class TemporalLeakageError(ValueError):
    """Raised when a forbidden temporal-leakage column is passed to a model."""


class StressCoordinateError(ValueError):
    """Raised when the stress coordinate cannot be constructed monotonically."""


class ForecastHorizonError(ValueError):
    """Raised when ``require_known_forecast_horizon`` cannot be satisfied."""


@dataclass
class AnchorDataset:
    """Container for anchor-level PINN training data.

    Attributes
    ----------
    frame : pd.DataFrame
        Row-per-anchor frame carrying identifiers, targets and preprocessed
        features. Only rows with ``next_rpt_Q_Ah`` present are retained. The
        frame also carries ``stress_current``, ``stress_next`` and
        ``stress_delta`` (aligned per the Stage-1 diagnosis fix).
    feature_columns : list[str]
        Ordered list of feature columns (leakage-audited) fed to the network.
    stress_column : str
        Physical stress coordinate column (always ``'stress_current'``).
    horizon_column : str
        Column carrying Δs in the same units as ``stress_column``.
    q0_column : str
        Reference-capacity column (``initial_rpt_Q_Ah``).
    q_current_column : str
        Current-anchor capacity column (``previous_rpt_Q_Ah`` or fallback).
    audit : pd.DataFrame
        Temporal-leakage audit table (one row per candidate feature). Includes
        ``status`` (allowed / rejected / unaudited) and a
        ``rejection_reason``.
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

def _classify_column(column: str) -> tuple[str, str]:
    """Return ``(status, reason)`` for a candidate column name.

    Statuses:
      * ``"allowed"``  — matches the explicit allow-list or a known-causal prefix.
      * ``"rejected"`` — matches the forbidden list or a known-leaky pattern.
      * ``"unaudited"``— no positive signal; rejected by default at call site.
    """
    if column in FORBIDDEN_COLUMNS or column in TARGETS:
        return "rejected", "future_derived_or_target"
    lowered = column.lower()
    for prefix in LEAKAGE_REJECT_PREFIXES:
        if lowered.startswith(prefix):
            return "rejected", f"leaky_prefix:{prefix}"
    for suffix in LEAKAGE_REJECT_SUFFIXES:
        if lowered.endswith(suffix):
            return "rejected", f"leaky_suffix:{suffix}"
    for token in LEAKAGE_REJECT_SUBSTRINGS:
        if token in lowered:
            return "rejected", f"leaky_substring:{token}"
    if column in CAUSAL_ALLOW_LIST:
        return "allowed", "allow_list"
    for prefix in CAUSAL_ALLOW_PREFIXES:
        if lowered.startswith(prefix):
            return "allowed", f"allow_prefix:{prefix}"
    return "unaudited", "no_positive_signal"


def load_provenance_manifest(preprocessing: str, *,
                              features_dir: Path | None = None,
                              explicit_path: Path | None = None
                              ) -> pd.DataFrame | None:
    """Load a per-preprocessing feature provenance CSV, if present.

    The CSV supersedes the name-based whitelist for any column it lists. Schema
    (each row describes one feature):

    ``feature``          — column name (must match the parquet).
    ``source_columns``   — comma-separated raw columns the feature depends on.
    ``uses_target_rpt``  — bool; True → rejected (target-derived).
    ``uses_future_rows`` — bool; True → rejected (any forward lookup).
    ``rolling_direction``— {'backward','none','',NaN} allowed;
                           {'forward','centered'} → rejected.
    ``shift_count``      — non-positive → allowed (past only), positive → rejected.
    ``known_at_anchor``  — bool; False → rejected unless
                           ``planned_at_inference`` is also True.
    ``planned_at_inference`` — bool; forecast-horizon-style features may be
                           unknown at anchor time but planned.
    ``allowed``          — optional explicit override (bool). If present it wins.
    """
    candidate = explicit_path
    if candidate is None and features_dir is not None:
        candidate = features_dir / f"{preprocessing}.provenance.csv"
    if candidate is None or not Path(candidate).exists():
        return None
    return pd.read_csv(candidate)


def _classify_from_provenance(row: pd.Series) -> tuple[str, str]:
    """Return ``(status, reason)`` from a provenance manifest row."""
    def _truthy(v: Any) -> bool:
        if v is None:
            return False
        if isinstance(v, float) and np.isnan(v):
            return False
        s = str(v).strip().lower()
        return s in {"1", "true", "yes", "y", "t"}

    # Explicit override wins.
    if "allowed" in row.index and pd.notna(row.get("allowed")):
        if _truthy(row.get("allowed")):
            return "allowed", "provenance:explicit_allowed"
        return "rejected", "provenance:explicit_rejected"

    if _truthy(row.get("uses_target_rpt")):
        return "rejected", "provenance:uses_target_rpt"
    if _truthy(row.get("uses_future_rows")):
        return "rejected", "provenance:uses_future_rows"

    direction = str(row.get("rolling_direction", "") or "").strip().lower()
    if direction in {"forward", "centered", "look_ahead", "lookahead"}:
        return "rejected", f"provenance:rolling_direction={direction}"

    shift = row.get("shift_count", 0)
    try:
        shift_int = int(float(shift)) if shift is not None and str(shift) != "" else 0
    except (TypeError, ValueError):
        shift_int = 0
    if shift_int > 0:
        return "rejected", f"provenance:shift_count={shift_int}"

    planned = _truthy(row.get("planned_at_inference"))
    known = _truthy(row.get("known_at_anchor"))
    if not known and not planned:
        return "rejected", "provenance:not_known_at_anchor"

    return "allowed", "provenance:causal"


def build_temporal_feature_audit(candidate_columns: Iterable[str], *,
                                 preprocessing: str,
                                 allow_unaudited: bool = False,
                                 provenance: pd.DataFrame | None = None,
                                 ) -> pd.DataFrame:
    """Classify each candidate feature column as allowed / rejected / unaudited.

    When ``provenance`` is supplied (see :func:`load_provenance_manifest`), it
    is the authoritative source: for any listed feature the provenance row
    decides the status directly. Columns absent from provenance fall back to
    the name-based whitelist (with the same ``allow_unaudited`` semantics).
    """
    provenance_map: dict[str, pd.Series] = {}
    if provenance is not None and not provenance.empty and "feature" in provenance.columns:
        provenance_map = {
            str(row["feature"]): row for _, row in provenance.iterrows()
        }

    rows: list[dict[str, Any]] = []
    for column in candidate_columns:
        prov_row = provenance_map.get(column)
        if prov_row is not None:
            status, reason = _classify_from_provenance(prov_row)
            provenance_source = "provenance_manifest"
            planned_at_inference = bool(
                str(prov_row.get("planned_at_inference", "")).strip().lower()
                in {"1", "true", "yes", "y", "t"}
            )
            known_at_anchor = bool(
                str(prov_row.get("known_at_anchor", "")).strip().lower()
                in {"1", "true", "yes", "y", "t"}
            )
        else:
            status, reason = _classify_column(column)
            provenance_source = "column_name_registry"
            planned_at_inference = column in {"next_rpt_horizon_days", "horizon_days_planned"}
            known_at_anchor = bool(status == "allowed")

        if status == "allowed":
            allowed = True
            final_reason = ""
        elif status == "rejected":
            allowed = False
            final_reason = reason
        else:  # unaudited
            allowed = bool(allow_unaudited)
            final_reason = "" if allowed else "unaudited (no positive provenance signal)"
        rows.append({
            "feature": column,
            "preprocessing": preprocessing,
            "status": status,
            "provenance_source": provenance_source,
            "measurement_end": "anchor_time_or_earlier" if status == "allowed"
                else ("future_derived" if status == "rejected" else "unknown"),
            "known_at_anchor": known_at_anchor,
            "planned_at_inference": planned_at_inference,
            "future_derived": bool(status == "rejected"),
            "allowed_as_input": bool(allowed),
            "rejection_reason": final_reason,
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


def _chronological_sort(frame: pd.DataFrame) -> pd.DataFrame:
    """Sort anchors chronologically within each cell using the most precise
    time key available: ``file_start_time`` > ``global_cycle`` > ``visit``.
    """
    keys: list[str] = ["cell"]
    if "file_start_time" in frame.columns:
        frame = frame.copy()
        frame["_file_start_time_sort"] = pd.to_datetime(
            frame["file_start_time"], errors="coerce"
        )
        keys.append("_file_start_time_sort")
    if "global_cycle" in frame.columns:
        keys.append("global_cycle")
    if "visit" in frame.columns and "visit" not in keys:
        keys.append("visit")
    sorted_frame = frame.sort_values(keys, kind="mergesort").reset_index(drop=True)
    if "_file_start_time_sort" in sorted_frame.columns:
        sorted_frame = sorted_frame.drop(columns=["_file_start_time_sort"])
    return sorted_frame


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


def _compute_stress_alignment(frame: pd.DataFrame, raw_stress_column: str,
                              *, epsilon: float = 1e-6,
                              require_known_forecast_horizon: bool = True
                              ) -> pd.DataFrame:
    """Attach physically-aligned ``stress_current``, ``stress_next``,
    ``stress_delta``.

    For anchor k in chronological order within a cell:
        stress_next[k]    = raw_stress[k]                    (end of current CYC block)
        stress_current[k] = raw_stress[k-1]  or  0 for k=0   (previous RPT time)
        stress_delta[k]   = stress_next[k] - stress_current[k]

    Any nonpositive ``stress_delta`` triggers a :class:`StressCoordinateError`
    because the current CYC block must accumulate strictly positive exposure.
    """
    out = frame.copy()
    stress_current = np.zeros(len(out), dtype=float)
    stress_next = np.zeros(len(out), dtype=float)

    for cell, sub in out.groupby("cell", sort=False):
        indices = sub.index.to_numpy()
        raw = sub[raw_stress_column].to_numpy(dtype=float)
        if np.isnan(raw).any():
            raise StressCoordinateError(
                f"cell {cell} still contains NaN raw stress after causal fill"
            )
        stress_next[indices] = raw
        # shift previous end-of-block downward by one row within the cell;
        # first anchor's stress_current defaults to 0 (initial RPT is at cell start).
        stress_current[indices[0]] = 0.0
        if len(indices) > 1:
            stress_current[indices[1:]] = raw[:-1]

    stress_delta = stress_next - stress_current
    if (stress_delta < 0).any():
        bad = out.loc[stress_delta < 0, ["cell"]].head(5).to_dict(orient="records")
        raise StressCoordinateError(
            f"negative stress_delta encountered after alignment: {bad}"
        )
    if (stress_delta <= 0).any():
        # A zero-length transition means the current CYC block accumulated no
        # exposure — the sample is not informative and would collapse the PDE.
        zeros = out.loc[stress_delta <= 0, ["cell"]].head(5).to_dict(orient="records")
        raise StressCoordinateError(
            f"nonpositive stress_delta encountered after alignment: {zeros}"
        )
    out["stress_current"] = stress_current
    out["stress_next"] = stress_next
    out["stress_delta"] = stress_delta
    if require_known_forecast_horizon and not np.all(np.isfinite(stress_delta)):
        raise ForecastHorizonError(
            "require_known_forecast_horizon=True but some stress_delta values are "
            "nonfinite; the retrospective block-to-RPT task requires every anchor "
            "to have a determined current-block exposure."
        )
    return out


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------

def build_anchor_dataset(preprocessing_frame: pd.DataFrame, *,
                         preprocessing: str,
                         stress_cfg: Mapping[str, Any],
                         audit_cfg: Mapping[str, Any] | None = None,
                         audit_path: Path | None = None,
                         provenance_manifest: pd.DataFrame | None = None,
                         features_dir: Path | None = None) -> AnchorDataset:
    """Filter ``preprocessing_frame`` to eligible anchors and audit features.

    ``stress_cfg`` corresponds to ``config.pinn.stress`` and
    ``audit_cfg`` corresponds to ``config.pinn.audit`` (may be ``None``).
    """
    if "is_prediction_anchor" not in preprocessing_frame.columns:
        raise ValueError("preprocessing frame is missing 'is_prediction_anchor'")
    anchors = preprocessing_frame[
        preprocessing_frame["is_prediction_anchor"].fillna(False)
    ].copy()
    anchors = anchors.dropna(subset=["next_rpt_Q_Ah", "initial_rpt_Q_Ah"]).copy()
    anchors = anchors[anchors["initial_rpt_Q_Ah"] > 0].copy()
    if anchors.empty:
        raise ValueError(f"no eligible anchor rows in preprocessing '{preprocessing}'")

    # Chronological sort per cell — mandatory before any per-cell temporal op.
    anchors = _chronological_sort(anchors)

    # Q_current: capacity measured at the RPT preceding the anchor cycle.
    if "previous_rpt_Q_Ah" in anchors.columns:
        anchors["q_current_Ah"] = anchors["previous_rpt_Q_Ah"].fillna(
            anchors["initial_rpt_Q_Ah"]
        )
    else:
        anchors["q_current_Ah"] = anchors["initial_rpt_Q_Ah"]

    # Normalized capacities.
    anchors["u_current"] = (
        anchors["q_current_Ah"].astype(float)
        / anchors["initial_rpt_Q_Ah"].astype(float)
    )
    anchors["u_true_next"] = (
        anchors["next_rpt_Q_Ah"].astype(float)
        / anchors["initial_rpt_Q_Ah"].astype(float)
    )

    # Stress coordinate: causal forward-fill only (no bfill!).
    raw_stress_column = _select_stress_column(anchors, stress_cfg)
    if anchors[raw_stress_column].isna().any():
        anchors[raw_stress_column] = anchors.groupby("cell")[raw_stress_column].transform(
            lambda s: s.ffill()
        )
    if anchors[raw_stress_column].isna().any():
        first_bad = anchors.loc[anchors[raw_stress_column].isna(), "cell"].unique()[:5]
        raise StressCoordinateError(
            f"raw stress column {raw_stress_column!r} still NaN after causal "
            f"forward-fill for cells: {list(first_bad)}"
        )
    _ensure_per_cell_monotone(anchors, raw_stress_column)

    epsilon = float(stress_cfg.get("epsilon", 1e-6))
    require_known = bool(stress_cfg.get("require_known_forecast_horizon", True))
    anchors = _compute_stress_alignment(
        anchors, raw_stress_column,
        epsilon=epsilon,
        require_known_forecast_horizon=require_known,
    )

    # ---- Feature columns.
    #
    # For the ``unified`` preprocessing the feature set comes from the fixed
    # registry, not from column discovery. That is what makes the NaPINN-Q
    # feature matrix identical to the one Ridge and ExtraTrees receive: previously
    # the PINN inferred its own candidate list by dtype and then filtered it
    # through a causal allow-list, while the classical pipeline applied its own
    # fold-local variance/correlation pruning, so the two models never saw the
    # same columns.
    #
    # ``stress_coordinate`` and ``initial_state`` are excluded because they enter
    # the model as the differentiable coordinate and the initial condition
    # respectively; duplicating them in the auxiliary tensor is what the revision
    # forbids.
    if preprocessing == UNIFIED_PREPROCESSING:
        from ..preprocessing.unified import build_temporal_audit
        from ..preprocessing.unified_feature_registry import (
            model_feature_columns, stress_coordinate_column,
        )

        expected_stress = stress_coordinate_column()
        if raw_stress_column != expected_stress:
            raise StressCoordinateError(
                f"unified registry declares {expected_stress!r} as the stress "
                f"coordinate but the dataset selected {raw_stress_column!r}; "
                f"align config.pinn.stress.preferred_coordinate with the registry"
            )
        allowed = model_feature_columns(
            exclude_roles=("stress_coordinate", "initial_state"),
        )
        missing = [c for c in allowed if c not in anchors.columns]
        if missing:
            raise ValueError(
                f"unified.parquet is missing registered feature columns: {missing}"
            )
        audit = build_temporal_audit(anchors)
        # Reshape to the columns downstream audit consumers expect.
        audit = audit.assign(
            preprocessing=preprocessing,
            status="allowed",
            allowed_as_input=audit["feature"].isin(allowed),
            known_at_anchor=True,
            planned_at_inference=True,
            future_derived=False,
            rejection_reason="",
            provenance_source="unified_feature_registry",
        )
    else:
        numeric = anchors.select_dtypes(include=[np.number]).columns.tolist()
        identifiers = set(IDENTIFIER_COLUMNS) | {
            "initial_rpt_Q_Ah", "previous_rpt_Q_Ah", "previous_rpt_SOH_pct",
            "q_current_Ah", "u_current", "u_true_next",
            "stress_current", "stress_delta", "stress_next",
            # Exclude the raw stress column so it does not appear both as the
            # model's stress input AND inside the feature matrix.
            raw_stress_column,
        }
        candidate_columns = [
            c for c in numeric if c not in identifiers and c not in TARGETS
        ]

        allow_unaudited = bool((audit_cfg or {}).get("allow_unaudited", False))
        if provenance_manifest is None and features_dir is not None:
            provenance_manifest = load_provenance_manifest(
                preprocessing, features_dir=features_dir,
            )
        audit = build_temporal_feature_audit(
            candidate_columns, preprocessing=preprocessing,
            allow_unaudited=allow_unaudited,
            provenance=provenance_manifest,
        )
        allowed = audit[audit["allowed_as_input"]]["feature"].tolist()

    if audit_path is not None:
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit.to_csv(audit_path, index=False)

    if not allowed:
        raise TemporalLeakageError(
            "no leakage-safe feature columns remain after temporal audit; "
            "set pinn.audit.allow_unaudited=true to permit unregistered features"
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


def inner_fold_indices(outer_train_frame: pd.DataFrame, *,
                       n_inner: int = 3,
                       seed: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    """Cell-grouped k-fold partition of the outer-training rows.

    Returns ``n_inner`` ``(inner_train, inner_val)`` index pairs whose
    validation halves are disjoint and together cover every outer-training cell
    exactly once. This is the primitive for nested selection: choosing the epoch
    count on a single split is unstable when the validation half holds only four
    or five cells, which is the regime here (26 development cells, 5 outer
    folds).

    ``seed`` controls only the shuffle of cells before they are dealt into
    folds, and is deliberately decoupled from the model-training seed so the
    inner partition can be held fixed while seeds vary.
    """
    cells = sorted(outer_train_frame["cell"].astype(str).unique())
    if len(cells) < 2:
        raise ValueError("need at least two cells for inner validation")
    n_folds = int(max(2, min(n_inner, len(cells))))
    shuffled = np.random.default_rng(seed).permutation(cells)
    # Deal cells round-robin so fold sizes differ by at most one.
    fold_of_cell = {cell: i % n_folds for i, cell in enumerate(shuffled)}
    outer_cells = outer_train_frame["cell"].astype(str).to_numpy()
    assignment = np.array([fold_of_cell[c] for c in outer_cells])
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for k in range(n_folds):
        val_mask = assignment == k
        train_mask = ~val_mask
        if not train_mask.any() or not val_mask.any():
            continue
        folds.append((np.flatnonzero(train_mask), np.flatnonzero(val_mask)))
    if not folds:
        raise ValueError("could not build any usable inner fold")
    return folds


def inner_split_indices(outer_train_frame: pd.DataFrame, *,
                        n_inner: int = 3,
                        seed: int = 0,
                        inner_fold: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """One ``(inner_train, inner_val)`` pair from the grouped k-fold partition.

    ``inner_fold`` selects which fold of :func:`inner_fold_indices` to return.
    This used to be a single group-*shuffle* split, so repeated calls could
    never cover the outer-training cells systematically; it is now fold ``k`` of
    a proper partition, which keeps the existing single-split callers working
    while making nested selection expressible.
    """
    folds = inner_fold_indices(outer_train_frame, n_inner=n_inner, seed=seed)
    return folds[int(inner_fold) % len(folds)]


# ---------------------------------------------------------------------------
# Fold-local scaling
# ---------------------------------------------------------------------------

@dataclass
class FoldScaler:
    """Fold-local imputation + scaler.

    Estimated on the *inner-training* subset only (never the inner-validation
    or outer-validation folds — Stage 2 diagnosis fix #12).

    Stress interval normalization
    -----------------------------
    A stress increment Δs is dimensionally the difference of two stress values;
    subtracting the mean of Δs would produce arbitrary negative "normalized"
    intervals. We therefore scale only:

        stress_normalized       = (s - stress_mean) / stress_std
        stress_delta_normalized = Δs / stress_std

    (Stage 1 diagnosis fix #4.)
    """

    feature_medians: np.ndarray | None = None
    feature_mean: np.ndarray | None = None
    feature_std: np.ndarray | None = None
    feature_columns_used: list[str] | None = None
    dropped_all_nan_columns: list[str] = None  # type: ignore[assignment]
    stress_mean: float = 0.0
    stress_std: float = 1.0
    u_max: float = 1.0
    epsilon_rec: float = 0.02
    #: When set, the feature block is delegated to the shared
    #: UnifiedFoldPreprocessor so NaPINN-Q and the classical models apply
    #: byte-identical imputation and scaling to identical columns.
    unified: Any | None = None

    def fit(self, frame: pd.DataFrame, feature_columns: list[str], *,
            u_max_source: str, u_max_constant: float, u_max_margin: float,
            tolerance_source: str, tolerance_constant: float,
            tolerance_quantile: float,
            use_unified: bool = False) -> "FoldScaler":
        if use_unified:
            # Registry-driven path. No fold-local column dropping: a feature
            # that happens to be entirely missing on one fold keeps its slot
            # (imputed to the fold median, or 0.0 when undefined) so the matrix
            # width and column order are fold-invariant and identical to the
            # classical pipeline's. Dropping columns per fold, as the legacy
            # path does, silently changes feature_dim between folds.
            from ..preprocessing.fold_transform import UnifiedFoldPreprocessor

            preprocessor = UnifiedFoldPreprocessor(
                exclude_roles=("stress_coordinate", "initial_state"),
                scale=True,
            )
            # Restrict to the requested columns while preserving registry order.
            expected = preprocessor.expected_columns()
            if list(feature_columns) != expected:
                raise ValueError(
                    "unified feature columns disagree with the registry order; "
                    f"got {len(feature_columns)} columns, expected {len(expected)}"
                )
            preprocessor.fit(frame)
            self.unified = preprocessor
            self.feature_columns_used = list(expected)
            self.dropped_all_nan_columns = []
        else:
            # Drop all-NaN columns fold-locally (legacy P1/P2/P3 path).
            features_raw = frame[feature_columns].to_numpy(dtype=float)
            all_nan = np.isnan(features_raw).all(axis=0)
            used_columns = [c for c, drop in zip(feature_columns, all_nan) if not drop]
            dropped = [c for c, drop in zip(feature_columns, all_nan) if drop]
            self.feature_columns_used = list(used_columns)
            self.dropped_all_nan_columns = list(dropped)
            if not used_columns:
                raise ValueError(
                    "all candidate feature columns are entirely NaN on this fold"
                )
            features = frame[used_columns].to_numpy(dtype=float)
            self.feature_medians = np.nanmedian(features, axis=0)
            # No column can still be all-NaN here because we dropped those
            # above, so feature_medians has no NaN entries.
            replaced = np.where(np.isnan(features), self.feature_medians, features)
            self.feature_mean = replaced.mean(axis=0)
            self.feature_std = replaced.std(axis=0) + 1e-8

        stress = frame["stress_current"].to_numpy(dtype=float)
        stress_next = frame["stress_next"].to_numpy(dtype=float)
        pool = np.concatenate([stress, stress_next])
        self.stress_mean = float(np.mean(pool))
        self.stress_std = float(np.std(pool) + 1e-8)

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
        # Callers pass the original candidate list; we honour the fold-local
        # ``feature_columns_used`` chosen at fit-time to guarantee shape stability.
        del feature_columns  # ignored — we always use fit-time selection
        if self.feature_columns_used is None:
            raise RuntimeError("FoldScaler.transform_features called before fit")
        if self.unified is not None:
            return self.unified.transform(frame)
        features = frame[self.feature_columns_used].to_numpy(dtype=float)
        replaced = np.where(np.isnan(features), self.feature_medians, features)
        return (replaced - self.feature_mean) / self.feature_std

    def feature_hash(self) -> str:
        """Hash of the exact feature contract, recorded per fold."""
        import hashlib
        import json

        if self.unified is not None:
            return self.unified.feature_hash()
        payload = json.dumps(
            {"feature_columns": list(self.feature_columns_used or [])},
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def transform_stress(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=float) - self.stress_mean) / self.stress_std

    def transform_horizon(self, values: np.ndarray) -> np.ndarray:
        # An interval is dimensionally (s2 - s1); centering it would be wrong.
        return np.asarray(values, dtype=float) / self.stress_std

    def state_dict(self) -> dict[str, Any]:
        return {
            "feature_columns_used": list(self.feature_columns_used or []),
            "dropped_all_nan_columns": list(self.dropped_all_nan_columns or []),
            "feature_medians": None if self.feature_medians is None else self.feature_medians.tolist(),
            "feature_mean": None if self.feature_mean is None else self.feature_mean.tolist(),
            "feature_std": None if self.feature_std is None else self.feature_std.tolist(),
            "stress_mean": self.stress_mean,
            "stress_std": self.stress_std,
            "u_max": self.u_max,
            "epsilon_rec": self.epsilon_rec,
            "unified": None if self.unified is None else self.unified.state_dict(),
            "feature_hash": self.feature_hash() if self.feature_columns_used else None,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "FoldScaler":
        obj = cls()
        unified_state = state.get("unified")
        if unified_state:
            from ..preprocessing.fold_transform import UnifiedFoldPreprocessor

            obj.unified = UnifiedFoldPreprocessor.from_state_dict(unified_state)
        obj.feature_columns_used = list(state.get("feature_columns_used") or [])
        obj.dropped_all_nan_columns = list(state.get("dropped_all_nan_columns") or [])
        obj.feature_medians = None if state["feature_medians"] is None else np.asarray(state["feature_medians"])
        obj.feature_mean = None if state["feature_mean"] is None else np.asarray(state["feature_mean"])
        obj.feature_std = None if state["feature_std"] is None else np.asarray(state["feature_std"])
        obj.stress_mean = float(state["stress_mean"])
        obj.stress_std = float(state["stress_std"])
        obj.u_max = float(state["u_max"])
        obj.epsilon_rec = float(state["epsilon_rec"])
        return obj
