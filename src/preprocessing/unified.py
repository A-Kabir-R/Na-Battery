"""The single ``unified`` preprocessing pipeline.

One anchor-level feature table, consumed unchanged by Ridge, ExtraTrees and
NaPINN-Q. Replaces the P1/P2/P3 ladder for all final training; the older modules
remain importable for traceability only.

Structure
---------
1. Cycle-level base: validated complete cycles, safe derived quantities, EFC.
2. Targets attached on the cycle frame, which marks the prediction anchors
   (exactly one per cycling block: its last complete cycle).
3. Anchor-level assembly. Because there are only ~236 anchors, every expensive
   feature is computed *at the anchors only*:
     * Group C temporal summaries over the trailing complete cycles of the cell,
     * Group D/E/F curve + thermal features from the raw ``.ird`` waveforms of
       the anchor's own block (first vs last complete cycle),
     * Group G resistance joined from the *previous* RPT visit.
4. Derived interactions, registry projection, audit artifacts.

Causality
---------
Every feature is a function of data recorded at or before the anchor cycle. The
anchor is the last complete cycle of an already-finished block, so the block's
own first cycle is a legitimate causal reference. No future cycle, no future
RPT, no centered window, no whole-dataset statistic is used. The temporal audit
in :func:`build_temporal_audit` records the reference window of every column and
:func:`assert_causal` fails the build if a column cannot be proven causal.
"""
from __future__ import annotations

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from ..io.ird_parser import parse_ird
from ..io.loaders import (
    load_config,
    load_inventory,
    load_rpt_measurements,
    load_steps_table,
)
from .curve_features import (
    delta_q_of_v,
    discharge_qv_curve,
    icdv_reference_shift,
    safe_divide,
    signed_dqdv_peaks,
    thermal_features,
)
from .dcir import previous_rpt_dcir
from .p1_aggregate import build_p1
from .protocol_segmentation import segment_cyc_protocol, summarize_steps
from .unified_feature_registry import (
    CAUSAL_REFERENCE_WINDOWS,
    FEATURE_SPECS,
    IDENTITY_COLUMNS,
    MIN_TEMPORAL_OBSERVATIONS,
    REGISTRY_VERSION,
    TARGET_COLUMNS,
    continuous_feature_columns,
    flag_feature_columns,
    model_feature_columns,
    registry_hash,
    registry_manifest,
)

PREPROCESSING_NAME = "unified"

#: Trailing window, in complete cycles, for Group C features.
TEMPORAL_WINDOW = 10


class UnifiedBuildError(RuntimeError):
    """Raised when the unified table cannot be built safely."""


# ---------------------------------------------------------------------------
# Cycle-level base
# ---------------------------------------------------------------------------
def build_cycle_base(cycle_metrics: pd.DataFrame | None = None) -> pd.DataFrame:
    """Complete-cycle base frame with safe derived quantities and EFC."""
    cfg = load_config()
    nominal = float(cfg["protocol"]["nominal_capacity_Ah"])
    base = build_p1(cycle_metrics)

    base["energy_loss_Wh"] = base["E_charge_Wh"] - base["E_discharge_Wh"]
    base["mean_charge_voltage_V"] = [
        safe_divide(e, q)
        for e, q in zip(base["E_charge_Wh"], base["Q_charge_Ah"])
    ]
    base["mean_discharge_voltage_V"] = [
        safe_divide(e, q)
        for e, q in zip(base["E_discharge_Wh"], base["Q_discharge_Ah"])
    ]
    base["voltage_hysteresis_V"] = (
        base["mean_charge_voltage_V"] - base["mean_discharge_voltage_V"]
    )
    base["temperature_span_C"] = base["T_max"] - base["T_min"]
    base["cycle_duration_s"] = base["duration_s"]

    # EFC accumulates only past discharge throughput within the cell.
    base = base.sort_values(["condition", "cell", "visit", "cycle_in_file"])
    base["EFC_cum"] = (
        base.groupby(["condition", "cell"])["Q_discharge_Ah"]
        .transform(lambda s: s.fillna(0.0).cumsum())
        / nominal
    )
    return base.replace([np.inf, -np.inf], np.nan).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Group C — temporal summaries evaluated at the anchors only
# ---------------------------------------------------------------------------
def _robust_slope(values: np.ndarray) -> float:
    """Theil-Sen slope (median of pairwise slopes) against cycle position.

    Robust to the occasional capacity-regeneration outlier, and cheap for the
    10-point windows used here.
    """
    y = np.asarray(values, dtype=float)
    finite = np.isfinite(y)
    y = y[finite]
    if y.size < 2:
        return float("nan")
    x = np.arange(y.size, dtype=float)
    slopes: list[float] = []
    for i in range(y.size - 1):
        dx = x[i + 1:] - x[i]
        slopes.extend(((y[i + 1:] - y[i]) / dx).tolist())
    return float(np.median(slopes)) if slopes else float("nan")


def temporal_features_at_anchor(history: pd.DataFrame, *,
                                window: int = TEMPORAL_WINDOW) -> dict[str, float]:
    """Group C features from the trailing ``window`` complete cycles.

    ``history`` must already be restricted to cycles at or before the anchor for
    a single cell, in chronological order, with the anchor as the final row.
    """
    tail = history.tail(int(window))
    n_valid = int(tail["Q_discharge_Ah"].notna().sum())
    available = float(n_valid >= MIN_TEMPORAL_OBSERVATIONS)

    if not available:
        return {
            "Q_discharge_Ah_r10_slope": np.nan,
            "Q_discharge_Ah_r10_std": np.nan,
            "Q_discharge_Ah_dev_r10": np.nan,
            "coulombic_efficiency_r10_mean": np.nan,
            "coulombic_efficiency_r10_slope": np.nan,
            "temporal_window_available": 0.0,
        }

    q = tail["Q_discharge_Ah"]
    ce = tail["coulombic_efficiency"]
    anchor_q = float(history["Q_discharge_Ah"].iloc[-1])
    rolling_mean = float(q.mean())
    return {
        "Q_discharge_Ah_r10_slope": _robust_slope(q.to_numpy()),
        "Q_discharge_Ah_r10_std": float(q.std(ddof=1)) if n_valid >= 2 else np.nan,
        "Q_discharge_Ah_dev_r10": (
            anchor_q - rolling_mean if np.isfinite(anchor_q) else np.nan
        ),
        "coulombic_efficiency_r10_mean": float(ce.mean()),
        "coulombic_efficiency_r10_slope": _robust_slope(ce.to_numpy()),
        "temporal_window_available": 1.0,
    }


def build_temporal_block(base: pd.DataFrame, anchors: pd.DataFrame, *,
                         window: int = TEMPORAL_WINDOW) -> pd.DataFrame:
    """Group C features for every anchor, using only that cell's past cycles."""
    ordered = base.sort_values(["condition", "cell", "visit", "cycle_in_file"])
    rows: list[dict[str, Any]] = []
    for cell_key, history in ordered.groupby(["condition", "cell"], sort=False):
        cell_anchors = anchors[
            anchors["condition"].eq(cell_key[0]) & anchors["cell"].eq(cell_key[1])
        ]
        for anchor in cell_anchors.itertuples(index=False):
            # Strictly at-or-before the anchor cycle within this cell.
            past = history[
                (history["visit"] < anchor.visit)
                | (
                    history["visit"].eq(anchor.visit)
                    & history["cycle_in_file"].le(anchor.cycle_in_file)
                )
            ]
            if past.empty:
                continue
            features = temporal_features_at_anchor(past, window=window)
            features["anchor_id"] = anchor.anchor_id
            rows.append(features)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Groups D/E/F — waveform features from the anchor's own cycling block
# ---------------------------------------------------------------------------
def _block_waveform_features(path: Path, row: dict[str, Any],
                             cfg: dict[str, Any]) -> dict[str, Any]:
    """Delta-Q(V), IC/DV shift and thermal features for one cycling file.

    Only two cycles are needed: the block's first complete cycle (causal
    reference) and its last complete cycle (the prediction anchor).
    """
    protocol = cfg["protocol"]
    preprocessing = cfg["preprocessing"]
    parsed = parse_ird(
        path,
        strict=True,
        max_invalid_fraction=float(protocol["max_invalid_fraction"]),
        max_sampling_gap_s=float(protocol["max_sampling_gap_s"]),
    )
    if parsed.data is None:
        raise UnifiedBuildError(f"parse returned no data for {path}")
    data = parsed.data.copy()
    for column, limits in protocol["physical_bounds"].items():
        invalid = (data[column] < limits[0]) | (data[column] > limits[1])
        data.loc[invalid, column] = np.nan

    steps = summarize_steps(
        data, current_deadband_A=float(protocol["current_deadband_A"])
    )
    nominal = float(protocol["nominal_capacity_Ah"])
    segments = segment_cyc_protocol(
        data,
        expected_charge_current_A=nominal * float(row["C_ch"]),
        expected_discharge_current_A=nominal * float(row["C_dis"]),
        current_relative_tolerance=float(protocol["cycling_current_relative_tolerance"]),
        minimum_phase_capacity_Ah=float(protocol["minimum_phase_capacity_Ah"]),
        capacity_ratio_bounds=tuple(protocol["capacity_ratio_bounds"]),
        minimum_cycle_voltage_span_V=float(
            protocol.get("minimum_cycle_voltage_span_V", 0.0)
        ),
        minimum_phase_voltage_span_V=float(
            protocol.get("minimum_phase_voltage_span_V", 0.0)
        ),
        steps=steps,
    )
    complete = [
        segment for segment in segments
        if segment.complete
        and segment.discharge_step_index is not None
        and segment.charge_step_index is not None
    ]
    if len(complete) < 2:
        raise UnifiedBuildError(
            f"{path.name}: need >=2 complete cycles, found {len(complete)}"
        )
    complete.sort(key=lambda segment: segment.cycle_in_file)
    reference_segment, anchor_segment = complete[0], complete[-1]

    def phase_frames(segment) -> tuple[pd.DataFrame, pd.DataFrame]:
        discharge_step = steps[segment.discharge_step_index]
        charge_step = steps[segment.charge_step_index]
        return (
            data.iloc[discharge_step.start:discharge_step.stop],
            data.iloc[charge_step.start:charge_step.stop],
        )

    anchor_discharge, anchor_charge = phase_frames(anchor_segment)
    reference_discharge, _ = phase_frames(reference_segment)

    sigma = float(preprocessing["dqdv_smooth_sigma"])
    n_grid = int(preprocessing.get("dqv_grid_points",
                                   preprocessing["vq_resample_points"]))

    v_anchor, q_anchor = discharge_qv_curve(
        anchor_discharge["Voltage"].to_numpy(),
        anchor_discharge["Ah_counter"].to_numpy(),
    )
    v_reference, q_reference = discharge_qv_curve(
        reference_discharge["Voltage"].to_numpy(),
        reference_discharge["Ah_counter"].to_numpy(),
    )

    features: dict[str, Any] = {}
    features.update(delta_q_of_v(
        v_anchor, q_anchor, v_reference, q_reference, n_points=n_grid,
    ))
    anchor_peaks = signed_dqdv_peaks(v_anchor, q_anchor, sigma=sigma)
    reference_peaks = signed_dqdv_peaks(v_reference, q_reference, sigma=sigma)
    features.update(icdv_reference_shift(anchor_peaks, reference_peaks))
    features["icdv_dis_peak1_V"] = anchor_peaks["peak1_V"]
    features["icdv_dis_peak1_fwhm_V"] = anchor_peaks["peak1_fwhm_V"]

    q_dis = anchor_discharge["Ah_counter"].to_numpy(dtype=float)
    features.update(thermal_features(
        discharge_time=anchor_discharge["Time"].to_numpy(dtype=float),
        discharge_temperature=anchor_discharge["Temperature"].to_numpy(dtype=float),
        discharge_capacity=q_dis,
        charge_time=anchor_charge["Time"].to_numpy(dtype=float),
        charge_temperature=anchor_charge["Temperature"].to_numpy(dtype=float),
    ))

    features.update({
        "file_id": row.get("file_id"),
        "condition": row["condition"],
        "cell": row["cell"],
        "visit": int(row["visit"]),
        "waveform_reference_cycle_in_file": int(reference_segment.cycle_in_file),
        "waveform_anchor_cycle_in_file": int(anchor_segment.cycle_in_file),
        "waveform_complete_cycle_count": int(len(complete)),
    })
    return features


def _waveform_worker(row: dict[str, Any]) -> dict[str, Any]:
    cfg = load_config()
    path = Path(row["path"])
    try:
        return {
            "status": "valid", "file_id": row.get("file_id"),
            "error": None, "features": _block_waveform_features(path, row, cfg),
        }
    except Exception as exc:  # noqa: BLE001 - recorded per file, never silent
        return {
            "status": "failed", "file_id": row.get("file_id"),
            "error": f"{type(exc).__name__}: {exc}", "features": None,
        }


def build_waveform_block(anchor_file_ids: set[str], *,
                         workers: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Waveform features for every cycling file that carries an anchor."""
    cfg = load_config()
    workers = workers or int(cfg["experiment"]["parallel_workers"])
    inventory = load_inventory()
    tasks = inventory[
        inventory["test_type"].eq("CYC")
        & inventory["file_id"].isin(anchor_file_ids)
        & inventory.get("parse_status", "valid").eq("valid")
    ].to_dict(orient="records")
    if not tasks:
        raise UnifiedBuildError("no parseable cycling files carry an anchor")

    rows: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        iterator = executor.map(_waveform_worker, tasks, chunksize=1)
        bar = tqdm(iterator, total=len(tasks), desc="[unified] waveforms", unit="file")
        for result in bar:
            coverage.append({
                "file_id": result["file_id"],
                "status": result["status"],
                "error": result["error"],
            })
            if result["features"] is not None:
                rows.append(result["features"])
            failures = sum(entry["status"] != "valid" for entry in coverage)
            bar.set_postfix_str(f"ok={len(rows)} failed={failures}")
    return pd.DataFrame(rows), pd.DataFrame(coverage)


# ---------------------------------------------------------------------------
# Derived interactions
# ---------------------------------------------------------------------------
def add_interactions(frame: pd.DataFrame) -> pd.DataFrame:
    """Group H — a small, interpretable interaction set. No polynomials."""
    out = frame.copy()
    out["temperature_K"] = out["T_degC"] + 273.15
    out["inverse_temperature_1_K"] = [
        safe_divide(1.0, value) for value in out["temperature_K"]
    ]
    dod_fraction = out["DOD_pct"] / 100.0
    out["stress_x_DOD_fraction"] = out["EFC_cum"] * dod_fraction
    out["stress_x_inverse_temperature"] = (
        out["EFC_cum"] * out["inverse_temperature_1_K"]
    )
    out["stress_x_C_dis"] = out["EFC_cum"] * out["C_dis"]
    return out


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------
def build_temporal_audit(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-feature causality, coverage and unit audit."""
    rows: list[dict[str, Any]] = []
    n_rows = max(len(frame), 1)
    for spec in FEATURE_SPECS:
        present = spec.name in frame.columns
        if present:
            column = frame[spec.name]
            missing_rate = float(column.isna().mean())
            finite_rate = float(
                np.isfinite(pd.to_numeric(column, errors="coerce")).sum() / n_rows
            )
        else:
            missing_rate = 1.0
            finite_rate = 0.0
        # Derive the causality verdict from the declared reference window and
        # source columns. These were previously hardcoded True/False literals,
        # which made assert_causal's unsafe filter empty by construction and
        # unable to reject anything.
        backward = spec.reference in CAUSAL_REFERENCE_WINDOWS
        target_derived = bool(set(spec.sources) & set(TARGET_COLUMNS))
        reads_next_rpt = target_derived or "next" in spec.reference.lower()
        rows.append({
            "feature": spec.name,
            "group": spec.group,
            "kind": spec.kind,
            "role": spec.role,
            "unit": spec.unit,
            "source_columns": ";".join(spec.sources),
            "reference_window": spec.reference,
            "uses_only_past_or_current": backward,
            "uses_future_cycles": not backward,
            "uses_future_rpt": reads_next_rpt,
            "target_derived": target_derived,
            "present_in_table": present,
            "missingness_rate": missing_rate,
            "finite_rate": finite_rate,
            "enabled_in_registry": True,
            "registry_version": REGISTRY_VERSION,
        })
    return pd.DataFrame(rows)


def assert_causal(audit: pd.DataFrame) -> None:
    """Fail the build when any registered feature cannot be proven causal."""
    missing = audit.loc[~audit["present_in_table"], "feature"].tolist()
    if missing:
        raise UnifiedBuildError(
            f"registered features absent from the unified table: {missing}"
        )
    unsafe = audit[
        audit["uses_future_cycles"]
        | audit["uses_future_rpt"]
        | audit["target_derived"]
        | ~audit["uses_only_past_or_current"]
    ]
    if not unsafe.empty:
        raise UnifiedBuildError(
            f"non-causal features rejected: {unsafe['feature'].tolist()}"
        )
    empty = audit.loc[audit["finite_rate"] <= 0.0, "feature"].tolist()
    if empty:
        raise UnifiedBuildError(
            f"registered features are entirely missing/non-finite: {empty}"
        )


def build_coverage(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-feature coverage, split by condition, for the report."""
    rows: list[dict[str, Any]] = []
    for spec in FEATURE_SPECS:
        if spec.name not in frame.columns:
            continue
        column = frame[spec.name]
        rows.append({
            "feature": spec.name, "group": spec.group, "scope": "all",
            "n_rows": int(len(column)),
            "n_present": int(column.notna().sum()),
            "coverage": float(column.notna().mean()),
        })
        for condition, part in frame.groupby("condition"):
            values = part[spec.name]
            rows.append({
                "feature": spec.name, "group": spec.group, "scope": str(condition),
                "n_rows": int(len(values)),
                "n_present": int(values.notna().sum()),
                "coverage": float(values.notna().mean()),
            })
    return pd.DataFrame(rows)


def table_hash(frame: pd.DataFrame) -> str:
    digest = pd.util.hash_pandas_object(frame, index=True, categorize=True)
    return hashlib.sha256(digest.to_numpy().tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def build_unified(cycle_metrics: pd.DataFrame | None = None,
                  cu_cache: pd.DataFrame | None = None,
                  *, workers: int | None = None,
                  attach_targets=None) -> dict[str, Any]:
    """Build the unified anchor table and its audit artifacts."""
    cfg = load_config()
    nominal = float(cfg["protocol"]["nominal_capacity_Ah"])
    initial_visit = int(
        cfg.get("targets", {}).get("next_rpt", {}).get("initial_visit", 0)
    )

    if attach_targets is None:
        from ..targets.next_rpt import add_target as add_next_rpt

        def attach_targets(frame: pd.DataFrame) -> pd.DataFrame:
            return add_next_rpt(frame, cu_cache=cu_cache)

    base = build_cycle_base(cycle_metrics)
    with_targets = attach_targets(base)
    anchors = with_targets[
        with_targets["is_prediction_anchor"].fillna(False)
    ].copy().reset_index(drop=True)
    if anchors.empty:
        raise UnifiedBuildError("no prediction anchors were identified")
    if anchors["anchor_id"].duplicated().any():
        raise UnifiedBuildError("anchor_id is not unique")

    # --- Group C
    temporal = build_temporal_block(base, anchors)
    anchors = anchors.merge(temporal, on="anchor_id", how="left", validate="one_to_one")

    # --- Groups D/E/F
    waveform, waveform_coverage = build_waveform_block(
        set(anchors["file_id"].dropna()), workers=workers,
    )
    join_columns = [
        column for column in waveform.columns
        if column not in {"condition", "cell", "visit"}
    ]
    anchors = anchors.merge(
        waveform[join_columns], on="file_id", how="left", validate="one_to_one",
    )
    for flag in ("dqv_available", "icdv_available"):
        anchors[flag] = anchors[flag].fillna(0.0)

    # --- Group G: previous-RPT resistance only
    steps = load_steps_table()
    resistance = previous_rpt_dcir(
        steps[steps["test_type"].ne("CYC")],
        cu_cache if cu_cache is not None else load_rpt_measurements(),
        nominal_capacity_Ah=nominal,
        initial_visit=initial_visit,
    )
    # Join on the previous RPT's file identity, not its visit label. Visit
    # labels are not unique (five cells carry two files both labelled V16), so a
    # visit join deduplicates distinct files and binds anchors to whichever file
    # happened to sort first -- stale in the observed data, and the target RPT
    # itself under a different file ordering.
    resistance = resistance.rename(columns={"path": "previous_rpt_path"})[[
        "previous_rpt_path", "dcir_discharge_ohm",
        "dcir_delta_from_baseline_ohm", "dcir_available",
    ]].drop_duplicates("previous_rpt_path")
    anchors = anchors.merge(
        resistance, on="previous_rpt_path", how="left", validate="many_to_one",
    )
    anchors = anchors.rename(columns={
        "dcir_discharge_ohm": "dcir_prev_rpt_discharge_ohm",
        "dcir_delta_from_baseline_ohm": "dcir_prev_rpt_delta_from_baseline_ohm",
    })
    anchors["dcir_available"] = anchors["dcir_available"].fillna(0.0)

    # --- Group H + registry aliases
    anchors["visit_index"] = anchors["visit"].astype(float)
    anchors = add_interactions(anchors)
    anchors = anchors.replace([np.inf, -np.inf], np.nan)

    # --- Project onto the registry
    features = model_feature_columns()
    absent = [column for column in features if column not in anchors.columns]
    if absent:
        raise UnifiedBuildError(
            f"unified build did not produce registered features: {absent}"
        )
    for flag in flag_feature_columns():
        anchors[flag] = anchors[flag].fillna(0.0).astype(float)

    identity = [c for c in IDENTITY_COLUMNS if c in anchors.columns]
    targets = [c for c in TARGET_COLUMNS if c in anchors.columns]
    extras = [
        column for column in (
            "previous_rpt_visit", "next_rpt_target_status",
            "target_unavailable_reason", "waveform_reference_cycle_in_file",
            "waveform_anchor_cycle_in_file", "waveform_complete_cycle_count",
            "dqv_overlap_span_V", "dqv_valid_points",
        )
        if column in anchors.columns and column not in identity + targets + features
    ]
    unified = anchors[identity + targets + features + extras].copy()

    audit = build_temporal_audit(unified)
    assert_causal(audit)
    coverage = build_coverage(unified)

    manifest = {
        "preprocessing": PREPROCESSING_NAME,
        "registry_version": REGISTRY_VERSION,
        "registry_hash": registry_hash(),
        "n_model_features": len(features),
        "n_continuous_features": len(continuous_feature_columns()),
        "n_flag_features": len(flag_feature_columns()),
        "n_anchors": int(len(unified)),
        "n_cells": int(unified["cell"].nunique()),
        "n_conditions": int(unified["condition"].nunique()),
        "primary_target": "next_rpt_Q_Ah",
        "temporal_window_cycles": TEMPORAL_WINDOW,
        "minimum_temporal_observations": MIN_TEMPORAL_OBSERVATIONS,
        "dqv_reference": "block_first_complete_cycle",
        "dcir_source": "previous_rpt_0.5C_discharge_pulse_12s",
        "relaxation_features": "not_implemented_protocol_scoped_out",
        "feature_columns": features,
        "table_hash": table_hash(unified),
        "features": registry_manifest(),
    }
    return {
        "unified": unified,
        "audit": audit,
        "coverage": coverage,
        "manifest": manifest,
        "waveform_coverage": waveform_coverage,
    }


def write_unified(result: dict[str, Any], out_dir: Path) -> None:
    """Persist unified.parquet and every required audit artifact."""
    out_dir.mkdir(parents=True, exist_ok=True)
    result["unified"].to_parquet(out_dir / "unified.parquet", index=False)
    result["audit"].to_csv(out_dir / "unified_temporal_audit.csv", index=False)
    result["coverage"].to_csv(out_dir / "unified_feature_coverage.csv", index=False)
    pd.DataFrame(registry_manifest()).to_csv(
        out_dir / "unified_feature_manifest.csv", index=False
    )
    (out_dir / "unified_feature_manifest.json").write_text(
        json.dumps(result["manifest"], indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    coverage = result.get("waveform_coverage")
    if coverage is not None and not coverage.empty:
        coverage.to_csv(out_dir / "unified_waveform_coverage.csv", index=False)
