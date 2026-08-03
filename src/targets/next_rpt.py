"""Leakage-safe next-RPT reference-capacity target construction."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..io.ird_parser import parse_ird
from ..io.loaders import load_config, load_inventory
from ..preprocessing.protocol_segmentation import extract_reference_discharge, summarize_steps

TARGET = "next_rpt_Q_Ah"
SOH_TARGET = "next_rpt_SOH_pct"
DELTA_Q_TARGET = "delta_next_rpt_Q_Ah"
DELTA_SOH_TARGET = "delta_next_rpt_SOH_pct"


class DuplicateRPTError(ValueError):
    """Raised when same-visit RPT files cannot be ordered unambiguously."""


def _cu_capacity_by_visit(inv: pd.DataFrame) -> pd.DataFrame:
    """Extract one validated full-range 0.5C discharge from each CU file.

    The returned cache retains invalid files and their QC status. Consumers
    must use only ``rpt_qc_status == 'valid'`` rows.
    """
    cfg = load_config()
    protocol = cfg["protocol"]
    required = {"cell", "visit", "path", "test_type"}
    missing = required - set(inv.columns)
    if missing:
        raise ValueError(f"inventory is missing columns: {sorted(missing)}")
    cu = inv.loc[inv["test_type"] == "CU"].copy()
    rows: list[dict[str, Any]] = []
    for _, row in cu.iterrows():
        parsed = parse_ird(
            Path(row["path"]),
            strict=False,
            max_invalid_fraction=float(protocol["max_invalid_fraction"]),
            max_sampling_gap_s=float(protocol["max_sampling_gap_s"]),
        )
        base = {
            "condition": row.get("condition"),
            "cell": row["cell"],
            "cu_visit": int(row["visit"]),
            "cu_path": str(row["path"]),
            "cu_test_id": int(row["test_id"]) if "test_id" in row and pd.notna(row["test_id"]) else np.nan,
            "cu_start_time": parsed.metadata.get("starttime") if parsed.metadata else pd.NaT,
        }
        if parsed.data is None or not parsed.qc.get("parse_ok"):
            rows.append({
                **base,
                "cu_Q_Ah": np.nan,
                "rpt_qc_status": "parse_failed",
                "rpt_qc_flags": parsed.qc.get("parse_error_code", "parse_failed"),
            })
            continue
        try:
            steps = summarize_steps(
                parsed.data,
                current_deadband_A=float(protocol["current_deadband_A"]),
            )
            reference = extract_reference_discharge(
                parsed.data,
                reference_current_A=float(protocol["rpt_reference_current_A"]),
                current_relative_tolerance=float(protocol["rpt_current_relative_tolerance"]),
                charged_voltage_min_V=float(protocol["rpt_charged_voltage_min_V"]),
                discharged_voltage_max_V=float(protocol["rpt_discharged_voltage_max_V"]),
                capacity_bounds_Ah=tuple(protocol["rpt_capacity_bounds_Ah"]),
                steps=steps,
            )
            rows.append({
                **base,
                "cu_Q_Ah": reference.capacity_Ah,
                "rpt_qc_status": reference.status,
                "rpt_qc_flags": ";".join(reference.qc_flags),
            })
        except Exception as exc:
            rows.append({
                **base,
                "cu_Q_Ah": np.nan,
                "rpt_qc_status": "extraction_failed",
                "rpt_qc_flags": f"{type(exc).__name__}: {exc}",
            })
    columns = [
        "condition", "cell", "cu_visit", "cu_path", "cu_test_id", "cu_start_time",
        "cu_Q_Ah", "rpt_qc_status", "rpt_qc_flags",
    ]
    return pd.DataFrame(rows, columns=columns)


def _normalize_rpt_cache(cu_cache: pd.DataFrame) -> pd.DataFrame:
    cache = cu_cache.copy()
    rename = {
        "visit": "cu_visit",
        "path": "cu_path",
        "test_id": "cu_test_id",
        "file_start_time": "cu_start_time",
        "reference_capacity_Ah": "cu_Q_Ah",
    }
    for source, target in rename.items():
        if source in cache.columns and target not in cache.columns:
            cache = cache.rename(columns={source: target})
    if "rpt_qc_status" not in cache.columns:
        cache["rpt_qc_status"] = np.where(cache["cu_Q_Ah"].notna(), "valid", "missing")
    if "condition" not in cache.columns:
        cache["condition"] = pd.NA
    if "cu_start_time" in cache.columns:
        cache["cu_start_time"] = pd.to_datetime(cache["cu_start_time"], errors="coerce")
    return cache


def _validate_unique_rpts(cache: pd.DataFrame) -> None:
    """Allow visit-label collisions only when timestamps distinguish events."""
    valid = cache[cache["rpt_qc_status"] == "valid"]
    key_cols = ["cell", "cu_visit"]
    if valid["condition"].notna().any():
        key_cols.insert(0, "condition")
    duplicate = valid.duplicated(key_cols, keep=False)
    if not duplicate.any():
        return
    ambiguous: list[dict[str, object]] = []
    for _, group in valid.loc[duplicate].groupby(key_cols, dropna=False):
        timestamps = group["cu_start_time"] if "cu_start_time" in group else pd.Series(dtype="datetime64[ns]")
        paths = group["cu_path"] if "cu_path" in group else pd.Series(dtype="object")
        if timestamps.isna().any() or timestamps.duplicated().any() or paths.duplicated().any():
            columns = [*key_cols, "cu_path"]
            if "cu_start_time" in group:
                columns.append("cu_start_time")
            ambiguous.extend(group[columns].to_dict(orient="records"))
    if ambiguous:
        raise DuplicateRPTError(
            "same-visit RPT files lack unique chronology: "
            f"{ambiguous}"
        )


def add_target(df: pd.DataFrame, cu_cache: pd.DataFrame | None = None) -> pd.DataFrame:
    """Attach the next RPT only to the final complete cycle of each CYC file.

    Chronology uses file start timestamps when available. Visit order is an
    explicit fallback and is recorded in ``next_rpt_match_method``.
    """
    cache = _normalize_rpt_cache(
        cu_cache if cu_cache is not None else _cu_capacity_by_visit(load_inventory())
    )
    _validate_unique_rpts(cache)

    cfg = load_config()
    initial_visit = int(cfg.get("targets", {}).get("next_rpt", {}).get("initial_visit", 0))

    out = df.copy()
    for column in (
        "initial_rpt_Q_Ah", "previous_rpt_Q_Ah", "previous_rpt_SOH_pct",
        TARGET, SOH_TARGET, DELTA_Q_TARGET, DELTA_SOH_TARGET,
    ):
        out[column] = np.nan
    out["initial_rpt_path"] = pd.NA
    out["initial_rpt_visit"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    out["initial_rpt_match_method"] = pd.NA
    out["previous_rpt_path"] = pd.NA
    out["previous_rpt_visit"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    out["previous_rpt_match_method"] = pd.NA
    out["next_rpt_path"] = pd.NA
    out["next_rpt_visit"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    out["next_rpt_match_method"] = pd.NA
    out["next_rpt_horizon_days"] = np.nan
    out["is_prediction_anchor"] = False
    out["anchor_id"] = pd.NA
    out["next_rpt_target_status"] = "not_anchor"
    out["next_rpt_soh_status"] = "not_anchor"
    out["next_rpt_delta_status"] = "not_anchor"
    out["delta_next_rpt_Q_status"] = "not_anchor"
    out["delta_next_rpt_SOH_status"] = "not_anchor"
    out["target_unavailable_reason"] = "not_final_complete_cycle"
    if cache.empty:
        return out

    valid = cache[cache["rpt_qc_status"] == "valid"].copy()
    if valid.empty:
        return out
    if "file_start_time" in out.columns:
        out["file_start_time"] = pd.to_datetime(out["file_start_time"], errors="coerce")

    grouping = [column for column in ["condition", "cell", "path"] if column in out.columns]
    if "path" not in grouping:
        grouping = [column for column in ["condition", "cell", "visit"] if column in out.columns]
    order_column = "global_cycle" if "global_cycle" in out.columns else "cycle_in_file"

    for _, cycle_file in out.groupby(grouping, dropna=False, sort=False):
        eligible = cycle_file
        if "cycle_complete" in eligible.columns:
            eligible = eligible[eligible["cycle_complete"].fillna(False)]
        if eligible.empty:
            continue
        anchor_index = eligible[order_column].astype(float).idxmax()
        anchor = out.loc[anchor_index]
        out.at[anchor_index, "is_prediction_anchor"] = True
        anchor_key = anchor.get("file_id")
        if pd.isna(anchor_key):
            anchor_key = anchor.get("path")
        if pd.isna(anchor_key):
            anchor_key = str(anchor_index)
        out.at[anchor_index, "anchor_id"] = f"{anchor_key}::{anchor.get(order_column)}"
        out.at[anchor_index, "next_rpt_target_status"] = "unavailable"
        out.at[anchor_index, "next_rpt_soh_status"] = "unavailable"
        out.at[anchor_index, "next_rpt_delta_status"] = "unavailable"
        out.at[anchor_index, "delta_next_rpt_Q_status"] = "unavailable"
        out.at[anchor_index, "delta_next_rpt_SOH_status"] = "unavailable"
        out.at[anchor_index, "target_unavailable_reason"] = "no_valid_rpt_for_cell"
        candidates = valid[valid["cell"] == anchor["cell"]].copy()
        if "condition" in out.columns and valid["condition"].notna().any():
            candidates = candidates[candidates["condition"] == anchor["condition"]]
        if candidates.empty:
            continue

        candidates = candidates.sort_values(
            ["cu_start_time", "cu_test_id"], na_position="last"
        )
        initial_candidates = candidates[candidates["cu_visit"] == initial_visit]
        initial_rpt = initial_candidates.iloc[0] if not initial_candidates.empty else None
        initial_q = np.nan

        file_start_time = anchor.get("file_start_time", pd.NaT)
        cycle_end_s = anchor.get("cycle_end_time_s", np.nan)
        if pd.notna(file_start_time) and pd.notna(cycle_end_s):
            anchor_time = file_start_time + pd.to_timedelta(float(cycle_end_s), unit="s")
            time_method = "cycle_end_time"
        else:
            anchor_time = file_start_time
            time_method = "file_start_time_fallback"
        if initial_rpt is not None:
            initial_time = initial_rpt.get("cu_start_time", pd.NaT)
            if pd.notna(anchor_time) and pd.notna(initial_time):
                initial_is_causal = initial_time < anchor_time
                initial_method = "timestamp"
            else:
                initial_is_causal = int(initial_rpt["cu_visit"]) < int(anchor["visit"])
                initial_method = "visit_fallback_missing_timestamp"
            if initial_is_causal:
                initial_q = float(initial_rpt["cu_Q_Ah"])
                out.at[anchor_index, "initial_rpt_Q_Ah"] = initial_q
                out.at[anchor_index, "initial_rpt_path"] = initial_rpt.get("cu_path")
                out.at[anchor_index, "initial_rpt_visit"] = int(initial_rpt["cu_visit"])
                out.at[anchor_index, "initial_rpt_match_method"] = initial_method
        timed_candidates = (
            candidates[candidates["cu_start_time"].notna()]
            if "cu_start_time" in candidates.columns else candidates.iloc[0:0]
        )
        visit_future = candidates[candidates["cu_visit"] > int(anchor["visit"])].sort_values(
            ["cu_visit", "cu_test_id"], na_position="last"
        )
        visit_previous = candidates[candidates["cu_visit"] < int(anchor["visit"])].sort_values(
            ["cu_visit", "cu_test_id"], na_position="last"
        )
        previous: pd.Series | None = None
        previous_method: str | None = None
        if pd.notna(anchor_time) and not timed_candidates.empty:
            timed_previous = timed_candidates[
                timed_candidates["cu_start_time"] < anchor_time
            ].sort_values(["cu_start_time", "cu_test_id"], na_position="last")
            if not timed_previous.empty:
                previous = timed_previous.iloc[-1]
                previous_method = "timestamp"
        if not visit_previous.empty:
            visit_candidate = visit_previous.iloc[-1]
            visit_candidate_time = visit_candidate.get("cu_start_time", pd.NaT)
            visit_candidate_is_causal = (
                pd.isna(anchor_time) or pd.isna(visit_candidate_time)
                or visit_candidate_time < anchor_time
            )
            if visit_candidate_is_causal and (
                previous is None
                or (
                    pd.isna(visit_candidate_time)
                    and int(visit_candidate["cu_visit"]) >= int(previous["cu_visit"])
                )
            ):
                previous = visit_candidate
                previous_method = "visit_fallback_missing_timestamp"
        if previous is not None:
            previous_q = float(previous["cu_Q_Ah"])
            out.at[anchor_index, "previous_rpt_Q_Ah"] = previous_q
            out.at[anchor_index, "previous_rpt_path"] = previous.get("cu_path")
            out.at[anchor_index, "previous_rpt_visit"] = int(previous["cu_visit"])
            out.at[anchor_index, "previous_rpt_match_method"] = previous_method
            if np.isfinite(initial_q) and initial_q > 0:
                out.at[anchor_index, "previous_rpt_SOH_pct"] = 100.0 * previous_q / initial_q
        if pd.notna(anchor_time) and not timed_candidates.empty:
            future = timed_candidates[timed_candidates["cu_start_time"] > anchor_time].sort_values(
                ["cu_start_time", "cu_test_id"], na_position="last"
            )
            method = time_method
            untimed_future = visit_future[visit_future["cu_start_time"].isna()]
            if not untimed_future.empty and (
                future.empty or int(untimed_future.iloc[0]["cu_visit"]) < int(future.iloc[0]["cu_visit"])
            ):
                future = untimed_future
                method = "visit_fallback_missing_timestamp"
        else:
            future = visit_future
            method = "visit_fallback_missing_timestamp"
        if future.empty:
            out.at[anchor_index, "target_unavailable_reason"] = "no_future_valid_rpt"
            continue
        next_rpt = future.iloc[0]
        next_q = float(next_rpt["cu_Q_Ah"])
        out.at[anchor_index, TARGET] = next_q
        out.at[anchor_index, "next_rpt_path"] = next_rpt.get("cu_path")
        out.at[anchor_index, "next_rpt_visit"] = int(next_rpt["cu_visit"])
        out.at[anchor_index, "next_rpt_match_method"] = method
        out.at[anchor_index, "next_rpt_target_status"] = "available"
        out.at[anchor_index, "target_unavailable_reason"] = pd.NA
        if np.isfinite(initial_q) and initial_q > 0:
            next_soh = 100.0 * next_q / initial_q
            out.at[anchor_index, SOH_TARGET] = next_soh
            out.at[anchor_index, "next_rpt_soh_status"] = "available"
        else:
            next_soh = np.nan
            out.at[anchor_index, "next_rpt_soh_status"] = "missing_initial_rpt"
        if previous is not None:
            previous_q = float(previous["cu_Q_Ah"])
            out.at[anchor_index, DELTA_Q_TARGET] = next_q - previous_q
            out.at[anchor_index, "delta_next_rpt_Q_status"] = "available"
            if np.isfinite(next_soh) and np.isfinite(out.at[anchor_index, "previous_rpt_SOH_pct"]):
                out.at[anchor_index, DELTA_SOH_TARGET] = (
                    next_soh - float(out.at[anchor_index, "previous_rpt_SOH_pct"])
                )
            delta_soh_available = np.isfinite(out.at[anchor_index, DELTA_SOH_TARGET])
            out.at[anchor_index, "delta_next_rpt_SOH_status"] = (
                "available" if delta_soh_available else "missing_initial_rpt"
            )
            out.at[anchor_index, "next_rpt_delta_status"] = (
                "available" if delta_soh_available else "partial_capacity_only"
            )
        else:
            out.at[anchor_index, "next_rpt_delta_status"] = "missing_previous_rpt"
            out.at[anchor_index, "delta_next_rpt_Q_status"] = "missing_previous_rpt"
            out.at[anchor_index, "delta_next_rpt_SOH_status"] = "missing_previous_rpt"
        if method in {"cycle_end_time", "file_start_time_fallback"}:
            out.at[anchor_index, "next_rpt_horizon_days"] = (
                next_rpt["cu_start_time"] - anchor_time
            ).total_seconds() / 86400.0
    return _drop_non_adjacent_blocks(out)


def _drop_non_adjacent_blocks(out: pd.DataFrame) -> pd.DataFrame:
    """Retain one anchor per (cell, next RPT): the block adjacent to the target.

    A cell may run several CYC blocks with no RPT between them (observed for
    ``S2000FM42G``: CYC V03, V04, V05 all sit between CU V02 and CU V06). Every
    such block is an anchor pointing at the same next RPT, which gives duplicate
    targets on non-independent rows, and for the earlier blocks the features
    cover only part of the previous-RPT-to-next-RPT interval while the stress
    delta spans all of it. Only the last block before the target satisfies the
    contract, so the earlier ones are demoted to unavailable rather than
    silently regressed on.
    """
    targeted = out["next_rpt_path"].notna()
    if not targeted.any():
        return out
    # Closest to the target in time wins; visit order breaks ties when the
    # horizon is unavailable (visit-fallback matches carry no horizon).
    order = out.loc[targeted].sort_values(
        ["next_rpt_horizon_days", "visit"],
        ascending=[True, False],
        na_position="last",
        kind="mergesort",
    )
    keep = order.groupby(["condition", "cell", "next_rpt_path"], sort=False).head(1).index
    superseded = out.index[targeted].difference(keep)
    if len(superseded) == 0:
        return out
    for column in (TARGET, SOH_TARGET, DELTA_Q_TARGET, DELTA_SOH_TARGET):
        out.loc[superseded, column] = np.nan
    out.loc[superseded, "next_rpt_target_status"] = "unavailable"
    out.loc[superseded, "target_unavailable_reason"] = "superseded_by_adjacent_block"
    return out
