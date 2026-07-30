"""Build feature parquets and cache them under artifacts/features/.

``unified.parquet`` is the table every retained model trains on. P1/P2/P3 are
still built for traceability and for the reported feature-family ablation, but
they are not used for final training.

Usage
-----
    python3 scripts/01_build_features.py              # everything
    python3 scripts/01_build_features.py --unified-only
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import pandas as pd
from tqdm.auto import tqdm

from src.io.loaders import (
    load_config,
    load_cycle_metrics,
    load_inventory,
    load_rpt_measurements,
)
from src.preprocessing.p1_aggregate import build_p1
from src.preprocessing.p2_rolling_delta import build_p2
from src.preprocessing.p3_waveform import build_p3
from src.preprocessing.unified import build_unified, write_unified
from src.targets.failure_label import add_target as add_failure
from src.targets.next_rpt import add_target as add_next_rpt
from src.targets.rul import add_target as add_rul


# Columns that are never features: identifiers, metadata, targets, and
# physics-coordinate columns that are computed by the PINN trainer at
# training time rather than stored in the feature parquets.
_NON_FEATURE_COLUMNS: frozenset[str] = frozenset({
    "condition", "cell", "file_id", "visit", "test_id", "cycle_in_file",
    "anchor_id", "is_prediction_anchor", "split", "rpt_qc_status",
    "target_unavailable_reason", "file_start_time", "path",
    # Target columns — must never appear as input features
    "next_rpt_Q_Ah", "next_rpt_SOH_pct", "delta_next_rpt_Q_Ah", "delta_next_rpt_SOH_pct",
    "next_rpt_visit", "next_rpt_horizon_days",
    "rul", "time_to_failure", "is_failed", "failure_type",
    # RPT-derived columns that are not features but ride along in the frame
    "reference_capacity_Ah", "rpt_SOH_pct", "initial_rpt_Q_Ah",
    "previous_rpt_Q_Ah", "delta_rpt_Q_Ah", "delta_rpt_SOH_pct",
    # PINN physics-coordinate columns (computed at training time)
    "stress_current", "stress_next", "stress_delta", "u_current", "u_true_next",
    "q0_column", "q_current_column",
})


def _write_provenance(df: pd.DataFrame, path: Path, preprocessing: str) -> None:
    """Write a provenance CSV with one row per numeric feature column in df.

    Attributes are derived conservatively from column naming conventions.
    All P1/P2/P3 features are marked uses_target_rpt=False and
    uses_future_rows=False — they must be causal quantities computed only
    from cycling data available at the anchor cycle.
    """
    rows = []
    for col in df.columns:
        if col in _NON_FEATURE_COLUMNS:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        rolling_direction = ""
        shift_count = 0
        col_lower = col.lower()
        if any(kw in col_lower for kw in ("rolling", "window", "lag", "shift")):
            rolling_direction = "backward"
            # Extract shift count from trailing integer, e.g. "_w5" -> 5
            import re as _re
            m = _re.search(r"_w(\d+)$", col)
            if m:
                shift_count = int(m.group(1))
        rows.append({
            "feature": col,
            "source_columns": col,
            "preprocessing": preprocessing,
            "uses_target_rpt": False,
            "uses_future_rows": False,
            "rolling_direction": rolling_direction,
            "shift_count": shift_count,
            "known_at_anchor": True,
            "planned_at_inference": True,
            "allowed": True,
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def _attach_targets(df: pd.DataFrame, cu_cache: pd.DataFrame) -> pd.DataFrame:
    df = add_rul(df)
    df = add_next_rpt(df, cu_cache=cu_cache)
    df = add_failure(df)
    return df


def _enrich_rpt_measurements(frame: pd.DataFrame, initial_visit: int) -> pd.DataFrame:
    out = frame.copy()
    out["file_start_time"] = pd.to_datetime(out["file_start_time"], errors="coerce")
    out = out.sort_values(["condition", "cell", "file_start_time", "test_id"]).reset_index(drop=True)
    valid = out["rpt_qc_status"].eq("valid")
    baseline = (
        out[valid & out["visit"].eq(initial_visit)]
        .groupby(["condition", "cell"])["reference_capacity_Ah"]
        .first()
        .rename("initial_rpt_Q_Ah")
    )
    out = out.join(baseline, on=["condition", "cell"])
    out["rpt_SOH_pct"] = float("nan")
    has_baseline = valid & out["initial_rpt_Q_Ah"].gt(0)
    out.loc[has_baseline, "rpt_SOH_pct"] = (
        100.0 * out.loc[has_baseline, "reference_capacity_Ah"]
        / out.loc[has_baseline, "initial_rpt_Q_Ah"]
    )
    out["previous_rpt_Q_Ah"] = float("nan")
    out["delta_rpt_Q_Ah"] = float("nan")
    out["delta_rpt_SOH_pct"] = float("nan")
    valid_rows = out[valid].copy()
    valid_rows["previous_rpt_Q_Ah"] = valid_rows.groupby(
        ["condition", "cell"]
    )["reference_capacity_Ah"].shift(1)
    valid_rows["delta_rpt_Q_Ah"] = (
        valid_rows["reference_capacity_Ah"] - valid_rows["previous_rpt_Q_Ah"]
    )
    valid_rows["delta_rpt_SOH_pct"] = valid_rows.groupby(
        ["condition", "cell"]
    )["rpt_SOH_pct"].diff()
    out.loc[valid_rows.index, [
        "previous_rpt_Q_Ah", "delta_rpt_Q_Ah", "delta_rpt_SOH_pct",
    ]] = valid_rows[[
        "previous_rpt_Q_Ah", "delta_rpt_Q_Ah", "delta_rpt_SOH_pct",
    ]]
    return out


def _target_eligibility(inventory: pd.DataFrame, p2: pd.DataFrame,
                        cu_cache: pd.DataFrame) -> pd.DataFrame:
    cells = inventory[["condition", "cell"]].drop_duplicates().copy()
    cyc = inventory[inventory["test_type"] == "CYC"].groupby(["condition", "cell"]).agg(
        cycling_file_count=("file_id", "size"),
        parseable_cycling_file_count=("parse_status", lambda values: int((values == "valid").sum())),
    ).reset_index()
    rpt = cu_cache.groupby(["condition", "cell"]).agg(
        rpt_file_count=("file_id", "size"),
        valid_rpt_count=("rpt_qc_status", lambda values: int((values == "valid").sum())),
        ambiguous_rpt_count=("rpt_qc_status", lambda values: int((values == "ambiguous").sum())),
        missing_rpt_count=("rpt_qc_status", lambda values: int((values == "missing").sum())),
    ).reset_index()
    anchors = p2[p2["is_prediction_anchor"].fillna(False)].groupby(
        ["condition", "cell"]
    ).agg(
        anchor_count=("anchor_id", "size"),
        eligible_next_rpt_count=("next_rpt_Q_Ah", "count"),
        eligible_next_rpt_soh_count=("next_rpt_SOH_pct", "count"),
        target_unavailable_reason=(
            "target_unavailable_reason",
            lambda values: ";".join(sorted({str(value) for value in values.dropna()})),
        ),
    ).reset_index()
    result = cells.merge(cyc, on=["condition", "cell"], how="left")
    result = result.merge(rpt, on=["condition", "cell"], how="left")
    result = result.merge(anchors, on=["condition", "cell"], how="left")
    count_columns = [column for column in result if column.endswith("_count")]
    result[count_columns] = result[count_columns].fillna(0).astype(int)
    result["supervised_target_available"] = result["eligible_next_rpt_count"] > 0
    no_cycles = result["parseable_cycling_file_count"].eq(0)
    no_anchor = result["anchor_count"].eq(0) & ~no_cycles
    result.loc[no_cycles, "target_unavailable_reason"] = "cycling_parse_failed"
    result.loc[no_anchor, "target_unavailable_reason"] = "no_complete_cycle_anchor"
    result.loc[result["supervised_target_available"], "target_unavailable_reason"] = pd.NA
    return result.sort_values(["condition", "cell"]).reset_index(drop=True)


def _cycling_file_eligibility(inventory: pd.DataFrame, p2: pd.DataFrame) -> pd.DataFrame:
    files = inventory[inventory["test_type"].eq("CYC")][[
        "file_id", "condition", "cell", "visit", "relative_path", "parse_status", "qc_status",
    ]].copy()
    anchors = p2[p2["is_prediction_anchor"].fillna(False)][[
        "file_id", "anchor_id", "next_rpt_Q_Ah", "next_rpt_SOH_pct",
        "target_unavailable_reason",
    ]].drop_duplicates("file_id")
    result = files.merge(anchors, on="file_id", how="left", validate="one_to_one")
    result["has_complete_cycle_anchor"] = result["anchor_id"].notna()
    result["supervised_target_available"] = result["next_rpt_Q_Ah"].notna()
    result.loc[result["parse_status"].ne("valid"), "target_unavailable_reason"] = (
        "cycling_parse_failed"
    )
    result.loc[
        result["parse_status"].eq("valid") & ~result["has_complete_cycle_anchor"],
        "target_unavailable_reason",
    ] = "no_complete_cycle_anchor"
    result.loc[result["supervised_target_available"], "target_unavailable_reason"] = pd.NA
    return result.sort_values(["condition", "cell", "visit"]).reset_index(drop=True)


def _build_unified_stage(out: Path, cycle_metrics: pd.DataFrame,
                         cu_cache: pd.DataFrame) -> None:
    t0 = time.time()
    result = build_unified(cycle_metrics, cu_cache)
    write_unified(result, out)
    manifest = result["manifest"]
    tqdm.write(
        f"[build] unified: {result['unified'].shape} "
        f"({manifest['n_model_features']} model features, "
        f"{manifest['n_anchors']} anchors, {manifest['n_cells']} cells) "
        f"({time.time()-t0:.1f}s)"
    )
    coverage = result["coverage"]
    weak = coverage[
        coverage["scope"].eq("all") & coverage["coverage"].lt(0.95)
    ].sort_values("coverage")
    if not weak.empty:
        tqdm.write("[build] unified features with <95% coverage:")
        for row in weak.itertuples(index=False):
            tqdm.write(f"[build]     {row.feature:<42} {row.coverage:6.1%}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--unified-only", action="store_true",
        help="Build only cu_cache and unified.parquet (skip P1/P2/P3).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_config()
    out = Path(cfg["paths"]["artifacts"]) / "features"
    out.mkdir(parents=True, exist_ok=True)

    stages = (["cu_cache", "unified"] if args.unified_only
              else ["cu_cache", "P1", "P2", "P3", "unified"])
    pbar = tqdm(stages, desc="[build] stages", unit="stage")

    # -------- CU cache
    pbar.set_postfix_str("cu_cache")
    t0 = time.time()
    cu_cache = _enrich_rpt_measurements(
        load_rpt_measurements(),
        int(cfg.get("targets", {}).get("next_rpt", {}).get("initial_visit", 0)),
    )
    cu_cache.to_parquet(out / "cu_capacity.parquet")
    tqdm.write(f"[build] cu_cache: {cu_cache.shape} ({time.time()-t0:.1f}s)")
    pbar.update(1)

    if args.unified_only:
        pbar.set_postfix_str("unified")
        _build_unified_stage(out, load_cycle_metrics(), cu_cache)
        pbar.update(1)
        pbar.close()
        print("[build] done (unified only).")
        return

    # -------- P1
    pbar.set_postfix_str("P1")
    t0 = time.time()
    cm = load_cycle_metrics()
    p1 = build_p1(cm)
    p1 = _attach_targets(p1, cu_cache)
    p1.to_parquet(out / "p1.parquet")
    _write_provenance(p1, out / "p1.provenance.csv", "P1")
    tqdm.write(f"[build] P1: {p1.shape} ({time.time()-t0:.1f}s)")
    pbar.update(1)

    # -------- P2
    pbar.set_postfix_str("P2")
    t0 = time.time()
    p2 = build_p2(cm)
    p2 = _attach_targets(p2, cu_cache)
    p2.to_parquet(out / "p2.parquet")
    _write_provenance(p2, out / "p2.provenance.csv", "P2")
    anchors = p2[p2.get("is_prediction_anchor", False)].copy()
    anchors.to_parquet(out / "anchor_samples.parquet", index=False)
    anchors[anchors["next_rpt_Q_Ah"].notna()].to_parquet(
        out / "eligible_anchor_samples.parquet", index=False
    )
    eligibility = _target_eligibility(load_inventory(), p2, cu_cache)
    eligibility.to_parquet(out / "target_eligibility.parquet", index=False)
    eligibility.to_csv(out / "target_eligibility.csv", index=False)
    file_eligibility = _cycling_file_eligibility(load_inventory(), p2)
    file_eligibility.to_parquet(out / "cycling_file_target_eligibility.parquet", index=False)
    file_eligibility.to_csv(out / "cycling_file_target_eligibility.csv", index=False)
    tqdm.write(f"[build] P2: {p2.shape} ({time.time()-t0:.1f}s)")
    pbar.update(1)

    # -------- P3: waveform features from raw .ird, joined onto P2 base
    pbar.set_postfix_str("P3 (raw .ird)")
    t0 = time.time()
    wf = build_p3()
    wf.to_parquet(out / "p3_waveform_only.parquet")
    tqdm.write(f"[build] P3 waveform rows: {wf.shape} ({time.time()-t0:.1f}s)")

    merge_keys = ["condition", "cell", "file_id", "cycle_in_file"]
    if wf.empty or not all(k in wf.columns for k in merge_keys):
        raise RuntimeError("P3 waveform extraction returned no mergeable rows")
    if wf.duplicated(merge_keys).any():
        raise ValueError(f"P3 waveform keys are not unique: {merge_keys}")
    p3 = p2.merge(
        wf.drop(columns=[column for column in ("visit", "path") if column in wf.columns]),
        on=merge_keys,
        how="left",
        validate="one_to_one",
    )
    waveform_columns = [column for column in wf.columns if column.startswith("wf_")]
    matched = p3[waveform_columns].notna().any(axis=1)
    matched_fraction = float(matched.mean()) if len(matched) else 0.0
    required_fraction = float(cfg["preprocessing"]["minimum_p3_cycle_coverage"])
    if matched_fraction < required_fraction:
        raise RuntimeError(
            f"P3 merged cycle coverage {matched_fraction:.3f} is below {required_fraction:.3f}"
        )
    p3.to_parquet(out / "p3.parquet")
    _write_provenance(p3, out / "p3.provenance.csv", "P3")
    tqdm.write(f"[build] P3 final: {p3.shape}")
    pbar.update(1)

    # -------- unified: the single table used for final training
    pbar.set_postfix_str("unified")
    _build_unified_stage(out, cm, cu_cache)
    pbar.update(1)
    pbar.close()

    print("[build] done.")


if __name__ == "__main__":
    main()
