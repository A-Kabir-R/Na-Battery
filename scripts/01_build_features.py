"""Build feature parquets and cache them under artifacts/features/.

``unified.parquet`` is the table every retained model trains on.

Usage
-----
    python3 scripts/01_build_features.py
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
    load_rpt_measurements,
)
from src.preprocessing.unified import build_unified, write_unified


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


def main() -> None:
    cfg = load_config()
    out = Path(cfg["paths"]["artifacts"]) / "features"
    out.mkdir(parents=True, exist_ok=True)

    stages = ["cu_cache", "unified"]
    pbar = tqdm(stages, desc="[build] stages", unit="stage")

    pbar.set_postfix_str("cu_cache")
    t0 = time.time()
    cu_cache = _enrich_rpt_measurements(
        load_rpt_measurements(),
        int(cfg.get("targets", {}).get("next_rpt", {}).get("initial_visit", 0)),
    )
    cu_cache.to_parquet(out / "cu_capacity.parquet")
    tqdm.write(f"[build] cu_cache: {cu_cache.shape} ({time.time()-t0:.1f}s)")
    pbar.update(1)

    pbar.set_postfix_str("unified")
    _build_unified_stage(out, load_cycle_metrics(), cu_cache)
    pbar.update(1)
    pbar.close()
    print("[build] done.")


if __name__ == "__main__":
    main()
