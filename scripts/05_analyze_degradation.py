"""Build RPT-SOH degradation and explicitly extrapolated time-to-80% tables."""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import pandas as pd

from src.evaluation.degradation import (
    build_rpt_longitudinal,
    cell_degradation_summary,
    condition_degradation_summary,
)
from src.io.loaders import load_config, load_cycle_metrics, load_rpt_measurements


def main() -> None:
    cfg = load_config()
    artifacts = Path(cfg["paths"]["artifacts"])
    features = artifacts / "features"
    results = artifacts / "results"
    features.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    longitudinal = build_rpt_longitudinal(
        load_rpt_measurements(), load_cycle_metrics(),
        initial_visit=int(cfg["targets"]["next_rpt"].get("initial_visit", 0)),
        nominal_capacity_Ah=float(cfg["protocol"]["nominal_capacity_Ah"]),
    )
    cells = cell_degradation_summary(
        longitudinal,
        threshold_pct=float(cfg["targets"]["capacity_eol"]["soh_threshold_pct"]),
    )
    conditions = condition_degradation_summary(cells)
    longitudinal.to_parquet(features / "rpt_longitudinal.parquet", index=False)
    cells.to_csv(results / "rpt_cell_degradation.csv", index=False)
    conditions.to_csv(results / "rpt_condition_degradation.csv", index=False)
    print(
        f"[degradation] RPT rows={len(longitudinal)} cells={len(cells)} "
        f"observed_80pct_events={int(cells['observed_80pct_event'].sum())}"
    )


if __name__ == "__main__":
    main()
